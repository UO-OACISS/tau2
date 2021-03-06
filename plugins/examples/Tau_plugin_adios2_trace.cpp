/***************************************************************************
 * *   Plugin Testing
 * *   This plugin will provide iterative output of TAU profile data to an
 * *   ADIOS2 BP file.
 * *
 * *************************************************************************/

#if defined(TAU_ADIOS2)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#if TAU_MPI
#include "mpi.h"
#endif

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauMetaData.h>

#include <adios2.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <array>
#include <string>
#include <set>
#include <stack>
#include <list>
#include <iterator>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>

// signal handling to dump provenance
#include <signal.h>
#include <unistd.h>

// get program name
#include <errno.h>

#define CONVERT_TO_USEC 1.0/1000000.0 // hopefully the compiler will precompute this.
#define TAU_ADIOS2_PERIODIC_DEFAULT false
#define TAU_ADIOS2_PERIOD_DEFAULT 2000000 // microseconds
#define TAU_ADIOS2_USE_SELECTION_DEFAULT false
#define TAU_ADIOS2_FILENAME "tau-metrics"
#define TAU_ADIOS2_ONE_FILE_DEFAULT false
#define TAU_ADIOS2_ENGINE "BPFile"
#define TAU_ADIOS2_CONFIG_FILE_DEFAULT "./adios2.xml"

// This will enable some checking to make sure we don't have call stack violations.
// #define DO_VALIDATION

/* Some forward declarations that we need */
tau::Profiler *Tau_get_timer_at_stack_depth(int);
tau::Profiler *Tau_get_timer_at_stack_depth_task(int, int);
int Tau_plugin_adios2_function_exit(
    Tau_plugin_event_function_exit_data_t* data);
void Tau_dump_ADIOS2_metadata(adios2::IO& bpIO, int tid);

static bool enabled{false};
static bool done{false};
static bool _threaded{false};
static int global_comm_size = 1;
static int global_comm_rank = 0;
std::mutex _my_mutex;

/* These two variables are used to create something like a "counting semaphore".
 * When dumping, we want to prevent any threads from adding events to the buffers.
 * But when adding to the buffers, the threads don't need exclusive access.
 * Therefore, when "not dumping", the threads can increment the active_threads
 * counter, and the dumping thread will wait until those counters reach 0
 * before assuming exclusive access to the buffers.
 */
static atomic<bool> dumping{false};
static atomic<bool> in_async_write{false};
static atomic<size_t> active_threads{0};

std::condition_variable _my_cond;
std::thread * worker_thread = nullptr;

namespace tau_plugin {

char *_program_path()
{
#if defined(__APPLE__)
    return NULL;
#else
    char path[PATH_MAX] = {0};
    int len = readlink("/proc/self/exe", path, PATH_MAX);
    if (len == -1) {
        return NULL;
    }
    // the string from readlink isn't null terminated!
    path[len] = '\0';
    char *executable = (char*)calloc(PATH_MAX, sizeof(char));
    std::string tmp(path);
    size_t i = tmp.rfind('/', tmp.length());
    if (i != string::npos) {
        sprintf(executable, "%s", tmp.substr(i+1, ((tmp.length() - i) - 1)).c_str());
    }
    return executable;
#endif
}

class plugin_options {
    private:
        plugin_options(void) :
            env_periodic(TAU_ADIOS2_PERIODIC_DEFAULT),
            env_period(TAU_ADIOS2_PERIOD_DEFAULT),
            env_use_selection(TAU_ADIOS2_USE_SELECTION_DEFAULT),
            env_filename(TAU_ADIOS2_FILENAME),
            env_one_file(TAU_ADIOS2_ONE_FILE_DEFAULT),
            env_engine(TAU_ADIOS2_ENGINE),
            env_config_file("")
            {}
    public:
        int env_periodic;
        int env_period;
        bool env_use_selection;
        std::string env_filename;
        int env_one_file;
        std::string env_engine;
        std::string env_config_file;
        std::set<std::string> included_timers;
        std::set<std::string> excluded_timers;
        std::set<std::string> included_timers_with_wildcards;
        std::set<std::string> excluded_timers_with_wildcards;
        std::set<std::string> included_counters;
        std::set<std::string> excluded_counters;
        std::set<std::string> included_counters_with_wildcards;
        std::set<std::string> excluded_counters_with_wildcards;
        static plugin_options& thePluginOptions() {
            static plugin_options tpo;
            return tpo;
        }
};

inline plugin_options& thePluginOptions() {
    return plugin_options::thePluginOptions();
}

/* We need a way to prevent screwing up the timestamps when processing
 * events.  The first time we see a timer or counter, we create define
 * an attribute in ADIOS2, which IS INSTRUMENTED, so we can possibly get
 * a pair of start/stop events for IO::DefineAttribute in our trace.
 * To prevent that, ignore the plugin callbacks when we are processing
 * a plugin callback. */
bool& inPlugin() {
    static thread_local bool in(false);
    return in;
}

void Tau_ADIOS2_parse_environment_variables(void);
void Tau_ADIOS2_parse_selection_file(const char * filename);
bool Tau_ADIOS2_contains(std::set<std::string>& myset,
        const char * key, bool if_empty);

void * Tau_ADIOS2_thread_function(void);

/*********************************************************************
 * Parse a boolean value
 ********************************************************************/
static bool parse_bool(const char *str, bool default_value = false) {
  if (str == NULL) {
    return default_value;
  }
  static char strbuf[128];
  char *ptr = strbuf;
  strncpy(strbuf, str, 128);
  while (*ptr) {
    *ptr = tolower(*ptr);
    ptr++;
  }
  if (strcmp(strbuf, "yes") == 0  ||
      strcmp(strbuf, "true") == 0 ||
      strcmp(strbuf, "on") == 0 ||
      strcmp(strbuf, "1") == 0) {
    return true;
  } else {
    return false;
  }
}

/*********************************************************************
 * Parse an integer value
 ********************************************************************/
static int parse_int(const char *str, int default_value = 0) {
  if (str == NULL) {
    return default_value;
  }
  int tmp = atoi(str);
  if (tmp < 0) {
    return default_value;
  }
  return tmp;
}

void Tau_ADIOS2_parse_environment_variables(void) {
    char * tmp = NULL;
    tmp = getenv("TAU_ADIOS2_PERIODIC");
    if (parse_bool(tmp, TAU_ADIOS2_PERIODIC_DEFAULT)) {
      thePluginOptions().env_periodic = true;
      tmp = getenv("TAU_ADIOS2_PERIOD");
      thePluginOptions().env_period = parse_int(tmp, TAU_ADIOS2_PERIOD_DEFAULT);
    }
    tmp = getenv("TAU_ADIOS2_FILENAME");
    if (tmp != NULL) {
      thePluginOptions().env_filename = strdup(tmp);
    }
    tmp = getenv("TAU_ADIOS2_ONE_FILE");
    thePluginOptions().env_one_file = parse_bool(tmp);
    tmp = getenv("TAU_ADIOS2_ENGINE");
    if (tmp != NULL) {
      thePluginOptions().env_engine = strdup(tmp);
    }
    tmp = getenv("TAU_ADIOS2_CONFIG_FILE");
    if (tmp != NULL) {
      thePluginOptions().env_config_file = strdup(tmp);
    } else {
      if( access(TAU_ADIOS2_CONFIG_FILE_DEFAULT, F_OK ) != -1 ) {
          // file exists
          thePluginOptions().env_config_file = strdup(TAU_ADIOS2_CONFIG_FILE_DEFAULT);
      } else {
          // file doesn't exist
          thePluginOptions().env_config_file = "";
      }
    }
}

void Tau_ADIOS2_parse_selection_file(const char * filename) {
    std::ifstream file(filename);
    std::string str;
    bool including_timers = false;
    bool excluding_timers = false;
    bool including_counters = false;
    bool excluding_counters = false;
    thePluginOptions().env_use_selection = true;
    while (std::getline(file, str)) {
        // trim right whitespace
        str.erase(str.find_last_not_of(" \n\r\t")+1);
        // trim left whitespace
        str.erase(0, str.find_first_not_of(" \n\r\t"));
        // skip blank lines
        if (str.size() == 0) {
            continue;
        }
        // skip comments
        if (str.find("#", 0) == 0) {
            continue;
        }
        if (str.compare("BEGIN_INCLUDE_TIMERS") == 0) {
            including_timers = true;
        } else if (str.compare("END_INCLUDE_TIMERS") == 0) {
            including_timers = false;
        } else if (str.compare("BEGIN_EXCLUDE_TIMERS") == 0) {
            excluding_timers = true;
        } else if (str.compare("END_EXCLUDE_TIMERS") == 0) {
            excluding_timers = false;
        } else if (str.compare("BEGIN_INCLUDE_COUNTERS") == 0) {
            including_counters = true;
        } else if (str.compare("END_INCLUDE_COUNTERS") == 0) {
            including_counters = false;
        } else if (str.compare("BEGIN_EXCLUDE_COUNTERS") == 0) {
            excluding_counters = true;
        } else if (str.compare("END_EXCLUDE_COUNTERS") == 0) {
            excluding_counters = false;
        } else {
            if (including_timers) {
                if (str.find("#") == string::npos && str.find("?") == string::npos) {
                    thePluginOptions().included_timers.insert(str);
                } else {
                    thePluginOptions().included_timers_with_wildcards.insert(str);
                }
            } else if (excluding_timers) {
                if (str.find("#") == string::npos && str.find("?") == string::npos) {
                    thePluginOptions().excluded_timers.insert(str);
                } else {
                    thePluginOptions().excluded_timers_with_wildcards.insert(str);
                }
            } else if (including_counters) {
                if (str.find("#") == string::npos && str.find("?") == string::npos) {
                    thePluginOptions().included_counters.insert(str);
                } else {
                    thePluginOptions().included_counters_with_wildcards.insert(str);
                }
            } else if (excluding_counters) {
                if (str.find("#") == string::npos && str.find("?") == string::npos) {
                    thePluginOptions().excluded_counters.insert(str);
                } else {
                    thePluginOptions().excluded_counters_with_wildcards.insert(str);
                }
            } else {
                std::cerr << "Warning, selection outside of include/exclude section: "
                    << str << std::endl;
            }
        }
    }
}

typedef std::vector< std::pair< unsigned long,
        std::array<unsigned long, 5> > >
        timer_values_array_t;
typedef std::vector< std::pair< unsigned long,
        std::array<unsigned long, 5> > >
        counter_values_array_t;
typedef std::vector< std::pair< unsigned long,
        std::array<unsigned long, 7> > >
        comm_values_array_t;

class step_data_t {
public:
    step_data_t(int num_threads) : threads(num_threads),
        primer_stacks{nullptr}, step_of_events(nullptr) { }
    ~step_data_t() {
        for (int i=0 ; i < threads ; i++) {
            if (primer_stacks[i] != nullptr) {
                delete primer_stacks[i];
            }
        }
        if (step_of_events != nullptr) {
            delete step_of_events;
        }
    }
    int programs;
    int comm_ranks;
    int threads;
    int event_types;
    int timers;
    size_t num_timer_values;
    int counters;
    size_t num_counter_values;
    size_t num_comm_values;
    timer_values_array_t* primer_stacks[TAU_MAX_THREADS];
    std::vector<unsigned long>* step_of_events;
    std::vector<unsigned long> step_of_counters;
    std::vector<unsigned long> step_of_comms;
};

/* Class containing ADIOS archive info */
class adios {
    private:
        bool opened;
        adios2::Engine bpWriter;
        adios2::Variable<int> program_count;
        adios2::Variable<int> comm_size;
        adios2::Variable<int> thread_count;
        adios2::Variable<int> event_type_count;
        adios2::Variable<int> timer_count;
        adios2::Variable<size_t> timer_event_count;
        adios2::Variable<int> counter_count;
        adios2::Variable<size_t> counter_event_count;
        adios2::Variable<size_t> comm_count;
        adios2::Variable<unsigned long> event_timestamps;
        adios2::Variable<unsigned long> counter_values;
        adios2::Variable<unsigned long> comm_timestamps;
/* from SOS object */
        int prog_name_index;
        int comm_rank_index;
        int value_name_index;
        int value_index;
        int frame_index;
        int total_valid;
        int time_index;
        int max_threads;
    public:
        std::unordered_map<std::string, int> prog_names;
        std::unordered_map<std::string, int> value_names;
        std::unordered_map<std::string, int> metadata_keys;
        std::unordered_map<std::string, int> groups;
        std::unordered_map<std::string, int> timers;
        std::unordered_map<std::string, int> event_types;
        std::unordered_map<std::string, int> counters;
        adios2::ADIOS ad;
        adios2::IO _bpIO;
        adios() :
            opened(false),
            prog_name_index(-1),
            comm_rank_index(-1),
            value_name_index(-1),
            value_index(-1),
            frame_index(-1),
            time_index(-1),
            max_threads(0)
        {
            initialize();
            open();
            define_variables();
        };
        ~adios() {
            close();
        };
        void initialize();
        void define_variables();
        void open();
        void close();
        void define_attribute(const std::string& name,
            const std::string& value, adios2::IO& _bpIO, bool force);
        void write_variables(void);
/* from sos object */
        int check_prog_name(char * prog_name);
        int check_event_type(const std::string& event_type);
        int check_timer(const char * timer);
        int check_counter(const char * counter);
        // do we need this?
        void check_thread(int thread);
        int get_prog_count(void) { return prog_names.size(); }
        int get_value_name_count(void) { return value_names.size(); }
        int get_timer_count(void) { return timers.size(); }
        int get_event_type_count(void) { return event_types.size(); }
        int get_counter_count(void) { return counters.size(); }
        int get_thread_count(void) { return max_threads+1; } // zero-indexed.
        // This is an array (one per thread)
        // of vectors (one per timestamp)
        // of pairs (timestamp, values)...
        timer_values_array_t timer_values_array[TAU_MAX_THREADS];
        counter_values_array_t counter_values_array[TAU_MAX_THREADS];
        comm_values_array_t comm_values_array[TAU_MAX_THREADS];
        // for validation
#ifdef DO_VALIDATION
        std::stack<unsigned long> timer_stack[TAU_MAX_THREADS];
        std::stack<unsigned long> pre_timer_stack[TAU_MAX_THREADS];
#endif
        unsigned long previous_timestamp[TAU_MAX_THREADS];
};

void adios::initialize() {
#if TAU_MPI
    // Get the rank and size in the original communicator
    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm adios_comm;

    if (thePluginOptions().env_one_file) {
        PMPI_Comm_dup(MPI_COMM_WORLD, &adios_comm);
    } else {
        PMPI_Comm_dup(MPI_COMM_SELF, &adios_comm);
    }
    Tau_global_incr_insideTAU();
    if (thePluginOptions().env_config_file != "") {
        ad = adios2::ADIOS(thePluginOptions().env_config_file, adios_comm, true);
    } else {
        ad = adios2::ADIOS(adios_comm, true);
    }
#else
    /** ADIOS class factory of IO class objects, DebugON is recommended */
    if (thePluginOptions().env_config_file != "") {
        ad = adios2::ADIOS(thePluginOptions().env_config_file, true);
    } else {
        ad = adios2::ADIOS(true);
    }
#endif
    /*** IO class object: settings and factory of Settings: Variables,
     * Parameters, Transports, and Execution: Engines */
    _bpIO = ad.DeclareIO("TAU trace data");

    if (thePluginOptions().env_config_file == "") {
        // if not defined by user, we can change the default settings
        // BPFile is the default engine
        _bpIO.SetEngine(thePluginOptions().env_engine);
        // bpIO.SetParameters({{"num_threads", "2"}});
        // don't wait on readers to connect
        _bpIO.SetParameters({{"RendezvousReaderCount", "0"}});
        _bpIO.SetParameters({{"num_threads", "1"}});

        // ISO-POSIX file output is the default transport (called "File")
        // Passing parameters to the transport
        _bpIO.AddTransport("File", {{"Library", "POSIX"}});
    }
    Tau_global_decr_insideTAU();
}

void adios::define_variables(void) {
    program_count = _bpIO.DefineVariable<int>("program_count");
    comm_size = _bpIO.DefineVariable<int>("comm_rank_count");
    thread_count = _bpIO.DefineVariable<int>("thread_count");
    event_type_count = _bpIO.DefineVariable<int>("event_type_count");
    timer_count = _bpIO.DefineVariable<int>("timer_count");
    timer_event_count = _bpIO.DefineVariable<size_t>("timer_event_count");
    counter_count = _bpIO.DefineVariable<int>("counter_count");
    counter_event_count = _bpIO.DefineVariable<size_t>("counter_event_count");
    comm_count = _bpIO.DefineVariable<size_t>("comm_count");
    /* These are 2 dimensional variables, so they get special treatment */
    event_timestamps = _bpIO.DefineVariable<unsigned long>("event_timestamps", {1, 6}, {0, 0}, {1, 6});
    counter_values = _bpIO.DefineVariable<unsigned long>("counter_values", {1, 6}, {0, 0}, {1, 6});
    comm_timestamps = _bpIO.DefineVariable<unsigned long>("comm_timestamps", {1, 8}, {0, 0}, {1, 8});
}

/* Open the ADIOS archive */
void adios::open() {
    if (!opened) {
        Tau_global_incr_insideTAU();
        std::stringstream ss;
        ss << thePluginOptions().env_filename;
        char * program = _program_path();
        if (program != NULL) {
            ss << "-" << program;
            free(program);
        }
        if (!thePluginOptions().env_one_file) {
            ss << "-" << global_comm_rank;
        }
        ss << ".bp";
        //printf("Writing %s\n", ss.str().c_str());
        std::string tmp{ss.str()};
        bpWriter = _bpIO.Open(tmp, adios2::Mode::Write);
        opened = true;
        Tau_global_decr_insideTAU();
    }
}

/* Close the ADIOS archive */
void adios::close() {
    if (opened) {
        Tau_global_incr_insideTAU();
        bpWriter.Close();
        opened = false;
        enabled = false;
        Tau_global_decr_insideTAU();
    }
};

/* Define an attribute (TAU metadata) */
void adios::define_attribute(const std::string& name, const std::string& value, adios2::IO& bpIO, bool force) {
    static std::unordered_set<std::string> seen;
    if (seen.count(name) == 0 || force) {
        seen.insert(name);
        bpIO.DefineAttribute<std::string>(name, value);
    }
}

/* Write the arrays of timestamps and values for this step */
void adios::write_variables(void)
{
    int programs = get_prog_count();
    int comm_ranks = global_comm_size;
    int event_types = get_event_type_count();
    _my_mutex.lock();
    int threads = get_thread_count();
    int timers = get_timer_count();
    int counters = get_counter_count();
    _my_mutex.unlock();

    Tau_global_incr_insideTAU();
    bpWriter.BeginStep();

    /* There's a tiny chance of a race condition because we have to update
     * and check two variables.  So acquire the lock before waiting until it's
     * safe to proceed with the dump.
     */
    _my_mutex.lock();
    // set the dumping flag
    dumping = true;
    // wait for all active threads to finish doing what they do
    while(active_threads > 0) {}
    /* Release the lock, we've got control */
    _my_mutex.unlock();

    /* sort into one big vector from all threads */
    // make a list from the first thread of data - copying the data in!
    std::list<std::pair<unsigned long, std::array<unsigned long, 5> > >
        merged_timers(timer_values_array[0].begin(),
                      timer_values_array[0].end());
    // this clear will empty the vector and destroy the objects!
    timer_values_array[0].clear();
    // copy the current primer stack
    for (int t = 1 ; t < threads ; t++) {
        // is there any data on this thread?
        if (timer_values_array[t].size() == 0) { continue; }
        // make a list from the next thread of data - copying the data in!
        std::list<std::pair<unsigned long, std::array<unsigned long, 5> > >
            next_thread(timer_values_array[t].begin(),
                        timer_values_array[t].end());
        // this clear will empty the vector and destroy the objects!
        timer_values_array[t].clear();
        // start at the head of the list
        auto it = merged_timers.begin();
        // start at the head of the vector
        auto head = next_thread.begin();
        auto tail = next_thread.end();
        do {
            // if the next event on thread n is less than the current timestamp
            if (head->first < it->first) {
                merged_timers.insert(it, *head);
                head++;
            } else {
                it++;
                // if we're at the end of the list, append the rest of this thread
                if (it == merged_timers.end()) {
                    merged_timers.insert (it,head,tail);
                    break;
                }
            }
        } while (head != tail);
    }
    size_t num_timer_values = merged_timers.size();

    std::vector<unsigned long> all_timers(6,0);
    for (auto it = merged_timers.begin() ; it != merged_timers.end() ; it++) {
        all_timers.push_back(it->second[0]);
        all_timers.push_back(it->second[1]);
        all_timers.push_back(it->second[2]);
        all_timers.push_back(it->second[3]);
        all_timers.push_back(it->second[4]);
        all_timers.push_back(it->first);
#ifdef DO_VALIDATION
        if (it->second[3] == 0) {
            // on entry
            timer_stack[it->second[2]].push(it->second[4]);
        } else if (it->second[3] == 1) {
            // on exit
            if (timer_stack[it->second[2]].size() == 0) {
                TAU_VERBOSE("Writing: Stack violation.\n");
                TAU_VERBOSE("Writing: Stack for thread %lu is empty, timestamp %lu.\n",
                    it->second[2], it->first);
            } else {
                if (timer_stack[it->second[2]].top() != it->second[4]) {
                    TAU_VERBOSE("Writing: Stack violation.\n");
                    TAU_VERBOSE("Writing: thread %lu, %lu != %lu, timestamp %lu\n",
                        it->second[2], timer_stack[it->second[2]].top(),
                        it->second[4], it->first);
                }
                timer_stack[it->second[2]].pop();
            }
        }
#endif
    }

    //printf("%d %s %d\n", global_comm_rank, __func__, __LINE__); fflush(stdout);

    /* sort into one big vector from all threads */
    std::vector<std::pair<unsigned long, std::array<unsigned long, 5> > >
        merged_counters(counter_values_array[0]);
    for (int t = 1 ; t < threads ; t++) {
        merged_counters.insert(merged_counters.end(),
            counter_values_array[t].begin(),
            counter_values_array[t].end());
    }
    std::sort(merged_counters.begin(), merged_counters.end());
    size_t num_counter_values = merged_counters.size();

    std::vector<unsigned long> all_counters(6,0);
    for (auto it = merged_counters.begin() ;
         it != merged_counters.end() ; it++) {
        all_counters.push_back(it->second[0]);
        all_counters.push_back(it->second[1]);
        all_counters.push_back(it->second[2]);
        all_counters.push_back(it->second[3]);
        all_counters.push_back(it->second[4]);
        all_counters.push_back(it->first);
    }

    for (int t = 0 ; t < threads ; t++) {
        counter_values_array[t].clear();
    }

    /* sort into one big vector from all threads */
    std::vector<std::pair<unsigned long, std::array<unsigned long, 7> > >
        merged_comms(comm_values_array[0]);
    for (int t = 1 ; t < threads ; t++) {
        merged_comms.insert(merged_comms.end(),
            comm_values_array[t].begin(),
            comm_values_array[t].end());
    }
    std::sort(merged_comms.begin(), merged_comms.end());
    size_t num_comm_values = merged_comms.size();

    std::vector<unsigned long> all_comms(8,0);
    for (auto it = merged_comms.begin() ;
         it != merged_comms.end() ; it++) {
        all_comms.push_back(it->second[0]);
        all_comms.push_back(it->second[1]);
        all_comms.push_back(it->second[2]);
        all_comms.push_back(it->second[3]);
        all_comms.push_back(it->second[4]);
        all_comms.push_back(it->second[5]);
        all_comms.push_back(it->second[6]);
        all_comms.push_back(it->first);
    }

    for (int t = 0 ; t < threads ; t++) {
        comm_values_array[t].clear();
    }

    // Need to release the "dumping" flag so that ADIOS2 calls that
    // result in TAU timers won't deadlock when they enter this
    // plugin
    dumping = false;

    bpWriter.Put(program_count, &programs);
    bpWriter.Put(comm_size, &comm_ranks);
    bpWriter.Put(thread_count, &threads);
    bpWriter.Put(event_type_count, &event_types);
    bpWriter.Put(timer_count, &timers);
    bpWriter.Put(timer_event_count, &num_timer_values);
    bpWriter.Put(counter_count, &counters);
    bpWriter.Put(counter_event_count, &num_counter_values);
    bpWriter.Put(comm_count, &num_comm_values);

    if (num_timer_values > 0) {
        event_timestamps.SetShape({num_timer_values, 6});
        /* These dimensions need to change for 1-file case! */
        const adios2::Dims timer_start{0, 0};
        const adios2::Dims timer_count{static_cast<size_t>(num_timer_values), 6};
        const adios2::Box<adios2::Dims> timer_selection{timer_start, timer_count};
        event_timestamps.SetSelection(timer_selection);
        bpWriter.Put(event_timestamps, all_timers.data());
    }

    if (num_counter_values > 0) {
        counter_values.SetShape({num_counter_values, 6});
        /* These dimensions need to change for 1-file case! */
        const adios2::Dims counter_start{0, 0};
        const adios2::Dims counter_count{static_cast<size_t>(num_counter_values), 6};
        const adios2::Box<adios2::Dims> counter_selection{counter_start, counter_count};
        counter_values.SetSelection(counter_selection);
        bpWriter.Put(counter_values, all_counters.data());
    }

    if (num_comm_values > 0) {
        comm_timestamps.SetShape({num_comm_values, 8});
        /* These dimensions need to change for 1-file case! */
        const adios2::Dims comm_start{0, 0};
        const adios2::Dims comm_count{static_cast<size_t>(num_comm_values), 8};
        const adios2::Box<adios2::Dims> comm_selection{comm_start, comm_count};
        comm_timestamps.SetSelection(comm_selection);
        bpWriter.Put(comm_timestamps, all_comms.data());
    }

    bpWriter.EndStep();
    // if we aren't storing history, free the arrays now.  ADIOS should be
    // done with them.  Because they are stack variables, they'll go out
    // of scope.
    Tau_global_decr_insideTAU();
}

/* sos object methods */

    /* Keep a map of program names to indexes */
    int adios::check_prog_name(char * prog_name) {
        if (prog_names.count(prog_name) == 0) {
            std::stringstream ss;
            int num = prog_names.size();
            ss << "program_name " << num;
            prog_names[prog_name] = num;
            define_attribute(ss.str(), prog_name, _bpIO, false);
        }
        return prog_names[prog_name];
    }

    /* Keep a map of event types to indexes */
    int adios::check_event_type(const std::string& event_type) {
        if (event_types.count(event_type) == 0) {
            std::stringstream ss;
            int num = event_types.size();
            ss << "event_type " << num;
            event_types[event_type] = num;
            define_attribute(ss.str(), event_type, _bpIO, false);
        }
        return event_types[event_type];
    }

    /* Keep a map of timers to indexes */
    int adios::check_timer(const char * timer) {
        bool new_timer{false};
        std::string tmp(timer);
        if (timers.count(tmp) == 0) {
            _my_mutex.lock();
            // check to make sure another thread didn't create it already
            if (timers.count(tmp) == 0) {
                int num = timers.size();
                timers[tmp] = num;
                // printf("%d = %s\n", num, timer);
                new_timer = true;
            }
            _my_mutex.unlock();
            // Because ADIOS is instrumented with TAU calls, make sure the
            // lock is released before defining the attribute.
            if (new_timer) {
                std::stringstream ss;
                ss << "timer " << timers[tmp];
                define_attribute(ss.str(), tmp, _bpIO, false);
            }
        }
        return timers[tmp];
    }

    /* Keep a map of counters to indexes */
    int adios::check_counter(const char * counter) {
        bool new_counter{false};
        std::string tmp(counter);
        if (counters.count(tmp) == 0) {
            _my_mutex.lock();
            // check to make sure another thread didn't create it already
            if (counters.count(tmp) == 0) {
                int num = counters.size();
                counters[tmp] = num;
                new_counter = true;
            }
            _my_mutex.unlock();
            if (new_counter) {
                std::stringstream ss;
                ss << "counter " << counters[tmp];
                define_attribute(ss.str(), tmp, _bpIO, false);
            }
        }
        return counters[tmp];
    }

    /* Keep a map of max number of threads per process */
    void adios::check_thread(int thread) {
        if (thread > max_threads) {
            max_threads = thread;
        }
    }

/* Necessary to use const char * because UserEvents use TauSafeString objects,
 * not std::string. We use the "if_empty" parameter to tell us how to treat
 * an empty set.  For exclude lists, it's false, for include lists, it's true */
bool Tau_ADIOS2_contains(std::set<std::string>& myset,
        const char * key, bool if_empty) {
    // if the set has contents, and we are in the set, then return true.
    std::string _key(key);
    if (myset.size() == 0) {
        return if_empty;
    } else if (myset.find(_key) == myset.end()) {
        return false;
    }
    // otherwise, return false.
    return true;
}

}; // end namespace tau_plugin

tau_plugin::adios& my_adios() {
  static tau_plugin::adios _my_adios;
  return _my_adios;
}

int Tau_plugin_adios2_dump(Tau_plugin_event_dump_data_t* data) {
    if (!enabled) return 0;
    tau_plugin::inPlugin() = true;
    static int iter = 0;
    TAU_VERBOSE("%d TAU PLUGIN ADIOS2 Dump %d \n", global_comm_rank, iter); fflush(stdout);
    //Tau_pure_start(__func__);
    Tau_global_incr_insideTAU();
    my_adios().write_variables();
    Tau_global_decr_insideTAU();
    //Tau_pure_stop(__func__);
    TAU_VERBOSE("%d TAU PLUGIN ADIOS2 Dump exit %d\n", global_comm_rank, iter++); fflush(stdout);
    tau_plugin::inPlugin() = false;
    return 0;
}

void Tau_ADIOS2_stop_worker(void) {
    if (!enabled) return;
    if (!_threaded) return;
    done = true;
    if (tau_plugin::thePluginOptions().env_periodic && _threaded) {
        TAU_VERBOSE("TAU ADIOS2 thread joining...\n"); fflush(stderr);
        while(in_async_write) {/* wait for the async thread to finish writing, if necessary */}
        _my_cond.notify_all();
        if (worker_thread != nullptr && worker_thread->joinable()) {
            worker_thread->join();
        }
    }
    _threaded = false;
}

void * Tau_ADIOS2_thread_function(void) {
	Tau_register_thread();
    std::chrono::microseconds period(tau_plugin::thePluginOptions().env_period);
    while (!done) {
        in_async_write = false;
        {
            // scoped region for lock
            std::unique_lock<std::mutex> lk(_my_mutex);
            if (_my_cond.wait_for(lk, period, [] {return done == true;})) {
                // done executing
                break;
            }
        }
        in_async_write = true;
        // timeout, dump data
        TAU_VERBOSE("%d Sending data from TAU thread...\n", RtsLayer::myNode()); fflush(stderr);
        Tau_plugin_event_dump_data_t dummy_data;
        Tau_plugin_adios2_dump(&dummy_data);
        TAU_VERBOSE("%d Done.\n", RtsLayer::myNode()); fflush(stderr);
    }
    in_async_write = false;
	return(NULL);
}

void Tau_plugin_adios2_signal_handler(int signal) {
    printf("In Provenance signal handler\n");
    // the next time we write data, dump history first
}

extern "C" int Tau_is_thread_fake(int tid);

int do_teardown(bool pre) {
    /* only complete this function once! */
    static bool done_once = false;
    if (done_once) { return 0; }
    TAU_VERBOSE("%d:%d %s...\n", global_comm_rank, RtsLayer::myThread(), __func__); fflush(stdout);
    /* Don't do these instructions more than once */
    done_once = true;
    Tau_ADIOS2_stop_worker();
    /* Stop any outstanding timers on the stack(s) */
    if (pre) {
        Tau_plugin_event_function_exit_data_t exit_data;
        int tid = 0; // only do thread 0
        int depth = Tau_get_current_stack_depth(tid);
        for (int i = depth ; i > -1 ; i--) {
            tau::Profiler *profiler = Tau_get_timer_at_stack_depth(i);
            // not sure how this can happen
            if (profiler == NULL) { continue; }
            exit_data.timer_name = profiler->ThisFunction->GetName();
            // not sure how this can happen
            if (exit_data.timer_name == NULL) { continue; }
            exit_data.timer_group = profiler->ThisFunction->GetAllGroups();
            exit_data.tid = tid;
            double CurrentTime[TAU_MAX_COUNTERS] = { 0 };
            RtsLayer::getUSecD(tid, CurrentTime);
            exit_data.timestamp = (x_uint64)CurrentTime[0];    // USE COUNTER1 for tracing
            //printf("%d Stopping %s\n", tid, exit_data.timer_name);
            Tau_plugin_adios2_function_exit(&exit_data);
        }
    }
    /* write those last events... */
    Tau_plugin_event_dump_data_t dummy_data;
    Tau_plugin_adios2_dump(&dummy_data);
    /* Close ADIOS archive */
    my_adios().close();
#if TAU_MPI
    /* Don't let any processes exit early - it could terminate our processing. */
    TAU_VERBOSE("%d TAU ADIOS2 plugin Exiting\n", global_comm_rank);
    PMPI_Barrier(MPI_COMM_WORLD);
#endif
    return 0;
}

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_adios2_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    if (!enabled || data->tid != 0) return 0;
    TAU_VERBOSE("TAU PLUGIN ADIOS2 Pre-Finalize\n"); fflush(stdout);
    return do_teardown(false);
}

/* This happens from Profiler.cpp, when data is written out. */
int Tau_plugin_adios2_end_of_execution(Tau_plugin_event_end_of_execution_data_t* data) {
    if (!enabled || data->tid != 0) return 0;
    TAU_VERBOSE("TAU PLUGIN ADIOS2 Finalize\n"); fflush(stdout);
    int rc = do_teardown(false);
    return rc;
}

void Tau_dump_ADIOS2_metadata(adios2::IO& bpIO, int tid) {
    // ok to do this - it happens after TAU initialization is done
    //if (!enabled) return;
    tau_plugin::inPlugin() = true;
    //int tid = RtsLayer::myThread();
    int nodeid = TAU_PROFILE_GET_NODE();
    Tau_global_incr_insideTAU();
    for (MetaDataRepo::iterator it = Tau_metadata_getMetaData(tid).begin();
         it != Tau_metadata_getMetaData(tid).end(); it++) {
        // check for executable name
        if (strcmp(it->first.name, "Executable") == 0) {
            my_adios().check_prog_name(it->second->data.cval);
        }
        std::stringstream ss;
        ss << "MetaData:" << global_comm_rank << ":" << tid << ":" << it->first.name;
        switch(it->second->type) {
            case TAU_METADATA_TYPE_STRING:
                my_adios().define_attribute(ss.str(), std::string(it->second->data.cval), bpIO, true);
                /* If this is a ROCm queue, use the same metadata that the
                 * Chimbuko anomaly detection is expecting - the CUDA metadata */
                if (strcmp(it->first.name, "ROCM_GPU_ID") == 0) {
                    ss.str("");
                    ss << "MetaData:" << global_comm_rank << ":" << tid << ":CUDA Device";
                    my_adios().define_attribute(ss.str(), std::string(it->second->data.cval), bpIO, true);
                    ss.str();
                    ss << "MetaData:" << global_comm_rank << ":" << tid << ":CUDA Context";
                    my_adios().define_attribute(ss.str(), "0", bpIO, true);
                } else if (strcmp(it->first.name, "ROCM_QUEUE_ID") == 0) {
                    std::stringstream ss2;
                    ss << "MetaData:" << global_comm_rank << ":" << tid << ":CUDA Stream";
                    my_adios().define_attribute(ss.str(), std::string(it->second->data.cval), bpIO, true);
                }
                break;
            case TAU_METADATA_TYPE_INTEGER:
                my_adios().define_attribute(ss.str(), std::to_string(it->second->data.ival), bpIO, true);
                break;
            case TAU_METADATA_TYPE_DOUBLE:
                my_adios().define_attribute(ss.str(), std::to_string(it->second->data.dval), bpIO, true);
                break;
            case TAU_METADATA_TYPE_TRUE:
                my_adios().define_attribute(ss.str(), std::string("true"), bpIO, true);
                break;
            case TAU_METADATA_TYPE_FALSE:
                my_adios().define_attribute(ss.str(), std::string("false"), bpIO, true);
                break;
            case TAU_METADATA_TYPE_NULL:
                my_adios().define_attribute(ss.str(), std::string("(null)"), bpIO, true);
                break;
            default:
                break;
        }
    }
    Tau_global_decr_insideTAU();
    tau_plugin::inPlugin() = false;
}

/* This happens when a Metadata field is saved. */
int Tau_plugin_metadata_registration_complete_func(Tau_plugin_event_metadata_registration_data_t* data) {
    if (!enabled) return 0;
    tau_plugin::inPlugin() = true;
    Tau_global_incr_insideTAU();
    //fprintf(stdout, "TAU Metadata registration\n"); fflush(stdout);
    std::stringstream ss;
    ss << "MetaData:" << global_comm_rank << ":" << data->tid << ":" << data->name;
    switch(data->value->type) {
        case TAU_METADATA_TYPE_STRING:
            my_adios().define_attribute(ss.str(), std::string(data->value->data.cval), my_adios()._bpIO, false);
            /* If this is a ROCm queue, use the same metadata that the
             * Chimbuko anomaly detection is expecting - the CUDA metadata */
            if (strcmp(data->name, "ROCM_GPU_ID") == 0) {
                ss.str("");
                ss << "MetaData:" << global_comm_rank << ":" << data->tid << ":CUDA Device";
                my_adios().define_attribute(ss.str(), std::string(data->value->data.cval), my_adios()._bpIO, false);
                ss.str("");
                ss << "MetaData:" << global_comm_rank << ":" << data->tid << ":CUDA Context";
                my_adios().define_attribute(ss.str(), "0", my_adios()._bpIO, false);
            } else if (strcmp(data->name, "ROCM_QUEUE_ID") == 0) {
                ss.str("");
                ss << "MetaData:" << global_comm_rank << ":" << data->tid << ":CUDA Stream";
                my_adios().define_attribute(ss.str(), std::string(data->value->data.cval), my_adios()._bpIO, false);
            }
            break;
        case TAU_METADATA_TYPE_INTEGER:
            my_adios().define_attribute(ss.str(), std::to_string(data->value->data.ival), my_adios()._bpIO, false);
            break;
        case TAU_METADATA_TYPE_DOUBLE:
            my_adios().define_attribute(ss.str(), std::to_string(data->value->data.dval), my_adios()._bpIO, false);
            break;
        case TAU_METADATA_TYPE_TRUE:
            my_adios().define_attribute(ss.str(), std::string("true"), my_adios()._bpIO, false);
            break;
        case TAU_METADATA_TYPE_FALSE:
            my_adios().define_attribute(ss.str(), std::string("false"), my_adios()._bpIO, false);
            break;
        case TAU_METADATA_TYPE_NULL:
            my_adios().define_attribute(ss.str(), std::string("(null)"), my_adios()._bpIO, false);
            break;
        default:
            break;
    }
    Tau_global_decr_insideTAU();
    tau_plugin::inPlugin() = false;
    return 0;
}

/* This happens on MPI_Send events (and similar) */
int Tau_plugin_adios2_send(Tau_plugin_event_send_data_t* data) {
    if (!enabled) return 0;
    if (tau_plugin::inPlugin()) return 0;
    tau_plugin::inPlugin() = true;
    static std::string sendstr("SEND");
    int event_index = my_adios().check_event_type(sendstr);
    my_adios().check_thread(data->tid);
    std::array<unsigned long, 7> tmparray;
    tmparray[0] = 0UL;
    tmparray[1] = (unsigned long)(global_comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(event_index);
    tmparray[4] = (unsigned long)(data->message_tag);
    tmparray[5] = (unsigned long)(data->destination);
    tmparray[6] = (unsigned long)(data->bytes_sent);

    /* There's a tiny chance of a race condition because we have to update
     * and check two variables.  So acquire the lock before waiting until it's
     * safe to proceed with the dump.
     */
    _my_mutex.lock();
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    _my_mutex.unlock();

    unsigned long ts = my_adios().previous_timestamp[data->tid] >
        data->timestamp ? my_adios().previous_timestamp[data->tid] :
            data->timestamp;
    my_adios().previous_timestamp[data->tid] = ts;

    auto &tmp = my_adios().comm_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    active_threads--;
    tau_plugin::inPlugin() = false;
    return 0;
}

/* This happens on MPI_Recv events (and similar) */
int Tau_plugin_adios2_recv(Tau_plugin_event_recv_data_t* data) {
    if (!enabled) return 0;
    if (tau_plugin::inPlugin()) return 0;
    tau_plugin::inPlugin() = true;
    static std::string recvstr("RECV");
    int event_index = my_adios().check_event_type(recvstr);
    my_adios().check_thread(data->tid);
    std::array<unsigned long, 7> tmparray;
    tmparray[0] = 0UL;
    tmparray[1] = (unsigned long)(global_comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(event_index);
    tmparray[4] = (unsigned long)(data->message_tag);
    tmparray[5] = (unsigned long)(data->source);
    tmparray[6] = (unsigned long)(data->bytes_received);

    /* There's a tiny chance of a race condition because we have to update
     * and check two variables.  So acquire the lock before waiting until it's
     * safe to proceed with the dump.
     */
    _my_mutex.lock();
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    _my_mutex.unlock();

    unsigned long ts = my_adios().previous_timestamp[data->tid] >
        data->timestamp ? my_adios().previous_timestamp[data->tid] :
            data->timestamp;
    my_adios().previous_timestamp[data->tid] = ts;

    auto &tmp = my_adios().comm_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    active_threads--;
    tau_plugin::inPlugin() = false;
    return 0;
}

bool skip_timer(const char * key);

/* This happens on Tau_start() */
int Tau_plugin_adios2_function_entry(Tau_plugin_event_function_entry_data_t* data) {
    if (!enabled) return 0;
    /* First, check to see if we are including/excluding this timer */
    if (skip_timer(data->timer_name)) {
        return 0;
    }
    if (tau_plugin::inPlugin()) return 0;
    tau_plugin::inPlugin() = true;
    /* todo: filter on group */
    int timer_index = my_adios().check_timer(data->timer_name);
    static std::string entrystr("ENTRY");
    int event_index = my_adios().check_event_type(entrystr);
    my_adios().check_thread(data->tid);
    std::array<unsigned long, 5> tmparray;
    tmparray[0] = 0;
    tmparray[1] = (unsigned long)(global_comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(event_index);
    tmparray[4] = (unsigned long)(timer_index);

    /* There's a tiny chance of a race condition because we have to update
     * and check two variables.  So acquire the lock before waiting until it's
     * safe to proceed with the dump.
     */
    _my_mutex.lock();
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    _my_mutex.unlock();

    auto &tmp = my_adios().timer_values_array[data->tid];
    unsigned long ts = my_adios().previous_timestamp[data->tid] > data->timestamp ?
        my_adios().previous_timestamp[data->tid] : data->timestamp;
    my_adios().previous_timestamp[data->tid] = ts;
#ifdef DO_VALIDATION
    my_adios().pre_timer_stack[data->tid].push(timer_index);
    TAU_VERBOSE("Enter:   %d %d %lu %lu %s %s\n", global_comm_rank, data->tid, data->timestamp, ts, (data->timestamp != ts ? "FIXED" : ""), data->timer_name);
#endif
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    active_threads--;
    tau_plugin::inPlugin() = false;
    return 0;
}

/* This happens on Tau_stop() */
int Tau_plugin_adios2_function_exit(Tau_plugin_event_function_exit_data_t* data) {
    if (!enabled) return 0;
    /* First, check to see if we are including/excluding this timer */
    if (skip_timer(data->timer_name)) {
        return 0;
    }
    if (tau_plugin::inPlugin()) return 0;
    tau_plugin::inPlugin() = true;
    /* todo: filter on group */
    int timer_index = my_adios().check_timer(data->timer_name);
    static std::string exitstr("EXIT");
    int event_index = my_adios().check_event_type(exitstr);
    std::array<unsigned long, 5> tmparray;
    tmparray[0] = 0UL;
    tmparray[1] = (unsigned long)(global_comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(event_index);
    tmparray[4] = (unsigned long)(timer_index);

    /* There's a tiny chance of a race condition because we have to update
     * and check two variables.  So acquire the lock before waiting until it's
     * safe to proceed with the dump.
     */
    _my_mutex.lock();
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    _my_mutex.unlock();

    auto &tmp = my_adios().timer_values_array[data->tid];
    unsigned long ts = my_adios().previous_timestamp[data->tid] >
        data->timestamp ? my_adios().previous_timestamp[data->tid] :
            data->timestamp;
    my_adios().previous_timestamp[data->tid] = ts;
#ifdef DO_VALIDATION
    TAU_VERBOSE("Exit:    %d %d %lu %lu %s %s\n", global_comm_rank, data->tid, data->timestamp, ts, (data->timestamp != ts ? "FIXED" : ""), data->timer_name);
    if (my_adios().pre_timer_stack[data->tid].size() == 0) {
      TAU_VERBOSE("%d Pre: Stack violation. %s\n", getpid(), data->timer_name);
      TAU_VERBOSE("%d Pre: Stack for thread %lu is empty, timestamp %lu.\n",
        getpid(), tmparray[2], data->timestamp);
      active_threads--;
      tau_plugin::inPlugin() = false;
      return 0;
    } else {
        unsigned long lhs = (unsigned long)(my_adios().pre_timer_stack[data->tid].top());
        unsigned long rhs = (unsigned long)(timer_index);
        if (lhs != rhs) {
            TAU_VERBOSE("Pre: Stack violation. %s\n", data->timer_name);
            TAU_VERBOSE("Pre: thread %lu, %lu != %lu, timestamp %lu\n",
                tmparray[2], lhs, rhs, data->timestamp);
        }
        my_adios().pre_timer_stack[data->tid].pop();
    }
#endif
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    active_threads--;
    tau_plugin::inPlugin() = false;
    return 0;
}

bool skip_counter(const char * key);

/* This happens on Tau_userevent() */
int Tau_plugin_adios2_atomic_trigger(Tau_plugin_event_atomic_event_trigger_data_t* data) {
    if (!enabled) return 0;
    /* First, check to see if we are including/excluding this counter */
    if (skip_counter(data->counter_name)) {
        return 0;
    }
    if (tau_plugin::inPlugin()) return 0;
    tau_plugin::inPlugin() = true;
    int counter_index = my_adios().check_counter(data->counter_name);
    my_adios().check_thread(data->tid);
    std::array<unsigned long, 5> tmparray;
    tmparray[0] = 0UL;
    tmparray[1] = (unsigned long)(global_comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(counter_index);
    tmparray[4] = (unsigned long)(data->value);

    /* There's a tiny chance of a race condition because we have to update
     * and check two variables.  So acquire the lock before waiting until it's
     * safe to proceed with the dump.
     */
    _my_mutex.lock();
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    _my_mutex.unlock();

    unsigned long ts = my_adios().previous_timestamp[data->tid] >
        data->timestamp ? my_adios().previous_timestamp[data->tid] :
            data->timestamp;
    my_adios().previous_timestamp[data->tid] = ts;
#ifdef DO_VALIDATION
    TAU_VERBOSE("Counter: %d %d %lu %lu %s %s\n", global_comm_rank, data->tid, data->timestamp, ts, (data->timestamp != ts ? "FIXED" : ""), data->counter_name);
#endif

    auto &tmp = my_adios().counter_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    active_threads--;
    tau_plugin::inPlugin() = false;
    return 0;
}

extern x_uint64 TauTraceGetTimeStamp(int tid);

/* This happens after MPI_Init, and after all TAU metadata variables have been
 * read */
int Tau_plugin_adios2_post_init(Tau_plugin_event_post_init_data_t* data) {
    tau_plugin::inPlugin() = true;
    /* Open the ADIOS archive */
    printf("Making my_adios %d\n", __LINE__);
    for (int i = 0 ; i < my_adios().get_thread_count() ; i++) {
        Tau_dump_ADIOS2_metadata(my_adios()._bpIO, i);
    }
    my_adios().check_event_type(std::string("ENTRY"));
    my_adios().check_event_type(std::string("EXIT"));
    my_adios().check_event_type(std::string("SEND"));
    my_adios().check_event_type(std::string("RECV"));

    /* spawn the thread if doing periodic */
    if (tau_plugin::thePluginOptions().env_periodic) {
        _threaded = true;
        TAU_VERBOSE("Spawning thread for ADIOS2.\n");
        worker_thread = new std::thread(Tau_ADIOS2_thread_function);
    } else {
        _threaded = false;
    }

    /* If we are tracing, we need to "start" all of the timers on the stack */
    RtsLayer::LockDB();
    // Do this now, otherwise we don't get enter events!
    enabled = true;
    //int tid = RtsLayer::myThread();
    for (int tid = RtsLayer::getTotalThreads()-1 ; tid >= 0 ; tid--) {
      if (Tau_is_thread_fake(tid) == 1) { continue; }
    //int tid = 0; // only do thread 0
      Tau_plugin_event_function_entry_data_t entry_data;
      // safe to assume 0?
      int depth = Tau_get_current_stack_depth(tid);
      for (int i = 0 ; i <= depth ; i++) {
        tau::Profiler *profiler = Tau_get_timer_at_stack_depth_task(i, tid);
        // not sure how this can happen...
        if (profiler == NULL) {
#ifdef DO_VALIDATION
        TAU_VERBOSE("%d,%d,%d,%d NULL profiler!\n", getpid(), tid, i, depth);
#endif
        continue; }
        entry_data.timer_name = profiler->ThisFunction->GetName();
        // not sure how this can happen...
        if (entry_data.timer_name == NULL) {
#ifdef DO_VALIDATION
        TAU_VERBOSE("%d,%d,%d,%d Missing timer name!\n", getpid(), tid, i, depth);
#endif
        continue; }
        entry_data.timer_group = profiler->ThisFunction->GetAllGroups();
        entry_data.tid = tid;
        entry_data.timestamp = (x_uint64)profiler->StartTime[0];
#ifdef DO_VALIDATION
        TAU_VERBOSE("%d,%d,%d,%d Starting %s\n", getpid(), tid, i, depth, entry_data.timer_name);
#endif
        Tau_plugin_adios2_function_entry(&entry_data);
      }
    }
    RtsLayer::UnLockDB();

    if (signal(SIGUSR1, Tau_plugin_adios2_signal_handler) == SIG_ERR) {
      perror("failed to register TAU profile dump signal handler");
    }

    tau_plugin::inPlugin() = false;
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks_t cb;
    TAU_VERBOSE("TAU PLUGIN ADIOS2 Init\n"); fflush(stdout);
    tau_plugin::Tau_ADIOS2_parse_environment_variables();
#if TAU_MPI
    PMPI_Comm_size(MPI_COMM_WORLD, &global_comm_size);
    PMPI_Comm_rank(MPI_COMM_WORLD, &global_comm_rank);
#endif
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(&cb);

    /* Required event support */
    if (!tau_plugin::thePluginOptions().env_periodic) {
        cb.Dump = Tau_plugin_adios2_dump;
    }
    cb.MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_func;
    cb.PostInit = Tau_plugin_adios2_post_init;
    cb.PreEndOfExecution = Tau_plugin_adios2_pre_end_of_execution;
    cb.EndOfExecution = Tau_plugin_adios2_end_of_execution;
    /* Trace events */
    cb.Send = Tau_plugin_adios2_send;
    cb.Recv = Tau_plugin_adios2_recv;
    cb.FunctionEntry = Tau_plugin_adios2_function_entry;
    cb.FunctionExit = Tau_plugin_adios2_function_exit;
    cb.AtomicEventTrigger = Tau_plugin_adios2_atomic_trigger;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(&cb, id);

    return 0;
}

// C++ program to implement wildcard
// pattern matching algorithm
// from: https://www.geeksforgeeks.org/wildcard-pattern-matching/
#if defined(__APPLE__) && defined(__clang__)
// do nothing
#else
#include <bits/stdc++.h>
#endif
using namespace std;

// Function that matches input str with
// given wildcard pattern
bool strmatch(const char str[], const char pattern[],
              int n, int m)
{
    // empty pattern can only match with
    // empty string
    if (m == 0)
        return (n == 0);

    // lookup table for storing results of
    // subproblems
    bool lookup[n + 1][m + 1]; // = {false};
	// PGI compiler doesn't like initialization during declaration...
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= m; j++) {
            lookup[i][j] = false;
		}
	}

    // initailze lookup table to false
    //memset(lookup, false, sizeof(lookup));

    // empty pattern can match with empty string
    lookup[0][0] = true;

    // Only '#' can match with empty string
    for (int j = 1; j <= m; j++)
        if (pattern[j - 1] == '#')
            lookup[0][j] = lookup[0][j - 1];

    // fill the table in bottom-up fashion
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            // Two cases if we see a '#'
            // a) We ignore ‘#’ character and move
            //    to next  character in the pattern,
            //     i.e., ‘#’ indicates an empty sequence.
            // b) '#' character matches with ith
            //     character in input
            if (pattern[j - 1] == '#')
                lookup[i][j] = lookup[i][j - 1] ||
                               lookup[i - 1][j];

            // Current characters are considered as
            // matching in two cases
            // (a) current character of pattern is '?'
            // (b) characters actually match
            else if (pattern[j - 1] == '?' ||
                    str[i - 1] == pattern[j - 1])
                lookup[i][j] = lookup[i - 1][j - 1];

            // If characters don't match
            else lookup[i][j] = false;
        }
    }

    return lookup[n][m];
}

using namespace tau_plugin;

bool skip_timer(const char * key) {
    // are we filtering at all?
    if (!thePluginOptions().env_use_selection) {
        return false;
    }
    // check to see if this label is excluded
    if (Tau_ADIOS2_contains(thePluginOptions().excluded_timers, key, false)) {
        return true;
    // check to see if this label is included
    } else if (Tau_ADIOS2_contains(thePluginOptions().included_timers, key, false)) {
        return false;
    } else {
        // check to see if it's in the excluded wildcards
        for (std::set<std::string>::iterator
                it=thePluginOptions().excluded_timers_with_wildcards.begin();
             it!=thePluginOptions().excluded_timers_with_wildcards.end(); ++it) {
            if (strmatch(key, it->c_str(), strlen(key), it->length())) {
                // make the lookup faster next time
                thePluginOptions().excluded_timers.insert(key);
                return true;
            }
        }
        // check to see if it's in the included wildcards
        for (std::set<std::string>::iterator
                it=thePluginOptions().included_timers_with_wildcards.begin();
             it!=thePluginOptions().included_timers_with_wildcards.end(); ++it) {
            if (strmatch(key, it->c_str(), strlen(key), it->length())) {
                // make the lookup faster next time
                thePluginOptions().included_timers.insert(key);
                return false;
            }
        }
    }
    // neither included nor excluded?
    // do we have an inclusion list? If so, then skip (because we didn't match it).
    if (!thePluginOptions().included_timers.empty() ||
        !thePluginOptions().included_timers_with_wildcards.empty()) {
        return true;
    }
    // by default, don't skip it.
    return false;
}

bool skip_counter(const char * key) {
    // are we filtering at all?
    if (!thePluginOptions().env_use_selection) {
        return false;
    }
    // check to see if this label is excluded
    if (Tau_ADIOS2_contains(thePluginOptions().excluded_counters, key, false)) {
        return true;
    // check to see if this label is included
    } else if (Tau_ADIOS2_contains(thePluginOptions().included_counters, key, false)) {
        return false;
    } else {
        // check to see if it's in the excluded wildcards
        for (std::set<std::string>::iterator
                it=thePluginOptions().excluded_counters_with_wildcards.begin();
             it!=thePluginOptions().excluded_counters_with_wildcards.end(); ++it) {
            if (strmatch(key, it->c_str(), strlen(key), it->length())) {
                // make the lookup faster next time
                thePluginOptions().excluded_counters.insert(key);
                return true;
            }
        }
        // check to see if it's in the included wildcards
        for (std::set<std::string>::iterator
                it=thePluginOptions().included_counters_with_wildcards.begin();
             it!=thePluginOptions().included_counters_with_wildcards.end(); ++it) {
            if (strmatch(key, it->c_str(), strlen(key), it->length())) {
                // make the lookup faster next time
                thePluginOptions().included_counters.insert(key);
                return false;
            }
        }
    }
    // neither included nor excluded?
    // do we have an inclusion list? If so, then skip (because we didn't match it).
    if (!thePluginOptions().included_counters.empty() ||
        !thePluginOptions().included_counters_with_wildcards.empty()) {
        return true;
    }
    return false;
}


#endif // TAU_ADIOS2
