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

// signal handling to dump provenance
#include <signal.h>
#include <unistd.h>

#define CONVERT_TO_USEC 1.0/1000000.0 // hopefully the compiler will precompute this.
#define TAU_ADIOS2_PERIODIC_DEFAULT false
#define TAU_ADIOS2_PERIOD_DEFAULT 2000000 // microseconds
#define TAU_ADIOS2_PROVENANCE_HISTORY_DEFAULT 5 // 5 steps
#define TAU_ADIOS2_USE_SELECTION_DEFAULT false
#define TAU_ADIOS2_FILENAME "tau-metrics"
#define TAU_ADIOS2_ONE_FILE_DEFAULT false
#define TAU_ADIOS2_ENGINE "BPFile"
#define TAU_ADIOS2_CONFIG_FILE_DEFAULT "./adios2.xml"

// This will enable some checking to make sure we don't have call stack violations.
//#define DO_VALIDATION

/* Some forward declarations that we need */
tau::Profiler *Tau_get_timer_at_stack_depth(int);
int Tau_plugin_adios2_function_exit(
    Tau_plugin_event_function_exit_data_t* data);
void Tau_dump_ADIOS2_metadata(adios2::IO& bpIO, int tid);
void Tau_plugin_adios2_dump_history(void);

static bool enabled{false};
static bool done{false};
static bool _threaded{false};
static int global_comm_size = 1;
static int global_comm_rank = 0;
pthread_mutex_t _my_mutex; // for initialization, termination, semaphore access (see below)
/* These two variables are used to create something like a "counting semaphore".
 * When dumping, we want to prevent any threads from adding events to the buffers.
 * But when adding to the buffers, the threads don't need exclusive access.
 * Therefore, when "not dumping", the threads can increment the active_threads
 * counter, and the dumping thread will wait until those counters reach 0
 * before assuming exclusive access to the buffers.
 */
static atomic<bool> dumping{false};
static atomic<size_t> active_threads{0};

pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;
static atomic<bool> dump_history{false};

namespace tau_plugin {

char *_program_path()
{
#if defined(__APPLE__)
    return NULL;
#else
    char *path = (char*)malloc(PATH_MAX);
    if (path != NULL) {
        if (readlink("/proc/self/exe", path, PATH_MAX) == -1) {
            free(path);
            path = NULL;
        }
        std::string tmp(path);
        size_t i = tmp.rfind('/', tmp.length());
        if (i != string::npos) {
            sprintf(path, "%s", tmp.substr(i+1, tmp.length() - i).c_str());
        }
    }
    return path;
#endif
}

class plugin_options {
    private:
        plugin_options(void) :
            env_periodic(TAU_ADIOS2_PERIODIC_DEFAULT),
            env_period(TAU_ADIOS2_PERIOD_DEFAULT),
            env_history_depth(TAU_ADIOS2_PROVENANCE_HISTORY_DEFAULT),
            env_use_selection(TAU_ADIOS2_USE_SELECTION_DEFAULT),
            env_filename(TAU_ADIOS2_FILENAME),
            env_one_file(TAU_ADIOS2_ONE_FILE_DEFAULT),
            env_engine(TAU_ADIOS2_ENGINE),
            env_config_file(nullptr)
            {}
    public:
        int env_periodic;
        int env_period;
        int env_history_depth;
        bool env_use_selection;
        std::string env_filename;
        int env_one_file;
        std::string env_engine;
        char * env_config_file;
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

void Tau_ADIOS2_parse_environment_variables(void);
void Tau_ADIOS2_parse_selection_file(const char * filename);
bool Tau_ADIOS2_contains(std::set<std::string>& myset,
        const char * key, bool if_empty);

void * Tau_ADIOS2_thread_function(void* data);

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
    tmp = getenv("TAU_ADIOS2_PROVENANCE_HISTORY");
    thePluginOptions().env_history_depth = parse_int(tmp, TAU_ADIOS2_PROVENANCE_HISTORY_DEFAULT);
    tmp = getenv("TAU_ADIOS2_SELECTION_FILE");
    if (tmp != NULL) {
      Tau_ADIOS2_parse_selection_file(tmp);
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
          thePluginOptions().env_config_file = nullptr;
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

/* Circular buffer for storing provenance */
class circular_buffer {
public:
    explicit circular_buffer(size_t size) :
        max_size_(size) {
            buf_.reserve(size);
            for(int i = 0 ; i < size ; i++) {
                buf_[i] = nullptr;
            }
        }

    void put(step_data_t* item) {
        // destroy any existing data at this index
        if (buf_[head_] != nullptr) {
            delete buf_[head_];
        }
        buf_[head_] = item;
        if(full_) {
            tail_ = (tail_ + 1) % max_size_;
        }
        head_ = (head_ + 1) % max_size_;
        full_ = head_ == tail_;
    }

    step_data_t* get() {
        if(empty()) {
            return nullptr;
        }

        //Read data and advance the tail (we now have a free space)
        auto val = buf_[tail_];
        buf_[tail_] = nullptr;
        full_ = false;
        tail_ = (tail_ + 1) % max_size_;
        return val;
    }

    void reset() {
        head_ = tail_;
        full_ = false;
    }

    bool empty() const {
        //if head and tail are equal, we are empty
        return (!full_ && (head_ == tail_));
    }

    bool full() const {
        //If tail is ahead the head by 1, we are full
        return full_;
    }

    size_t capacity() const { return max_size_; }

    size_t size() const {
        size_t size = max_size_;
        if(!full_) {
            if(head_ >= tail_) {
                size = head_ - tail_;
            } else {
                size = max_size_ + head_ - tail_;
            }
        }
        return size;
    }

private:
    std::vector<step_data_t*> buf_;
    size_t head_ = 0;
    size_t tail_ = 0;
    const size_t max_size_;
    bool full_ = 0;
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
            max_threads(0),
            step_history(thePluginOptions().env_history_depth)
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
        // provenance history containers
        timer_values_array_t current_primer_stack[TAU_MAX_THREADS];
        circular_buffer step_history;
        // for validation
#ifdef DO_VALIDATION
        std::stack<unsigned long> timer_stack[TAU_MAX_THREADS];
        std::stack<unsigned long> pre_timer_stack[TAU_MAX_THREADS];
        unsigned long previous_timestamp[TAU_MAX_THREADS];
#endif
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
    if (thePluginOptions().env_config_file != nullptr) {
        ad = adios2::ADIOS(thePluginOptions().env_config_file, adios_comm, true);
    } else {
        ad = adios2::ADIOS(adios_comm, true);
    }
#else
    /** ADIOS class factory of IO class objects, DebugON is recommended */
    if (thePluginOptions().env_config_file != nullptr) {
        ad = adios2::ADIOS(thePluginOptions().env_config_file, true);
    } else {
        ad = adios2::ADIOS(true);
    }
#endif
    /*** IO class object: settings and factory of Settings: Variables,
     * Parameters, Transports, and Execution: Engines */
    _bpIO = ad.DeclareIO("TAU trace data");

    if (thePluginOptions().env_config_file == nullptr) {
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
        TAU_VERBOSE("Writing %s\n", ss.str().c_str());
        bpWriter = _bpIO.Open(ss.str(), adios2::Mode::Write);
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
    pthread_mutex_lock(&_my_mutex);
    int threads = get_thread_count();
    int timers = get_timer_count();
    int counters = get_counter_count();
    pthread_mutex_unlock(&_my_mutex);

    Tau_global_incr_insideTAU();
    bpWriter.BeginStep();
    step_data_t* this_step = new step_data_t(threads);

    /* There's a tiny chance of a race condition because we have to update
     * and check two variables.  So acquire the lock before waiting until it's
     * safe to proceed with the dump.
     */
    pthread_mutex_lock(&_my_mutex);
    // set the dumping flag
    dumping = true;
    // wait for all active threads to finish doing what they do
    while(active_threads > 0) {}
    /* Release the lock, we've got control */
    pthread_mutex_unlock(&_my_mutex);

    /* sort into one big vector from all threads */
    // make a list from the first thread of data - copying the data in!
    std::list<std::pair<unsigned long, std::array<unsigned long, 5> > >
        merged_timers(timer_values_array[0].begin(),
                      timer_values_array[0].end());
    // this clear will empty the vector and destroy the objects!
    timer_values_array[0].clear();
    // copy the current primer stack
    this_step->primer_stacks[0] = new timer_values_array_t(current_primer_stack[0]);
    for (int t = 1 ; t < threads ; t++) {
        // is there any data on this thread?
        if (timer_values_array[t].size() == 0) { continue; }
        // make a list from the next thread of data - copying the data in!
        std::list<std::pair<unsigned long, std::array<unsigned long, 5> > >
            next_thread(timer_values_array[t].begin(),
                        timer_values_array[t].end());
        // this clear will empty the vector and destroy the objects!
        timer_values_array[t].clear();
        // copy the current primer stack
        this_step->primer_stacks[t] = new timer_values_array_t(current_primer_stack[t]);
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

    //printf("%d %s %d\n", global_comm_rank, __func__, __LINE__); fflush(stdout);

    std::vector<unsigned long>* all_timers = new std::vector<unsigned long>(6,0);
    all_timers->reserve(6*merged_timers.size());
    int timer_value_index = 0;
    for (auto it = merged_timers.begin() ; it != merged_timers.end() ; it++) {
        (*all_timers)[timer_value_index++] = it->second[0];
        (*all_timers)[timer_value_index++] = it->second[1];
        (*all_timers)[timer_value_index++] = it->second[2];
        (*all_timers)[timer_value_index++] = it->second[3];
        (*all_timers)[timer_value_index++] = it->second[4];
        (*all_timers)[timer_value_index++] = it->first;
#ifdef DO_VALIDATION
        if (it->second[3] == 0) {
            // on entry
            timer_stack[it->second[2]].push(it->second[4]);
        } else if (it->second[3] == 1) {
            // on exit
            if (timer_stack[it->second[2]].size() == 0) {
                fprintf(stderr, "Writing: Stack violation.\n");
                fprintf(stderr, "Writing: Stack for thread %lu is empty, timestamp %lu.\n",
                    it->second[2], it->first);
            } else {
                if (timer_stack[it->second[2]].top() != it->second[4]) {
                    fprintf(stderr, "Writing: Stack violation.\n");
                    fprintf(stderr, "Writing: thread %lu, %lu != %lu, timestamp %lu\n",
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

    std::vector<unsigned long> all_counters(6,0);;
    all_counters.reserve(6*merged_counters.size());
    int counter_value_index = 0;
    for (auto it = merged_counters.begin() ;
         it != merged_counters.end() ; it++) {
        all_counters[counter_value_index++] = it->second[0];
        all_counters[counter_value_index++] = it->second[1];
        all_counters[counter_value_index++] = it->second[2];
        all_counters[counter_value_index++] = it->second[3];
        all_counters[counter_value_index++] = it->second[4];
        all_counters[counter_value_index++] = it->first;
    }

    for (int t = 0 ; t < threads ; t++) {
        counter_values_array[t].clear();
    }

    //printf("%d %s %d\n", global_comm_rank, __func__, __LINE__); fflush(stdout);

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

    std::vector<unsigned long> all_comms(8,0);;
    all_comms.reserve(8*merged_comms.size());
    int comm_value_index = 0;
    for (auto it = merged_comms.begin() ;
         it != merged_comms.end() ; it++) {
        all_comms[comm_value_index++] = it->second[0];
        all_comms[comm_value_index++] = it->second[1];
        all_comms[comm_value_index++] = it->second[2];
        all_comms[comm_value_index++] = it->second[3];
        all_comms[comm_value_index++] = it->second[4];
        all_comms[comm_value_index++] = it->second[5];
        all_comms[comm_value_index++] = it->second[6];
        all_comms[comm_value_index++] = it->first;
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

    this_step->programs = programs;
    this_step->comm_ranks = comm_ranks;
    this_step->threads = threads;
    this_step->event_types = event_types;
    this_step->timers = timers;
    this_step->num_timer_values = num_timer_values;
    this_step->counters = counters;
    this_step->num_counter_values = num_counter_values;
    this_step->num_comm_values = num_comm_values;

    //printf("%d %s %d\n", global_comm_rank, __func__, __LINE__); fflush(stdout);

    if (num_timer_values > 0) {
        event_timestamps.SetShape({num_timer_values, 6});
        /* These dimensions need to change for 1-file case! */
        const adios2::Dims timer_start{0, 0};
        const adios2::Dims timer_count{static_cast<size_t>(num_timer_values), 6};
        const adios2::Box<adios2::Dims> timer_selection{timer_start, timer_count};
        event_timestamps.SetSelection(timer_selection);
        bpWriter.Put(event_timestamps, all_timers->data());
    }

    /* save the current set of events in this step */
    this_step->step_of_events = all_timers;

    if (num_counter_values > 0) {
        counter_values.SetShape({num_counter_values, 6});
        /* These dimensions need to change for 1-file case! */
        const adios2::Dims counter_start{0, 0};
        const adios2::Dims counter_count{static_cast<size_t>(num_counter_values), 6};
        const adios2::Box<adios2::Dims> counter_selection{counter_start, counter_count};
        counter_values.SetSelection(counter_selection);
        bpWriter.Put(counter_values, all_counters.data());
    }

    /* save the current set of counters in this step */
    this_step->step_of_counters = std::move(all_counters);

    if (num_comm_values > 0) {
        comm_timestamps.SetShape({num_comm_values, 8});
        /* These dimensions need to change for 1-file case! */
        const adios2::Dims comm_start{0, 0};
        const adios2::Dims comm_count{static_cast<size_t>(num_comm_values), 8};
        const adios2::Box<adios2::Dims> comm_selection{comm_start, comm_count};
        comm_timestamps.SetSelection(comm_selection);
        bpWriter.Put(comm_timestamps, all_comms.data());
    }

    /* save the current set of counters in this step */
    this_step->step_of_comms = std::move(all_comms);
    step_history.put(this_step);

    //printf("%d %s %d\n", global_comm_rank, __func__, __LINE__); fflush(stdout);

    bpWriter.EndStep();
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
            pthread_mutex_lock(&_my_mutex);
            // check to make sure another thread didn't create it already
            if (timers.count(tmp) == 0) {
                int num = timers.size();
                timers[tmp] = num;
                // printf("%d = %s\n", num, timer);
                new_timer = true;
            }
            pthread_mutex_unlock(&_my_mutex);
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
            pthread_mutex_lock(&_my_mutex);
            // check to make sure another thread didn't create it already
            if (counters.count(tmp) == 0) {
                int num = counters.size();
                counters[tmp] = num;
                new_counter = true;
            }
            pthread_mutex_unlock(&_my_mutex);
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

static tau_plugin::adios * my_adios{nullptr};

void init_lock(pthread_mutex_t * _mutex) {
    if (!_threaded) return;
    pthread_mutexattr_t Attr;
    pthread_mutexattr_init(&Attr);
    pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_ERRORCHECK);
    int rc;
    if ((rc = pthread_mutex_init(_mutex, &Attr)) != 0) {
        errno = rc;
        perror("pthread_mutex_init error");
        exit(1);
    }
    if ((rc = pthread_cond_init(&_my_cond, NULL)) != 0) {
        errno = rc;
        perror("pthread_cond_init error");
        exit(1);
    }
}

int Tau_plugin_adios2_dump(Tau_plugin_event_dump_data_t* data) {
    if (!enabled) return 0;
    static int iter = 0;
    TAU_VERBOSE("%d TAU PLUGIN ADIOS2 Dump %d \n", global_comm_rank, iter); fflush(stdout);
    if (dump_history) {
        Tau_plugin_adios2_dump_history();
        // reset for next time
        dump_history = false;
    }
    //Tau_pure_start(__func__);
    Tau_global_incr_insideTAU();
    my_adios->write_variables();
    Tau_global_decr_insideTAU();
    //Tau_pure_stop(__func__);
    TAU_VERBOSE("%d TAU PLUGIN ADIOS2 Dump exit %d\n", global_comm_rank, iter++); fflush(stdout);
    return 0;
}

void Tau_ADIOS2_stop_worker(void) {
    if (!enabled) return;
    if (!_threaded) return;
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    if (tau_plugin::thePluginOptions().env_periodic) {
        TAU_VERBOSE("TAU ADIOS2 thread joining...\n"); fflush(stderr);
        pthread_cond_signal(&_my_cond);
        int ret = pthread_join(worker_thread, NULL);
        if (ret != 0) {
            switch (ret) {
                case ESRCH:
                    // already exited.
                    break;
                case EINVAL:
                    // Didn't exist?
                    break;
                case EDEADLK:
                    // trying to join with itself?
                    break;
                default:
                    errno = ret;
                    perror("Warning: pthread_join failed\n");
                    break;
            }
        }
    }
    _threaded = false;
}

/* This happens after MPI_Init, and after all TAU metadata variables have been
 * read */
int Tau_plugin_adios2_post_init(Tau_plugin_event_post_init_data_t* data) {
    if (!enabled) return 0;
    return 0;
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
    if (my_adios != nullptr) {
        my_adios->close();
        my_adios = nullptr;
    }
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
    /* Not really necessary. */
    if (tau_plugin::thePluginOptions().env_periodic) {
        pthread_cond_destroy(&_my_cond);
        pthread_mutex_destroy(&_my_mutex);
    }
    return rc;
}

void Tau_dump_ADIOS2_metadata(adios2::IO& bpIO, int tid) {
    if (!enabled) return;
    //int tid = RtsLayer::myThread();
    int nodeid = TAU_PROFILE_GET_NODE();
    Tau_global_incr_insideTAU();
    for (MetaDataRepo::iterator it = Tau_metadata_getMetaData(tid).begin();
         it != Tau_metadata_getMetaData(tid).end(); it++) {
        // check for executable name
        if (strcmp(it->first.name, "Executable") == 0) {
            my_adios->check_prog_name(it->second->data.cval);
        }
        std::stringstream ss;
        ss << "MetaData:" << global_comm_rank << ":" << tid << ":" << it->first.name;
        switch(it->second->type) {
            case TAU_METADATA_TYPE_STRING:
                my_adios->define_attribute(ss.str(), std::string(it->second->data.cval), bpIO, true);
                break;
            case TAU_METADATA_TYPE_INTEGER:
                my_adios->define_attribute(ss.str(), std::to_string(it->second->data.ival), bpIO, true);
                break;
            case TAU_METADATA_TYPE_DOUBLE:
                my_adios->define_attribute(ss.str(), std::to_string(it->second->data.dval), bpIO, true);
                break;
            case TAU_METADATA_TYPE_TRUE:
                my_adios->define_attribute(ss.str(), std::string("true"), bpIO, true);
                break;
            case TAU_METADATA_TYPE_FALSE:
                my_adios->define_attribute(ss.str(), std::string("false"), bpIO, true);
                break;
            case TAU_METADATA_TYPE_NULL:
                my_adios->define_attribute(ss.str(), std::string("(null)"), bpIO, true);
                break;
            default:
                break;
        }
    }
    Tau_global_decr_insideTAU();
}

/* This happens when a Metadata field is saved. */
int Tau_plugin_metadata_registration_complete_func(Tau_plugin_event_metadata_registration_data_t* data) {
    if (!enabled) return 0;
    Tau_global_incr_insideTAU();
    //fprintf(stdout, "TAU Metadata registration\n"); fflush(stdout);
    std::stringstream ss;
    ss << "MetaData:" << global_comm_rank << ":" << data->tid << ":" << data->name;
    switch(data->value->type) {
        case TAU_METADATA_TYPE_STRING:
            my_adios->define_attribute(ss.str(), std::string(data->value->data.cval), my_adios->_bpIO, false);
            break;
        case TAU_METADATA_TYPE_INTEGER:
            my_adios->define_attribute(ss.str(), std::to_string(data->value->data.ival), my_adios->_bpIO, false);
            break;
        case TAU_METADATA_TYPE_DOUBLE:
            my_adios->define_attribute(ss.str(), std::to_string(data->value->data.dval), my_adios->_bpIO, false);
            break;
        case TAU_METADATA_TYPE_TRUE:
            my_adios->define_attribute(ss.str(), std::string("true"), my_adios->_bpIO, false);
            break;
        case TAU_METADATA_TYPE_FALSE:
            my_adios->define_attribute(ss.str(), std::string("false"), my_adios->_bpIO, false);
            break;
        case TAU_METADATA_TYPE_NULL:
            my_adios->define_attribute(ss.str(), std::string("(null)"), my_adios->_bpIO, false);
            break;
        default:
            break;
    }
    Tau_global_decr_insideTAU();
    return 0;
}

/* This happens on MPI_Send events (and similar) */
int Tau_plugin_adios2_send(Tau_plugin_event_send_data_t* data) {
    if (!enabled) return 0;
    static std::string sendstr("SEND");
    int event_index = my_adios->check_event_type(sendstr);
    my_adios->check_thread(data->tid);
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
    pthread_mutex_lock(&_my_mutex);
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    pthread_mutex_unlock(&_my_mutex);

    auto &tmp = my_adios->comm_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp,
            std::move(tmparray)
        )
    );
    active_threads--;
    return 0;
}

/* This happens on MPI_Recv events (and similar) */
int Tau_plugin_adios2_recv(Tau_plugin_event_recv_data_t* data) {
    if (!enabled) return 0;
    static std::string recvstr("RECV");
    int event_index = my_adios->check_event_type(recvstr);
    my_adios->check_thread(data->tid);
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
    pthread_mutex_lock(&_my_mutex);
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    pthread_mutex_unlock(&_my_mutex);

    auto &tmp = my_adios->comm_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp,
            std::move(tmparray)
        )
    );
    active_threads--;
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
    /* todo: filter on group */
    int timer_index = my_adios->check_timer(data->timer_name);
    static std::string entrystr("ENTRY");
    int event_index = my_adios->check_event_type(entrystr);
    my_adios->check_thread(data->tid);
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
    pthread_mutex_lock(&_my_mutex);
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    pthread_mutex_unlock(&_my_mutex);

    auto &tmp = my_adios->timer_values_array[data->tid];
#ifdef DO_VALIDATION
    unsigned long ts = my_adios->previous_timestamp[data->tid] > data->timestamp ?
        my_adios->previous_timestamp[data->tid] + 1 : data->timestamp;
    my_adios->previous_timestamp[data->tid] = ts;
    my_adios->pre_timer_stack[data->tid].push(timer_index);
#else
    unsigned long ts = data->timestamp;
#endif
    // push this timer on the stack for provenance output, make a copy
    my_adios->current_primer_stack[data->tid].push_back(std::make_pair(ts, tmparray));
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    active_threads--;
    return 0;
}

/* This happens on Tau_stop() */
int Tau_plugin_adios2_function_exit(Tau_plugin_event_function_exit_data_t* data) {
    if (!enabled) return 0;
    /* First, check to see if we are including/excluding this timer */
    if (skip_timer(data->timer_name)) {
        return 0;
    }
    /* todo: filter on group */
    int timer_index = my_adios->check_timer(data->timer_name);
    static std::string exitstr("EXIT");
    int event_index = my_adios->check_event_type(exitstr);
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
    pthread_mutex_lock(&_my_mutex);
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    pthread_mutex_unlock(&_my_mutex);

    auto &tmp = my_adios->timer_values_array[data->tid];
#ifdef DO_VALIDATION
    unsigned long ts = my_adios->previous_timestamp[data->tid] >
        data->timestamp ? my_adios->previous_timestamp[data->tid] + 1 :
            data->timestamp;
    my_adios->previous_timestamp[data->tid] = ts;
    if (my_adios->pre_timer_stack[data->tid].size() == 0) {
      fprintf(stderr, "Pre: Stack violation. %s\n", data->timer_name);
      fprintf(stderr, "Pre: Stack for thread %lu is empty, timestamp %lu.\n",
        tmparray[2], data->timestamp);
	  if (my_adios->current_primer_stack[data->tid].size() > 0) {
          my_adios->current_primer_stack[data->tid].pop_back();
	  }
      active_threads--;
      return 0;
    } else {
        unsigned long lhs = (unsigned long)(my_adios->pre_timer_stack[data->tid].top());
        unsigned long rhs = (unsigned long)(timer_index);
        if (lhs != rhs) {
            fprintf(stderr, "Pre: Stack violation. %s\n", data->timer_name);
            fprintf(stderr, "Pre: thread %lu, %lu != %lu, timestamp %lu\n",
                tmparray[2], lhs, rhs, data->timestamp);
        }
        my_adios->pre_timer_stack[data->tid].pop();
    }
#else
    unsigned long ts = data->timestamp;
#endif
    // pop this timer off the stack for provenance output
	// For some reason, at the end of execution we are popping too many.
	// This is a safety check, but not great for performance.
	if (my_adios->current_primer_stack[data->tid].size() > 0) {
        my_adios->current_primer_stack[data->tid].pop_back();
	}
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    active_threads--;
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
    int counter_index = my_adios->check_counter(data->counter_name);
    my_adios->check_thread(data->tid);
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
    pthread_mutex_lock(&_my_mutex);
    while(dumping) {}
    active_threads++;
    /* Release the lock, we've got control */
    pthread_mutex_unlock(&_my_mutex);

    auto &tmp = my_adios->counter_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp,
            std::move(tmparray)
        )
    );
    active_threads--;
    return 0;
}

void * Tau_ADIOS2_thread_function(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;
	//Tau_create_top_level_timer_if_necessary();
	Tau_register_thread();

    while (!done) {
        // wait x microseconds for the next batch.
        gettimeofday(&tp, NULL);
        const int one_second = 1000000;
        // first, add the period to the current microseconds
        int tmp_usec = tp.tv_usec + tau_plugin::thePluginOptions().env_period;
        int flow_sec = 0;
        if (tmp_usec > one_second) { // did we overflow?
            flow_sec = tmp_usec / one_second; // how many seconds?
            tmp_usec = tmp_usec % one_second; // get the remainder
        }
        ts.tv_sec  = (tp.tv_sec + flow_sec);
        ts.tv_nsec = (1000 * tmp_usec);
        pthread_mutex_lock(&_my_mutex);
        int rc = pthread_cond_timedwait(&_my_cond, &_my_mutex, &ts);
        if (rc == ETIMEDOUT) {
            TAU_VERBOSE("%d Sending data from TAU thread...\n", RtsLayer::myNode()); fflush(stderr);
            Tau_plugin_event_dump_data_t dummy_data;
            Tau_plugin_adios2_dump(&dummy_data);
            TAU_VERBOSE("%d Done.\n", RtsLayer::myNode()); fflush(stderr);
        } else if (rc == EINVAL) {
            TAU_VERBOSE("Invalid timeout!\n"); fflush(stderr);
        } else if (rc == EPERM) {
            TAU_VERBOSE("Mutex not locked!\n"); fflush(stderr);
        }
    }
    // unlock after being signalled.
    pthread_mutex_unlock(&_my_mutex);
    pthread_exit((void*)0L);
	return(NULL);
}

void Tau_plugin_adios2_signal_handler(int signal) {
    printf("In Provenance signal handler\n");
    // the next time we write data, dump history first
    dump_history = true;
}

extern x_uint64 TauTraceGetTimeStamp(int tid);

void Tau_plugin_adios2_dump_history(void) {
    RtsLayer::LockDB();
    printf("In Provenance history dump\n");

    /* open an ADIOS archive */
    adios2::IO bpIO = my_adios->ad.DeclareIO("TAU trace data window");
    bpIO.SetEngine("BPFile");
    bpIO.SetParameters({{"RendezvousReaderCount", "0"}});
    std::stringstream ss;
    ss << tau_plugin::thePluginOptions().env_filename << "-window";
    ss << "-" << global_comm_rank;
    ss << ".bp";
    TAU_VERBOSE("Writing %s\n", ss.str().c_str());
    adios2::Engine bpWriter = bpIO.Open(ss.str(), adios2::Mode::Write);
    adios2::Variable<int> program_count = bpIO.DefineVariable<int>("program_count");
    adios2::Variable<int> comm_size = bpIO.DefineVariable<int>("comm_rank_count");
    adios2::Variable<int> thread_count = bpIO.DefineVariable<int>("thread_count");
    adios2::Variable<int> event_type_count = bpIO.DefineVariable<int>("event_type_count");
    adios2::Variable<int> timer_count = bpIO.DefineVariable<int>("timer_count");
    adios2::Variable<size_t> timer_event_count = bpIO.DefineVariable<size_t>("timer_event_count");
    adios2::Variable<int> counter_count = bpIO.DefineVariable<int>("counter_count");
    adios2::Variable<size_t> counter_event_count = bpIO.DefineVariable<size_t>("counter_event_count");
    adios2::Variable<size_t> comm_count = bpIO.DefineVariable<size_t>("comm_count");
    /* These are 2 dimensional variables, so they get special treatment */
    adios2::Variable<unsigned long> event_timestamps =
        bpIO.DefineVariable<unsigned long>("event_timestamps", {1, 6}, {0, 0}, {1, 6});
    adios2::Variable<unsigned long> counter_values =
        bpIO.DefineVariable<unsigned long>("counter_values", {1, 6}, {0, 0}, {1, 6});
    adios2::Variable<unsigned long> comm_timestamps =
        bpIO.DefineVariable<unsigned long>("comm_timestamps", {1, 8}, {0, 0}, {1, 8});
    /* write the metadata */
    for (int i = 0 ; i < my_adios->get_thread_count() ; i++) {
        Tau_dump_ADIOS2_metadata(bpIO, i);
    }
    /* write the program name */
    for (auto iter : my_adios->prog_names) {
        auto prog_name = iter.first;
        std::stringstream ss;
        ss << "program_name " << my_adios->prog_names[prog_name];
        my_adios->define_attribute(ss.str(), prog_name, bpIO, true);
    }
    /* write the event types */
    for (auto iter : my_adios->event_types) {
        auto event_type = iter.first;
        std::stringstream ss;
        ss << "event_type " << my_adios->event_types[event_type];
        my_adios->define_attribute(ss.str(), event_type, bpIO, true);
    }
    /* write the timer names */
    for (auto iter : my_adios->timers) {
        auto timer = iter.first;
        std::stringstream ss;
        ss << "timer " << my_adios->timers[timer];
        my_adios->define_attribute(ss.str(), timer, bpIO, true);
    }
    /* write the counter names */
    for (auto iter : my_adios->counters) {
        auto counter = iter.first;
        std::stringstream ss;
        ss << "counter " << my_adios->counters[counter];
        my_adios->define_attribute(ss.str(), counter, bpIO, true);
    }

    bool first = true;
    while (!my_adios->step_history.empty()) {
        auto step_data = my_adios->step_history.get();
        /* write the primer events? */
        if (first) {
            first = false;
            /* sort into one big vector from all threads */
            // make a list from the first thread of data - copying the data in!
            std::list<std::pair<unsigned long, std::array<unsigned long, 5> > >
                merged_timers(step_data->primer_stacks[0]->begin(),
                            step_data->primer_stacks[0]->end());
            for (int t = 1 ; t < step_data->threads ; t++) {
                // make a list from the next thread of data - copying the data in!
                std::list<std::pair<unsigned long, std::array<unsigned long, 5> > >
                    next_thread(step_data->primer_stacks[t]->begin(),
                                step_data->primer_stacks[t]->end());
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
            all_timers.reserve(6*merged_timers.size());
            int timer_value_index = 0;
            for (auto it = merged_timers.begin() ; it != merged_timers.end() ; it++) {
                (all_timers)[timer_value_index++] = it->second[0];
                (all_timers)[timer_value_index++] = it->second[1];
                (all_timers)[timer_value_index++] = it->second[2];
                (all_timers)[timer_value_index++] = it->second[3];
                (all_timers)[timer_value_index++] = it->second[4];
                (all_timers)[timer_value_index++] = it->first;
            }

            bpWriter.BeginStep();
            size_t num_counter_values{0};
            size_t num_comm_values{0};
            bpWriter.Put(program_count, &step_data->programs);
            bpWriter.Put(comm_size, &step_data->comm_ranks);
            bpWriter.Put(thread_count, &step_data->threads);
            bpWriter.Put(event_type_count, &step_data->event_types);
            bpWriter.Put(timer_count, &step_data->timers);
            bpWriter.Put(timer_event_count, &num_timer_values);
            bpWriter.Put(counter_count, &step_data->counters);
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
            bpWriter.EndStep();
            // this is all we want from this frame, so advance to the next one.
            continue;
        }

        bpWriter.BeginStep();
        bpWriter.Put(program_count, &step_data->programs);
        bpWriter.Put(comm_size, &step_data->comm_ranks);
        bpWriter.Put(thread_count, &step_data->threads);
        bpWriter.Put(event_type_count, &step_data->event_types);
        bpWriter.Put(timer_count, &step_data->timers);
        bpWriter.Put(timer_event_count, &step_data->num_timer_values);
        bpWriter.Put(counter_count, &step_data->counters);
        bpWriter.Put(counter_event_count, &step_data->num_counter_values);
        bpWriter.Put(comm_count, &step_data->num_comm_values);

        if (step_data->num_timer_values > 0) {
            event_timestamps.SetShape({step_data->num_timer_values, 6});
            /* These dimensions need to change for 1-file case! */
            const adios2::Dims timer_start{0, 0};
            const adios2::Dims timer_count{static_cast<size_t>(step_data->num_timer_values), 6};
            const adios2::Box<adios2::Dims> timer_selection{timer_start, timer_count};
            event_timestamps.SetSelection(timer_selection);
            bpWriter.Put(event_timestamps, step_data->step_of_events->data());
        }

        if (step_data->num_counter_values > 0) {
            counter_values.SetShape({step_data->num_counter_values, 6});
            /* These dimensions need to change for 1-file case! */
            const adios2::Dims counter_start{0, 0};
            const adios2::Dims counter_count{static_cast<size_t>(step_data->num_counter_values), 6};
            const adios2::Box<adios2::Dims> counter_selection{counter_start, counter_count};
            counter_values.SetSelection(counter_selection);
            bpWriter.Put(counter_values, step_data->step_of_counters.data());
        }

        if (step_data->num_comm_values > 0) {
            comm_timestamps.SetShape({step_data->num_comm_values, 8});
            /* These dimensions need to change for 1-file case! */
            const adios2::Dims comm_start{0, 0};
            const adios2::Dims comm_count{static_cast<size_t>(step_data->num_comm_values), 8};
            const adios2::Box<adios2::Dims> comm_selection{comm_start, comm_count};
            comm_timestamps.SetSelection(comm_selection);
            bpWriter.Put(comm_timestamps, step_data->step_of_comms.data());
        }

        bpWriter.EndStep();
        // if this is the last step to write, stop the current timers
        if (my_adios->step_history.empty()) {
            /* sort into one big vector from all threads */
            // make a list from the first thread of data - copying the data in!
            std::list<std::pair<unsigned long, std::array<unsigned long, 5> > >
                merged_timers(step_data->primer_stacks[0]->rbegin(),
                            step_data->primer_stacks[0]->rend());
            for (int t = 1 ; t < step_data->threads ; t++) {
                // make a list from the next thread of data - copying the data in!
                std::list<std::pair<unsigned long, std::array<unsigned long, 5> > >
                    next_thread(step_data->primer_stacks[t]->rbegin(),
                                step_data->primer_stacks[t]->rend());
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
            all_timers.reserve(6*merged_timers.size());
            int timer_value_index = 0;
            for (auto it = merged_timers.begin() ; it != merged_timers.end() ; it++) {
                (all_timers)[timer_value_index++] = it->second[0];
                (all_timers)[timer_value_index++] = it->second[1];
                (all_timers)[timer_value_index++] = it->second[2];
                (all_timers)[timer_value_index++] = it->second[3];
                (all_timers)[timer_value_index++] = it->second[4];
                (all_timers)[timer_value_index++] = TauTraceGetTimeStamp(0);
            }

            bpWriter.BeginStep();
            size_t num_counter_values{0};
            size_t num_comm_values{0};
            bpWriter.Put(program_count, &step_data->programs);
            bpWriter.Put(comm_size, &step_data->comm_ranks);
            bpWriter.Put(thread_count, &step_data->threads);
            bpWriter.Put(event_type_count, &step_data->event_types);
            bpWriter.Put(timer_count, &step_data->timers);
            bpWriter.Put(timer_event_count, &num_timer_values);
            bpWriter.Put(counter_count, &step_data->counters);
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
            bpWriter.EndStep();
         }
        delete step_data;
    }
    bpWriter.Close();
    RtsLayer::UnLockDB();
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

    /* Open the ADIOS archive */
    my_adios = new tau_plugin::adios();
    enabled = true;
    for (int i = 0 ; i < my_adios->get_thread_count() ; i++) {
        Tau_dump_ADIOS2_metadata(my_adios->_bpIO, i);
    }
    my_adios->check_event_type(std::string("ENTRY"));
    my_adios->check_event_type(std::string("EXIT"));
    my_adios->check_event_type(std::string("SEND"));
    my_adios->check_event_type(std::string("RECV"));

    /* spawn the thread if doing periodic */
    if (tau_plugin::thePluginOptions().env_periodic) {
        _threaded = true;
        init_lock(&_my_mutex);
        TAU_VERBOSE("Spawning thread for ADIOS2.\n");
        int ret = pthread_create(&worker_thread, NULL, &Tau_ADIOS2_thread_function, NULL);
        if (ret != 0) {
            errno = ret;
            perror("Error: pthread_create (1) fails\n");
            exit(1);
        }
    } else {
        _threaded = false;
    }

    /* If we are tracing, we need to "start" all of the timers on the stack */
    RtsLayer::LockDB();
    //int tid = RtsLayer::myThread();
    //for (int tid = RtsLayer::getTotalThreads()-1 ; tid >= 0 ; tid--) {
    //  if (Tau_is_thread_fake(tid) == 1) { continue; }
    int tid = 0; // only do thread 0
      Tau_plugin_event_function_entry_data_t entry_data;
      // safe to assume 0?
      int depth = Tau_get_current_stack_depth(tid);
      for (int i = 0 ; i <= depth ; i++) {
        tau::Profiler *profiler = Tau_get_timer_at_stack_depth(i);
        // not sure how this can happen...
        if (profiler == NULL) { continue; }
        entry_data.timer_name = profiler->ThisFunction->GetName();
        // not sure how this can happen...
        if (entry_data.timer_name == NULL) { continue; }
        entry_data.timer_group = profiler->ThisFunction->GetAllGroups();
        entry_data.tid = tid;
        entry_data.timestamp = (x_uint64)profiler->StartTime[0];
        //printf("%d,%d,%d,%d Starting %s\n", getpid(), tid, i, depth, entry_data.timer_name);
        //fflush(stdout);
        Tau_plugin_adios2_function_entry(&entry_data);
      }
    //}

    if (signal(SIGUSR1, Tau_plugin_adios2_signal_handler) == SIG_ERR) {
      perror("failed to register TAU profile dump signal handler");
    }

    RtsLayer::UnLockDB();

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
