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

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauMetaData.h>
#if TAU_MPI
#include "mpi.h"
#endif

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

#define CONVERT_TO_USEC 1.0/1000000.0 // hopefully the compiler will precompute this.
#define TAU_ADIOS2_PERIODIC_DEFAULT false
#define TAU_ADIOS2_PERIOD_DEFAULT 2000000 // microseconds
#define TAU_ADIOS2_USE_SELECTION_DEFAULT false
#define TAU_ADIOS2_FILENAME "tau-metrics"
#define TAU_ADIOS2_ONE_FILE_DEFAULT false
#define TAU_ADIOS2_ENGINE "BPFile"

// This will enable some checking to make sure we don't have call stack violations.
// #define DO_VALIDATION

/* Some forward declarations that we need */
tau::Profiler *Tau_get_timer_at_stack_depth(int);
int Tau_plugin_adios2_function_exit(
    Tau_plugin_event_function_exit_data_t* data);
void Tau_dump_ADIOS2_metadata(void);

static bool enabled{false};
static bool initialized{false};
static bool done{false};
static bool _threaded{false};
static int global_comm_size = 1;
static int global_comm_rank = 0;
pthread_mutex_t _my_mutex; // for initialization, termination
// for controlling access (per thread) to vectors of data:
pthread_mutex_t _vector_mutex[TAU_MAX_THREADS];
pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;

namespace tau_plugin {

class plugin_options {
    private:
        plugin_options(void) :
            env_periodic(TAU_ADIOS2_PERIODIC_DEFAULT),
            env_period(TAU_ADIOS2_PERIOD_DEFAULT),
            env_use_selection(TAU_ADIOS2_USE_SELECTION_DEFAULT),
            env_filename(TAU_ADIOS2_FILENAME),
            env_one_file(TAU_ADIOS2_ONE_FILE_DEFAULT),
            env_engine(TAU_ADIOS2_ENGINE)
            {}
    public:
        int env_periodic;
        int env_period;
        bool env_use_selection;
        std::string env_filename;
        int env_one_file;
        std::string env_engine;
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

/* Class containing ADIOS archive info */
class adios {
    private:
        bool opened;
        adios2::ADIOS ad;
        adios2::IO bpIO;
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
        std::unordered_map<std::string, int> prog_names;
        std::unordered_map<std::string, int> value_names;
        std::unordered_map<std::string, int> metadata_keys;
        std::unordered_map<std::string, int> groups;
        std::unordered_map<std::string, int> timers;
        std::unordered_map<std::string, int> event_types;
        std::unordered_map<std::string, int> counters;
    public:
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
            for(int i = 0 ; i < TAU_MAX_THREADS ; i++) {
                timer_values_array[i].reserve(1024);
                counter_values_array[i].reserve(1024);
                comm_values_array[i].reserve(1024);
            }
        };
        ~adios() {
            close();
        };
        void initialize();
        void define_variables();
        void open();
        void close();
        void define_attribute(const std::string& name,
            const std::string& value);
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
        std::vector<
            std::pair<
                unsigned long, 
                std::array<unsigned long, 5> > > 
            timer_values_array[TAU_MAX_THREADS];
        std::vector<
            std::pair<
                unsigned long,
                std::array<unsigned long, 5> > > 
            counter_values_array[TAU_MAX_THREADS];
        std::vector<
            std::pair<
                unsigned long,
                std::array<unsigned long, 7> > > 
            comm_values_array[TAU_MAX_THREADS];
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
        // Get the group of processes in MPI_COMM_WORLD
        MPI_Group world_group;
        PMPI_Comm_group(MPI_COMM_WORLD, &world_group);

        int n = 1;
        const int ranks[1] = {world_rank};

        MPI_Group adios_group;
        PMPI_Group_incl(world_group, 1, ranks, &adios_group);
        PMPI_Comm_create_group(MPI_COMM_WORLD, adios_group, 0, &adios_comm);
        PMPI_Group_free(&world_group);
        PMPI_Group_free(&adios_group);
    }
    Tau_global_incr_insideTAU();
    ad = adios2::ADIOS(adios_comm, true);
#else
    /** ADIOS class factory of IO class objects, DebugON is recommended */
    ad = adios2::ADIOS(true);
#endif
    /*** IO class object: settings and factory of Settings: Variables,
     * Parameters, Transports, and Execution: Engines */
    bpIO = ad.DeclareIO("TAU trace data");
    // if not defined by user, we can change the default settings
    // BPFile is the default engine
    bpIO.SetEngine(thePluginOptions().env_engine);
    bpIO.SetParameters({{"num_threads", "2"}});

    // ISO-POSIX file output is the default transport (called "File")
    // Passing parameters to the transport
    bpIO.AddTransport("File", {{"Library", "POSIX"}});
    Tau_global_decr_insideTAU();
}

void adios::define_variables(void) {
    program_count = bpIO.DefineVariable<int>("program_count");
    comm_size = bpIO.DefineVariable<int>("comm_rank_count");
    thread_count = bpIO.DefineVariable<int>("thread_count");
    event_type_count = bpIO.DefineVariable<int>("event_type_count");
    timer_count = bpIO.DefineVariable<int>("timer_count");
    timer_event_count = bpIO.DefineVariable<size_t>("timer_event_count");
    counter_count = bpIO.DefineVariable<int>("counter_count");
    counter_event_count = bpIO.DefineVariable<size_t>("counter_event_count");
    comm_count = bpIO.DefineVariable<size_t>("comm_count");
    /* These are 2 dimensional variables, so they get special treatment */
    event_timestamps = bpIO.DefineVariable<unsigned long>("event_timestamps", {1, 6}, {0, 0}, {1, 6});
    counter_values = bpIO.DefineVariable<unsigned long>("counter_values", {1, 6}, {0, 0}, {1, 6});
    comm_timestamps = bpIO.DefineVariable<unsigned long>("comm_timestamps", {1, 8}, {0, 0}, {1, 8});
}

/* Open the ADIOS archive */
void adios::open() {
    if (!opened) {
        Tau_global_incr_insideTAU();
        std::stringstream ss;
        ss << thePluginOptions().env_filename; 
        if (!thePluginOptions().env_one_file) {
            ss << "-" << global_comm_rank;
        }
        ss << ".bp";
        printf("Writing %s\n", ss.str().c_str());
        bpWriter = bpIO.Open(ss.str(), adios2::Mode::Write);
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
void adios::define_attribute(const std::string& name, const std::string& value) {
    static std::unordered_set<std::string> seen;
    if (seen.count(name) == 0) {
        seen.insert(name);
        bpIO.DefineAttribute<std::string>(name, value);
    }
}

/* Write the arrays of timestamps and values for this step */
void adios::write_variables(void)
{
    int programs = get_prog_count();
    int comm_ranks = global_comm_size;
    int threads = get_thread_count();
    int event_types = get_event_type_count();
    int timers = get_timer_count();
    int counters = get_counter_count();

    Tau_global_incr_insideTAU();
    bpWriter.BeginStep();

    /* sort into one big vector from all threads */
#if 0
    std::vector<std::pair<unsigned long, std::array<unsigned long, 5> > > 
        merged_timers(timer_values_array[0]);
    timer_values_array[0].clear();
    for (int t = 1 ; t < threads ; t++) {
        merged_timers.insert(merged_timers.end(),
            timer_values_array[t].begin(),
            timer_values_array[t].end());
        timer_values_array[t].clear();
    }
    std::sort(merged_timers.begin(), merged_timers.end());
#else
    pthread_mutex_lock(&_vector_mutex[0]);
    // make a list from the first thread of data - copying the data in!
    std::list<std::pair<unsigned long, std::array<unsigned long, 5> > > 
        merged_timers(timer_values_array[0].begin(),
                      timer_values_array[0].end());
    // this clear will empty the vector and destroy the objects!
    timer_values_array[0].clear();
    pthread_mutex_unlock(&_vector_mutex[0]);
    for (int t = 1 ; t < threads ; t++) {
        pthread_mutex_lock(&_vector_mutex[t]);
        // make a list from the next thread of data - copying the data in!
        std::list<std::pair<unsigned long, std::array<unsigned long, 5> > > 
            next_thread(timer_values_array[t].begin(),
                        timer_values_array[t].end());
        // this clear will empty the vector and destroy the objects!
        timer_values_array[t].clear();
        pthread_mutex_unlock(&_vector_mutex[t]);
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
#endif
    size_t num_timer_values = merged_timers.size();

    std::vector<unsigned long> all_timers(6,0);;
    all_timers.reserve(6*merged_timers.size());
    int timer_value_index = 0;
    for (auto it = merged_timers.begin() ; it != merged_timers.end() ; it++) {
        all_timers[timer_value_index++] = it->second[0];
        all_timers[timer_value_index++] = it->second[1];
        all_timers[timer_value_index++] = it->second[2];
        all_timers[timer_value_index++] = it->second[3];
        all_timers[timer_value_index++] = it->second[4];
        all_timers[timer_value_index++] = it->first;
#ifdef DO_VALIDATION
        if (it->second[3] == 0) {
            // on entry
            timer_stack[it->second[2]].push(it->second[4]);
        } else if (it->second[3] == 1) {
            // on exit
            if (timer_stack[it->second[2]].top() != it->second[4]) {
                fprintf(stderr, "Stack violation.\n");
                fprintf(stderr, "thread %lu, %lu != %lu, timestamp %lu\n", it->second[2], timer_stack[it->second[2]].top(), it->second[4], it->first);
            } else if (timer_stack[it->second[2]].size() == 0) {
                fprintf(stderr, "Stack violation.\n");
                fprintf(stderr, "Stack for thread %lu is empty, timestamp %lu.\n", it->second[2], it->first);
            }
            timer_stack[it->second[2]].pop();
        }
#endif
    }


    /* sort into one big vector from all threads */
    pthread_mutex_lock(&_vector_mutex[0]);
    std::vector<std::pair<unsigned long, std::array<unsigned long, 5> > > 
        merged_counters(counter_values_array[0]);
    pthread_mutex_unlock(&_vector_mutex[0]);
    for (int t = 1 ; t < threads ; t++) {
        pthread_mutex_lock(&_vector_mutex[t]);
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
        pthread_mutex_unlock(&_vector_mutex[t]);
    }

    /* sort into one big vector from all threads */
    pthread_mutex_lock(&_vector_mutex[0]);
    std::vector<std::pair<unsigned long, std::array<unsigned long, 7> > > 
        merged_comms(comm_values_array[0]);
    pthread_mutex_unlock(&_vector_mutex[0]);
    for (int t = 1 ; t < threads ; t++) {
        pthread_mutex_lock(&_vector_mutex[t]);
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
        pthread_mutex_unlock(&_vector_mutex[t]);
    }

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
            define_attribute(ss.str(), prog_name);
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
            define_attribute(ss.str(), event_type);
        }   
        return event_types[event_type];
    }   

    /* Keep a map of timers to indexes */
    int adios::check_timer(const char * timer) {
        std::string tmp(timer);
        if (timers.count(tmp) == 0) {
            std::stringstream ss; 
            int num = timers.size();
            ss << "timer " << num;
            pthread_mutex_lock(&_my_mutex);
            // check to make sure another thread didn't create it already
            if (timers.count(tmp) == 0) {
                timers[tmp] = num;
                // printf("%d = %s\n", num, timer);
                define_attribute(ss.str(), tmp);
            }
            pthread_mutex_unlock(&_my_mutex);
        }
        return timers[tmp];
    }

    /* Keep a map of counters to indexes */
    int adios::check_counter(const char * counter) {
        std::string tmp(counter);
        if (counters.count(tmp) == 0) {
            std::stringstream ss;
            int num = counters.size();
            ss << "counter " << num;
            pthread_mutex_lock(&_my_mutex);
            // check to make sure another thread didn't create it already
            if (counters.count(tmp) == 0) {
                counters[tmp] = num;
                define_attribute(ss.str(), tmp);
            }
            pthread_mutex_unlock(&_my_mutex);
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
    Tau_global_incr_insideTAU();
    pthread_mutex_lock(&_my_mutex);
    my_adios->write_variables();
    pthread_mutex_unlock(&_my_mutex);
    Tau_global_decr_insideTAU();
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

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_adios2_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    if (!enabled || data->tid != 0) return 0;
    fprintf(stdout, "TAU PLUGIN ADIOS2 Pre-Finalize\n"); fflush(stdout);
    Tau_ADIOS2_stop_worker();
    Tau_plugin_event_function_exit_data_t exit_data;
    // safe to assume 0?
    //int tid = exit_data.tid;
    RtsLayer::UnLockDB();
    for (int tid = TAU_MAX_THREADS-1 ; tid >= 0 ; tid--) {
      int depth = Tau_get_current_stack_depth(tid);
      for (int i = depth ; i > -1 ; i--) {
        tau::Profiler *profiler = Tau_get_timer_at_stack_depth(i);
        if (profiler->ThisFunction->GetName() == NULL) {
          // small memory leak, but at shutdown.
          exit_data.timer_name = strdup(".TAU application");
        } else {
          exit_data.timer_name = profiler->ThisFunction->GetName();
          exit_data.timer_group = profiler->ThisFunction->GetAllGroups();
        }
        exit_data.tid = tid;
        double CurrentTime[TAU_MAX_COUNTERS] = { 0 };
        RtsLayer::getUSecD(tid, CurrentTime);
        exit_data.timestamp = (x_uint64)CurrentTime[0];    // USE COUNTER1 for tracing
        //printf("%d Stopping %s\n", tid, exit_data.timer_name);
        Tau_plugin_adios2_function_exit(&exit_data);
      }
    }
    RtsLayer::UnLockDB();
    /* write those last events... */
    Tau_plugin_event_dump_data_t* dummy_data;
    Tau_plugin_adios2_dump(dummy_data);
    /* Close ADIOS archive */
    if (my_adios != nullptr) {
        my_adios->close();
        my_adios = nullptr;
    }
    return 0;
}

/* This happens after MPI_Init, and after all TAU metadata variables have been
 * read */
int Tau_plugin_adios2_post_init(Tau_plugin_event_post_init_data_t* data) {
    if (!enabled) return 0;
    return 0;
}

/* This happens from Profiler.cpp, when data is written out. */
int Tau_plugin_adios2_end_of_execution(Tau_plugin_event_end_of_execution_data_t* data) {
    if (!enabled || data->tid != 0) return 0;
    fprintf(stdout, "TAU PLUGIN ADIOS2 Finalize\n"); fflush(stdout);
    Tau_ADIOS2_stop_worker();
    Tau_plugin_event_dump_data_t* dummy_data;
    Tau_plugin_adios2_dump(dummy_data);
    enabled = false;
    /* Close ADIOS archive */
    if (my_adios != nullptr) {
        my_adios->close();
        my_adios = nullptr;
    }
    if (tau_plugin::thePluginOptions().env_periodic) {
        pthread_cond_destroy(&_my_cond);
        for (int i = 0 ; i < TAU_MAX_THREADS ; i++) {
            pthread_mutex_destroy(&(_vector_mutex[i]));
        }
        pthread_mutex_destroy(&_my_mutex);
    }
    return 0;
}

void Tau_dump_ADIOS2_metadata(void) {
    int tid = RtsLayer::myThread();
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
                my_adios->define_attribute(ss.str(), std::string(it->second->data.cval));
                break;
            case TAU_METADATA_TYPE_INTEGER:
                my_adios->define_attribute(ss.str(), std::to_string(it->second->data.ival));
                break;
            case TAU_METADATA_TYPE_DOUBLE:
                my_adios->define_attribute(ss.str(), std::to_string(it->second->data.dval));
                break;
            case TAU_METADATA_TYPE_TRUE:
                my_adios->define_attribute(ss.str(), std::string("true"));
                break;
            case TAU_METADATA_TYPE_FALSE:
                my_adios->define_attribute(ss.str(), std::string("false"));
                break;
            case TAU_METADATA_TYPE_NULL:
                my_adios->define_attribute(ss.str(), std::string("(null)"));
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
    ss << "MetaData:" << global_comm_rank << ":" << RtsLayer::myThread() << ":" << data->name;
    switch(data->value->type) {
        case TAU_METADATA_TYPE_STRING:
            my_adios->define_attribute(ss.str(), std::string(data->value->data.cval));
            break;
        case TAU_METADATA_TYPE_INTEGER:
            my_adios->define_attribute(ss.str(), std::to_string(data->value->data.ival));
            break;
        case TAU_METADATA_TYPE_DOUBLE:
            my_adios->define_attribute(ss.str(), std::to_string(data->value->data.dval));
            break;
        case TAU_METADATA_TYPE_TRUE:
            my_adios->define_attribute(ss.str(), std::string("true"));
            break;
        case TAU_METADATA_TYPE_FALSE:
            my_adios->define_attribute(ss.str(), std::string("false"));
            break;
        case TAU_METADATA_TYPE_NULL:
            my_adios->define_attribute(ss.str(), std::string("(null)"));
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
    pthread_mutex_lock(&_vector_mutex[data->tid]);
    auto &tmp = my_adios->comm_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp, 
            std::move(tmparray)
        )
    );
    pthread_mutex_unlock(&_vector_mutex[data->tid]);
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
    pthread_mutex_lock(&_vector_mutex[data->tid]);
    auto &tmp = my_adios->comm_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp, 
            std::move(tmparray)
        )
    );
    pthread_mutex_unlock(&_vector_mutex[data->tid]);
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
    pthread_mutex_lock(&_vector_mutex[data->tid]);
    auto &tmp = my_adios->timer_values_array[data->tid];
#ifdef DO_VALIDATION
    unsigned long ts = my_adios->previous_timestamp[data->tid] > data->timestamp ? my_adios->previous_timestamp[data->tid] + 1 : data->timestamp;
    my_adios->previous_timestamp[data->tid] = ts;
    my_adios->pre_timer_stack[tmparray[2]].push(tmparray[4]);
#else
    unsigned long ts = data->timestamp;
#endif
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    pthread_mutex_unlock(&_vector_mutex[data->tid]);
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
    pthread_mutex_lock(&_vector_mutex[data->tid]);
    auto &tmp = my_adios->timer_values_array[data->tid];
#ifdef DO_VALIDATION
    unsigned long ts = my_adios->previous_timestamp[data->tid] > data->timestamp ? my_adios->previous_timestamp[data->tid] + 1 : data->timestamp;
    my_adios->previous_timestamp[data->tid] = ts;
    if (my_adios->pre_timer_stack[tmparray[2]].top() != tmparray[4]) {
      fprintf(stderr, "Pre: Stack violation.\n");
      fprintf(stderr, "Pre: thread %lu, %lu != %lu, timestamp %lu\n", tmparray[2], my_adios->pre_timer_stack[tmparray[2]].top(), tmparray[4], data->timestamp);
    } else if (my_adios->pre_timer_stack[tmparray[2]].size() == 0) {
      fprintf(stderr, "Pre: Stack violation.\n");
      fprintf(stderr, "Pre: Stack for thread %lu is empty, timestamp %lu.\n", tmparray[2], data->timestamp);
    }
    my_adios->pre_timer_stack[tmparray[2]].pop();
#else
    unsigned long ts = data->timestamp;
#endif
    tmp.push_back(
        std::make_pair(
            ts,
            std::move(tmparray)
        )
    );
    pthread_mutex_unlock(&_vector_mutex[data->tid]);
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
    pthread_mutex_lock(&_vector_mutex[data->tid]);
    auto &tmp = my_adios->counter_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp, 
            std::move(tmparray)
        )
    );
    pthread_mutex_unlock(&_vector_mutex[data->tid]);
    return 0;
}

void * Tau_ADIOS2_thread_function(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;

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
            Tau_plugin_event_dump_data_t* dummy_data;
            Tau_plugin_adios2_dump(dummy_data);
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


/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv) {
    Tau_plugin_callbacks_t * cb = (Tau_plugin_callbacks_t*)malloc(sizeof(Tau_plugin_callbacks_t));
    fprintf(stdout, "TAU PLUGIN ADIOS2 Init\n"); fflush(stdout);
    tau_plugin::Tau_ADIOS2_parse_environment_variables();
#if TAU_MPI
    PMPI_Comm_size(MPI_COMM_WORLD, &global_comm_size);
    PMPI_Comm_rank(MPI_COMM_WORLD, &global_comm_rank);
#endif
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

    /* Required event support */
    cb->Dump = Tau_plugin_adios2_dump;
    cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_func;
    cb->PostInit = Tau_plugin_adios2_post_init;
    cb->PreEndOfExecution = Tau_plugin_adios2_pre_end_of_execution;
    cb->EndOfExecution = Tau_plugin_adios2_end_of_execution;
    /* Trace events */
    cb->Send = Tau_plugin_adios2_send;
    cb->Recv = Tau_plugin_adios2_recv;
    cb->FunctionEntry = Tau_plugin_adios2_function_entry;
    cb->FunctionExit = Tau_plugin_adios2_function_exit;
    cb->AtomicEventTrigger = Tau_plugin_adios2_atomic_trigger;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb);

    /* Open the ADIOS archive */
    my_adios = new tau_plugin::adios();
    enabled = true;
    Tau_dump_ADIOS2_metadata();
    my_adios->check_event_type(std::string("ENTRY"));
    my_adios->check_event_type(std::string("EXIT"));
    my_adios->check_event_type(std::string("SEND"));
    my_adios->check_event_type(std::string("RECV"));

    /* spawn the thread if doing periodic */
    if (tau_plugin::thePluginOptions().env_periodic) {
        _threaded = true;
        init_lock(&_my_mutex);
        for (int i = 0 ; i < TAU_MAX_THREADS ; i++) {
            init_lock(&(_vector_mutex[i]));
        }
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
    for (int tid = TAU_MAX_THREADS-1 ; tid >= 0 ; tid--) {
      Tau_plugin_event_function_entry_data_t entry_data;
      // safe to assume 0?
      int depth = Tau_get_current_stack_depth(tid);
      for (int i = 0 ; i <= depth ; i++) {
        tau::Profiler *profiler = Tau_get_timer_at_stack_depth(i);
        entry_data.timer_name = profiler->ThisFunction->GetName();
        entry_data.timer_group = profiler->ThisFunction->GetAllGroups();
        entry_data.tid = tid;
        entry_data.timestamp = (x_uint64)profiler->StartTime[0];
        //printf("%d,%d Starting %s\n", getpid(), tid, data.timer_name);
        Tau_plugin_adios2_function_entry(&entry_data);
      }
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
