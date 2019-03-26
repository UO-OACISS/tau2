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

/* Some forward declarations that we need */
tau::Profiler *Tau_get_timer_at_stack_depth(int);
int Tau_plugin_adios2_function_exit(
    Tau_plugin_event_function_exit_data_t* data);
void Tau_dump_ADIOS2_metadata(void);

static bool enabled(false);
static bool initialized(false);
static int comm_size = 1;
static int comm_rank = 0;

namespace tau_plugin {

/* Class containing ADIOS archive info */
class adios {
    private:
        bool opened;
        adios2::ADIOS ad;
        adios2::IO bpIO;
        adios2::Engine bpWriter;
        adios2::Variable<int> program_count;
        adios2::Variable<int> comm_rank_count;
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
        int max_comm_rank;
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
            max_comm_rank(0),
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
        int check_event_type(std::string& event_type);
        int check_timer(const char * timer);
        int check_counter(const char * counter);
        // do we need this?
        void check_comm_rank(int comm_rank);
        // do we need this?
        void check_thread(int thread);
        int get_prog_count(void) { return prog_names.size(); }
        int get_value_name_count(void) { return value_names.size(); }
        int get_timer_count(void) { return timers.size(); }
        int get_event_type_count(void) { return event_types.size(); }
        int get_counter_count(void) { return counters.size(); }
        int get_comm_rank_count(void) { return max_comm_rank+1; }
        int get_thread_count(void) { return max_threads+1; }
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
};

void adios::initialize() {
#if TAU_MPI
    // Get the rank and size in the original communicator
    int world_rank, world_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    PMPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int n = 1;
    const int ranks[1] = {world_rank};

    MPI_Comm adios_comm;
    MPI_Group adios_group;
    PMPI_Group_incl(world_group, 1, ranks, &adios_group);
    PMPI_Comm_create_group(MPI_COMM_WORLD, adios_group, 0, &adios_comm);
    PMPI_Group_free(&world_group);
    PMPI_Group_free(&adios_group);
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
    bpIO.SetEngine("BPFile");
    bpIO.SetParameters({{"num_threads", "2"}});

    // ISO-POSIX file output is the default transport (called "File")
    // Passing parameters to the transport
    bpIO.AddTransport("File", {{"Library", "POSIX"}});
}

void adios::define_variables(void) {
    program_count = bpIO.DefineVariable<int>("program_count");
    comm_rank_count = bpIO.DefineVariable<int>("comm_rank_count");
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
        std::stringstream ss;
        ss << "./tautrace."; 
        ss << comm_rank;
        ss << ".bp";
        printf("Writing %s\n", ss.str().c_str());
        bpWriter = bpIO.Open(ss.str(), adios2::Mode::Write);
        opened = true;
    }
}

/* Close the ADIOS archive */
void adios::close() {
    if (opened) {
        bpWriter.Close();
        opened = false;
        enabled = false;
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
    int comm_ranks = get_comm_rank_count();
    int threads = get_thread_count();
    int event_types = get_event_type_count();
    int timers = get_timer_count();
    int counters = get_counter_count();

    bpWriter.BeginStep();

    /* sort into one big vector from all threads */
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
    }

    /* sort into one big vector from all threads */
    std::vector<std::pair<unsigned long, std::array<unsigned long, 5> > > 
        merged_counters(counter_values_array[0]);
    counter_values_array[0].clear();
    for (int t = 1 ; t < threads ; t++) {
        merged_counters.insert(merged_counters.end(),
            counter_values_array[t].begin(),
            counter_values_array[t].end());
        counter_values_array[t].clear();
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

    /* sort into one big vector from all threads */
    std::vector<std::pair<unsigned long, std::array<unsigned long, 8> > > 
        merged_comms(comm_values_array[0]);
    comm_values_array[0].clear();
    for (int t = 1 ; t < threads ; t++) {
        merged_comms.insert(merged_comms.end(),
            comm_values_array[t].begin(),
            comm_values_array[t].end());
        comm_values_array[t].clear();
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

    bpWriter.Put(program_count, &programs);
    bpWriter.Put(comm_rank_count, &comm_ranks);
    bpWriter.Put(thread_count, &threads);
    bpWriter.Put(event_type_count, &event_types);
    bpWriter.Put(timer_count, &timers);
    bpWriter.Put(timer_event_count, &num_timer_values);
    bpWriter.Put(counter_count, &counters);
    bpWriter.Put(counter_event_count, &num_counter_values);
    bpWriter.Put(comm_count, &num_comm_values);

    if (num_timer_values > 0) {
        event_timestamps.SetShape({num_timer_values, 6});
        const adios2::Dims timer_start{0, 0};
        const adios2::Dims timer_count{static_cast<size_t>(num_timer_values), 6};
        const adios2::Box<adios2::Dims> timer_selection{timer_start, timer_count};
        event_timestamps.SetSelection(timer_selection);
        bpWriter.Put(event_timestamps, all_timers.data());
    }

    if (num_counter_values > 0) {
        counter_values.SetShape({num_counter_values, 6});
        const adios2::Dims counter_start{0, 0};
        const adios2::Dims counter_count{static_cast<size_t>(num_counter_values), 6};
        const adios2::Box<adios2::Dims> counter_selection{counter_start, counter_count};
        counter_values.SetSelection(counter_selection);
        bpWriter.Put(counter_values, all_counters.data());
    }

    if (num_comm_values > 0) {
        comm_timestamps.SetShape({num_comm_values, 8});
        const adios2::Dims comm_start{0, 0};
        const adios2::Dims comm_count{static_cast<size_t>(num_comm_values), 8};
        const adios2::Box<adios2::Dims> comm_selection{comm_start, comm_count};
        comm_timestamps.SetSelection(comm_selection);
        bpWriter.Put(comm_timestamps, all_comms.data());
    }

    bpWriter.EndStep();
}

/* sos object methods */

    /* Keep a map of program names to indexes */
    int adios::check_prog_name(char * prog_name) {
        if (prog_names.count(prog_name) == 0) {
            std::stringstream ss; 
            ss << "program_name " << prog_names.size();
            prog_names[prog_name] = prog_names.size();
            define_attribute(ss.str(), prog_name);
        }   
        return prog_names[prog_name];
    }   

    /* Keep a map of event types to indexes */
    int adios::check_event_type(std::string& event_type) {
        if (event_types.count(event_type) == 0) {
            std::stringstream ss; 
            ss << "event_type " << event_types.size();
            event_types[event_type] = event_types.size();
            define_attribute(ss.str(), event_type);
        }   
        return event_types[event_type];
    }   

    /* Keep a map of timers to indexes */
    int adios::check_timer(const char * timer) {
        std::string tmp(timer);
        if (timers.count(tmp) == 0) {
            std::stringstream ss; 
            ss << "timer " << timers.size();
            timers[tmp] = timers.size();
            define_attribute(ss.str(), tmp);
        }
        return timers[tmp];
    }

    /* Keep a map of counters to indexes */
    int adios::check_counter(const char * counter) {
        std::string tmp(counter);
        if (counters.count(tmp) == 0) {
            std::stringstream ss;
            ss << "counter " << counters.size();
            counters[tmp] = counters.size();
            define_attribute(ss.str(), tmp);
        }
        return counters[tmp];
    }

    /* Keep a map of max number of MPI communicators */
    void adios::check_comm_rank(int comm_rank) {
        if (comm_rank > max_comm_rank) {
            max_comm_rank = comm_rank;
        }
    }

    /* Keep a map of max number of threads per process */
    void adios::check_thread(int thread) {
        if (thread > max_threads) {
            max_threads = thread;
        }
    }

}; // end namespace tau_plugin

static tau_plugin::adios * my_adios{nullptr};

int Tau_plugin_adios2_dump(Tau_plugin_event_dump_data_t* data) {
    if (!enabled) return 0;
    my_adios->write_variables();
    return 0;
}

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_adios2_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    if (!enabled) return 0;
    fprintf(stdout, "TAU PLUGIN ADIOS2 Pre-Finalize\n"); fflush(stdout);
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
        //printf("%d,%d Stopping %s\n", getpid(), tid, data.timer_name);
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
    enabled = false;
    fprintf(stdout, "TAU PLUGIN ADIOS2 Finalize\n"); fflush(stdout);
    /* Close ADIOS archive */
    if (my_adios != nullptr) {
        my_adios->close();
        my_adios = nullptr;
    }
    return 0;
}

void Tau_dump_ADIOS2_metadata(void) {
    int tid = RtsLayer::myThread();
    int nodeid = TAU_PROFILE_GET_NODE();
    for (MetaDataRepo::iterator it = Tau_metadata_getMetaData(tid).begin();
         it != Tau_metadata_getMetaData(tid).end(); it++) {
        // check for executable name
        if (strcmp(it->first.name, "Executable") == 0) {
            my_adios->check_prog_name(it->second->data.cval);
        }
        std::stringstream ss;
        ss << "MetaData:" << comm_rank << ":" << tid << ":" << it->first.name;
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
}

/* This happens when a Metadata field is saved. */
int Tau_plugin_metadata_registration_complete_func(Tau_plugin_event_metadata_registration_data_t* data) {
    if (!enabled) return 0;
    //fprintf(stdout, "TAU Metadata registration\n"); fflush(stdout);
    std::stringstream ss;
    ss << "MetaData:" << comm_rank << ":" << RtsLayer::myThread() << ":" << data->name;
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
    tmparray[1] = (unsigned long)(comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(event_index);
    tmparray[4] = (unsigned long)(data->message_tag);
    tmparray[5] = (unsigned long)(data->destination);
    tmparray[6] = (unsigned long)(data->bytes_sent);
    auto &tmp = my_adios->comm_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp, 
            std::move(tmparray)
        )
    );
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
    tmparray[1] = (unsigned long)(comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(event_index);
    tmparray[4] = (unsigned long)(data->message_tag);
    tmparray[5] = (unsigned long)(data->destination);
    tmparray[6] = (unsigned long)(data->bytes_sent);
    auto &tmp = my_adios->comm_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp, 
            std::move(tmparray)
        )
    );
    return 0;
}

/* This happens on Tau_start() */
int Tau_plugin_adios2_function_entry(Tau_plugin_event_function_entry_data_t* data) {
    if (!enabled) return 0;
    /* First, check to see if we are including/excluding this timer */
#if 0
    if (skip_timer(data->timer_name)) {
        return 0;
    }
#endif
    /* todo: filter on group */
    int timer_index = my_adios->check_timer(data->timer_name);
    static std::string entrystr("ENTRY");
    int event_index = my_adios->check_event_type(entrystr);
    my_adios->check_thread(data->tid);
    std::array<unsigned long, 5> tmparray;
    tmparray[0] = 0;
    tmparray[1] = (unsigned long)(comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(event_index);
    tmparray[4] = (unsigned long)(timer_index);
    auto &tmp = my_adios->timer_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp, 
            std::move(tmparray)
        )
    );
    return 0;
}

/* This happens on Tau_stop() */
int Tau_plugin_adios2_function_exit(Tau_plugin_event_function_exit_data_t* data) {
    if (!enabled) return 0;
#if 0
    /* First, check to see if we are including/excluding this timer */
    if (skip_timer(data->timer_name)) {
        return 0;
    }
#endif
    /* todo: filter on group */
    int timer_index = my_adios->check_timer(data->timer_name);
    static std::string exitstr("EXIT");
    int event_index = my_adios->check_event_type(exitstr);
    std::array<unsigned long, 5> tmparray;
    tmparray[0] = 0UL;
    tmparray[1] = (unsigned long)(comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(event_index);
    tmparray[4] = (unsigned long)(timer_index);
    auto &tmp = my_adios->timer_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp, 
            std::move(tmparray)
        )
    );
    return 0;
}

/* This happens on Tau_userevent() */
int Tau_plugin_adios2_atomic_trigger(Tau_plugin_event_atomic_event_trigger_data_t* data) {
    if (!enabled) return 0;
#if 0
    /* First, check to see if we are including/excluding this counter */
    if (skip_counter(data->counter_name)) {
        return 0;
    }
#endif
    int counter_index = my_adios->check_counter(data->counter_name);
    my_adios->check_thread(data->tid);
    std::array<unsigned long, 5> tmparray;
    tmparray[0] = 0UL;
    tmparray[1] = (unsigned long)(comm_rank);
    tmparray[2] = (unsigned long)(data->tid);
    tmparray[3] = (unsigned long)(counter_index);
    tmparray[4] = (unsigned long)(data->value);
    auto &tmp = my_adios->counter_values_array[data->tid];
    tmp.push_back(
        std::make_pair(
            data->timestamp, 
            std::move(tmparray)
        )
    );
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv) {
    Tau_plugin_callbacks_t * cb = (Tau_plugin_callbacks_t*)malloc(sizeof(Tau_plugin_callbacks_t));
    fprintf(stdout, "TAU PLUGIN ADIOS2 Init\n"); fflush(stdout);
#if TAU_MPI
    PMPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    PMPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
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
    my_adios->check_comm_rank(comm_rank);
    enabled = true;
    Tau_dump_ADIOS2_metadata();

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


#endif // TAU_ADIOS2
