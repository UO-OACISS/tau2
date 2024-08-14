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
#include <assert.h>
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
#include "Tau_scoped_timer.h"

#define TAU_ADIOS2_PERIODIC_DEFAULT false
#define TAU_ADIOS2_PERIOD_DEFAULT 2000000 // microseconds
#define TAU_ADIOS2_ONE_FILE_DEFAULT true
#define TAU_ADIOS2_ENGINE "BPFile"
#define TAU_ADIOS2_FILENAME "tauprofile"

static bool enabled{false};
static bool initialized{false};
static bool opened{false};
static bool done{false};
static bool _threaded{false};
static int world_comm_size = 1;
static int world_comm_rank = 0;
static int adios_comm_size = 1;
static int adios_comm_rank = 0;
pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;

/* Some ADIOS variables */
adios2::ADIOS ad;
adios2::IO bpIO;
adios2::Engine bpWriter;
//std::map<std::string, adios2::Variable<double> > timers;
std::map<std::string, std::vector<double> > timers;
std::map<std::string, std::vector<double> > counters;
adios2::Variable<int> num_threads_var;
adios2::Variable<int> num_metrics_var;
adios2::Variable<int> num_ranks_var;

char *_program_path()
{
#if defined(__APPLE__)
    return NULL;
#else
    char path[PATH_MAX] = {0};
    if (readlink("/proc/self/exe", path, PATH_MAX) == -1) {
        return(NULL);
    }
    std::string tmp(path);
    size_t i = tmp.rfind('/', tmp.length());
    if (i != string::npos) {
        //sprintf(path, "%s", tmp.substr(i+1, tmp.length() - i).c_str());
        path[i] = '\0';
    }
    return strdup(path);
#endif
}

char *_program_name()
{
#if defined(__APPLE__)
    return NULL;
#else
    char path[PATH_MAX] = {0};
    if (readlink("/proc/self/exe", path, PATH_MAX) == -1) {
        return(NULL);
    }
    std::string tmp(path);
    size_t i = tmp.rfind('/', tmp.length());
    if (i != string::npos) {
        snprintf(path, sizeof(path),  "%s", tmp.substr(i+1, tmp.length() - i).c_str());
    }
    return strdup(path);
#endif
}

class plugin_options {
    private:
        plugin_options(void) :
            env_periodic(TAU_ADIOS2_PERIODIC_DEFAULT),
            env_period(TAU_ADIOS2_PERIOD_DEFAULT),
            env_one_file(TAU_ADIOS2_ONE_FILE_DEFAULT),
            env_engine(TAU_ADIOS2_ENGINE),
            env_filename(TAU_ADIOS2_FILENAME)
            {}
    public:
        int env_periodic;
        int env_period;
        int env_one_file;
        std::string env_engine;
        std::string env_filename;
        static plugin_options& thePluginOptions() {
            static plugin_options tpo;
            return tpo;
        }
};

inline plugin_options& thePluginOptions() {
    return plugin_options::thePluginOptions();
}

void Tau_ADIOS2_parse_environment_variables(void);

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
    tmp = getenv("TAU_ADIOS2_ONE_FILE");
    thePluginOptions().env_one_file = parse_bool(tmp, TAU_ADIOS2_ONE_FILE_DEFAULT);
    tmp = getenv("TAU_ADIOS2_ENGINE");
    if (tmp != NULL) {
      thePluginOptions().env_engine = strdup(tmp);
    }
    tmp = getenv("TAU_ADIOS2_FILENAME");
    if (tmp != NULL) {
      thePluginOptions().env_filename = strdup(tmp);
    } else {
        // add the program name to make it unique
        char * program = _program_name();
        if (program != NULL && strlen(program) > 0) {
            std::stringstream ss;
            ss << thePluginOptions().env_filename << "-" << program;
            free(program);
            thePluginOptions().env_filename = std::string(ss.str());
        }
    }
}

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

void Tau_ADIOS2_stop_worker(void) {
    if (!enabled) return;
    if (!_threaded) return;
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    if (thePluginOptions().env_periodic) {
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

void Tau_dump_ADIOS2_metadata() {
    tau::plugins::ScopedTimer(__func__);
    int tid = RtsLayer::myThread();
    int nodeid = TAU_PROFILE_GET_NODE();
    for (MetaDataRepo::iterator it = Tau_metadata_getMetaData(tid).begin();
         it != Tau_metadata_getMetaData(tid).end(); it++) {
        std::stringstream ss;
        ss << "TAU:" << nodeid << ":" << tid << ":MetaData:" << it->first.name;
        switch(it->second->type) {
            case TAU_METADATA_TYPE_STRING:
                bpIO.DefineAttribute<std::string>(ss.str(),
                    it->second->data.cval);
                break;
            case TAU_METADATA_TYPE_INTEGER:
                bpIO.DefineAttribute<int>(ss.str(), it->second->data.ival);
                break;
            case TAU_METADATA_TYPE_DOUBLE:
                bpIO.DefineAttribute<double>(ss.str(), it->second->data.dval);
                break;
            case TAU_METADATA_TYPE_TRUE:
                bpIO.DefineAttribute<std::string>(ss.str(),
                    std::string("true"));
                break;
            case TAU_METADATA_TYPE_FALSE:
                bpIO.DefineAttribute<std::string>(ss.str(),
                    std::string("false"));
                break;
            case TAU_METADATA_TYPE_NULL:
                bpIO.DefineAttribute<std::string>(ss.str(),
                    std::string("(null)"));
                break;
            default:
                break;
        }
    }
}

void Tau_plugin_adios2_init_adios(void) {
    try {
        char * config = getenv("TAU_ADIOS2_CONFIG_FILE");
        if (config == nullptr) {
            if( access( "./adios2.xml", F_OK ) != -1 ) {
                // file exists
                config = strdup("./adios2.xml");
            } else {
                // file doesn't exist
                config = nullptr;
            }
        }
        /** ADIOS class factory of IO class objects, DebugON is recommended */
#if TAU_MPI
        MPI_Comm adios_comm;
        if (thePluginOptions().env_one_file) {
            PMPI_Comm_dup(MPI_COMM_WORLD, &adios_comm);
        } else {
            PMPI_Comm_dup(MPI_COMM_SELF, &adios_comm);
        }
        PMPI_Comm_rank(adios_comm, &adios_comm_rank);
        PMPI_Comm_size(adios_comm, &adios_comm_size);
        if (config != nullptr) {
            ad = adios2::ADIOS(config, adios_comm);
        } else {
            ad = adios2::ADIOS(adios_comm);
        }
#else
        if (config != nullptr) {
            ad = adios2::ADIOS(config, true);
        } else {
            ad = adios2::ADIOS(true);
        }
#endif
        /*** IO class object: settings and factory of Settings: Variables,
        * Parameters, Transports, and Execution: Engines */
        bpIO = ad.DeclareIO("TAUProfileOutput");

        /* If there was no adios2.xml file, set some reasonable defaults */
        if (config == nullptr) {
            // if not defined by user, we can change the default settings
            // BPFile is the default engine
            bpIO.SetEngine(thePluginOptions().env_engine);
            bpIO.SetParameters({{"num_threads", "1"}});

            // ISO-POSIX file output is the default transport (called "File")
            // Passing parameters to the transport
            bpIO.AddTransport("File", {{"Library", "POSIX"}});
        }

        /* write the metadata as attributes */
        Tau_dump_ADIOS2_metadata();

        /* Create some "always used" variables */

        /** global array : name, { shape (total) }, { start (local) }, {
        * count (local) }, all are constant dimensions */
        const std::size_t Nx = 1;
        const adios2::Dims shape{static_cast<size_t>(Nx * adios_comm_size)};
        const adios2::Dims start{static_cast<size_t>(Nx * adios_comm_rank)};
        const adios2::Dims count{Nx};
        num_threads_var = bpIO.DefineVariable<int>(
            "num_threads", shape, start, count, adios2::ConstantDims);
        num_metrics_var = bpIO.DefineVariable<int>(
            "num_metrics", shape, start, count, adios2::ConstantDims);
        num_ranks_var = bpIO.DefineVariable<int>(
            "num_ranks", shape, start, count, adios2::ConstantDims);
    } catch (std::invalid_argument &e)
    {
        std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                  << world_comm_rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::ios_base::failure &e)
    {
        std::cout << "IO System base failure exception, STOPPING PROGRAM "
                     "from rank "
                  << world_comm_rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Exception, STOPPING PROGRAM from rank " << world_comm_rank << "\n";
        std::cout << e.what() << "\n";
    }
    initialized = true;
}

void Tau_plugin_adios2_open_file(void) {
    std::stringstream ss;
    const char * prefix = TauEnv_get_profile_prefix();
    ss << TauEnv_get_profiledir() << "/";
    if (prefix != NULL) {
        ss << TauEnv_get_profile_prefix();
    }
    ss << thePluginOptions().env_filename;
    if (!thePluginOptions().env_one_file) {
        ss << "-" << world_comm_rank;
    }
    ss << ".bp";
    //printf("Writing %s\n", ss.str().c_str());
    bpWriter = bpIO.Open(ss.str(), adios2::Mode::Write);
    opened = true;
}

void shorten_timer_name(std::string& name) {
    std::size_t index = name.find(" [{");
    if (index != std::string::npos) {
        name = name.substr(0,index);
    }
}

extern void Tau_sampling_finalizeProfile(int tid);

void Tau_plugin_adios2_define_variables(int numThreads, int numCounters,
    const char** counterNames) {
    const std::size_t Nx = numThreads;
    const adios2::Dims shape{static_cast<size_t>(Nx * adios_comm_size)};
    const adios2::Dims start{static_cast<size_t>(Nx * adios_comm_rank)};
    const adios2::Dims count{Nx};

    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    RtsLayer::LockDB();
    // UPDATE SAMPLE DATA!
    if (TauEnv_get_ebs_enabled()) {
        Tau_sampling_finalizeProfile(0);
    }
    /* Copy the function info database so we can release the lock */
    std::vector<FunctionInfo*> tmpTimers(TheFunctionDB());
    RtsLayer::UnLockDB();

    std::map<std::string, std::vector<double> >::iterator timer_map_it;

    //foreach: TIMER
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;

        stringstream ss;
        std::string shortName(fi->GetName());
        shorten_timer_name(shortName);
        ss << shortName << " / Calls";
        timer_map_it = timers.find(ss.str());
        if (timer_map_it == timers.end()) {
            // add the timer to the map
            timers.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads,0)));
            // define the variable for ADIOS
            bpIO.DefineVariable<double>(
                ss.str(), shape, start, count, adios2::ConstantDims);
            for (int i = 0 ; i < numCounters ; i++) {
                ss.str(std::string());
                ss << shortName << " / Inclusive " << counterNames[i];
                // add the timer to the map
                timers.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads,0.0)));
                // define the variable for ADIOS
                bpIO.DefineVariable<double>(
                    ss.str(), shape, start, count, adios2::ConstantDims);
                ss.str(std::string());
                ss << shortName << " / Exclusive " << counterNames[i];
                // add the timer to the map
                timers.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads,0.0)));
                // define the variable for ADIOS
                bpIO.DefineVariable<double>(
                    ss.str(), shape, start, count, adios2::ConstantDims);
            }
        }
    }

    RtsLayer::LockDB();
    tau::AtomicEventDB::const_iterator it2;
    std::map<std::string, std::vector<double> >::iterator counter_map_it;

    // do the same with counters.
    std::stringstream tmp_str;
    for (it2 = tau::TheEventDB().begin(); it2 != tau::TheEventDB().end(); it2++) {
        tau::TauUserEvent *ue = (*it2);
        if (ue == NULL) continue;
        stringstream ss;
	    std::string counter_name(ue->GetName().c_str());
        ss << counter_name << " / Num Events";
        counter_map_it = counters.find(ss.str());
        if (counter_map_it == counters.end()) {
            // add the counter to the map
            counters.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads)));
            // define the variable for ADIOS
            bpIO.DefineVariable<double>(
                ss.str(), shape, start, count, adios2::ConstantDims);
            ss.str(std::string());
            ss << counter_name << " / Min";
            // add the counter to the map
            counters.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads)));
            // define the variable for ADIOS
            bpIO.DefineVariable<double>(
                ss.str(), shape, start, count, adios2::ConstantDims);
            ss.str(std::string());
            ss << counter_name << " / Max";
            // add the counter to the map
            counters.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads)));
            // define the variable for ADIOS
            bpIO.DefineVariable<double>(
                ss.str(), shape, start, count, adios2::ConstantDims);
            ss.str(std::string());
            ss << counter_name << " / Mean";
            // add the counter to the map
            counters.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads)));
            // define the variable for ADIOS
            bpIO.DefineVariable<double>(
                ss.str(), shape, start, count, adios2::ConstantDims);
            ss.str(std::string());
            ss << counter_name << " / Sum Squares";
            // add the counter to the map
            counters.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads)));
            // define the variable for ADIOS
            bpIO.DefineVariable<double>(
                ss.str(), shape, start, count, adios2::ConstantDims);
        }
    }

    RtsLayer::UnLockDB();
}

void Tau_plugin_adios2_write_variables(int numThreads, int numCounters,
    const char** counterNames) {
    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    RtsLayer::LockDB();
    /* Copy the function info database so we can release the lock */
    std::vector<FunctionInfo*> tmpTimers(TheFunctionDB());
    RtsLayer::UnLockDB();

    // don't try to read more threads than this node has seen
    int numThreadsLocal = numThreads < RtsLayer::getTotalThreads() ? numThreads : RtsLayer::getTotalThreads();

    std::map<std::string, std::vector<double> >::iterator timer_map_it;

    //foreach: TIMER
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;

        stringstream ss;
        std::string shortName(fi->GetName());
        shorten_timer_name(shortName);
        ss << shortName << " / Calls";

        // Check if a timer showed up since we last defined variables.
        timer_map_it = timers.find(ss.str());
        if (timer_map_it == timers.end()) {
            continue;
        }

        try {
            // assign real data
            for (int tid = 0; tid < numThreadsLocal; tid++) {
                timers[ss.str()][tid] = (double)(fi->GetCalls(tid));
            }

            bpWriter.Put<double>(ss.str(), timers[ss.str()].data());

            for (int m = 0 ; m < numCounters ; m++) {
                stringstream incl;
                stringstream excl;
                incl << shortName << " / Inclusive " << counterNames[m];
                excl << shortName << " / Exclusive " << counterNames[m];
                // assign real data
                for (int tid = 0; tid < numThreadsLocal; tid++) {
                    if (fi->GetCalls(tid) == 0) {
                        timers[incl.str()][tid] = 0.0;
                        timers[excl.str()][tid] = 0.0;
		    } else {
                        timers[incl.str()][tid] = fi->getDumpInclusiveValues(tid)[m];
                        timers[excl.str()][tid] = fi->getDumpExclusiveValues(tid)[m];
		    }
                }
                // pad with zeroes - all ranks may not have the same num threads
                for (int tid = numThreadsLocal; tid < numThreadsLocal; tid++) {
                    timers[incl.str()][tid] = 0.0;
                    timers[excl.str()][tid] = 0.0;
                }
                bpWriter.Put<double>(incl.str(), timers[incl.str()].data());
                bpWriter.Put<double>(excl.str(), timers[excl.str()].data());
            }
        } catch (std::invalid_argument &e) {
            std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                    << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::ios_base::failure &e) {
            std::cout << "IO System base failure exception, STOPPING PROGRAM "
                        "from rank "
                    << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::exception &e) {
            std::cout << "Exception, STOPPING PROGRAM from rank " << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
        }
    }

    RtsLayer::LockDB();
    tau::AtomicEventDB::const_iterator it2;
    std::map<std::string, std::vector<double> >::iterator counter_map_it;

    // do the same with counters.
    for (it2 = tau::TheEventDB().begin(); it2 != tau::TheEventDB().end(); it2++) {
        tau::TauUserEvent *ue = (*it2);
        if (ue == NULL) continue;
	    std::string counter_name(ue->GetName().c_str());
        std::stringstream ss, mean, min, max, sumsqr;
        ss << counter_name << " / Num Events";
        mean << counter_name << " / Mean";
        min << counter_name << " / Min";
        max << counter_name << " / Max";
        sumsqr << counter_name << " / Sum Squares";

        // Check if a counter showed up since we last defined variables.
        counter_map_it = counters.find(ss.str());
        if (counter_map_it == counters.end()) {
            continue;
        }

        try {
            // assign real data
            for (int tid = 0; tid < numThreadsLocal; tid++) {
                counters[ss.str()][tid] = (double)(ue->GetNumEvents(tid));
                counters[mean.str()][tid] = ue->GetMean(tid);
                counters[max.str()][tid] = ue->GetMax(tid);
                counters[min.str()][tid] = ue->GetMin(tid);
                counters[sumsqr.str()][tid] = ue->GetSumSqr(tid);
            }
            // pad with zeroes - all ranks may not have the same num threads
            for (int tid = numThreadsLocal; tid < numThreadsLocal; tid++) {
                counters[ss.str()][tid] = 0.0;
                counters[mean.str()][tid] = 0.0;
                counters[max.str()][tid] = 0.0;
                counters[min.str()][tid] = 0.0;
                counters[sumsqr.str()][tid] = 0.0;
            }
            bpWriter.Put<double>(ss.str(), counters[ss.str()].data());
            bpWriter.Put<double>(mean.str(), counters[mean.str()].data());
            bpWriter.Put<double>(min.str(), counters[min.str()].data());
            bpWriter.Put<double>(max.str(), counters[max.str()].data());
            bpWriter.Put<double>(sumsqr.str(), counters[sumsqr.str()].data());
        } catch (std::invalid_argument &e) {
            std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                    << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::ios_base::failure &e) {
            std::cout << "IO System base failure exception, STOPPING PROGRAM "
                        "from rank "
                    << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::exception &e) {
            std::cout << "Exception, STOPPING PROGRAM from rank " << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
         }
    }
    RtsLayer::UnLockDB();
}

int Tau_plugin_adios2_dump(Tau_plugin_event_dump_data_t* data) {
    if (!enabled) return 0;
    TAU_VERBOSE("TAU PLUGIN ADIOS2: dump\n");
    tau::plugins::ScopedTimer(__func__);

    if (!initialized) {
        Tau_plugin_adios2_init_adios();
    }

    Tau_global_incr_insideTAU();
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();
    std::vector<int> numThreads = {RtsLayer::getTotalThreads()};
    const char **counterNames;
    std::vector<int> numCounters = {0};
    TauMetrics_getCounterList(&counterNames, &(numCounters[0]));
    std::vector<int> numRanks = {adios_comm_size};

    /* This logic breaks down if the number of threads changes during execution.
     * So, we'll just assume a fixed number of threads, that makes things easier.
     * ADIOS2 doesn't like it when different ranks have different numbers of
     * threads, either. */
    //int max_threads = RtsLayer::getTotalThreads();
    //const int max_threads = TAU_MAX_THREADS;
    const int max_threads = 1;
    numThreads[0] = max_threads;

    Tau_plugin_adios2_define_variables(numThreads[0], numCounters[0], counterNames);
    Tau_global_decr_insideTAU();

	if (!opened) {
       Tau_plugin_adios2_open_file();
    }

    if (opened) {
        try {
            bpWriter.BeginStep();
            bpWriter.Put<int>(num_threads_var, numThreads.data());
            bpWriter.Put<int>(num_metrics_var, numCounters.data());
            bpWriter.Put<int>(num_ranks_var, numRanks.data());
            Tau_plugin_adios2_write_variables(numThreads[0], numCounters[0], counterNames);
            bpWriter.EndStep();
        } catch (std::invalid_argument &e) {
            std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                    << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::ios_base::failure &e) {
            std::cout << "IO System base failure exception, STOPPING PROGRAM "
                        "from rank "
                    << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::exception &e) {
            std::cout << "Exception, STOPPING PROGRAM from rank " << world_comm_rank << "\n";
            std::cout << e.what() << "\n";
        }
    }

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
        int tmp_usec = tp.tv_usec + thePluginOptions().env_period;
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
            dummy_data.tid = 1;
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

/* This is a weird event, not sure what for */
int Tau_plugin_finalize(Tau_plugin_event_function_finalize_data_t* data) {
    return 0;
}

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_adios2_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    if (!enabled) return 0;
    TAU_VERBOSE("TAU PLUGIN ADIOS2 Pre-Finalize\n"); fflush(stdout);
#ifdef TAU_MPI
    Tau_ADIOS2_stop_worker();
    Tau_plugin_event_dump_data_t dummy;
    dummy.tid = 0;
    Tau_plugin_adios2_dump(&dummy);
    if (opened) {
        bpWriter.Close();
        opened = false;
    }
    enabled = false;
#endif
    return 0;
}

char * handle_socket_message(char * message) {
    std::stringstream reply;
    if (strcmp(message, "Get num threads") == 0) {
        reply << RtsLayer::getTotalThreads();
    }
    //free (message);
    return strdup(reply.str().c_str());
}

/* This happens after MPI_Init, and after all TAU metadata variables have been
 * read */
int Tau_plugin_adios2_post_init(Tau_plugin_event_post_init_data_t* data) {
    if (!enabled) return 0;
    Tau_plugin_adios2_init_adios();
    Tau_plugin_adios2_open_file();

    /* spawn the thread if doing periodic */
    if (thePluginOptions().env_periodic) {
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

    return 0;
}

/* This happens from Profiler.cpp, when data is written out. */
int Tau_plugin_adios2_end_of_execution(Tau_plugin_event_end_of_execution_data_t* data) {
    if (!enabled || data->tid != 0) return 0;
    TAU_VERBOSE("TAU PLUGIN ADIOS2 Finalize\n"); fflush(stdout);
#ifndef TAU_MPI
    Tau_ADIOS2_stop_worker();
    if (opened) {
        Tau_plugin_event_dump_data_t dummy;
        dummy.tid = 0;
        Tau_plugin_adios2_dump(&dummy);
        bpWriter.Close();
        opened = false;
    }
    if (thePluginOptions().env_periodic) {
        pthread_cond_destroy(&_my_cond);
        pthread_mutex_destroy(&_my_mutex);
    }
    enabled = false;
#endif
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks_t cb;
    TAU_VERBOSE("TAU PLUGIN ADIOS2 Init\n"); fflush(stdout);
    Tau_ADIOS2_parse_environment_variables();
    world_comm_rank = RtsLayer::myNode();
    world_comm_size = tau_totalnodes(0,1);
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(&cb);
    /* Required event support */
    cb.Dump = Tau_plugin_adios2_dump;
    cb.PostInit = Tau_plugin_adios2_post_init;
    cb.PreEndOfExecution = Tau_plugin_adios2_pre_end_of_execution;
    cb.EndOfExecution = Tau_plugin_adios2_end_of_execution;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(&cb, id);
    enabled = true;
    return 0;
}


#endif // TAU_ADIOS2
