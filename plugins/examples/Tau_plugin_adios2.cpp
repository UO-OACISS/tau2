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

static bool enabled(false);
static bool initialized(false);
static bool opened(false);

/* Some ADIOS variables */
adios2::ADIOS ad;
adios2::IO bpIO;
adios2::Engine bpWriter;
//std::map<std::string, adios2::Variable<double> > timers;
std::map<std::string, std::vector<double> > timers;
adios2::Variable<int> num_threads_var;
adios2::Variable<int> num_metrics_var;

/* Some MPI variables */
int comm_rank(0);
int comm_size(1);

void Tau_dump_ADIOS2_metadata(adios2::IO &bpIO) {
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
#if TAU_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
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
        PMPI_Comm_dup(MPI_COMM_WORLD, &adios_comm);
        if (config != nullptr) {
            ad = adios2::ADIOS(config, adios_comm, adios2::DebugON);
        } else {
            ad = adios2::ADIOS(adios_comm, adios2::DebugON);
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
        bpIO = ad.DeclareIO("TAU_profiles");
        // if not defined by user, we can change the default settings
        // BPFile is the default engine
        bpIO.SetEngine("BPFile");
        bpIO.SetParameters({{"num_threads", "1"}});

        // ISO-POSIX file output is the default transport (called "File")
        // Passing parameters to the transport
        bpIO.AddTransport("File", {{"Library", "POSIX"}});

        /* write the metadata as attributes */
        Tau_dump_ADIOS2_metadata(bpIO);

        /* Create some "always used" variables */
    
        /** global array : name, { shape (total) }, { start (local) }, {
        * count (local) }, all are constant dimensions */
        const std::size_t Nx = 1;
        const adios2::Dims shape{static_cast<size_t>(Nx * comm_size)};
        const adios2::Dims start{static_cast<size_t>(Nx * comm_rank)};
        const adios2::Dims count{Nx};
        num_threads_var = bpIO.DefineVariable<int>(
            "num_threads", shape, start, count, adios2::ConstantDims);
        num_metrics_var = bpIO.DefineVariable<int>(
            "num_metrics", shape, start, count, adios2::ConstantDims);
    } catch (std::invalid_argument &e)
    {
        std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                  << comm_rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::ios_base::failure &e)
    {
        std::cout << "IO System base failure exception, STOPPING PROGRAM "
                     "from rank "
                  << comm_rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Exception, STOPPING PROGRAM from rank " << comm_rank << "\n";
        std::cout << e.what() << "\n";
    }
    initialized = true;
}

void Tau_plugin_adios2_open_file(void) {
    std::stringstream ss;
    const char * prefix = TauEnv_get_profile_prefix();
    ss << TauEnv_get_profiledir() << "/";
    if (prefix != NULL) {
        ss << TauEnv_get_profile_prefix() << "-";
    }
    ss << "tauprofile.bp";
    printf("Writing %s\n", ss.str().c_str());
    bpWriter = bpIO.Open(ss.str(), adios2::Mode::Write);
    opened = true;
}

void Tau_plugin_adios2_define_variables(int numThreads, int numCounters,
    const char** counterNames) {
    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    RtsLayer::LockDB();

    std::map<std::string, std::vector<double> >::iterator timer_map_it;

    const std::size_t Nx = numThreads;
    const adios2::Dims shape{static_cast<size_t>(Nx * comm_size)};
    const adios2::Dims start{static_cast<size_t>(Nx * comm_rank)};
    const adios2::Dims count{Nx};

    //foreach: TIMER
    for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
        FunctionInfo *fi = *it;

        stringstream ss;
        ss << fi->GetName() << " / Calls";
        timer_map_it = timers.find(ss.str());
        if (timer_map_it == timers.end()) {
            // add the timer to the map
            timers.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads)));
            // define the variable for ADIOS
            bpIO.DefineVariable<double>(
                ss.str(), shape, start, count, adios2::ConstantDims);
            for (int i = 0 ; i < numCounters ; i++) {
                ss.str(std::string());
                ss << fi->GetName() << " / Inclusive " << counterNames[i];
                // add the timer to the map
                timers.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads)));
                // define the variable for ADIOS
                bpIO.DefineVariable<double>(
                    ss.str(), shape, start, count, adios2::ConstantDims);
                ss.str(std::string());
                ss << fi->GetName() << " / Exclusive " << counterNames[i];
                // add the timer to the map
                timers.insert(pair<std::string,vector<double> >(ss.str(), vector<double>(numThreads)));
                // define the variable for ADIOS
                bpIO.DefineVariable<double>(
                    ss.str(), shape, start, count, adios2::ConstantDims);
            }
        }
    }

    RtsLayer::UnLockDB();
}

void Tau_plugin_adios2_write_variables(int numThreads, int numCounters,
    const char** counterNames) {
    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    RtsLayer::LockDB();

    std::map<std::string, std::vector<double> >::iterator timer_map_it;

    //foreach: TIMER
    for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
        FunctionInfo *fi = *it;
        int tid = 0; // todo: get ALL thread data.

        stringstream ss;
        ss << fi->GetName() << " / Calls";

        // Check if a timer showed up since we last defined variables.
        timer_map_it = timers.find(ss.str());
        if (timer_map_it == timers.end()) {
            continue;
        }

        try {
            for (tid = 0; tid < numThreads; tid++) {
                timers[ss.str()][tid] = (double)(fi->GetCalls(tid));
            }

            bpWriter.Put<double>(ss.str(), timers[ss.str()].data());

            for (int m = 0 ; m < numCounters ; m++) {
                stringstream incl;
                stringstream excl;
                incl << fi->GetName() << " / Inclusive " << counterNames[m];
                excl << fi->GetName() << " / Exclusive " << counterNames[m];
                for (tid = 0; tid < numThreads; tid++) {
                    timers[incl.str()][tid] = fi->getDumpInclusiveValues(tid)[m];
                    timers[excl.str()][tid] = fi->getDumpExclusiveValues(tid)[m];
                }
                bpWriter.Put<double>(incl.str(), timers[incl.str()].data());
                bpWriter.Put<double>(excl.str(), timers[excl.str()].data());
            }
        } catch (std::invalid_argument &e) {
            std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                    << comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::ios_base::failure &e) {
            std::cout << "IO System base failure exception, STOPPING PROGRAM "
                        "from rank "
                    << comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::exception &e) {
            std::cout << "Exception, STOPPING PROGRAM from rank " << comm_rank << "\n";
            std::cout << e.what() << "\n";
        }
    }

    RtsLayer::UnLockDB();
}

int Tau_plugin_adios2_dump(Tau_plugin_event_dump_data_t* data) {
    if (!enabled) return 0;
    printf("TAU PLUGIN ADIOS2: dump\n");

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
            Tau_plugin_adios2_write_variables(numThreads[0], numCounters[0], counterNames);
            bpWriter.EndStep();
        } catch (std::invalid_argument &e) {
            std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                    << comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::ios_base::failure &e) {
            std::cout << "IO System base failure exception, STOPPING PROGRAM "
                        "from rank "
                    << comm_rank << "\n";
            std::cout << e.what() << "\n";
        } catch (std::exception &e) {
            std::cout << "Exception, STOPPING PROGRAM from rank " << comm_rank << "\n";
            std::cout << e.what() << "\n";
        }
    }

    return 0;
}

/* This is a weird event, not sure what for */
int Tau_plugin_finalize(Tau_plugin_event_function_finalize_data_t* data) {
    return 0;
}

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_adios2_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    if (!enabled) return 0;
    fprintf(stdout, "TAU PLUGIN ADIOS2 Pre-Finalize\n"); fflush(stdout);
#if 1
    Tau_plugin_event_dump_data_t * dummy;
    Tau_plugin_adios2_dump(dummy);
    if (opened) {
        bpWriter.Close();
        opened = false;
    }
#endif
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
#if 0
    Tau_plugin_event_dump_data_t * dummy;
    Tau_plugin_adios2_dump(dummy);
    if (opened) {
        bpWriter.Close();
        opened = false;
    }
#endif
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv) {
    Tau_plugin_callbacks_t * cb = (Tau_plugin_callbacks_t*)malloc(sizeof(Tau_plugin_callbacks_t));
    fprintf(stdout, "TAU PLUGIN ADIOS2 Init\n"); fflush(stdout);
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
    /* Required event support */
    cb->Dump = Tau_plugin_adios2_dump;
    cb->PostInit = Tau_plugin_adios2_post_init;
    cb->PreEndOfExecution = Tau_plugin_adios2_pre_end_of_execution;
    cb->EndOfExecution = Tau_plugin_adios2_end_of_execution;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb);
    enabled = true;
    Tau_plugin_adios2_init_adios();
    Tau_plugin_adios2_open_file();

    return 0;
}


#endif // TAU_ADIOS2
