/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

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
#if TAU_MPI
#include "mpi.h"
#endif

#include <adios2.h>

static bool enabled(false);
static bool initialized(false);
static bool opened(false);

/* Some ADIOS variables */
adios2::ADIOS ad;
adios2::Engine bpWriter;
std::map<std::string, adios2::Variable<double> >varT;
adios2::Variable<double> num_threads;
adios2::Variable<double> num_metrics;

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
    /** ADIOS class factory of IO class objects, DebugON is recommended */
    ad = adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugON);

    /*** IO class object: settings and factory of Settings: Variables,
     * Parameters, Transports, and Execution: Engines */
    adios2::IO bpIO = adios.DeclareIO("TAU_profiles");

    /* write the metadata as attributes */
    Tau_dump_ADIOS2_metadata(bpIO);

    /* Create some "always used" variables */
    
    /** global array : name, { shape (total) }, { start (local) }, {
     * count (local) }, all are constant dimensions */
    num_threads = bpIO.DefineVariable<double>(
        "num_threads", {1}, {0}, {0}, adios2::ConstantDims);
    num_metrics = bpIO.DefineVariable<double>(
        "num_metrics", {1}, {0}, {0}, adios2::ConstantDims);

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
    adios2::Engine bpFileWriter = bpIO.Open(ss.str(), adios2::Mode::Write);
}

void Tau_plugin_adios2_define_variables(int numThreads, int numCounters,
    char** counterNames) {
    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    RtsLayer::LockDB();

    //foreach: TIMER
    for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
        FunctionInfo *fi = *it;
        int tid = 0; // todo: get ALL thread data.
        int calls;
        double inclusive, exclusive;
        calls = 0;
        inclusive = 0.0;
        exclusive = 0.0;


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
    numThreads = RtsLayer::getTotalThreads();
    const char **counterNames;
    int numCounters;
    TauMetrics_getCounterList(&counterNames, &numCounters);
 
    Tau_plugin_adios2_define_variables(numThreads, numCounters, counterNames);

	if (!opened) {
       Tau_plugin_adios2_open_file();
    }

    bpWriter.BeginStep();
    //bpWriter.Put<double>(varT, ht.data());
    bpWriter.EndStep();

    return 0;
}

/* This is a weird event, not sure what for */
int Tau_plugin_finalize(Tau_plugin_event_function_finalize_data_t* data) {
    return 0;
}

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_adios2_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    if (!enabled) return 0;
    //fprintf(stdout, "TAU PLUGIN ADIOS2 Pre-Finalize\n"); fflush(stdout);
    if (opened) {
        bpWriter.Close();
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
    //fprintf(stdout, "TAU PLUGIN ADIOS2 Finalize\n"); fflush(stdout);
    if (opened) {
        bpWriter.Close();
    }
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

    return 0;
}


#endif // TAU_ADIOS2
