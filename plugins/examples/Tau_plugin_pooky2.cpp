/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#if TAU_MPI
#include "mpi.h"
#endif

#define CONVERT_TO_USEC 1.0/1000000.0 // hopefully the compiler will precompute this.

static bool enabled(false);
static std::ofstream tracefile;
int commrank = 0;
int commsize = 1;

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_sos_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    if (!enabled) return 0;
    Tau_global_incr_insideTAU();
    /* Close the file */
    if (tracefile.is_open()) {
        tracefile.close();
    }
    Tau_global_decr_insideTAU();
    return 0;
}

/* This happens after MPI_Init, and after all TAU metadata variables have been
 * read */
int Tau_plugin_sos_post_init(Tau_plugin_event_post_init_data_t* data) {
    if (!enabled) return 0;
    Tau_global_incr_insideTAU();
    /* Open a file for myself to write to */
#if TAU_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &commrank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
#endif
    std::stringstream filename;
    // make the directory
    struct stat st = {0};
    if (stat("pooky2", &st) == -1) {
        mkdir("pooky2", 0700);
    }
    filename << "pooky2/rank" << commrank << ".trace";
    tracefile.open(filename.str());
    Tau_global_decr_insideTAU();
    return 0;
}

/* This happens for special events from ADIOS, MPI */
int Tau_plugin_sos_current_timer_exit(Tau_plugin_event_current_timer_exit_data_t* data) {
    if (!enabled) return 0;
    Tau_global_incr_insideTAU();
    // get the current profiler
    tau::Profiler * p = Tau_get_current_profiler();
    // get the current time
    double current[TAU_MAX_THREADS];
    int tid = RtsLayer::myThread();
    RtsLayer::getUSecD(tid, current);
    // assume time is the first counter!
    // also convert it to microseconds
    double value = (current[0] - p->StartTime[0]) * CONVERT_TO_USEC;
    tracefile << TauMetrics_getTimeOfDay() << "\t"
              << std::fixed << value << "\t"
              << data->name_prefix << std::endl;
    Tau_global_decr_insideTAU();
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_global_incr_insideTAU();
    Tau_plugin_callbacks_t cb;
    //fprintf(stdout, "TAU PLUGIN SOS Init\n"); fflush(stdout);
    //Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
    TAU_VERBOSE("TAU PLUGIN Pooky2 Init\n");
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(&cb);
    /* Required event support */
    cb.PostInit = Tau_plugin_sos_post_init;
    cb.PreEndOfExecution = Tau_plugin_sos_pre_end_of_execution;
    cb.CurrentTimerExit = Tau_plugin_sos_current_timer_exit;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(&cb, id);
    enabled = true;
    Tau_global_decr_insideTAU();
    return 0;
}
