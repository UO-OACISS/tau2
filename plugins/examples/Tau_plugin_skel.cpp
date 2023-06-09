/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iomanip>
#include <mutex>          // std::mutex

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
static bool opened(false);
static std::ofstream tracefile;
static std::stringstream buffer;
static std::ostream *active_stream = nullptr;
int commrank = 0;
int commsize = 1;
std::mutex mtx;           // mutex for critical section
static int step = 0;

int Tau_plugin_skel_dump(Tau_plugin_event_dump_data_t* data) {
    step = step + 1;
    return 0;
}

static void open_file() {
    if (!enabled || opened) return;
    Tau_global_incr_insideTAU();
    mtx.lock();
    /* Open a file for myself to write to */
    commrank = RtsLayer::myNode();
    commsize = tau_totalnodes(0,1);

    std::stringstream filename;
    // make the directory
    struct stat st = {0};
    if (stat("skel", &st) == -1) {
        mkdir("skel", 0700);
    }
    filename << "skel/rank" << std::setfill('0')
             << std::setw(5) << commrank << ".trace";
    tracefile.open(filename.str());
    opened = true;
    tracefile << "[\n";
    tracefile << buffer.str();
    active_stream = &tracefile;
    mtx.unlock();
    Tau_global_decr_insideTAU();
}

static void close_file() {
    if (!enabled || !opened) return;
    Tau_global_incr_insideTAU();
    mtx.lock();
    uint64_t globalStart = TauMetrics_getInitialTimeStamp();
    uint64_t end = TauMetrics_getTimeOfDay() - globalStart;
    tracefile << "{\"ts\": " << std::fixed << end
              << ", \"dur\": " << std::fixed << 1
              << ", \"ph\": \"X\", \"tid\": 0"
              << ", \"pid\": " << std::fixed << Tau_get_node()
              << ", \"step\": " << std::fixed << step
              << ", \"cat\": \"TAU\""
              << ", \"name\": \"program exit\"}\n]\n";

    /* Close the file */
    if (tracefile.is_open()) {
        tracefile.flush();
        tracefile.close();
    }
    opened = false;
    mtx.unlock();
    Tau_global_decr_insideTAU();
}

int Tau_plugin_skel_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    close_file();
    return 0;
}

int Tau_plugin_skel_end_of_execution(Tau_plugin_event_end_of_execution_data_t* data) {
    close_file();
    return 0;
}

/* This happens after MPI_Init, and after all TAU metadata variables have been
 * read */
int Tau_plugin_skel_post_init(Tau_plugin_event_post_init_data_t* data) {
    open_file();
    return 0;
}

/* This happens for special events from ADIOS, MPI */
int Tau_plugin_skel_current_timer_exit(Tau_plugin_event_current_timer_exit_data_t* data) {
    if (!enabled) return 0;
    /* We're going to use I/O, so make sure the iowrapper doesn't track what we do */
    thread_local static bool inside{false};
    if (inside) return 0;
    inside = true;
    Tau_global_incr_insideTAU();
    // get the current profiler
    tau::Profiler * p = Tau_get_current_profiler();
    // get the current time
    // assume time is the first counter!
    // also convert it to microseconds
    static uint64_t globalStart = TauMetrics_getInitialTimeStamp();
    uint64_t start = p->StartTime[0];
    uint64_t end = TauMetrics_getTimeOfDay();
    //double value = (end - start) * CONVERT_TO_USEC;
    uint64_t value = end - start;
    start = start - globalStart; // normalize to 0
    mtx.lock();
    (*active_stream) << "{\"ts\": " << std::fixed << start
              << ", \"dur\": " << std::fixed << value
              << ", \"ph\": \"X\", \"tid\": " << RtsLayer::myThread()
              << ", \"pid\": " << std::fixed << (Tau_get_node() == -1 ? 0 : Tau_get_node())
              << ", \"step\": " << std::fixed << step
              << ", " << data->name_prefix << "},\n";
    active_stream->flush();
    mtx.unlock();
    Tau_global_decr_insideTAU();
    inside = false;
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_global_incr_insideTAU();
    Tau_plugin_callbacks_t cb;
    TAU_VERBOSE("TAU PLUGIN Skel Init\n");
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(&cb);
    /* Required event support */
    cb.PostInit = Tau_plugin_skel_post_init;
    //cb.PreEndOfExecution = Tau_plugin_skel_pre_end_of_execution;
    cb.EndOfExecution = Tau_plugin_skel_end_of_execution;
    cb.CurrentTimerExit = Tau_plugin_skel_current_timer_exit;
    cb.Dump = Tau_plugin_skel_dump;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(&cb, id);
    enabled = true;
    // so we can capture events before the file is opened,
    // which we can't do until we know our rank
    active_stream = &buffer;
    Tau_global_decr_insideTAU();
    return 0;
}
