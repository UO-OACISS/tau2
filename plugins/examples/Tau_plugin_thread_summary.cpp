/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <Profile/Profiler.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <string>
#include <vector>
#include <set>

class TimerData {
    public:
        uint64_t _calls;
        uint64_t _inclusive;
        uint64_t _exclusive;
        std::string _name;
        TimerData(uint64_t calls, uint64_t inclusive,
            uint64_t exclusive, std::string name) :
            _calls(calls), _inclusive(inclusive),
            _exclusive(exclusive), _name(name) { }
};

void dump_summary() {
    if (Tau_get_node() > 0) return;
    Tau_global_incr_insideTAU();
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();

    RtsLayer::LockDB();
    /* Copy the function info database so we can release the lock */
    std::vector<FunctionInfo*> tmpTimers(TheFunctionDB());
    RtsLayer::UnLockDB();

    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    const char **counterNames;
    int numCounters;
    TauMetrics_getCounterList(&counterNames, &numCounters);
    std::map<uint64_t, TimerData > timers;

    //foreach: TIMER
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;
        if (strncmp(fi->GetName(), ".TAU application", 16) == 0) continue;
        // get the number of calls
        int tid = 0; // todo: get ALL thread data.
        uint64_t calls;
        uint64_t inclusive, exclusive;
        calls = 0;
        inclusive = 0.0;
        exclusive = 0.0;

        //foreach: THREAD
        for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
            calls += (uint64_t)fi->GetCalls(tid);
            // skip this timer if this thread didn't call it.
            // for data-reduction reasons.
            if (calls == 0) continue;
            // iterate over metrics 
            //for (int m = 0; m < Tau_Global_numCounters; m++) {
            for (int m = 0; m < 1; m++) {
                inclusive += (uint64_t)fi->getDumpInclusiveValues(tid)[m];
                exclusive += (uint64_t)fi->getDumpExclusiveValues(tid)[m];
            }
        }
        timers.insert(std::pair<uint64_t, TimerData>(
                inclusive, TimerData(calls, inclusive, exclusive, fi->GetName())));
    }
    printf("%d threads, profile totals across all threads: \n", RtsLayer::getTotalThreads());
    printf("   Calls  Inc_msec.  Exc_msec. Name\n");
    printf("-------- ---------- ---------- ------------------------------------------------------------\n");
    int index = 0;
    for (std::map<uint64_t, TimerData >::reverse_iterator iter = timers.rbegin() ; 
            iter != timers.rend() ; ++iter) {
        TimerData& t = iter->second;
        printf("%8lu %10lu %10lu %.60s\n", t._calls, t._inclusive/1000, t._exclusive/1000, t._name.c_str());
        if (++index > 5) break;
    }
}

int Tau_plugin_event_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t *data) {
    if (data->tid != 0) return 0;
    dump_summary();
    return 0;
}

int Tau_plugin_my_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {
    if (data->tid != 0) return 0;
    dump_summary();
    return 0;
}

/* Only dump data to SOS if we aren't doing periodic dumps */
int Tau_plugin_my_event_trigger(Tau_plugin_event_trigger_data_t* data) {
    printf("TAU PLUGIN SOS: trigger\n"); fflush(stdout);
    dump_summary();
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks_t * cb = (Tau_plugin_callbacks_t*)malloc(sizeof(Tau_plugin_callbacks_t));
  fprintf(stdout, "TAU PLUGIN Thread Summary Init\n"); fflush(stdout);
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->Trigger = Tau_plugin_my_event_trigger;
  cb->PreEndOfExecution = Tau_plugin_event_pre_end_of_execution;
  //cb->EndOfExecution = Tau_plugin_my_event_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);
  return 0;
}

