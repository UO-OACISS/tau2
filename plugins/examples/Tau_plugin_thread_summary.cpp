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

void dump_summary() {
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
    //printf("Num Counters: %d, Counter[0]: %s\n", numCounters, counterNames[0]);

    printf("Calls   Inclusive  Exclusive  Name\n");
    //foreach: TIMER
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;
        // get the number of calls
        int tid = 0; // todo: get ALL thread data.
        int calls;
        double inclusive, exclusive;
        calls = 0;
        inclusive = 0.0;
        exclusive = 0.0;

        //foreach: THREAD
        for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
            calls += fi->GetCalls(tid);
            // skip this timer if this thread didn't call it.
            // for data-reduction reasons.
            if (calls == 0) continue;
            // iterate over metrics 
            //for (int m = 0; m < Tau_Global_numCounters; m++) {
            for (int m = 0; m < 1; m++) {
                inclusive += fi->getDumpInclusiveValues(tid)[m];
                exclusive += fi->getDumpExclusiveValues(tid)[m];
            }
        }
        printf("%4d %4.f %4.f %s\n", calls, inclusive, exclusive, fi->GetName());
    }
}

int Tau_plugin_event_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t *data) {
    dump_summary();
    return 0;
}

int Tau_plugin_my_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {
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
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  fprintf(stdout, "TAU PLUGIN Thread Summary Init\n"); fflush(stdout);
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->Trigger = Tau_plugin_my_event_trigger;
  cb->PreEndOfExecution = Tau_plugin_event_pre_end_of_execution;
  cb->EndOfExecution = Tau_plugin_my_event_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);
  return 0;
}

