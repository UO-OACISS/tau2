#include <TAU.h>

void phase() {
  TAU_PHASE("phase","",TAU_DEFAULT);
}

void dynamic_profile() {
  TAU_DYNAMIC_PROFILE("dynamic_profile","",TAU_DEFAULT);
}

void dynamic_phase() {
  TAU_DYNAMIC_PHASE("dynamic_phase","",TAU_DEFAULT);
}

void basic_timer() {
  TAU_PROFILE_TIMER(timer, "basic_timer", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_STOP(timer);
}

void dynamic_timer() {
  TAU_PROFILE_TIMER_DYNAMIC(timer, "basic_timer", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_STOP(timer);
}

void dynamic_auto() {
  TAU_PROFILE_CREATE_DYNAMIC_AUTO(timer, "dynamic auto", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_STOP(timer);
}

void static_phase() {
  TAU_PHASE_CREATE_STATIC(timer, "static phase", "", TAU_DEFAULT);
  TAU_PHASE_START(timer);
  TAU_PHASE_STOP(timer);
}

void dynamic_phase_c() {
  TAU_PHASE_CREATE_DYNAMIC(timer, "dynamic phase", "", TAU_DEFAULT);
  TAU_PHASE_START(timer);
  TAU_PHASE_STOP(timer);
}

void dynamic_phase_auto() {
  TAU_PHASE_CREATE_DYNAMIC_AUTO(timer, "dynamic phase", "", TAU_DEFAULT);
  TAU_PHASE_START(timer);
  TAU_PHASE_STOP(timer);
}

void set_attribs() {
  TAU_PROFILE_TIMER(timer, "set_attribs", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_STOP(timer);
}

void param() {
  TAU_PROFILE_TIMER(timer, "param", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_PARAM1L(50, "parameter");
  TAU_PROFILE_STOP(timer);
}

void stmt() {
  int x;

  TAU_PROFILE_STMT("x=5");
}

void db_access() {
  int numOfFunctions;
  const char ** functionList;
  int numOfCounters;
  const char ** counterList;

  double **counterExclusiveValues;
  double **counterInclusiveValues;
  int *numOfCalls;
  int *numOfSubRoutines;



  const char **eventList;
  int numEvents;
  int *numSamples;
  double *max;
  double *min;
  double *mean;
  double *sumSqr;


  TAU_GET_FUNC_NAMES(functionList, numOfFunctions);
  TAU_GET_COUNTER_NAMES(counterList, numOfCounters);

  TAU_GET_FUNC_VALS(functionList, numOfFunctions,
		    counterExclusiveValues,
		    counterInclusiveValues,
		    numOfCalls,
		    numOfSubRoutines,
		    counterList,
		    numOfCounters);

  TAU_DUMP_FUNC_VALS(functionList, numOfFunctions);
  TAU_DUMP_FUNC_VALS_INCR(functionList, numOfFunctions);


  
  TAU_GET_EVENT_NAMES(eventList, numEvents);
  TAU_GET_EVENT_VALS(eventList, numEvents, numSamples, max, min, mean, sumSqr);


}



void user_events() {
  TAU_REGISTER_EVENT(tau_event, "user_event");
  TAU_EVENT(tau_event, 50);
  TAU_EVENT_SET_NAME(tau_event, "user_event (new name)");
  TAU_EVENT_DISABLE_MIN(tau_event);
  TAU_EVENT_DISABLE_MAX(tau_event);
  TAU_EVENT_DISABLE_MEAN(tau_event);
  TAU_EVENT_DISABLE_STDDEV(tau_event);

  TAU_REPORT_STATISTICS();
  TAU_REPORT_THREAD_STATISTICS();
}

void context_events() {
  TAU_REGISTER_CONTEXT_EVENT(tau_context_event, "user_event");
  TAU_CONTEXT_EVENT(tau_context_event, 50);
  TAU_DISABLE_CONTEXT_EVENT(tau_context_event);
  TAU_ENABLE_CONTEXT_EVENT(tau_context_event);
}


TAU_GLOBAL_TIMER_EXTERNAL(gtimer);
TAU_GLOBAL_TIMER(gtimer, "global_timer", "", TAU_DEFAULT);

TAU_GLOBAL_PHASE_EXTERNAL(gphase);
TAU_GLOBAL_PHASE(gphase, "global_phase", "", TAU_DEFAULT);


int main (int argc, char **argv) {


  TAU_PROFILE("profile","",TAU_DEFAULT);
  TAU_INIT(&argc, &argv);
  TAU_PROFILE_SET_NODE(0);
  TAU_PROFILE_SET_CONTEXT(0);
  TAU_PROFILE_SET_THREAD(0);


  TAU_PROFILE_SET_GROUP_NAME("foobar");


  TAU_START("basic");
  TAU_STOP("basic");

  TAU_DYNAMIC_TIMER_START("dynamic_timer");
  TAU_DYNAMIC_TIMER_STOP("dynamic_timer");

  TAU_STATIC_PHASE_START("static_phase");
  TAU_STATIC_PHASE_STOP("static_phase");

  TAU_DYNAMIC_PHASE_START("dynamic_phase");
  TAU_DYNAMIC_PHASE_STOP("dynamic_phase");

  TAU_GLOBAL_TIMER_START(gtimer);
  TAU_GLOBAL_TIMER_STOP();

  TAU_GLOBAL_PHASE_START(gphase);
  TAU_GLOBAL_PHASE_STOP(gphase);

  user_events();
  context_events();
  phase();
  basic_timer();
  dynamic_timer();
  dynamic_auto();
  static_phase();
  dynamic_phase();
  dynamic_phase_c();
  dynamic_phase_auto();

  set_attribs();
  db_access();

  TAU_DB_DUMP();
  TAU_DB_DUMP_PREFIX("prefix");
  TAU_DB_DUMP_INCR();
  TAU_DUMP_FUNC_NAMES();


  //TAU_REGISTER_THREAD()
  //TAU_REGISTER_FORK(?,?)
  TAU_DISABLE_INSTRUMENTATION();
  TAU_ENABLE_INSTRUMENTATION();

  TAU_DISABLE_GROUP(TAU_DEFAULT);
  TAU_ENABLE_GROUP(TAU_DEFAULT);

  TAU_DISABLE_GROUP_NAME("TAU_DEFAULT");
  TAU_ENABLE_GROUP_NAME("TAU_DEFAULT");

  TAU_DISABLE_ALL_GROUPS();
  TAU_ENABLE_ALL_GROUPS();

  TAU_GET_PROFILE_GROUP("TAU_DEFAULT");


  TAU_SET_INTERRUPT_INTERVAL(10);

  TAU_PROFILE_SNAPSHOT("foo");
  TAU_PROFILE_SNAPSHOT_1L("foo", 5);
  TAU_METADATA("name","value");
  TAU_CONTEXT_METADATA("name","value");
  TAU_PHASE_METADATA("name", "value");

  void *handle;
  TAU_PROFILER_CREATE(handle, "handle", "", TAU_DEFAULT);
  TAU_PROFILER_START(handle);
  TAU_PROFILER_STOP(handle);

  double data[500];
  TAU_PROFILER_GET_INCLUSIVE_VALUES(handle, data);
  TAU_PROFILER_GET_EXCLUSIVE_VALUES(handle, data);

  long number;
  TAU_PROFILER_GET_CALLS(handle, &number);
  TAU_PROFILER_GET_CHILD_CALLS(handle, &number);

  const char **counters;
  int numcounters;
  TAU_PROFILER_GET_COUNTER_INFO(&counters, &numcounters);



  TAU_ENABLE_TRACKING_MEMORY();
  TAU_DISABLE_TRACKING_MEMORY();
    
  TAU_DISABLE_TRACKING_MEMORY_HEADROOM();
  TAU_ENABLE_TRACKING_MEMORY_HEADROOM();

  TAU_TRACK_MEMORY();
  TAU_TRACK_MEMORY_HEADROOM();
  TAU_TRACK_MEMORY_HERE();
  TAU_TRACK_MEMORY_HEADROOM_HERE();


    
  TAU_DB_PURGE();
  TAU_PROFILE_EXIT("message");
  return 0;
}


