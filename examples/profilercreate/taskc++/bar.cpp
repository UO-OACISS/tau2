#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <TAU.h>

int bar(int x)
{
  printf("Inside bar\n");
  sleep(x);
  return 0;
}

int foo(int x)
{
  printf("Inside foo\n");
  sleep(x-1);
  bar(x-1);
  return 0;
}

int main(int argc, char **argv)
{
   /* Initialize */
  

  void *ptr, *top;
  long calls, childcalls;
  double incl[MAX_TAU_COUNTERS], excl[MAX_TAU_COUNTERS];
  const char **counters;
  int numcounters, i, j, taskid; 
  TAU_INIT(&argc, &argv);
  TAU_PROFILE_SET_NODE(0);
  TAU_CREATE_TASK(taskid);
  TAU_PROFILER_CREATE(top, "Top-level-timer","", TAU_USER);
  TAU_PROFILER_CREATE(ptr, "foo","", TAU_USER);

  TAU_PROFILER_START_TASK(top, taskid);
  TAU_PROFILER_START_TASK(ptr, taskid);
  foo(2);
  TAU_PROFILER_STOP_TASK(ptr, taskid);

  TAU_PROFILER_CREATE(ptr, "bar", "", TAU_USER);

  
  for (i=0; i < 5; i++) {
    TAU_PROFILER_START_TASK(ptr, taskid);
      bar(1);
    TAU_PROFILER_STOP_TASK(ptr, taskid);
  }
  TAU_PROFILER_GET_CALLS_TASK(ptr, &calls, taskid);
  TAU_PROFILER_GET_CHILD_CALLS_TASK(ptr, &childcalls, taskid);
  TAU_PROFILER_GET_INCLUSIVE_VALUES_TASK(ptr, &incl, taskid);
  TAU_PROFILER_GET_EXCLUSIVE_VALUES_TASK(ptr, &excl, taskid);

  TAU_PROFILER_GET_COUNTER_INFO_TASK(&counters, &numcounters, taskid);
  printf("Calls = %ld, child = %ld\n", calls, childcalls);
  printf("numcounters = %d\n", numcounters);
  for (j = 0; j < numcounters ; j++)
  {
    printf(">>>");
    printf("counter [%d] = %s\n", j, counters[j]);
    printf(" excl [%d] = %g, incl [%d] = %g\n", j, excl[j], j, incl[j]);
  }

  printf("Before setting calls: %d\n", calls);
  TAU_PROFILER_SET_CALLS_TASK(ptr, 1024, taskid);
  TAU_PROFILER_GET_CALLS_TASK(ptr, &calls, taskid);
  printf("After setting child calls: %d\n", calls);

  printf("Adding 200 s to exclusive time value of bar in the task \n");
  excl[0] += 200000000.0;
  TAU_PROFILER_SET_EXCLUSIVE_VALUES_TASK(ptr, excl, taskid);

  TAU_PROFILER_STOP_TASK(top, taskid);

  return 0;
}
