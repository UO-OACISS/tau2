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
  

  void *ptr;
  long calls, childcalls;
  double incl[MAX_TAU_COUNTERS], excl[MAX_TAU_COUNTERS];
  int i, j;
  const char **counters;
  int numcounters;
  TAU_INIT(&argc, &argv);
  TAU_PROFILE_SET_NODE(0);
  TAU_CREATE_PROFILER(ptr, "foo","", TAU_USER);

  TAU_PROFILER_START(ptr);
  foo(2);
  TAU_PROFILER_STOP(ptr);

  TAU_CREATE_PROFILER(ptr, "bar", "", TAU_USER);
  
  for (i=0; i < 5; i++) {
    TAU_PROFILER_START(ptr);
      bar(1);
    TAU_PROFILER_STOP(ptr);
  }
  TAU_PROFILER_GET_CALLS(ptr, &calls);
  TAU_PROFILER_GET_CHILD_CALLS(ptr, &childcalls);
  TAU_PROFILER_GET_INCLUSIVE_VALUES(ptr, &incl);
  TAU_PROFILER_GET_EXCLUSIVE_VALUES(ptr, &excl);

  TAU_PROFILER_GET_COUNTER_INFO(&counters, &numcounters);
  printf("Calls = %ld, child = %ld\n", calls, childcalls);
  printf("numcounters = %d\n", numcounters);
  for (j = 0; j < numcounters ; j++) 
  {
    printf(">>>");
    printf("counter [%d] = %s\n", j, counters[j]);
    printf(" excl [%d] = %g, incl [%d] = %g\n", j, excl[j], j, incl[j]);
  }
  printf("after");
  return 0;
}
