#include <Profile/Profiler.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>


int work(int data);

void * threaded_func(void *data)
{
  TAU_REGISTER_THREAD();
  TAU_PROFILE("threaded_func()", "int ()", TAU_DEFAULT);
  work((int) *((int *) data)); /* work done by this thread */
  return NULL;
}


