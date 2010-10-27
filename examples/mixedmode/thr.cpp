#include <Profile/Profiler.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>


int work(int data);
extern pthread_barrier_t bar;

void * threaded_func(void *data)
{
  work((int) *((int *) data)); /* work done by this thread */
  pthread_barrier_wait(&bar);
  return NULL;
}


