/* This demonstrates how data cache misses can affect the performance of an 
application. We show how the time/counts for a simple matrix multiplication
algorithm dramatically reduce when we employ a strip mining optimization. */
#include <Profile/Profiler.h>
#include <pthread.h>
#include <sched.h>
#include <iostream>
using namespace std;
#include <stdlib.h>

#define SIZE 50
#define CACHE 64


double multiply(void)
{
  double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
  int i, j, k, n, m;
  int vl, sz, strip;
  TAU_PROFILE("multiply", "void (void)", TAU_USER);
  TAU_PROFILE_TIMER(t1,"multiply-regular", "void (void)", TAU_USER);
  TAU_PROFILE_TIMER(strip_timer,"multiply-with-strip-mining-optimization", "void (void)", TAU_USER);


  for (n = 0; n < SIZE; n++)
    for (m = 0; m < SIZE; m++)
     {
      A[n][m] = B[n][m] = n + m ;
      C[n][m] = 0;
     }
  TAU_PROFILE_START(t1);
  for (i = 0; i < SIZE; i ++)
  { 
    for (j = 0; j < SIZE; j++)
    {
      for (k = 0; k < SIZE; k++)
  	C[i][j] += A[i][k] * B[k][j];
    }
    sched_yield();
  }
  TAU_PROFILE_STOP(t1);

  /* Now we employ the strip mining optimization */

  for(n = 0; n < SIZE; n++)
    for(m = 0; m < SIZE; m++)
      C[n][m] = 0; 
  
  TAU_PROFILE_START(strip_timer);
  for(i=0; i < SIZE; i++)
    for(k=0; k < SIZE; k++)
      for(sz = 0; sz < SIZE; sz+=CACHE)
      {
    	sched_yield();
        //vl = min(SIZE-sz, CACHE);
  	vl = (SIZE - sz < CACHE ? SIZE - sz : CACHE); 
        for(strip = sz; strip < sz+vl; strip++)
          C[i][strip] += A[i][k]*B[k][strip];
      }
  TAU_PROFILE_STOP(strip_timer);

 
  return C[SIZE-10][SIZE-10];
  // So KCC doesn't optimize this loop away.
}

extern "C" void * threaded_func(void *data)
{
  TAU_REGISTER_THREAD();
  TAU_PROFILE("threaded_func()", "int ()", TAU_DEFAULT);
  multiply(); // work done by this thread
  return NULL;
}

       
       
int main(int argc, char **argv)
{
  TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);
  int ret;
  pthread_attr_t  attr;
  pthread_t       tid1, tid2, tid3;

  pthread_attr_init(&attr);

  cout <<"Multiplying "<<SIZE<<" x "<<SIZE<< " Matrices in three threads..." <<endl;


  if (ret = pthread_create(&tid1, NULL, threaded_func, NULL) )
  {
    cerr << " pthread_create fails ret = " << ret <<endl;
    exit(1);
  }
  if (ret = pthread_create(&tid2, NULL, threaded_func, NULL) )
  {
    cerr << " pthread_create fails ret = " << ret <<endl;
    exit(1);
  }
  if (ret = pthread_create(&tid3, NULL, threaded_func, NULL) )
  {
    cerr << " pthread_create fails ret = " << ret <<endl;
    exit(1);
  }

  if (ret = pthread_join(tid1, NULL) )
  {
    cerr << " pthread_join error  ret = "<< ret << endl;
    exit(1);
  }
  if (ret = pthread_join(tid2, NULL) )
  {
    cerr << " pthread_join error  ret = "<< ret << endl;
    exit(1);
  }
  if (ret = pthread_join(tid3, NULL) )
  {
    cerr << " pthread_join error  ret = "<< ret << endl;
    exit(1);
  }
  cout <<"Exiting..." <<endl;

  return 0;
}
