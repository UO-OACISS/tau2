#ifdef TAU_DOT_H_LESS_HEADERS 
#include <iostream> 
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#include <pthread.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <Profile/Profiler.h>

TAU_REGISTER_EVENT(iters, "Number of Iterates");
TAU_REGISTER_EVENT(mem, "Memory allocated by arrays");

int fourth(void)
{
  TAU_PROFILE("fourth()", "int ()", TAU_DEFAULT);
  TAU_EVENT(iters, 1);
  cout <<"Reached fourth " << endl;
  return 0;
}

int third(void)
{
  TAU_PROFILE("third()", "int ()", TAU_DEFAULT);
  cout <<"third calling fourth " <<endl;
  TAU_EVENT(mem, 1024);
  TAU_EVENT(iters, 1);
  fourth();
  return 0;
}

int second(void)
{
  TAU_PROFILE("second()", "int ()", TAU_DEFAULT);
  cout <<"second calling third " <<endl;
  TAU_EVENT(mem, 16*1024);
  TAU_EVENT(iters, 1);
  third();
  return 0;
}

int first(void)
{ 
  TAU_PROFILE("first()", "int ()", TAU_DEFAULT);
  cout << "first.. calling second. " << endl;
  TAU_EVENT(iters, 1);
  second();
  return 0;
}

int work (void)
{
  TAU_PROFILE("work()", "int ()", TAU_DEFAULT);
  cout << " Hello this is thread "<< endl;
  TAU_EVENT(iters, 1);
  TAU_EVENT(mem, 4096);
  sleep(5);
  first(); 
  return 0;
}
void * threaded_func(void *data)
{
  TAU_REGISTER_THREAD();
  TAU_PROFILE("threaded_func()", "int ()", TAU_DEFAULT);
  TAU_EVENT(iters, 1);
  work(); // work done by this thread 
  return NULL;
}

int main (int argc, char **argv)
{
  TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(0);
  int ret;
  pthread_attr_t  attr;
  pthread_t	  tid;

  pthread_attr_init(&attr);
  
  cout <<"Started Main..." <<endl;

  if (ret = pthread_create(&tid, NULL, threaded_func, NULL) ) 
  { 
    cerr << " pthread_create fails ret = " << ret <<endl;
    exit(1);
  }
  TAU_EVENT(iters, 1);

  TAU_EVENT(mem, 2048);
  //first();

  if (ret = pthread_join(tid, NULL) ) 
  {
    cerr << " pthread_join error  ret = "<< ret << endl;
    exit(1); 
  }
  //TAU_REPORT_THREAD_STATISTICS();
  TAU_REPORT_STATISTICS();

  cout <<"Exiting main ..."<<endl;
  return 0;
}
