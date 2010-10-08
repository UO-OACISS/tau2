
// Example code showing TAU monitoring a threaded application.
// This program spawns one thread that calls a bunch of dummy functions
// one of which sleeps for five seconds. It demonstrates that data can be
// collected for the two threads and displayed according to each thread

#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

int fourth(void)
{
  cout <<"Reached fourth " << endl;
  return 0;
}

int third(void)
{
  cout <<"third calling fourth " <<endl;
  fourth();
  return 0;
}

int second(void)
{
  cout <<"second calling third " <<endl;
  third();
  return 0;
}

int first(void)
{ 
  cout << "first.. calling second. " << endl;
  second();
  return 0;
}

int fifth(void)
{
  cout << " fifth called after work "<< endl;
  return 0;
}

int work (void)
{
  cout << " Hello this is thread "<< endl;
  sleep(5);
  first(); 
  return 0;
}

void * threaded_func(void *data)
{
  work(); // work done by this thread 
  fifth(); 
  return NULL;
}

int main (int argc, char **argv)
{
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
  //first();

  if (ret = pthread_join(tid, NULL) ) 
  {
    cerr << " pthread_join error  ret = "<< ret << endl;
    exit(1); 
  }

  cout <<"Exiting main ..."<<endl;
  return 0;
}

