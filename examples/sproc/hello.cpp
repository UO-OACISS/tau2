
//Example code showing TAU monitoring a threaded application.

//Threading example code updated to include Windows Threads.
//Original code by Sameer Shende.


#ifdef TAU_DOT_H_LESS_HEADERS 
#include <iostream> 
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <Profile/Profiler.h>
#include <abi_mutex.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <ulocks.h>
#include <unistd.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <sys/signal.h>
#include <sys/sysmp.h>
#include <sys/syssgi.h>
#include <sys/time.h>
#include <sys/types.h>

int GetTid(void)
{
  return RtsLayer::myThread();
}

int fourth(void)
{
  TAU_PROFILE("fourth()", "int ()", TAU_DEFAULT);
//  cout <<"Reached fourth Thread:" << GetTid()<<endl;
  return 0;
}

int third(void)
{
  TAU_PROFILE("third()", "int ()", TAU_DEFAULT);
//  cout <<"third calling fourth " <<GetTid() << endl;
  for (int i=0; i < 1000; i++)
  fourth();
  TAU_DB_DUMP();
  return 0;
}

int second(void)
{
  TAU_PROFILE("second()", "int ()", TAU_DEFAULT);
  cout <<"second calling third " <<GetTid()<<endl;
  third();
  return 0;
}

int first(void)
{ 
  TAU_PROFILE("first()", "int ()", TAU_DEFAULT);
  cout << "first.. calling second. " <<GetTid()<< endl;
  TAU_DB_DUMP();
  second();
  return 0;
}

int work (void)
{
  TAU_PROFILE("work()", "int ()", TAU_DEFAULT);
  cout << " Hello this is thread "<< GetTid()<< endl;

/*
  double d = 1.0;
  for(int i =  0; i < 100; i++)
   for (int j=0; j < 100; j++)
    d *= 1.00001; 
*/
  first(); 
  return 0;
}

void threaded_func(void *data, size_t sz)
{
  TAU_REGISTER_THREAD();
  TAU_PROFILE("threaded_func()", "int ()", TAU_DEFAULT);
  work(); // work done by this thread 
}

struct Thread_private {
  int pid;
  caddr_t sp;
  caddr_t stackbot;
  size_t stacklen;
  usema_t* startup;
  usema_t* done;
  usema_t* delete_ready;
  int bstacksize;
};


int main (int argc, char **argv)
{
  TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(0);
  Thread_private *priv_ = new Thread_private; 
  priv_->stacklen = 128*1024; 

  usptr_t *arena = usinit("/dev/zero");

  if (!arena)
    cout <<"Arena not initialized!"<<endl;

  usconfig(CONF_ARENATYPE, US_SHAREDONLY);
  usconfig(CONF_INITSIZE, 30*1024*1024);
  usconfig(CONF_INITUSERS, (unsigned int)140);
  arena=usinit("/dev/zero");

  int devzero_fd=open("/dev/zero", O_RDWR);
  if(devzero_fd == -1)
  { 
    perror("open(/dev/zero/) ERROR:");
    return 1;
  }     

  int tid = GetTid();
  cout <<"Main: tid = "<<tid<<endl;
  

  priv_->stackbot=(caddr_t)mmap(0, priv_->stacklen, PROT_READ|PROT_WRITE,
                                   MAP_PRIVATE, devzero_fd, 0);
  priv_->sp=priv_->stackbot+priv_->stacklen-1;
  if((long)priv_->sp == -1)
  {
    perror("stack pointer ERROR in mmap ");
    return 1;
  }
  priv_->bstacksize=0;
  sginap(40);

  priv_->pid=sprocsp(threaded_func, PR_SALL, priv_, priv_->sp, priv_->stacklen);    if(priv_->pid == -1)
  { 
    perror("sprocsp ERROR");
    return 1;
  }
  sginap(40);

  priv_->pid=sprocsp(threaded_func, PR_SALL, priv_, priv_->sp, priv_->stacklen);    if(priv_->pid == -1)
  { 
    perror("sprocsp ERROR");
    return 1;
  }
  
  sginap(40);

  priv_->pid=sprocsp(threaded_func, PR_SALL, priv_, priv_->sp, priv_->stacklen);    if(priv_->pid == -1)
  { 
    perror("sprocsp ERROR");
    return 1;
  }
  cout <<"Started Main... tid = "<< GetTid() <<endl;
  

  sginap(40);
  double d = 1.0;
  for(int i =  0; i < 1000; i++)
   for (int j=0; j < 1000; j++)
    d *= 1.00001; 

  TAU_DB_DUMP();
  fourth();


  cout <<"Exiting main ..."<<endl;
  return 0;
}

