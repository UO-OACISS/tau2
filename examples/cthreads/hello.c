#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <Profile/Profiler.h>

int user_square(int count)
{
  TAU_REGISTER_EVENT(ue1, "UserSquare Event");
  TAU_EVENT(ue1, count * count);
  return 0;
}

int fourth(void)
{
  int i;
  TAU_PROFILE_TIMER(tautimer,"fourth()", "int ()", TAU_DEFAULT);
  TAU_PROFILE_START(tautimer);
  printf("Reached fourth\n");

  for (i = 0; i < 100; i++)
    user_square(i);

  TAU_PROFILE_STOP(tautimer);
  return 0;
}

int third(void)
{
  TAU_PROFILE_TIMER(tautimer,"third()", "int ()", TAU_DEFAULT);
  TAU_PROFILE_START(tautimer);
  printf("third calling fourth()\n");
  fourth();
  TAU_PROFILE_STOP(tautimer);
  return 0;
}

int second(void)
{
  TAU_PROFILE_TIMER(tautimer,"second()", "int ()", TAU_DEFAULT);
  TAU_PROFILE_START(tautimer);
  printf("second calling third\n");
  third();
  TAU_PROFILE_STOP(tautimer);
  return 0;
}

int first(void)
{ 
  TAU_PROFILE_TIMER(tautimer,"first()", "int ()", TAU_DEFAULT);
  TAU_PROFILE_START(tautimer);
  printf("first.. calling second \n");
  second();
  TAU_PROFILE_STOP(tautimer);
  return 0;
}

int work (void)
{
  TAU_PROFILE_TIMER(tautimer, "work()", "int ()", TAU_DEFAULT);
  TAU_PROFILE_START(tautimer);
  printf("Hello. This is thread work calling first\n");
  sleep(5);
  first(); 
  TAU_PROFILE_STOP(tautimer);
  return 0;
}
void * threaded_func(void *data)
{
  TAU_REGISTER_THREAD();
  { /**** NOTE WE START ANOTHER BLOCK IN THREAD */
    TAU_PROFILE_TIMER(tautimer, "threaded_func()", "int ()", TAU_DEFAULT);
    TAU_PROFILE_START(tautimer);
    work(); /* work done by this thread */
    TAU_PROFILE_STOP(tautimer);
  }
  return NULL;
}

int main (int argc, char **argv)
{
  int ret, i;
  pthread_attr_t  attr;
  pthread_t	  tid;
  TAU_PROFILE_TIMER(tautimer,"main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_START(tautimer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(0);

  pthread_attr_init(&attr);
  
  printf("Started Main...\n");

  for (i = 0; i < 10; i++)
	user_square(i);
  if (ret = pthread_create(&tid, NULL, threaded_func, NULL) ) 
  { 
    printf(" pthread_create fails ret = %d\n", ret); 
    TAU_PROFILE_EXIT("pthread_create");
    exit(1);
  }


  if (ret = pthread_join(tid, NULL) ) 
  {
    printf(" pthread_join error  ret = %d\n", ret);
    TAU_PROFILE_EXIT("pthread_join");
    exit(1); 
  }

  /* prior to exiting, print statistics related to user defined events */
  TAU_REPORT_THREAD_STATISTICS();
  printf("Exiting main...\n");
  TAU_PROFILE_STOP(tautimer);
  return 0;
}
/***************************************************************************
 * $RCSfile: hello.c,v $   $Author: sameer $
 * $Revision: 1.5 $   $Date: 1999/06/18 17:09:54 $
 * POOMA_VERSION_ID: $Id: hello.c,v 1.5 1999/06/18 17:09:54 sameer Exp $
 ***************************************************************************/

