

#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <Profile/Profiler.h>
#include <TAU.h>

int f1();
int f2();
int f3();
int f4();
int f5();

int f1(void)
{
  TAU_PROFILE("f1()", " ", TAU_USER);
  printf("Inside f1: sleeps 1 sec, calls f2, f4\n");
  sleep(1);
  f2();
  f4();
  return 0;
}

int f2(void)
{
  TAU_PROFILE("f2()", "(sleeps 2 sec, calls f3)", TAU_USER);
  printf("Inside f2: sleeps 2 sec, calls f3\n");
  sleep(2);
  f3();
  return 0;
}

int f3(void)
{
  TAU_PROFILE("f3()", "(sleeps 3 sec)", TAU_USER);
  printf("Inside f3: sleeps 3 sec\n");
  sleep(3);
  return 0;
}

int f4(void)
{
  TAU_PROFILE("f4()", "(sleeps 4 sec, calls f2)", TAU_USER);
  printf("Inside f4: sleeps 4 sec, calls f2\n");
  sleep(4);
  f2();
  return 0;
}

int f5(void)
{
  TAU_PROFILE("f5()", "(sleeps 5 sec)", TAU_USER);
  printf("Inside f5: sleeps 5 sec\n");
  sleep(5);
  return 0;
}

void * threaded_func(void *data){
  TAU_REGISTER_THREAD();
  TAU_PROFILE("threaded_func()", "int ()", TAU_DEFAULT);
  f2();
  return NULL;
}

int main(int argc, char **argv)
{
  TAU_PROFILE("main()", "(calls f1, f5)", TAU_DEFAULT);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(0);
  printf("Inside main: calls f1, f5\n");

  int ret;
  pthread_attr_t  attr;
  pthread_t	  tid;

  pthread_attr_init(&attr);
  
  printf("Starting thread ...\n");
  if (ret = pthread_create(&tid, NULL, threaded_func, NULL) ){ 
    printf("pthread_create fails ret = %d ...\n",ret);
    exit(1);
  }

  f1();
  f5();

  if (ret = pthread_join(tid, NULL) ) {
    printf(" pthread_join error  ret = %d ...\n",ret);
    exit(1); 
  }

}
