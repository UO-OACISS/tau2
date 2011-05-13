// testTau.cpp : Defines the entry point for the console application.
//


#define TAU_WINDOWS
#define TAU_DOT_H_LESS_HEADERS
#define PROFILING_ON

#include <Profile/Profiler.h>


extern void dll_func(int a);

int main(int argc, char **argv) {

  TAU_PROFILE_TIMER(timer,"int main(int, char **)", " ", TAU_DEFAULT);
  //TAU_PROFILE_SET_NODE(0);

  TAU_PROFILE_START(timer);
  //printf ("main here, calling dll\n");
  //dll_func(5);
  //printf ("main here, finished\n");

  TAU_PROFILE_STOP(timer);
  return 0;
}

