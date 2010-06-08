#include <stdio.h>

/* This is the wrapper for dgemm. We add a call to TAU_PROFILE_PARAM1L to
distinguish the various invocations of dgemm based on the parameter (size) */
int __wrap_dgemm(int size) {
  int ret;   
  TAU_PROFILE_PARAM1L(size,"size");
  printf("Inside __wrap_dgemm: size = %d\n", size);
  ret = __real_dgemm(size);
  return ret;
}
