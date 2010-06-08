#include <stdio.h>

int dgemm(int size)
{
  printf("Inside dgemm: size=%d\n", size);
  return size-1;
}
