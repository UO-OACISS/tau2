#include <stdio.h>
#include <stdlib.h>

int foo1(int x)
{
  printf("Inside foo1: x = %d\n", x);
  sleep(x); 
  return x+1;
}

void foo2(int b, int c)
{
  printf("Inside foo2: b = %d, c = %d\n", b, c);
  sleep(b-c);
  return;
}
