#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <TAU.h>

int bar(int x)
{
  printf("calling sleep: x = %d\n", x);
  if (x > 0) sleep(x);
  return x;
}

int foo(int x)
{
  printf("inside foo: calling bar: x = %d\n", x);
  printf("before calling bar in foo\n");
  bar(x-1); /* 17 */
  printf("after calling bar in foo\n");
  return x;
}

int foo1(int x)
{
  printf("inside foo1: calling bar: x = %d\n", x);
  printf("before calling bar in foo1\n");
  bar(x-1); /* 26 */
  printf("after calling bar in foo1\n");
  return x;
}


int main(int argc, char **argv)
{
   int i;

   printf("inside main: calling foo\n");
   for (i=0; i < 5; i++)
   {
     foo(i);
     foo1(i);
   }
   return 0;
}
