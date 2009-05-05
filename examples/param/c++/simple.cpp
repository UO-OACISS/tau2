#include <stdio.h>
#include <unistd.h>
#include <TAU.h>

int f2(int y)
{
  TAU_PROFILE("f2()", "", TAU_USER);
  TAU_PROFILE_PARAM1L((long) y, "y");
  printf("Inside f2: sleeping for %d seconds\n", y/2);
  sleep(y/2);
}

int f1(int x)
{
  TAU_PROFILE("f1()", "", TAU_USER);
  TAU_PROFILE_PARAM1L((long) x, "x");
  printf("Inside f1: sleeping for %d seconds, calling f2\n", x);
  sleep(x);
  f2(x);
  return 0;
}

int main(int argc, char **argv)
{
  TAU_PROFILE("main()", "(calls f1, f5)", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);
  printf("Inside main: calling f1\n");

  f1(2);
  f1(4);
  f1(3);
  f1(2);
  f1(2);
}
