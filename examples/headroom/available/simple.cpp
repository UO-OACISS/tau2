#include <stdio.h>
#include <unistd.h>
#include <TAU.h>

int f1();
int f2();
int f3();
int f4();
int f5();

int f1(void)
{
  double *ary; 
  TAU_PROFILE("f1()", "(sleeps 1 sec, calls f2, f4)", TAU_USER);
  printf("Inside f1: sleeps 1 sec, calls f2, f4\n");
  ary = new double [1024*1024*50]; 
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

int main(int argc, char **argv)
{
  TAU_PROFILE("main()", "(calls f1, f5)", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);
  printf("Inside main: calls f1, f5\n");

  f1();
  f5();
}
