#include <Profile/Profiler.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <TAU.h>

int bar(int x)
{

	TAU_PROFILE_TIMER(tautimer, "int bar(int) C [{foo.c} {6,1}-{11,1}]", " ", TAU_USER);
	TAU_PROFILE_START(tautimer);

{
  printf("calling sleep: x = %d\n", x);
  if (x > 0) sleep(x);
  { int tau_ret_val =  x;  TAU_PROFILE_STOP(tautimer); return (tau_ret_val); }


}
	
	TAU_PROFILE_STOP(tautimer);

}

int foo(int x)
{

	TAU_PROFILE_TIMER(tautimer, "int foo(int) C [{foo.c} {13,1}-{20,1}]", " ", TAU_USER);
	TAU_PROFILE_START(tautimer);

{
  printf("inside foo: calling bar: x = %d\n", x);
  printf("before calling bar in foo\n");
  bar(x-1); /* 17 */
  printf("after calling bar in foo\n");
  { int tau_ret_val =  x;  TAU_PROFILE_STOP(tautimer); return (tau_ret_val); }


}
	
	TAU_PROFILE_STOP(tautimer);

}

int foo1(int x)
{

	TAU_PROFILE_CREATE_DYNAMIC_AUTO(tautimer, "int foo1(int) C [{foo.c} {22,1}-{29,1}]", " ", TAU_USER);
	TAU_PROFILE_START(tautimer);

{
  printf("inside foo1: calling bar: x = %d\n", x);
  printf("before calling bar in foo1\n");
  bar(x-1); /* 26 */
  printf("after calling bar in foo1\n");
  { int tau_ret_val =  x;  TAU_PROFILE_STOP(tautimer); return (tau_ret_val); }


}
	
	TAU_PROFILE_STOP(tautimer);

}


int main(int argc, char **argv)
{

	TAU_PROFILE_TIMER(tautimer, "int main(int, char **) C [{foo.c} {32,1}-{43,1}]", " ", TAU_DEFAULT);
  TAU_INIT(&argc, &argv); 
#ifndef TAU_MPI
#ifndef TAU_SHMEM
  TAU_PROFILE_SET_NODE(0);
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */
	TAU_PROFILE_START(tautimer);

{
   int i;

   printf("inside main: calling foo\n");
   for (i=0; i < 5; i++)
   {
     foo(i);
     foo1(i);
   }
   { int tau_ret_val =  0;  TAU_PROFILE_STOP(tautimer); return (tau_ret_val); }


}
	
	TAU_PROFILE_STOP(tautimer);

}
