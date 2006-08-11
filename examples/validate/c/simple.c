
#include <TAU.h>
#include <stdio.h>


int main (int argc, char **argv) {
  TAU_PROFILE_TIMER(tautimer,"main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_START(tautimer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(0);

  printf ("Hello, world\n");

  TAU_PROFILE_STOP(tautimer);
  return 0;
}
