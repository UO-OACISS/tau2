
#include <stdio.h>
#include <TAU.h>

int main (int argc, char **argv) {
#ifdef TAU_MPI
  TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI */
  printf ("Hello, world\n");
  return 0;
}
