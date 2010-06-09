
#define _XOPEN_SOURCE 600
#include <stdlib.h>


int main(int argc, char **argv) {
  int *p;

  posix_memalign((void**)&p, 8, 256);
  free(p);
  return 0;
}
