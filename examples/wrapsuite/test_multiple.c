
#define _XOPEN_SOURCE 600
#include <stdlib.h>


int main(int argc, char **argv) {
  int *p;

  posix_memalign((void**)&p, 8, 250); // malloc: 250

  char *m = (char *) malloc(500); // malloc: 750

  char *n = (char *) calloc(250,1); // malloc: 1000
  
  free (p); // malloc: 1000, free: 250

  m = (char *) realloc (m, 1500); // malloc: 2500, free: 750

  free(m); // malloc: 2500, free: 2250
  return 0;
}
