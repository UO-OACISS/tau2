#include <stdlib.h>

int main(int argc, char **argv) {
  int *p = calloc(500,1);
  free(p);
  return 0;
}
