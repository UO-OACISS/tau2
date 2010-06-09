#include <stdlib.h>

int main(int argc, char **argv) {
  int *p = malloc(500);
  int *q = malloc(250);
  int *r = malloc(125);
  free(q);
  return 0;
}
