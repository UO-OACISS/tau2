#include <stdlib.h>

int main(int argc, char **argv) {
  int *p = (int *)malloc(500);
  int *q = (int *)malloc(250);
  int *r = (int *)malloc(125);
  free(q);
  return 0;
}
