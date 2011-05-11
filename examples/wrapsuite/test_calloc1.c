#include <stdlib.h>

int main(int argc, char **argv) {
  int *p = (int *)calloc(500,1);
  free(p);
  return 0;
}
