#include <stdlib.h>

int main(int argc, char **argv) {
  int *p = malloc(500);
  free(p);
  return 0;
}
