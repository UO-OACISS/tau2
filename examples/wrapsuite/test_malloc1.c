#include <stdlib.h>

int foo(int x) {
  int *p = (int *) malloc(x);
  free(p);
}

int main(int argc, char **argv) {
  foo(500);
  return 0;
}
