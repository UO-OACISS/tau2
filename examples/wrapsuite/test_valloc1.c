#include <stdlib.h>

int main(int argc, char **argv) {
  int *p = valloc(125);
  free(p);
  return 0;
}
