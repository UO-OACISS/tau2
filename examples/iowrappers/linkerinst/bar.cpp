#include <stdio.h>

extern "C" int bar(int x) {
  printf("Inside bar: x = %d\n", x);
  return 42  - x;
}
