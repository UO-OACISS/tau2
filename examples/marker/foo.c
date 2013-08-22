#include <TAU.h>

int foo(int x) {
  TAU_REGISTER_EVENT(te, "foo counter");
  printf("Inside foo: x = %d\n", x);

  TAU_EVENT(te, (double) x);

  return x;
}
  

int main(int argc, char **argv) {

  foo(300);
  foo(301);
  foo(304);
  foo(299);
  foo(2000);
  foo(21);

  foo(5);
  return 0;
}
