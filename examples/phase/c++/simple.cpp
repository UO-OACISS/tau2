#include <stdio.h>
#include <unistd.h>
#include <TAU.h>


int f1(void) {
  TAU_PROFILE("f1()", "", TAU_USER);
  sleep(1);
  return 0;
}


int input(void) {
  TAU_PROFILE("input", "", TAU_USER);
  sleep(1);
  return 0;
}

int output(void) {
  TAU_PROFILE("output", "", TAU_USER);
  sleep(1);
  return 0;
}

int f2(void) {
  TAU_PROFILE("f2()", "", TAU_USER);

  TAU_PHASE_CREATE_STATIC(t,"IO Phase", "", TAU_USER);
  TAU_PHASE_START(t);
  input();
  output();
  TAU_PHASE_STOP(t);
  return 0;
}


int f3(int x) {
  TAU_PROFILE("f3()", "", TAU_USER);
  sleep(x);
  return 0;
}

int f4(void) {
  TAU_PROFILE("f4()", "", TAU_USER);
  sleep(1);
  return 0;
}

int main(int argc, char **argv) {
  TAU_PROFILE("main()", "", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);

  for (int i=0; i<5; i++) {
    char buf[32];
    snprintf(buf, sizeof(buf),  "Iteration %d", i);

    TAU_PHASE_CREATE_DYNAMIC(phase, buf, "", TAU_USER);
    TAU_PHASE_START(phase);


    printf("Iteration %d\n", i);
    f1();
    f2();
    f3(i);
    
    TAU_PHASE_STOP(phase);
  }

  f1();
  f4();
  return 0;
}
