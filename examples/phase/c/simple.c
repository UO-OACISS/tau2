#include <stdio.h>
#include <unistd.h>
#include <TAU.h>


int f1(void)
{
  TAU_PROFILE_TIMER(t,"f1()", "", TAU_USER);
  TAU_PROFILE_START(t);
  sleep(1);
  TAU_PROFILE_STOP(t);
  return 0;
}


int input(void) {
  TAU_PROFILE_TIMER(t,"input", "", TAU_USER);
  TAU_PROFILE_START(t);
  sleep(1);
  TAU_PROFILE_STOP(t);
}

int output(void) {
  TAU_PROFILE_TIMER(t,"output", "", TAU_USER);
  TAU_PROFILE_START(t);
  sleep(1);
  TAU_PROFILE_STOP(t);

}

int f2(void)
{
  TAU_PROFILE_TIMER(t,"f2()", "", TAU_USER);
  TAU_PROFILE_START(t);

  TAU_PHASE_CREATE_STATIC(t2,"IO Phase", "", TAU_USER);
  TAU_PHASE_START(t2);
  input();
  output();
  TAU_PHASE_STOP(t2);
  TAU_PROFILE_STOP(t);
  return 0;
}


int f3(int x) {
  TAU_PROFILE_TIMER(t,"f3()", "", TAU_USER);
  TAU_PROFILE_START(t);
  sleep(x);
  TAU_PROFILE_STOP(t);
  return 0;
}

int f4(void) {
  TAU_PROFILE_TIMER(t,"f4()", "", TAU_USER);
  TAU_PROFILE_START(t);
  sleep(1);
  TAU_PROFILE_STOP(t);
  return 0;
}


int main(int argc, char **argv) {
  int i;
  TAU_PROFILE_TIMER(t,"main()", "", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);
  TAU_PROFILE_START(t);

  for (i=0; i<5; i++) {
    char buf[32];
    snprintf(buf, sizeof(buf),  "Iteration %d", i);

    TAU_PHASE_CREATE_DYNAMIC(phase, buf, "", TAU_USER);
    TAU_PHASE_START(phase);


    // Alternatively, we could use a dynamically name timer, but we would not get phase info
    //    TAU_PROFILE_TIMER_DYNAMIC(t2, buf, "", TAU_USER);

    //    TAU_PROFILE_START(t2);
    printf("Iteration %d\n", i);
    f1();
    f2();
    f3(i);
 
    //    TAU_PROFILE_STOP(t2);

    TAU_PHASE_STOP(phase);
  }

  f1();
  f4();
  TAU_PROFILE_STOP(t);
}
