#include <sys/time.h>
#include <stdio.h>

void mytimer_(int* arg) {
  static struct timeval t1, t2;
  double elapsed;

  if (arg && *arg) {
    /* -- arg contains numpe, stop timing -- */
    gettimeofday(&t2, 0);
    elapsed = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 1.0e-6;
    fprintf(stdout, "... %6.3f sec on %d pe(s)\n", elapsed, *arg);
    fflush(stdout);
  } else {
    /* -- start timing -- */
    fprintf(stdout, "start ...\n");
    fflush(stdout);
    gettimeofday(&t1, 0);
  }
}
