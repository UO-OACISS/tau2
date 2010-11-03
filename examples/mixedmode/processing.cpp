#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
//#include "TauWrapper.h" /* For syscalls */


int work(int rank)
{
char buf[64];
  sleep(2);
  sprintf(buf,"Inside work (called from threaded_func): rank %d, pid = %d\n", rank, getpid());
  write(1, buf, strlen(buf));
  return 0;
}


