#include <Profile/Profiler.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdio.h>
#include <fcntl.h>


/* Wrapper routines */

  int tau__write(int fd, char * buffer, int bytes)
  {
    TAU_PROFILE("write()", "int (int, char *, int)", TAU_USER);
    return write(fd, buffer, bytes);
  }

  pid_t tau__getpid(void)
  {
    TAU_PROFILE("getpid()", "pid_t ()", TAU_USER);
    return getpid();
  }

  int tau__sleep(int time)
  {
    TAU_PROFILE("sleep()", "int (int)", TAU_USER);
    return sleep(time);
  }


