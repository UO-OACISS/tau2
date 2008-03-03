#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <TAU.h>
#include <string.h>
#include <stdlib.h>

#define TAU_READ TAU_IO
#define TAU_WRITE TAU_IO

int TauWrapperFsync( int fd)
{
  int ret;
  TAU_PROFILE_TIMER(t, "fsync()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = fsync(fd);

  TAU_PROFILE_STOP(t);
  return ret;
}

int TauWrapperOpen(const char *pathname, int flags)
{
  int ret;
  TAU_PROFILE_TIMER(t, "open()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = open(pathname, flags);

  TAU_PROFILE_STOP(t);
  return ret;
}

size_t TauWrapperRead(int fd, void *buf, size_t nbytes)
{
  int ret;
  double currentRead = 0.0;
  struct timeval t1, t2; 
  TAU_PROFILE_TIMER(t, "read()", " ", TAU_READ);
  TAU_REGISTER_EVENT(re, "READ Bandwidth (MB/s)");
  TAU_REGISTER_EVENT(bytesread, "Bytes Read");
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = read(fd, buf, nbytes);
  gettimeofday(&t2, 0);


  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_usec - t1.tv_usec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  TAU_EVENT(re, nbytes/currentRead);
  TAU_EVENT(bytesread, nbytes);

  TAU_PROFILE_STOP(t);
  
  
  return ret;
}

size_t TauWrapperWrite(int fd, void *buf, size_t nbytes)
{
  int ret;
  double currentWrite = 0.0;
  struct timeval t1, t2; 
  TAU_PROFILE_TIMER(t, "write()", " ", TAU_WRITE);
  TAU_REGISTER_EVENT(wb, "WRITE Bandwidth (MB/s)");
  TAU_REGISTER_EVENT(byteswritten, "Bytes Written");
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = write(fd, buf, nbytes);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_usec - t1.tv_usec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  TAU_EVENT(wb, nbytes/currentWrite);
  TAU_EVENT(byteswritten, nbytes);

  TAU_PROFILE_STOP(t);

  return ret;
}
size_t TauWrapperClose(int fd)
{
  int ret;
  TAU_PROFILE_TIMER(t, "close()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = close(fd);

  TAU_PROFILE_STOP(t);
  return ret;
}
