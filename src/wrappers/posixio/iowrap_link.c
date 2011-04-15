#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <TAU.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <stdlib.h>
#include <sys/socket.h>

#define dprintf TAU_VERBOSE

#define TAU_READ TAU_IO
#define TAU_WRITE TAU_IO
extern void Tau_iowrap_checkInit(void);

int __wrap_fsync( int fd)
{
  int ret;
  Tau_iowrap_checkInit();
  TAU_PROFILE_TIMER(t, "fsync()", " ", TAU_IO);
  TAU_PROFILE_START(t);
  ret = __real_fsync(fd);

  if (TauEnv_get_track_io_params()) {
    TAU_REGISTER_CONTEXT_EVENT(fsync_fd, "FSYNC fd");
    TAU_REGISTER_CONTEXT_EVENT(fsync_ret, "FSYNC ret");
    TAU_CONTEXT_EVENT(fsync_fd, fd);
    TAU_CONTEXT_EVENT(fsync_ret, ret);
  }
  TAU_PROFILE_STOP(t);

  dprintf("Fsync call with fd %d ret %d\n", fd, ret);

  return ret;
}

int __wrap_open(const char *pathname, int flags)
{
  int ret;
  Tau_iowrap_checkInit();
  TAU_PROFILE_TIMER(t, "open()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = __real_open(pathname, flags);


  if (TauEnv_get_track_io_params()) {
    TAU_REGISTER_CONTEXT_EVENT(open_fd, "OPEN flags");
    TAU_REGISTER_CONTEXT_EVENT(open_ret, "OPEN ret");
    TAU_CONTEXT_EVENT(open_fd, flags);
    TAU_CONTEXT_EVENT(open_ret, ret);
  }
  TAU_PROFILE_STOP(t);

  dprintf("Open call with pathname %s and flags %d: ret %d\n", pathname, flags, ret);

  return ret;
}

size_t __wrap_read(int fd, void *buf, size_t nbytes)
{
  int ret;
  double currentRead = 0.0;
  struct timeval t1, t2; 
  Tau_iowrap_checkInit();
  TAU_PROFILE_TIMER(t, "read()", " ", TAU_READ);
  TAU_REGISTER_CONTEXT_EVENT(re, "READ Bandwidth (MB/s)");
  TAU_REGISTER_CONTEXT_EVENT(bytesread, "READ Bytes Read");
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = __real_read(fd, buf, nbytes);
  gettimeofday(&t2, 0);


  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  /*  if (currentRead > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) nbytes/currentRead);
    }
  else {
  printf("TauWrapperRead: currentRead = %g\n", currentRead);
    }*/

  if (currentRead > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) nbytes/currentRead);
  }

  TAU_CONTEXT_EVENT(bytesread, nbytes);
  if (TauEnv_get_track_io_params()) {
    TAU_REGISTER_CONTEXT_EVENT(read_fd, "READ fd");
    TAU_REGISTER_CONTEXT_EVENT(read_ret, "READ ret");
    TAU_CONTEXT_EVENT(read_fd, fd);
    TAU_CONTEXT_EVENT(read_ret, ret);
  }

  dprintf("Read fd %d nbytes %d buf %ld ret %d\n", fd, nbytes, (long)buf, ret);

  TAU_PROFILE_STOP(t);

  return ret;
}

size_t __wrap_write(int fd, void *buf, size_t nbytes)
{
  int ret;
  double currentWrite = 0.0;
  struct timeval t1, t2; 
  Tau_iowrap_checkInit();
  TAU_PROFILE_TIMER(t, "write()", " ", TAU_WRITE);
  TAU_REGISTER_CONTEXT_EVENT(wb, "WRITE Bandwidth (MB/s)");
  TAU_REGISTER_CONTEXT_EVENT(byteswritten, "WRITE Bytes Written");
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = __real_write(fd, buf, nbytes);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events *//*
  if (currentWrite > 1e-12) {
    TAU_CONTEXT_EVENT(wb, (double) nbytes/currentWrite);
  }
  else {
    printf("TauWrapperWrite: currentWrite = %g\n", currentWrite);
    }*/

  if (currentWrite > 1e-12) {
    TAU_CONTEXT_EVENT(wb, (double) nbytes/currentWrite);
  }
  TAU_CONTEXT_EVENT(byteswritten, nbytes);
  if (TauEnv_get_track_io_params()) {
    TAU_REGISTER_CONTEXT_EVENT(write_fd, "WRITE fd");
    TAU_REGISTER_CONTEXT_EVENT(write_ret, "WRITE ret");
    TAU_CONTEXT_EVENT(write_fd, fd);
    TAU_CONTEXT_EVENT(write_ret, ret);
  }

  dprintf("Write fd %d nbytes %d buf %ld ret %d\n", fd, nbytes, (long)buf, ret);

  TAU_PROFILE_STOP(t);

  return ret;
}
size_t __wrap_close(int fd)
{
  int ret;
  Tau_iowrap_checkInit();
  TAU_PROFILE_TIMER(t, "close()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = __real_close(fd);

  if (TauEnv_get_track_io_params()) {
    TAU_REGISTER_CONTEXT_EVENT(close_fd, "CLOSE fd");
    TAU_REGISTER_CONTEXT_EVENT(close_ret, "CLOSE ret");
    TAU_CONTEXT_EVENT(close_fd, fd);
    TAU_CONTEXT_EVENT(close_ret, ret);
  }

  dprintf("Close fd %d ret %d\n", fd, ret);

  TAU_PROFILE_STOP(t);

  return ret;
}

int __wrap_select(int nfds, fd_set *readfds, fd_set *writefds, 
  fd_set *exceptfds, const struct timeval *timeout)
{
  int ret;
  Tau_iowrap_checkInit();
  TAU_PROFILE_TIMER(t, "select()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = __real_select(nfds, readfds, writefds, exceptfds, timeout); 

  if (TauEnv_get_track_io_params()) {
    TAU_REGISTER_CONTEXT_EVENT(select_nfds, "select nfds");
    TAU_REGISTER_CONTEXT_EVENT(select_ret, "select ret");
    TAU_CONTEXT_EVENT(select_nfds, nfds);
    TAU_CONTEXT_EVENT(select_ret, ret);
  }

  dprintf("Select nfds %d ret %d\n", nfds, ret);

  TAU_PROFILE_STOP(t);

  return ret;
}

