#define _GNU_SOURCE
#include <dlfcn.h>

#define _XOPEN_SOURCE 600 /* posix_memalign */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
  
#include <stdarg.h>
  
#include <aio.h>
  
#include <setjmp.h>
#include <TAU.h>
    
/* #include <malloc.h>
*/
    
int TAU_entered = 0; 

#define TAU_WRITE TAU_IO
#define TAU_READ TAU_IO

ssize_t write (int fd, const void *buf, size_t count) {
  static ssize_t (*_write)(int fd, const void *buf, size_t count) = NULL;
  ssize_t ret;

  double currentWrite = 0.0;
  struct timeval t1, t2;

  TAU_PROFILE_TIMER(t, "write()", " ", TAU_WRITE);
  TAU_REGISTER_CONTEXT_EVENT(wb, "WRITE Bandwidth (MB/s)");
  TAU_REGISTER_CONTEXT_EVENT(byteswritten, "Bytes Written");
  TAU_PROFILE_START(t);


  if (_write == NULL) {
    _write = ( ssize_t (*)(int fd, const void *buf, size_t count)) dlsym(RTLD_NEXT, "write");
  }


  gettimeofday(&t1, 0);
  ret = _write(fd, buf, count);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentWrite > 1e-12) {
    TAU_CONTEXT_EVENT(wb, (double) count/currentWrite);
  }
  else {
    printf("TauWrapperWrite: currentWrite = %g\n", currentWrite);
  }
  TAU_CONTEXT_EVENT(byteswritten, count);

  TAU_PROFILE_STOP(t);

  TAU_entered++;
  printf ("* TAU: write : %d bytes\n", ret);
  fflush(stdout);
  fflush(stderr);
  TAU_entered--;

  return ret;
}


ssize_t read (int fd, void *buf, size_t count) {
  static ssize_t (*_read)(int fd, void *buf, size_t count) = NULL;
  ssize_t ret; 

  double currentRead = 0.0;
  struct timeval t1, t2;
  TAU_PROFILE_TIMER(t, "read()", " ", TAU_READ);
  TAU_REGISTER_CONTEXT_EVENT(re, "READ Bandwidth (MB/s)");
  TAU_REGISTER_CONTEXT_EVENT(bytesread, "Bytes Read");
  TAU_PROFILE_START(t);


  if (_read == NULL) {
    _read = ( ssize_t (*)(int fd, void *buf, size_t count)) dlsym(RTLD_NEXT, "read");
  }


  gettimeofday(&t1, 0);
  ret = _read(fd, buf, count);
  gettimeofday(&t2, 0);


  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentRead > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) count/currentRead);
  }
  else {
    printf("TauWrapperRead: currentRead = %g\n", currentRead);
  }
  TAU_CONTEXT_EVENT(bytesread, count);

  TAU_PROFILE_STOP(t);

  TAU_entered++;
  printf ("* TAU: read : %d bytes\n", ret);
  fflush(stdout);
  fflush(stderr);
  TAU_entered--;

  return ret;
}


int open (const char *pathname, int flags, ...) { 
  static int (*_open)(const char *pathname, int flags, ...)  = NULL;
  mode_t mode; 
  va_list args;
  int ret;

  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_TIMER(t, "open()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  if (_open == NULL) { 
    _open = ( int (*)(const char *pathname, int flags, ...)) dlsym(RTLD_NEXT, "open"); 
  } 
  
  if (flags & O_CREAT) { 
    va_list args ;
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args); 
  }

  ret = _open(pathname, flags, mode); 
  TAU_PROFILE_STOP(t); 

  printf ("* open called on %s\n", pathname); 
  fflush(stdout); 
  fflush(stderr);
    
  return ret; 
} 


int open64 (const char *pathname, int flags, ...) { 
  static int (*_open64)(const char *pathname, int flags, ...)  = NULL;
  mode_t mode; 
  va_list args;
  int ret;

  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_TIMER(t, "open64()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  if (_open64 == NULL) { 
     _open64 = ( int (*)(const char *pathname, int flags, ...)) dlsym(RTLD_NEXT, "open64"); 
  } 
  
  if (flags & O_CREAT) { 
    va_list args ;
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args); 
  }

  ret = _open64(pathname, flags, mode); 
  TAU_PROFILE_STOP(t); 
  printf ("* open64 called on %s\n", pathname); 
  fflush(stdout); 
  fflush(stderr);
    
  return ret; 
} 



int close(int fd) {
  static int (*_close) (int fd) = NULL;
  int ret; 

  TAU_PROFILE_TIMER(t, "close()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  if (_close == NULL) {
    _close = (int (*) (int fd) ) dlsym(RTLD_NEXT, "close");
  }
  ret = _close(fd);
  TAU_PROFILE_STOP(t); 
  Tau_stop_top_level_timer_if_necessary();

  printf ("* close called on %d\n", fd);
  fflush(stdout);
  fflush(stderr);
  
  return ret;
}
  


