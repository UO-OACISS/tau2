/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: iowrap_shared.cpp  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : LD_PRELOAD IO wrapper                            **
**                                                                         **
****************************************************************************/

#define _GNU_SOURCE
#include <dlfcn.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <aio.h>
  
#include <stdarg.h>
  
#include <aio.h> 
#include <sys/uio.h>
  
#include <setjmp.h>
#include <TAU.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <Profile/TauInit.h>
    
#define dprintf TAU_VERBOSE
#define TAU_WRITE TAU_IO
#define TAU_READ TAU_IO

#define TAU_MAX_FILENAME_LEN 2048


#include <vector>
using namespace std;

/*********************************************************************
 * register different kinds of events here
 ********************************************************************/
#define NUM_EVENTS 4
typedef enum {
  WRITE_BW,
  WRITE_BYTES,
  READ_BW,
  READ_BYTES
} event_type;

const char *iowrap_event_names[NUM_EVENTS] = {"WRITE Bandwidth (MB/s)", "Bytes Written", "READ Bandwidth (MB/s)", "Bytes Read"};


/*********************************************************************
 * IOvector subclasses vector to provide custom constructor/destructor 
 * to enable/disable wrapping
 ********************************************************************/
static int lightsOut = 0;
class IOvector : public vector<vector<TauUserEvent*> > {
public:
  IOvector(int farg) : vector<vector<TauUserEvent*> >(farg) {
    lightsOut = 0;
  }
  ~IOvector() {
    lightsOut = 1;
  }
};

/*********************************************************************
 * Return the set of events, must be done as a static returned because
 * the wrapped routines may be called during the initializers of other
 * shared libraries and may be before our global_ctors_aux() call
 ********************************************************************/
static IOvector &TheIoWrapEvents() {
  static IOvector iowrap_events(4);
  return iowrap_events;
}

void *global_write_bandwidth=0, *global_read_bandwidth=0, *global_bytes_written=0, *global_bytes_read=0;

void __attribute__ ((constructor)) tau_iowrap_preload_init(void);
void __attribute__ ((destructor)) tau_iowrap_preload_fini(void);

/*********************************************************************
 * return whether we should pass through and not track the IO
 ********************************************************************/
static int Tau_iowrap_checkPassThrough() {
  if (Tau_global_get_insideTAU() > 0 || lightsOut) {
    return 1;
  } else {
    return 0;
  }
}

/*********************************************************************
 * register the different events (read/write/etc) for a file descriptor
 ********************************************************************/
static void Tau_iowrap_registerEvents(int fid, const char *pathname) {
  RtsLayer::LockDB();
  dprintf ("Asked to registering %d with %s (current size=%d)\n", fid, pathname, TheIoWrapEvents()[0].size());
  fid++; // skip the "unknown" descriptor

  for (int i=0; i<NUM_EVENTS; i++) {
    TauUserEvent *unknown_ptr = 0;
    if (TheIoWrapEvents()[i].size() >= 1) {
      unknown_ptr = TheIoWrapEvents()[i][0];
    }
    while (TheIoWrapEvents()[i].size() <= fid) {
      TheIoWrapEvents()[i].push_back(unknown_ptr);
      if (TheIoWrapEvents()[i].size()-1 != fid) {
	dprintf ("Registering %d with unknown\n", TheIoWrapEvents()[i].size()-2);
      }
    }
    void *event = 0;
    string name = string(iowrap_event_names[i]) + " <file=\"" + pathname + "\">";
    Tau_get_context_userevent(&event, strdup((char*)name.c_str()));
    TheIoWrapEvents()[i][fid] = (TauUserEvent*)event;
  }
  dprintf ("Registering %d with %s\n", fid-1, pathname);
  RtsLayer::UnLockDB();
}

/*********************************************************************
 * unregister events for a file descriptor
 ********************************************************************/
static void Tau_iowrap_unregisterEvents(int fid) {
  RtsLayer::LockDB();
  dprintf ("Un-registering %d\n", fid);
  fid++; // skip the "unknown" descriptor

  for (int i=0; i<NUM_EVENTS; i++) {
    TauUserEvent *unknown_ptr = 0;
    if (TheIoWrapEvents()[i].size() >= 1) {
      unknown_ptr = TheIoWrapEvents()[i][0];
    }
    while (TheIoWrapEvents()[i].size() <= fid) {
      TheIoWrapEvents()[i].push_back(unknown_ptr);
    }
    TheIoWrapEvents()[i][fid] = unknown_ptr;
  }
  RtsLayer::UnLockDB();
}


/*********************************************************************
 * Tau_iowrap_dupEvents takes care of the associating the events with the 
 * new file descriptor obtained by using dup/dup2 calls.
 ********************************************************************/
static void Tau_iowrap_dupEvents(int oldfid, int newfid) {
  RtsLayer::LockDB();
  dprintf ("dup (old=%d, new=%d)\n", oldfid, newfid);
  oldfid++; // skip the "unknown" descriptor
  newfid++;

  for (int i=0; i<NUM_EVENTS; i++) {
    while (TheIoWrapEvents()[i].size() <= newfid) {
      TheIoWrapEvents()[i].push_back(0);
    }
    TheIoWrapEvents()[i][newfid] = TheIoWrapEvents()[i][oldfid];
  }
  RtsLayer::UnLockDB();
}



/*********************************************************************
 * initializer
 ********************************************************************/
void Tau_iowrap_checkInit() {
  static int init = 0;
  if (init == 1) {
    return;
  }
  init = 1;

  global_write_bandwidth=0;
  global_read_bandwidth=0;
  global_bytes_written=0;
  global_bytes_read=0;

  Tau_init_initializeTAU();
  Tau_iowrap_registerEvents(-1, "unknown");
  Tau_iowrap_registerEvents(0, "stdin");
  Tau_iowrap_registerEvents(1, "stdout");
  Tau_iowrap_registerEvents(2, "stderr");
  Tau_get_context_userevent(&global_write_bandwidth, "WRITE Bandwidth (MB/s)");
  Tau_get_context_userevent(&global_read_bandwidth, "READ Bandwidth (MB/s)");
  Tau_get_context_userevent(&global_bytes_written, "Bytes Written");
  Tau_get_context_userevent(&global_bytes_read, "Bytes Read");
  Tau_create_top_level_timer_if_necessary();
}


/*********************************************************************
 * shared library constructor
 ********************************************************************/
void tau_iowrap_preload_init() {
  Tau_iowrap_checkInit();
}

/*********************************************************************
 * shared library destructor
 ********************************************************************/
void tau_iowrap_preload_fini() {
}

/*********************************************************************
 * Get the user event for the given type of event and file descriptor
 ********************************************************************/
static void *Tau_iowrap_getEvent(event_type type, int fid) {
  fid++; // skip the "unknown" descriptor
  if (fid >= TheIoWrapEvents()[(int)type].size()) {
    dprintf ("************** unknown fid! %d\n", fid-1);
    fid = 0; // use the "unknown" descriptor
  }
  return TheIoWrapEvents()[(int)type][fid];
}

#define TAU_GET_IOWRAP_EVENT(e, event, fid) void *e = Tau_iowrap_getEvent(event, fid);



/*********************************************************************
 * fopen 
 ********************************************************************/
extern "C" FILE *fopen(const char *path, const char *mode) {
  Tau_iowrap_checkInit();
  static FILE* (*_fopen)(const char *path, const char *mode) = NULL;
  FILE *ret;
  if (_fopen == NULL) {
    _fopen = ( FILE* (*)(const char *path, const char *mode)) dlsym(RTLD_NEXT, "fopen");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    return _fopen(path, mode);
  }

  TAU_PROFILE_TIMER(t, "fopen()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _fopen(path, mode);
  if (ret != NULL) {
    Tau_iowrap_registerEvents(fileno(ret), path);
  }
  TAU_PROFILE_STOP(t); 

  dprintf ("* fopen called on %s\n", path); 
  return ret; 
}

/*********************************************************************
 * fopen64 
 ********************************************************************/
extern "C" FILE *fopen64(const char *path, const char *mode) {
  Tau_iowrap_checkInit();
  static FILE* (*_fopen64)(const char *path, const char *mode) = NULL;
  FILE *ret;
  if (_fopen64 == NULL) {
    _fopen64 = ( FILE* (*)(const char *path, const char *mode)) dlsym(RTLD_NEXT, "fopen64");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    return _fopen64(path, mode);
  }

  TAU_PROFILE_TIMER(t, "fopen64()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _fopen64(path, mode);
  if (ret != NULL) {
    Tau_iowrap_registerEvents(fileno(ret), path);
  }
  TAU_PROFILE_STOP(t); 

  dprintf ("* fopen64 called on %s\n", path); 
  return ret; 
}


/*********************************************************************
 * fdopen 
 ********************************************************************/
extern "C" FILE *fdopen(int fd, const char *mode) {
  Tau_iowrap_checkInit();
  static FILE* (*_fdopen)(int fd, const char *mode) = NULL;
  FILE *ret;
  if (_fdopen == NULL) {
    _fdopen = ( FILE* (*)(int fd, const char *mode)) dlsym(RTLD_NEXT, "fdopen");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    return _fdopen(fd, mode);
  }

  TAU_PROFILE_TIMER(t, "fdopen()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _fdopen(fd, mode);
  TAU_PROFILE_STOP(t); 

  dprintf ("* fdopen called on %d\n", fd); 
  return ret; 
}

/*********************************************************************
 * freopen 
 ********************************************************************/
extern "C" FILE *freopen(const char *path, const char *mode, FILE *stream) {
  Tau_iowrap_checkInit();
  static FILE* (*_freopen)(const char *path, const char *mode, FILE *stream) = NULL;
  FILE *ret;
  if (_freopen == NULL) {
    _freopen = ( FILE* (*)(const char *path, const char *mode, FILE *stream)) dlsym(RTLD_NEXT, "freopen");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    return _freopen(path, mode, stream);
  }

  TAU_PROFILE_TIMER(t, "freopen()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _freopen(path, mode, stream);
  if (ret != NULL) {
    Tau_iowrap_registerEvents(fileno(ret), path);
  }
  TAU_PROFILE_STOP(t); 

  dprintf ("* freopen called on %s\n", path); 
  return ret; 
}

/*********************************************************************
 * fclose 
 ********************************************************************/
extern "C" int fclose(FILE *fp) {
  Tau_iowrap_checkInit();
  static int (*_fclose)(FILE *fp) = NULL;
  int ret;
  if (_fclose == NULL) {
    _fclose = ( int (*)(FILE *fp)) dlsym(RTLD_NEXT, "fclose");
  }
  
  int fd = fileno(fp);

  if (Tau_iowrap_checkPassThrough()) {
    return _fclose(fp);
  }

  TAU_PROFILE_TIMER(t, "fclose()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  Tau_iowrap_unregisterEvents (fd);
  ret = _fclose(fp);
  TAU_PROFILE_STOP(t); 

  dprintf ("* fclose(%d) called\n", fd); 
  return ret; 
}



/*********************************************************************
 * fprintf 
 ********************************************************************/
extern "C" int fprintf(FILE *stream, const char *format, ...) {
  Tau_iowrap_checkInit();
  va_list arg;

  static int (*_fprintf)(FILE *stream, const char *format, ...) = NULL;
  int ret;
  if (_fprintf == NULL) {
    _fprintf = ( int (*)(FILE *stream, const char *format, ...)) dlsym(RTLD_NEXT, "fprintf");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    va_start (arg, format);
    ret = vfprintf(stream, format, arg);
    va_end (arg);
    return ret;
  }
  Tau_global_incr_insideTAU();


  double currentWrite = 0.0;
  struct timeval t1, t2;
  double bw = 0.0;

  TAU_GET_IOWRAP_EVENT(wb, WRITE_BW, fileno(stream));
  TAU_GET_IOWRAP_EVENT(byteswritten, WRITE_BYTES, fileno(stream));

  TAU_PROFILE_TIMER(t, "fprintf()", " ", TAU_IO);
  TAU_PROFILE_START(t);
  gettimeofday(&t1, 0);

  va_start (arg, format);
  ret = vfprintf(stream, format, arg);
  va_end (arg);
  
  gettimeofday(&t2, 0);

  int count = ret;

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentWrite > 1e-12) {
    bw = (double) count/currentWrite; 
    TAU_CONTEXT_EVENT(wb, bw);
    TAU_CONTEXT_EVENT(global_write_bandwidth, bw);
  } else {
    dprintf("TauWrapperWrite: currentWrite = %g\n", currentWrite);
  }
  TAU_CONTEXT_EVENT(byteswritten, count);
  TAU_CONTEXT_EVENT(global_bytes_written, count);

  TAU_PROFILE_STOP(t); 

  dprintf ("* fprintf called\n"); 
  Tau_global_decr_insideTAU();
  return ret; 
}

/*********************************************************************
 * fscanf 
 ********************************************************************/
extern "C" int fscanf(FILE *stream, const char *format, ...) {
  Tau_iowrap_checkInit();
  va_list arg;

  static int (*_fscanf)(FILE *stream, const char *format, ...) = NULL;
  int ret;
  if (_fscanf == NULL) {
    _fscanf = ( int (*)(FILE *stream, const char *format, ...)) dlsym(RTLD_NEXT, "fscanf");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    va_start (arg, format);
    ret = vfscanf(stream, format, arg);
    va_end (arg);
    return ret;
  }

  double currentRead = 0.0;
  struct timeval t1, t2;
  double bw = 0.0;

  TAU_GET_IOWRAP_EVENT(rb, READ_BW, fileno(stream));
  TAU_GET_IOWRAP_EVENT(bytesread, READ_BYTES, fileno(stream));
  TAU_PROFILE_TIMER(t, "fscanf()", " ", TAU_IO);
  TAU_PROFILE_START(t);
  gettimeofday(&t1, 0);

  va_start (arg, format);
  ret = vfscanf(stream, format, arg);
  va_end (arg);
  
  gettimeofday(&t2, 0);

  int count = ret;

  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentRead > 1e-12) {
    bw = (double) count/currentRead; 
    TAU_CONTEXT_EVENT(rb, bw);
    TAU_CONTEXT_EVENT(global_read_bandwidth, bw);
  } else {
    dprintf("TauWrapperWrite: currentWrite = %g\n", currentRead);
  }
  TAU_CONTEXT_EVENT(bytesread, count);
  TAU_CONTEXT_EVENT(global_bytes_read, count);

  TAU_PROFILE_STOP(t); 

  dprintf ("* fscanf called\n"); 
  return ret; 
}

/*********************************************************************
 * fwrite 
 ********************************************************************/
extern "C" size_t fwrite( const void *ptr, size_t size, size_t nmemb, FILE *stream) {
  Tau_iowrap_checkInit();
  static size_t (*_fwrite)(const void *ptr, size_t size, size_t nmemb, FILE *stream) = NULL;
  size_t ret;
  if (_fwrite == NULL) {
    _fwrite = ( size_t (*)(const void *ptr, size_t size, size_t nmemb, FILE *stream)) dlsym(RTLD_NEXT, "fwrite");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    return _fwrite(ptr, size, nmemb, stream);
  }

  double currentWrite = 0.0;
  struct timeval t1, t2;
  double bw = 0.0;

  TAU_GET_IOWRAP_EVENT(wb, WRITE_BW, fileno(stream));
  TAU_GET_IOWRAP_EVENT(byteswritten, WRITE_BYTES, fileno(stream));
  TAU_PROFILE_TIMER(t, "fwrite()", " ", TAU_IO);
  TAU_PROFILE_START(t);
  gettimeofday(&t1, 0);
  ret = _fwrite(ptr, size, nmemb, stream);
  gettimeofday(&t2, 0);

  int count = ret * size;

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentWrite > 1e-12) {
    bw = (double) count/currentWrite; 
    TAU_CONTEXT_EVENT(wb, bw);
    TAU_CONTEXT_EVENT(global_write_bandwidth, bw);
  } else {
    dprintf("TauWrapperWrite: currentWrite = %g\n", currentWrite);
  }
  TAU_CONTEXT_EVENT(byteswritten, count);
  TAU_CONTEXT_EVENT(global_bytes_written, count);

  TAU_PROFILE_STOP(t); 

  dprintf ("* fwrite called\n"); 
  return ret; 
}

/*********************************************************************
 * fread 
 ********************************************************************/
extern "C" size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  Tau_iowrap_checkInit();
  static size_t (*_fread)(void *ptr, size_t size, size_t nmemb, FILE *stream) = NULL;
  int ret;
  if (_fread == NULL) {
    _fread = ( size_t (*)(void *ptr, size_t size, size_t nmemb, FILE *stream)) dlsym(RTLD_NEXT, "fread");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    return _fread(ptr, size, nmemb, stream);
  }

  double currentRead = 0.0;
  struct timeval t1, t2;
  TAU_PROFILE_TIMER(t, "read()", " ", TAU_READ|TAU_IO);
  TAU_GET_IOWRAP_EVENT(re, READ_BW, fileno(stream));
  TAU_GET_IOWRAP_EVENT(bytesread, READ_BYTES, fileno(stream));
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = _fread(ptr, size, nmemb, stream);
  gettimeofday(&t2, 0);
  int count = ret * size;

  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentRead > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) count/currentRead);
    TAU_CONTEXT_EVENT(global_read_bandwidth, (double) count/currentRead);
  } else {
    dprintf("TauWrapperRead: currentRead = %g\n", currentRead);
  }
  TAU_CONTEXT_EVENT(bytesread, count);
  TAU_CONTEXT_EVENT(global_bytes_read, count);

  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: read : %d bytes\n", ret);
  return ret; 
}

/*********************************************************************
 * fcntl
 ********************************************************************/
extern "C" int fcntl(int fd, int cmd, ...) {
  Tau_iowrap_checkInit();
  va_list ap;
  void *arg;

  static int (*_fcntl)(int fd, int cmd, ...) = NULL;
  int ret;
  if (_fcntl == NULL) {
    _fcntl = ( int (*)(int fd, int cmd, ...)) dlsym(RTLD_NEXT, "fcntl");   
  }

  switch (cmd) {
    /* No arg */
    case F_GETFD : /* From kernel source fs/fcntl.c:do_fcntl() */
    case F_GETFL :
#if defined(F_GETOWN)
    case F_GETOWN :
#endif
#if defined(F_GETSIG)
    case F_GETSIG :
#endif
#if defined(F_GETLEASE)
    case F_GETLEASE :
#endif
      ret = _fcntl(fd, cmd, 0);
      break;
    default :
      va_start (ap, cmd);
      arg = va_arg (ap, void *);
      va_end (ap);
      ret = _fcntl(fd, cmd, arg);
      break;
  }
  
  switch (cmd) {
    case F_DUPFD :
      Tau_iowrap_dupEvents(fd, ret);
      break;
  }
  dprintf ("* fcntl(fid=%d,cmd=%d...) called\n", fd, cmd);
  return ret;
}


/*********************************************************************
 * lseek
 ********************************************************************/
extern "C" off_t lseek(int fd, off_t offset, int whence) {
  Tau_iowrap_checkInit();
  static off_t (*_lseek)(int fd, off_t offset, int whence) = NULL;
  int ret;
  if (_lseek == NULL) {
    _lseek = ( off_t (*)(int fd, off_t offset, int whence)) dlsym(RTLD_NEXT, "lseek");   }

  if (Tau_iowrap_checkPassThrough()) {
    return _lseek(fd, offset, whence);
  }

  TAU_PROFILE_TIMER(t, "lseek()", " ", TAU_IO);
  TAU_PROFILE_START(t); 
  ret = _lseek(fd, offset, whence);
  TAU_PROFILE_STOP(t);

  dprintf ("* lseek called\n");
  return ret;
}

/*********************************************************************
 * lseek64
 ********************************************************************/
extern "C" off_t lseek64(int fd, off_t offset, int whence) {
  Tau_iowrap_checkInit();
  static off_t (*_lseek64)(int fd, off_t offset, int whence) = NULL;
  int ret;
  if (_lseek64 == NULL) {
    _lseek64 = ( off_t (*)(int fd, off_t offset, int whence)) dlsym(RTLD_NEXT, "lseek64");   }

  if (Tau_iowrap_checkPassThrough()) {
    return _lseek64(fd, offset, whence);
  }

  TAU_PROFILE_TIMER(t, "lseek64()", " ", TAU_IO);
  TAU_PROFILE_START(t);
  ret = _lseek64(fd, offset, whence);
  TAU_PROFILE_STOP(t);

  dprintf ("* lseek64 called\n");
  return ret;
}

/*********************************************************************
 * fseek 
 ********************************************************************/
extern "C" int fseek(FILE *stream, long offset, int whence) {
  Tau_iowrap_checkInit();
  static int (*_fseek)(FILE *stream, long offset, int whence) = NULL;
  int ret;
  if (_fseek == NULL) {
    _fseek = ( int (*)(FILE *stream, long offset, int whence)) dlsym(RTLD_NEXT, "fseek");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    return _fseek(stream, offset, whence);
  }

  TAU_PROFILE_TIMER(t, "fseek()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _fseek(stream, offset, whence);
  TAU_PROFILE_STOP(t); 

  dprintf ("* fseek called\n"); 
  return ret; 
}

/*********************************************************************
 * rewind 
 ********************************************************************/
extern "C" void rewind(FILE *stream) {
  Tau_iowrap_checkInit();
  static void (*_rewind)(FILE *stream) = NULL;
  int ret;
  if (_rewind == NULL) {
    _rewind = ( void (*)(FILE *stream)) dlsym(RTLD_NEXT, "rewind");
  }
  
  if (Tau_iowrap_checkPassThrough()) {
    _rewind(stream);
  }

  TAU_PROFILE_TIMER(t, "rewind()", " ", TAU_IO);
  TAU_PROFILE_START(t);
  _rewind(stream);
  TAU_PROFILE_STOP(t); 

  dprintf ("* rewind called\n"); 
  return;
}


/*********************************************************************
 * write 
 ********************************************************************/
extern "C" ssize_t write (int fd, const void *buf, size_t count) {
  Tau_iowrap_checkInit();
  static ssize_t (*_write)(int fd, const void *buf, size_t count) = NULL;
  ssize_t ret;
  if (_write == NULL) {
    _write = ( ssize_t (*)(int fd, const void *buf, size_t count)) dlsym(RTLD_NEXT, "write");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _write(fd, buf, count);
  }

  double currentWrite = 0.0;
  struct timeval t1, t2;
  double bw = 0.0;

  TAU_PROFILE_TIMER(t, "write()", " ", TAU_WRITE|TAU_IO);
  TAU_GET_IOWRAP_EVENT(wb, WRITE_BW, fd);
  TAU_GET_IOWRAP_EVENT(byteswritten, WRITE_BYTES, fd);
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = _write(fd, buf, count);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentWrite > 1e-12) {
    bw = (double) count/currentWrite; 
    TAU_CONTEXT_EVENT(wb, bw);
    TAU_CONTEXT_EVENT(global_write_bandwidth, bw);
  } else {
    dprintf("TauWrapperWrite: currentWrite = %g\n", currentWrite);
  }
  TAU_CONTEXT_EVENT(byteswritten, count);
  TAU_CONTEXT_EVENT(global_bytes_written, count);
 
  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: write : %d bytes, bandwidth %g \n", ret, bw);

  return ret;
}


/*********************************************************************
 * read 
 ********************************************************************/
extern "C" ssize_t read (int fd, void *buf, size_t count) {
  Tau_iowrap_checkInit();
  static ssize_t (*_read)(int fd, void *buf, size_t count) = NULL;
  ssize_t ret; 

  if (_read == NULL) {
    _read = ( ssize_t (*)(int fd, void *buf, size_t count)) dlsym(RTLD_NEXT, "read");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _read(fd, buf, count);
  }

  double currentRead = 0.0;
  struct timeval t1, t2;
  TAU_PROFILE_TIMER(t, "read()", " ", TAU_READ|TAU_IO);
  TAU_GET_IOWRAP_EVENT(re, READ_BW, fd);
  TAU_GET_IOWRAP_EVENT(bytesread, READ_BYTES, fd);
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = _read(fd, buf, count);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentRead > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) count/currentRead);
    TAU_CONTEXT_EVENT(global_read_bandwidth, (double) count/currentRead);
  } else {
    dprintf("TauWrapperRead: currentRead = %g\n", currentRead);
  }
  TAU_CONTEXT_EVENT(bytesread, count);
  TAU_CONTEXT_EVENT(global_bytes_read, count);

  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: read(%d) : %d bytes\n", fd, ret);

  return ret;
}


/*********************************************************************
 * readv 
 ********************************************************************/
extern "C" ssize_t readv (int fd, const struct iovec *vec, int count) {
  Tau_iowrap_checkInit();
  static ssize_t (*_readv)(int fd, const struct iovec *vec, int count) = NULL;
  ssize_t ret; 
  int i;
  size_t sumOfBytesRead = 0;

  if (_readv == NULL) {
    _readv = ( ssize_t (*)(int fd, const struct iovec *vec, int count)) dlsym(RTLD_NEXT, "readv");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _readv(fd, vec, count);
  }

  double currentRead = 0.0;
  struct timeval t1, t2;
  TAU_PROFILE_TIMER(t, "readv()", " ", TAU_READ|TAU_IO);
  TAU_GET_IOWRAP_EVENT(re, READ_BW, fd);
  TAU_GET_IOWRAP_EVENT(bytesread, READ_BYTES, fd);
  TAU_PROFILE_START(t);


  gettimeofday(&t1, 0);
  ret = _readv(fd, vec, count);
  gettimeofday(&t2, 0);
  if (ret >= 0 ) {
    for (i = 0; i < count; i++) {
      sumOfBytesRead += vec[i].iov_len; 
    }
  }

  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentRead > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) sumOfBytesRead/currentRead);
    TAU_CONTEXT_EVENT(global_read_bandwidth, (double) sumOfBytesRead/currentRead);
  } else {
    dprintf("TauWrapperRead: currentRead = %g\n", currentRead);
  }
  TAU_CONTEXT_EVENT(bytesread, sumOfBytesRead);
  TAU_CONTEXT_EVENT(global_bytes_read, sumOfBytesRead);

  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: read : %d bytes\n", ret);

  return ret;
}

/*********************************************************************
 * writev 
 ********************************************************************/
extern "C" ssize_t writev (int fd, const struct iovec *vec, int count) {
  Tau_iowrap_checkInit();
  static ssize_t (*_writev)(int fd, const struct iovec *vec, int count) = NULL;
  ssize_t ret;

  double currentWrite = 0.0;
  struct timeval t1, t2;
  double bw = 0.0;
  int i;
  size_t sumOfBytesWritten = 0;

  if (_writev == NULL) {
    _writev = ( ssize_t (*)(int fd, const struct iovec *vec, int count)) dlsym(RTLD_NEXT, "writev");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _writev(fd, vec, count);
  }

  TAU_PROFILE_TIMER(t, "writev()", " ", TAU_WRITE|TAU_IO);
  TAU_GET_IOWRAP_EVENT(wb, WRITE_BW, fd);
  TAU_GET_IOWRAP_EVENT(byteswritten, WRITE_BYTES, fd);
  TAU_PROFILE_START(t);


  gettimeofday(&t1, 0);
  ret = _writev(fd, vec, count);
  gettimeofday(&t2, 0);

  /* calculate the total bytes written */
  for (i = 0; i < count; i++) {
    sumOfBytesWritten += vec[i].iov_len; 
  }

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentWrite > 1e-12) {
    bw = (double) sumOfBytesWritten/currentWrite; 
    TAU_CONTEXT_EVENT(wb, bw);
    TAU_CONTEXT_EVENT(global_write_bandwidth, bw);
  } else {
    dprintf("TauWrapperWrite: currentWrite = %g\n", currentWrite);
  }
  TAU_CONTEXT_EVENT(byteswritten, sumOfBytesWritten);
  TAU_CONTEXT_EVENT(global_bytes_written, sumOfBytesWritten);
 
  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: writev(%d) : %d bytes, bandwidth %g \n", fd, sumOfBytesWritten, bw);

  return ret;
}

/*********************************************************************
 * mkstemp
 ********************************************************************/
extern "C" int mkstemp (char *templat) {
  Tau_iowrap_checkInit();
  static int (*_mkstemp)(char *templat)  = NULL;
  int ret;

  if (_mkstemp == NULL) {
    _mkstemp = ( int (*)(char *templat)) dlsym(RTLD_NEXT, "mkstemp");
  }

  TAU_PROFILE_TIMER(t, "mkstemp()", " ", TAU_IO);

  if (Tau_iowrap_checkPassThrough()) {
    TAU_PROFILE_START(t);
  }

  ret = _mkstemp(templat);

  if (ret != -1) {
    Tau_iowrap_registerEvents(ret, templat);
  }

  if (Tau_iowrap_checkPassThrough()) {
    TAU_PROFILE_STOP(t);
  }

  dprintf ("* mkstemp called on %s\n", templat);

  return ret;
}

/*********************************************************************
 * tmpfile
 ********************************************************************/
extern "C" FILE* tmpfile () {
  Tau_iowrap_checkInit();
  static FILE* (*_tmpfile)()  = NULL;
  FILE* ret;

  if (_tmpfile == NULL) {
    _tmpfile = ( FILE* (*)()) dlsym(RTLD_NEXT, "tmpfile");
  }

  TAU_PROFILE_TIMER(t, "tmpfile()", " ", TAU_IO);

  if (Tau_iowrap_checkPassThrough()) {
    TAU_PROFILE_START(t);
  }

  ret = _tmpfile();

  if (ret != NULL) {
    Tau_iowrap_registerEvents(fileno(ret), "tmpfile");
  }

  if (Tau_iowrap_checkPassThrough()) {
    TAU_PROFILE_STOP(t);
  }

  dprintf ("* tmpfile called\n");

  return ret;
}


/*********************************************************************
 * open 
 ********************************************************************/
extern "C" int open (const char *pathname, int flags, ...) { 
  Tau_iowrap_checkInit();
  static int (*_open)(const char *pathname, int flags, ...)  = NULL;
  mode_t mode; 
  va_list args;
  int ret;

  if (_open == NULL) { 
    _open = ( int (*)(const char *pathname, int flags, ...)) dlsym(RTLD_NEXT, "open"); 
  } 

  if (Tau_iowrap_checkPassThrough()) {
    va_start (args, flags);
    ret = _open(pathname, flags, args);
    va_end (args);
    return ret;
  }

  TAU_PROFILE_TIMER(t, "open()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  /* if the file is being created, get the third argument for specifying the 
     mode (e.g., 0644) */
  if (flags & O_CREAT) { 
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args); 
  }

  ret = _open(pathname, flags, mode); 
  if (ret != -1) {
    Tau_iowrap_registerEvents(ret, pathname);
  }
  TAU_PROFILE_STOP(t); 

  dprintf ("* open called on %s\n", pathname); 
    
  return ret; 
} 

/*********************************************************************
 * open64 
 ********************************************************************/
extern "C" int open64 (const char *pathname, int flags, ...) { 
  Tau_iowrap_checkInit();
  static int (*_open64)(const char *pathname, int flags, ...)  = NULL;
  mode_t mode; 
  va_list args;
  int ret;

  if (_open64 == NULL) { 
     _open64 = ( int (*)(const char *pathname, int flags, ...)) dlsym(RTLD_NEXT, "open64"); 
  } 

  if (Tau_iowrap_checkPassThrough()) {
    va_start (args, flags);
    ret = _open64(pathname, flags, args); 
    va_end(args);
    return ret;
  }

  TAU_PROFILE_TIMER(t, "open64()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  if (flags & O_CREAT) { 
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args); 
  }

  ret = _open64(pathname, flags, mode); 
  if (ret != -1) {
    Tau_iowrap_registerEvents(ret, pathname);
  }
  TAU_PROFILE_STOP(t); 
  dprintf ("* open64 called on %s\n", pathname); 
    
  return ret; 
} 

/*********************************************************************
 * creat 
 ********************************************************************/
extern "C" int creat(const char *pathname, mode_t mode) {
  Tau_iowrap_checkInit();
  static int (*_creat)(const char *pathname, mode_t mode) = NULL;
  int ret;

  if (_creat == NULL) {
     _creat = ( int (*)(const char *pathname, mode_t mode)) dlsym(RTLD_NEXT, "creat");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _creat(pathname, mode);
  }

  TAU_PROFILE_TIMER(t, "creat()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _creat(pathname, mode);
  if (ret != -1) {
    Tau_iowrap_registerEvents(ret, pathname);
  }
  TAU_PROFILE_STOP(t);
  dprintf ("* creat called on %s\n", pathname);

  return ret;
}

/*********************************************************************
 * creat64 
 ********************************************************************/
extern "C" int creat64(const char *pathname, mode_t mode) {
  Tau_iowrap_checkInit();
  static int (*_creat64)(const char *pathname, mode_t mode) = NULL;
  int ret;

  if (_creat64 == NULL) {
     _creat64 = ( int (*)(const char *pathname, mode_t mode)) dlsym(RTLD_NEXT, "creat64");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _creat64(pathname, mode);
  }

  TAU_PROFILE_TIMER(t, "creat64()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _creat64(pathname, mode);
  if (ret != -1) {
    Tau_iowrap_registerEvents(ret, pathname);
  }
  TAU_PROFILE_STOP(t);
  dprintf ("* creat64 called on %s\n", pathname);

  return ret;
}


/*********************************************************************
 * close 
 ********************************************************************/
extern "C" int close(int fd) {
  Tau_iowrap_checkInit();
  static int (*_close) (int fd) = NULL;
  int ret; 

  if (_close == NULL) {
    _close = (int (*) (int fd) ) dlsym(RTLD_NEXT, "close");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _close(fd);
  }

  TAU_PROFILE_TIMER(t, "close()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  Tau_iowrap_unregisterEvents(fd);
  ret = _close(fd);

  TAU_PROFILE_STOP(t); 

  dprintf ("* close called on %d\n", fd);
  return ret;
}


/*********************************************************************
 * pipe 
 ********************************************************************/
extern "C" int pipe(int filedes[2]) {
  Tau_iowrap_checkInit();
  static int (*_pipe) (int filedes[2]) = NULL;
  int ret;

  if (_pipe == NULL) {
    _pipe = (int (*) (int filedes[2]) ) dlsym(RTLD_NEXT, "pipe");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _pipe(filedes);
  }

  TAU_PROFILE_TIMER(t, "pipe()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _pipe(filedes);

  if (ret == 0) {
    Tau_iowrap_registerEvents(filedes[0], "pipe");
    Tau_iowrap_registerEvents(filedes[1], "pipe");
  }

  TAU_PROFILE_STOP(t);

  dprintf ("* pipe called\n");

  return ret;
}


/*********************************************************************
 * Tau_get_socketname returns the name of the socket (AF_INET/AF_UNIX) 
 ********************************************************************/
extern "C" char * Tau_get_socket_name(const struct sockaddr *sa, char *s, size_t len) {
  Tau_iowrap_checkInit();
  char addr[256];
  switch (sa->sa_family) {
    case AF_INET: 
      inet_ntop(AF_INET, &(((struct sockaddr_in *) sa)->sin_addr), addr, len);
      sprintf(s,"%s:%d",addr,ntohs((((struct sockaddr_in *)sa)->sin_port)));
      break;
    case AF_INET6: 
      inet_ntop(AF_INET6, &(((struct sockaddr_in6 *) sa)->sin6_addr), addr, len);
      sprintf(s,"%s:%d",addr,ntohs((((struct sockaddr_in6 *)sa)->sin6_port)));
      break;
    case AF_UNIX:
      strncpy(s, ((char *)(((struct sockaddr_un *) sa)->sun_path)), len);
      break;
    default:
      strncpy(s, "Unknown address family", len);
      return NULL;
  }
  return s;
}

/*********************************************************************
 * socket 
 ********************************************************************/
extern "C" int socket(int domain, int type, int protocol) {
  Tau_iowrap_checkInit();
  static int (*_socket) (int domain, int type, int protocol) = NULL;
  int ret;

  if (_socket == NULL) {
    _socket = (int (*) (int domain, int type, int protocol) ) dlsym(RTLD_NEXT, "socket");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _socket(domain, type, protocol);
  }

  TAU_PROFILE_TIMER(t, "socket()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _socket(domain, type, protocol);

  if (ret != -1) {
    Tau_iowrap_registerEvents(ret, "socket");
  }

  TAU_PROFILE_STOP(t);

  dprintf ("* socket called on domain %d, type %d, protocol %d, ret=%d\n", domain, type, protocol, ret);

  return ret;
}

/*********************************************************************
 * socketpair 
 ********************************************************************/
extern "C" int socketpair(int d, int type, int protocol, int sv[2]) {
  Tau_iowrap_checkInit();
  static int (*_socketpair) (int d, int type, int protocol, int sv[2]) = NULL;
  int ret;

  if (_socketpair == NULL) {
    _socketpair = (int (*) (int d, int type, int protocol, int sv[2]) ) dlsym(RTLD_NEXT, "socketpair");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _socketpair(d, type, protocol, sv);
  }

  TAU_PROFILE_TIMER(t, "socketpair()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _socketpair(d, type, protocol, sv);

  if (ret == 0) {
    Tau_iowrap_registerEvents(sv[0], "socketpair");
    Tau_iowrap_registerEvents(sv[1], "socketpair");
  }

  TAU_PROFILE_STOP(t);

  dprintf ("* socketpair called on domain %d, type %d, protocol %d, returned (%d,%d)\n", d, type, protocol, sv[0], sv[1]);

  return ret;
}


/*********************************************************************
 * bind 
 ********************************************************************/
extern "C" int bind(int socket, const struct sockaddr *address, socklen_t address_len) {
  Tau_iowrap_checkInit();
  static int (*_bind) (int socket, const struct sockaddr *address, socklen_t address_len) = NULL;
  int ret;
  char socketname[TAU_MAX_FILENAME_LEN];

  if (_bind == NULL) {
    _bind = (int (*) (int socket, const struct sockaddr *address, socklen_t address_len) ) dlsym(RTLD_NEXT, "bind");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _bind(socket, address, address_len);
  }

  TAU_PROFILE_TIMER(t, "bind()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _bind(socket, address, address_len);
  TAU_PROFILE_STOP(t);

  if (ret == 0) {
    Tau_get_socket_name(address, (char *)socketname, address_len);
    dprintf("socket name = %s\n", socketname);
    Tau_iowrap_registerEvents(socket, (const char *)socketname);
  }

  return ret;
}

/*********************************************************************
 * accept
 ********************************************************************/
#ifndef _AIX
extern "C" int accept(int socket, struct sockaddr *address, socklen_t* address_len) {
  Tau_iowrap_checkInit();
  static int (*_accept) (int socket, struct sockaddr *address, socklen_t* address_len) = NULL;
  int current;
  char socketname[TAU_MAX_FILENAME_LEN];

  if (_accept == NULL) {
    _accept = (int (*) (int socket, struct sockaddr *address, socklen_t* address_len) ) dlsym(RTLD_NEXT, "accept");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _accept(socket, address, address_len);
  }

  TAU_PROFILE_TIMER(t, "accept()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  current = _accept(socket, address, address_len);
  TAU_PROFILE_STOP(t);
  if (current != -1) {
    Tau_get_socket_name(address, (char *)socketname, *address_len);
    dprintf("socket name = %s\n", socketname);
    Tau_iowrap_registerEvents(current, (const char *)socketname);
  }
  return current;
}
#endif /* _AIX */

/*********************************************************************
 * connect
 ********************************************************************/
extern "C" int connect(int socket, const struct sockaddr *address, socklen_t address_len) {
  Tau_iowrap_checkInit();
  static int (*_connect) (int socket, const struct sockaddr *address, socklen_t address_len) = NULL;
  int current;
  char socketname[2048];

  if (_connect == NULL) {
    _connect = (int (*) (int socket, const struct sockaddr *address, socklen_t address_len) ) dlsym(RTLD_NEXT, "connect");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _connect(socket, address, address_len);
  }

  TAU_PROFILE_TIMER(t, "connect()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  current = _connect(socket, address, address_len);
  TAU_PROFILE_STOP(t);
  if (current != -1) {
    Tau_get_socket_name(address, (char *)socketname, address_len);
    dprintf("socket name = %s\n", socketname);
    Tau_iowrap_registerEvents(socket, (const char *)socketname);
  }

  return current;
}

/*********************************************************************
 * recv
 ********************************************************************/
extern "C" ssize_t recv (int fd, void *buf, size_t count, int flags) {
  Tau_iowrap_checkInit();
  static ssize_t (*_recv)(int fd, void *buf, size_t count, int flags) = NULL;
  ssize_t ret; 

  if (_recv == NULL) {
    _recv = ( ssize_t (*)(int fd, void *buf, size_t count, int flags)) dlsym(RTLD_NEXT, "recv");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _recv(fd, buf, count, flags);
  }

  double currentRead = 0.0;
  struct timeval t1, t2;
  TAU_PROFILE_TIMER(t, "recv()", " ", TAU_READ|TAU_IO);
  TAU_GET_IOWRAP_EVENT(re, READ_BW, fd);
  TAU_GET_IOWRAP_EVENT(bytesrecv, READ_BYTES, fd);
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = _recv(fd, buf, count, flags);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentRead > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) ret/currentRead);
    TAU_CONTEXT_EVENT(global_read_bandwidth, (double) ret/currentRead);
  } else {
    dprintf("TauWrapperRead: currentRead = %g\n", currentRead);
  }
  TAU_CONTEXT_EVENT(bytesrecv, ret);
  TAU_CONTEXT_EVENT(global_bytes_read, ret);

  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: recv : %d bytes\n", ret);

  return ret;
}

/*********************************************************************
 * send
 ********************************************************************/

extern "C" ssize_t send (int fd, const void *buf, size_t count, int flags) {
  Tau_iowrap_checkInit();
  static ssize_t (*_send)(int fd, const void *buf, size_t count, int flags) = NULL;
  ssize_t ret; 

  if (_send == NULL) {
    _send = ( ssize_t (*)(int fd, const void *buf, size_t count, int flags)) dlsym(RTLD_NEXT, "send");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _send(fd, buf, count, flags);
  }

  double currentWrite = 0.0;
  struct timeval t1, t2;
  TAU_PROFILE_TIMER(t, "send()", " ", TAU_WRITE|TAU_IO);
  TAU_GET_IOWRAP_EVENT(re, WRITE_BW, fd);
  TAU_GET_IOWRAP_EVENT(byteswritten, WRITE_BYTES, fd);
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = _send(fd, buf, count, flags);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentWrite > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) ret/currentWrite);
    TAU_CONTEXT_EVENT(global_write_bandwidth, (double) ret/currentWrite);
  } else {
    dprintf("TauWrapperRead: currentWrite = %g\n", currentWrite);
  }
  TAU_CONTEXT_EVENT(byteswritten, ret);
  TAU_CONTEXT_EVENT(global_bytes_written, ret);

  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: send : %d bytes\n", ret);

  return ret;
}


/*********************************************************************
 * sendto
 ********************************************************************/

extern "C" ssize_t sendto (int fd, const void *buf, size_t count, int flags, const struct sockaddr *to, socklen_t len) {
  Tau_iowrap_checkInit();
  static ssize_t (*_sendto)(int fd, const void *buf, size_t count, int flags, const struct sockaddr *to, socklen_t len) = NULL;
  ssize_t ret; 

  if (_sendto == NULL) {
    _sendto = ( ssize_t (*)(int fd, const void *buf, size_t count, int flags, const struct sockaddr *to, socklen_t len)) dlsym(RTLD_NEXT, "sendto");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _sendto(fd, buf, count, flags, to, len);
  }

  double currentWrite = 0.0;
  struct timeval t1, t2;
  TAU_PROFILE_TIMER(t, "sendto()", " ", TAU_WRITE|TAU_IO);
  TAU_GET_IOWRAP_EVENT(re, WRITE_BW, fd);
  TAU_GET_IOWRAP_EVENT(byteswritten, WRITE_BYTES, fd);
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = _sendto(fd, buf, count, flags, to, len);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentWrite > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) count/currentWrite);
    TAU_CONTEXT_EVENT(global_write_bandwidth, (double) count/currentWrite);
  } else {
    dprintf("TauWrapperRead: currentWrite = %g\n", currentWrite);
  }
  TAU_CONTEXT_EVENT(byteswritten, ret);
  TAU_CONTEXT_EVENT(global_bytes_written, ret);

  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: sendto : %d bytes\n", ret);

  return ret;
}


#ifndef _AIX
/*********************************************************************
 * recvfrom
 ********************************************************************/

extern "C" ssize_t recvfrom (int fd, void *buf, size_t count, int flags, struct sockaddr *from, socklen_t *len) {
  Tau_iowrap_checkInit();
  static ssize_t (*_recvfrom)(int fd, void *buf, size_t count, int flags, struct sockaddr *from, socklen_t * len) = NULL;
  ssize_t ret; 

  if (_recvfrom == NULL) {
    _recvfrom = ( ssize_t (*)(int fd, void *buf, size_t count, int flags, struct sockaddr * from, socklen_t * len)) dlsym(RTLD_NEXT, "recvfrom");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _recvfrom(fd, buf, count, flags, from, len);
  }

  double currentRead = 0.0;
  struct timeval t1, t2;
  TAU_PROFILE_TIMER(t, "recvfrom()", " ", TAU_READ|TAU_IO);
  TAU_GET_IOWRAP_EVENT(re, READ_BW, fd);
  TAU_GET_IOWRAP_EVENT(bytesrecvfrom, READ_BYTES, fd);
  TAU_PROFILE_START(t);

  gettimeofday(&t1, 0);
  ret = _recvfrom(fd, buf, count, flags, from, len);
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentRead = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  if (currentRead > 1e-12) {
    TAU_CONTEXT_EVENT(re, (double) ret/currentRead);
    TAU_CONTEXT_EVENT(global_read_bandwidth, (double) ret/currentRead);
  } else {
    dprintf("TauWrapperRead: currentRead = %g\n", currentRead);
  }
  TAU_CONTEXT_EVENT(bytesrecvfrom, ret);
  TAU_CONTEXT_EVENT(global_bytes_read, ret);

  TAU_PROFILE_STOP(t);

  dprintf ("* TAU: recvfrom : %d bytes\n", ret);

  return ret;
}
#endif /* _AIX */

/*********************************************************************
 * dup
 ********************************************************************/
extern "C" int dup(int oldfd) {
  Tau_iowrap_checkInit();
  static int (*_dup)(int oldfd) = NULL;
  int fd;

  if (_dup == NULL) {
    _dup = ( int(*)(int fd)) dlsym(RTLD_NEXT, "dup");   
  }

  fd = _dup(oldfd);

  Tau_iowrap_dupEvents(oldfd, fd);

  return fd;
}


/*********************************************************************
 * dup2
 ********************************************************************/
extern "C" int dup2(int oldfd, int newfd) {
  Tau_iowrap_checkInit();
  static int (*_dup2)(int oldfd, int newfd) = NULL;
  int fd;

  if (_dup2 == NULL) {
    _dup2 = ( int(*)(int fd, int newfd)) dlsym(RTLD_NEXT, "dup2");   
  }

  newfd = _dup2(oldfd, newfd);

  Tau_iowrap_dupEvents(oldfd, newfd);

  return newfd;
}


/*********************************************************************
 * popen
 ********************************************************************/
extern "C" FILE * popen (const char *command, const char *type) {
  Tau_iowrap_checkInit();
  static FILE * (*_popen)(const char *command, const char *type)  = NULL;
  FILE* ret;

  if (_popen == NULL) {
    _popen = ( FILE * (*)(const char *command, const char *type)) dlsym(RTLD_NEXT, "popen");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _popen(command, type);   }

  TAU_PROFILE_TIMER(t, "popen()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _popen(command, type);
  /* NOTE: We use int fileno(FILE *stream) to convert FILE * to int fd */
  Tau_iowrap_registerEvents(fileno(ret), command);
  TAU_PROFILE_STOP(t);

  dprintf ("* popen called on %s\n", command);

  return ret;
}

/*********************************************************************
 * pclose
 ********************************************************************/
extern "C" int pclose(FILE * stream) {
  Tau_iowrap_checkInit();
  static int (*_pclose) (FILE * stream) = NULL;
  int ret;

  if (_pclose == NULL) {
    _pclose = (int (*) (FILE * stream) ) dlsym(RTLD_NEXT, "pclose");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _pclose(stream);
  }

  TAU_PROFILE_TIMER(t, "pclose()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _pclose(stream);
  TAU_PROFILE_STOP(t);

  dprintf ("* pclose called on %d\n", stream);

  return ret;
}

/*********************************************************************
 * aio_read
 ********************************************************************/
extern "C" int aio_read(struct aiocb *aiocbp) {
  Tau_iowrap_checkInit();
  static int (*_aio_read) (struct aiocb *aiocbp) = NULL;
  int ret;

  if (_aio_read == NULL) {
    _aio_read = (int (*) (struct aiocb *aiocbp) ) dlsym(RTLD_NEXT, "aio_read");
  }

  TAU_PROFILE_TIMER(t, "aio_read()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _aio_read(aiocbp);
  TAU_PROFILE_STOP(t);

  dprintf ("* aio_read called\n");

  return ret;
}

/*********************************************************************
 * aio_write
 ********************************************************************/
extern "C" int aio_write(struct aiocb *aiocbp) {
  Tau_iowrap_checkInit();
  static int (*_aio_write) (struct aiocb *aiocbp) = NULL;
  int ret;

  if (_aio_write == NULL) {
    _aio_write = (int (*) (struct aiocb *aiocbp) ) dlsym(RTLD_NEXT, "aio_write");
  }

  TAU_PROFILE_TIMER(t, "aio_write()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _aio_write(aiocbp);
  TAU_PROFILE_STOP(t);

  dprintf ("* aio_write called\n");

  return ret;
}

/*********************************************************************
 * aio_error
 ********************************************************************/
extern "C" int aio_error(const struct aiocb *aiocbp) {
  Tau_iowrap_checkInit();
  static int (*_aio_error) (const struct aiocb *aiocbp) = NULL;
  int ret;

  if (_aio_error == NULL) {
    _aio_error = (int (*) (const struct aiocb *aiocbp) ) dlsym(RTLD_NEXT, "aio_error");
  }

  TAU_PROFILE_TIMER(t, "aio_error()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _aio_error(aiocbp);


  if (ret == 0) {
    // the request was completed
    if (aiocbp->aio_lio_opcode == LIO_READ) {
      TAU_GET_IOWRAP_EVENT(bytesread, READ_BYTES, aiocbp->aio_fildes);
      TAU_CONTEXT_EVENT(bytesread, aiocbp->aio_nbytes);
      TAU_CONTEXT_EVENT(global_bytes_read, aiocbp->aio_nbytes);
    } else if (aiocbp->aio_lio_opcode == LIO_WRITE) {
      TAU_GET_IOWRAP_EVENT(byteswritten, WRITE_BYTES, aiocbp->aio_fildes);
      TAU_CONTEXT_EVENT(byteswritten, aiocbp->aio_nbytes);
      TAU_CONTEXT_EVENT(global_bytes_written, aiocbp->aio_nbytes);
    }
  }

  TAU_PROFILE_STOP(t);

  dprintf ("* aio_error called\n");

  return ret;
}

/*********************************************************************
 * aio_return
 ********************************************************************/
extern "C" ssize_t aio_return(struct aiocb *aiocbp) {
  Tau_iowrap_checkInit();
  static ssize_t (*_aio_return) (struct aiocb *aiocbp) = NULL;
  ssize_t ret;

  if (_aio_return == NULL) {
    _aio_return = (ssize_t (*) (struct aiocb *aiocbp) ) dlsym(RTLD_NEXT, "aio_return");
  }

  TAU_PROFILE_TIMER(t, "aio_return()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _aio_return(aiocbp);
  TAU_PROFILE_STOP(t);

  dprintf ("* aio_return called\n");

  return ret;
}

/*********************************************************************
 * aio_suspend
 ********************************************************************/
extern "C" int aio_suspend(const struct aiocb * const cblist[], int n, const struct timespec *timeout) {
  Tau_iowrap_checkInit();
  static int (*_aio_suspend) (const struct aiocb * const cblist[], int n, const struct timespec *timeout) = NULL;
  int ret;

  if (_aio_suspend == NULL) {
    _aio_suspend = (int (*) (const struct aiocb * const cblist[], int n, const struct timespec *timeout) ) dlsym(RTLD_NEXT, "aio_suspend");
  }

  TAU_PROFILE_TIMER(t, "aio_suspend()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _aio_suspend(cblist, n, timeout);
  TAU_PROFILE_STOP(t);

  dprintf ("* aio_suspend called\n");

  return ret;
}

/*********************************************************************
 * aio_cancel
 ********************************************************************/
extern "C" int aio_cancel(int fd, struct aiocb *aiocbp) {
  Tau_iowrap_checkInit();
  static int (*_aio_cancel) (int fd, struct aiocb *aiocbp) = NULL;
  int ret;

  if (_aio_cancel == NULL) {
    _aio_cancel = (int (*) (int fd, struct aiocb *aiocbp) ) dlsym(RTLD_NEXT, "aio_cancel");
  }

  TAU_PROFILE_TIMER(t, "aio_cancel()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _aio_cancel(fd, aiocbp);
  TAU_PROFILE_STOP(t);

  dprintf ("* aio_cancel called\n");

  return ret;
}

/*********************************************************************
 * lio_listio
 ********************************************************************/

extern "C" int lio_listio(int mode, struct aiocb * const list[], int nent, struct sigevent *sig) {
  Tau_iowrap_checkInit();
  static int (*_lio_listio) (int mode, struct aiocb * const list[], int nent, struct sigevent *sig) = NULL;
  ssize_t ret;

  if (_lio_listio == NULL) {
    _lio_listio = (int (*) (int mode, struct aiocb * const list[], int nent, struct sigevent *sig)) dlsym(RTLD_NEXT, "lio_listio");
  }

  TAU_PROFILE_TIMER(t, "lio_listio()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _lio_listio(mode, list, nent, sig);
  TAU_PROFILE_STOP(t);

  dprintf ("* lio_listio called\n");

  return ret;
}


/*********************************************************************
 * EOF
 ********************************************************************/
