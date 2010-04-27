#define _GNU_SOURCE
#include <dlfcn.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
  
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
    
#define dprintf if(1) printf
#define TAU_WRITE TAU_IO
#define TAU_READ TAU_IO


#include <vector>
using namespace std;

static vector<string> fid_to_string_map; // not used (yet)

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
static vector<TauUserEvent*> iowrap_events[NUM_EVENTS];

void *global_write_bandwidth=0, *global_read_bandwidth=0, *global_bytes_written=0, *global_bytes_read=0;

void __attribute__ ((constructor)) tau_iowrap_preload_init(void);
void __attribute__ ((destructor)) tau_iowrap_preload_fini(void);

/*********************************************************************
 * register the different events (read/write/etc) for a file descriptor
 ********************************************************************/
static void Tau_iowrap_registerEvents(int fid, const char *pathname) {
  fid++; // skip the "unknown" descriptor
  while (fid_to_string_map.size() <= fid) {
    fid_to_string_map.push_back("unknown");
  }
  fid_to_string_map[fid] = pathname;

  for (int i=0; i<NUM_EVENTS; i++) {
    while (iowrap_events[i].size() <= fid) {
      iowrap_events[i].push_back(0);
    }
    void *event = 0;
    string name = string(iowrap_event_names[i]) + " <file=\"" + pathname + "\">";
    Tau_get_context_userevent(&event, strdup((char*)name.c_str()));
    iowrap_events[i][fid] = (TauUserEvent*)event;
  }
}

/*********************************************************************
 * shared library constructor, register the unknown, stdin, stdout, and stderr
 ********************************************************************/
void tau_iowrap_preload_init() {
  Tau_init_initializeTAU();
  Tau_iowrap_registerEvents(-1, "unknown");
  Tau_iowrap_registerEvents(0, "stdin");
  Tau_iowrap_registerEvents(1, "stdout");
  Tau_iowrap_registerEvents(2, "stderr");
  Tau_get_context_userevent(&global_write_bandwidth, "WRITE Bandwidth (MB/s)");
  Tau_get_context_userevent(&global_read_bandwidth, "READ Bandwidth (MB/s)");
  Tau_get_context_userevent(&global_bytes_written, "Bytes Written");
  Tau_get_context_userevent(&global_bytes_read, "Bytes Read");
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
  if (fid >= iowrap_events[(int)type].size()) {
    fid = 0; // use the "unknown" descriptor
  }
  return iowrap_events[(int)type][fid];
}

#define TAU_GET_IOWRAP_EVENT(e, event, fid) void *e = Tau_iowrap_getEvent(event, fid);


/*********************************************************************
 * write 
 ********************************************************************/
extern "C" ssize_t write (int fd, const void *buf, size_t count) {
  static ssize_t (*_write)(int fd, const void *buf, size_t count) = NULL;
  ssize_t ret;
  if (_write == NULL) {
    _write = ( ssize_t (*)(int fd, const void *buf, size_t count)) dlsym(RTLD_NEXT, "write");
  }

  if (Tau_global_get_insideTAU() > 0) {
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
  fflush(stdout);
  fflush(stderr);

  return ret;
}


/*********************************************************************
 * read 
 ********************************************************************/
extern "C" ssize_t read (int fd, void *buf, size_t count) {
  static ssize_t (*_read)(int fd, void *buf, size_t count) = NULL;
  ssize_t ret; 

  if (_read == NULL) {
    _read = ( ssize_t (*)(int fd, void *buf, size_t count)) dlsym(RTLD_NEXT, "read");
  }

  if (Tau_global_get_insideTAU() > 0) {
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

  dprintf ("* TAU: read : %d bytes\n", ret);
  fflush(stdout);
  fflush(stderr);

  return ret;
}


/*********************************************************************
 * readv 
 ********************************************************************/
extern "C" ssize_t readv (int fd, const struct iovec *vec, int count) {
  static ssize_t (*_readv)(int fd, const struct iovec *vec, int count) = NULL;
  ssize_t ret; 
  int i;
  size_t sumOfBytesRead = 0;

  if (_readv == NULL) {
    _readv = ( ssize_t (*)(int fd, const struct iovec *vec, int count)) dlsym(RTLD_NEXT, "readv");
  }

  if (Tau_global_get_insideTAU() > 0) {
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
  fflush(stdout);
  fflush(stderr);

  return ret;
}

/*********************************************************************
 * writev 
 ********************************************************************/
extern "C" ssize_t writev (int fd, const struct iovec *vec, int count) {
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

  if (Tau_global_get_insideTAU() > 0) {
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

  dprintf ("* TAU: write : %d bytes, bandwidth %g \n", sumOfBytesWritten, bw);
  fflush(stdout);
  fflush(stderr);

  return ret;
}

/*********************************************************************
 * open 
 ********************************************************************/
extern "C" int open (const char *pathname, int flags, ...) { 
  static int (*_open)(const char *pathname, int flags, ...)  = NULL;
  mode_t mode; 
  va_list args;
  int ret;

  if (_open == NULL) { 
    _open = ( int (*)(const char *pathname, int flags, ...)) dlsym(RTLD_NEXT, "open"); 
  } 

  if (Tau_global_get_insideTAU() > 0) {
    return _open(pathname, flags, mode); 
  }

  TAU_PROFILE_TIMER(t, "open()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  /* if the file is being created, get the third argument for specifying the 
     mode (e.g., 0644) */
  if (flags & O_CREAT) { 
    va_list args ;
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args); 
  }

  ret = _open(pathname, flags, mode); 
  Tau_iowrap_registerEvents(ret, pathname);
  TAU_PROFILE_STOP(t); 

  dprintf ("* open called on %s\n", pathname); 
  fflush(stdout); 
  fflush(stderr);
    
  return ret; 
} 

/*********************************************************************
 * open64 
 ********************************************************************/
extern "C" int open64 (const char *pathname, int flags, ...) { 
  static int (*_open64)(const char *pathname, int flags, ...)  = NULL;
  mode_t mode; 
  va_list args;
  int ret;

  if (_open64 == NULL) { 
     _open64 = ( int (*)(const char *pathname, int flags, ...)) dlsym(RTLD_NEXT, "open64"); 
  } 

  if (Tau_global_get_insideTAU() > 0) {
    return _open64(pathname, flags, mode); 
  }

  TAU_PROFILE_TIMER(t, "open64()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  if (flags & O_CREAT) { 
    va_list args ;
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args); 
  }

  ret = _open64(pathname, flags, mode); 
  Tau_iowrap_registerEvents(ret, pathname);
  TAU_PROFILE_STOP(t); 
  dprintf ("* open64 called on %s\n", pathname); 
  fflush(stdout); 
  fflush(stderr);
    
  return ret; 
} 

/*********************************************************************
 * creat 
 ********************************************************************/
extern "C" int creat(const char *pathname, mode_t mode) {
  static int (*_creat)(const char *pathname, mode_t mode) = NULL;
  int ret;

  if (_creat == NULL) {
     _creat = ( int (*)(const char *pathname, mode_t mode)) dlsym(RTLD_NEXT, "creat");
  }

  if (Tau_global_get_insideTAU() > 0) {
    return _creat(pathname, mode);
  }

  TAU_PROFILE_TIMER(t, "creat()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _creat(pathname, mode);
  Tau_iowrap_registerEvents(ret, pathname);
  TAU_PROFILE_STOP(t);
  dprintf ("* creat called on %s\n", pathname);
  fflush(stdout);
  fflush(stderr);

  return ret;
}

/*********************************************************************
 * creat64 
 ********************************************************************/
extern "C" int creat64(const char *pathname, mode_t mode) {
  static int (*_creat64)(const char *pathname, mode_t mode) = NULL;
  int ret;

  if (_creat64 == NULL) {
     _creat64 = ( int (*)(const char *pathname, mode_t mode)) dlsym(RTLD_NEXT, "creat64");
  }

  if (Tau_global_get_insideTAU() > 0) {
    return _creat64(pathname, mode);
  }

  TAU_PROFILE_TIMER(t, "creat64()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _creat64(pathname, mode);
  Tau_iowrap_registerEvents(ret, pathname);
  TAU_PROFILE_STOP(t);
  dprintf ("* creat64 called on %s\n", pathname);
  fflush(stdout);
  fflush(stderr);

  return ret;
}


/*********************************************************************
 * close 
 ********************************************************************/
extern "C" int close(int fd) {
  static int (*_close) (int fd) = NULL;
  int ret; 

  if (_close == NULL) {
    _close = (int (*) (int fd) ) dlsym(RTLD_NEXT, "close");
  }

  if (Tau_global_get_insideTAU() > 0) {
    return _close(fd);
  }

  TAU_PROFILE_TIMER(t, "close()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _close(fd);
  TAU_PROFILE_STOP(t); 

  dprintf ("* close called on %d\n", fd);
  fflush(stdout);
  fflush(stderr);
  
  return ret;
}

/*********************************************************************
 * Tau_get_socketname returns the name of the socket (AF_INET/AF_UNIX) 
 ********************************************************************/
extern "C" char * Tau_get_socket_name(const struct sockaddr *sa, char *s, size_t len) {
   switch (sa->sa_family) {
     case AF_INET: 
       inet_ntop(AF_INET, &(((struct sockaddr_in *) sa)->sin_addr), s, len);
       break;
     case AF_INET6: 
       inet_ntop(AF_INET6, &(((struct sockaddr_in6 *) sa)->sin6_addr), s, len);
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
  static int (*_socket) (int domain, int type, int protocol) = NULL;
  int ret;

  if (_socket == NULL) {
    _socket = (int (*) (int domain, int type, int protocol) ) dlsym(RTLD_NEXT, "socket");
  }

  if (Tau_global_get_insideTAU() > 0) {
    return _socket(domain, type, protocol);
  }

  TAU_PROFILE_TIMER(t, "socket()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _socket(domain, type, protocol);
  TAU_PROFILE_STOP(t);

  dprintf ("* socket called on domain %d, type %d, protocol %d\n", domain, type, protocol);
  fflush(stdout);
  fflush(stderr);

  return ret;
}


/*********************************************************************
 * bind 
 ********************************************************************/
extern "C" int bind(int socket, const struct sockaddr *address, socklen_t address_len) {
  static int (*_bind) (int socket, const struct sockaddr *address, socklen_t address_len) = NULL;
  int ret;
  char socketname[2048];

  if (_bind == NULL) {
    _bind = (int (*) (int socket, const struct sockaddr *address, socklen_t address_len) ) dlsym(RTLD_NEXT, "bind");
  }

  if (Tau_global_get_insideTAU() > 0) {
    return _bind(socket, address, address_len);
  }

  TAU_PROFILE_TIMER(t, "bind()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  ret = _bind(socket, address, address_len);
  TAU_PROFILE_STOP(t);

  Tau_get_socket_name(address, (char *)socketname, address_len);
  dprintf("socket name = %s\n", socketname);
  Tau_iowrap_registerEvents(socket, (const char *)socketname);

  fflush(stdout);
  fflush(stderr);

  return ret;

}
/*********************************************************************
 * accept
 ********************************************************************/
extern "C" int accept(int socket, struct sockaddr *address, socklen_t* address_len) {
  static int (*_accept) (int socket, struct sockaddr *address, socklen_t* address_len) = NULL;
  int current;
  char socketname[2048];

  if (_accept == NULL) {
    _accept = (int (*) (int socket, struct sockaddr *address, socklen_t* address_len) ) dlsym(RTLD_NEXT, "accept");
  }

  if (Tau_global_get_insideTAU() > 0) {
    return _accept(socket, address, address_len);
  }

  TAU_PROFILE_TIMER(t, "accept()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  current = _accept(socket, address, address_len);
  TAU_PROFILE_STOP(t);

  Tau_get_socket_name(address, (char *)socketname, *address_len);
  dprintf("socket name = %s\n", socketname);
  Tau_iowrap_registerEvents(current, (const char *)socketname);

  fflush(stdout);
  fflush(stderr);

  return current;

}


