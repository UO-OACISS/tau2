#ifndef _TAU_WRAPPER_H_
#define _TAU_WRAPPER_H_

#if (defined(PROFILING_ON) || (TRACING_ON))

/* System wrapper functions for sleep, write and getpid */
extern  int tau__sleep(int secs);
extern  int tau__write(int fd, char * buffer, int bytes);
extern  pid_t tau__getpid(void); 

#define getpid()                        tau__getpid() 
#define write(fd, buffer, bytes)        tau__write(fd, buffer, bytes)
#define sleep(secs)			tau__sleep(secs)

#endif /* PROFILING_ON || TRACING_ON */
#endif /* _TAU_WRAPPER_H_ */
