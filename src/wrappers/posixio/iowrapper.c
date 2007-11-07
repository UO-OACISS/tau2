#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <TAU.h>

#define TAU_READ TAU_IO
#define TAU_WRITE TAU_IO

static int CheckRead(size_t bytes)
{
  static double prevRead = 0.0;
  double currentRead;
  const char **inFuncs;
  double **counterExclusiveValues;
  double **counterInclusiveValues;
  int *numOfCalls;
  int *numOfChildCalls;
  const char **counterNames;
  int numOfCounters = 1;
  const char ** counterList;
  TAU_REGISTER_EVENT(re, "READ Bandwidth (MB/s)");
  TAU_REGISTER_EVENT(bytesread, "Bytes Read");

#ifndef PROFILING_ON
  return 0; /* profiling must be turned on for this to work */
#endif /* PROFILING_ON */
 
  inFuncs = (const char **) malloc(sizeof(char *));
  inFuncs[0] = strdup("read()");
  inFuncs[1] = '\0';
  TAU_GET_FUNC_VALS(inFuncs, 1, counterExclusiveValues, 
    counterInclusiveValues, numOfCalls, numOfChildCalls, counterNames, 
    numOfCounters);
  currentRead = counterInclusiveValues[0][0] - prevRead;
  prevRead = counterInclusiveValues[0][0];
  TAU_EVENT(re, bytes/currentRead);
  TAU_EVENT(bytesread, bytes);
#ifdef DEBUG
  printf("read = %g usecs %d calls bytes = %d\n", currentRead, numOfCalls[0], bytes);
#endif /* DEBUG */
/*
  free(inFuncs);
  free(counterExclusiveValues);
  free(counterInclusiveValues);
  free(numOfCalls);
  free(numOfChildCalls);
  free(numOfChildCalls);
  free(counterNames);
*/
  return 0;
}

static int CheckWrite(size_t bytes)
{
  static double prevWrite = 0.0;
  double currentWrite;
  const char **inFuncs;
  double **counterExclusiveValues;
  double **counterInclusiveValues;
  int *numOfCalls;
  int *numOfChildCalls;
  const char **counterNames;
  int numOfCounters;
  const char ** counterList;
#ifndef PROFILING_ON
  return 0; /* profiling must be turned on for this to work */
#endif /* PROFILING_ON */

  TAU_REGISTER_EVENT(wb, "WRITE Bandwidth (MB/s)");
  TAU_REGISTER_EVENT(byteswritten, "Bytes Written");

 
  inFuncs = (const char **) malloc(sizeof(char *));
  inFuncs[0] = strdup("write()");
  inFuncs[1] = '\0';
  TAU_GET_FUNC_VALS(inFuncs, 1, counterExclusiveValues, 
    counterInclusiveValues, numOfCalls, numOfChildCalls, counterNames, 
    numOfCounters);
  currentWrite = counterInclusiveValues[0][0] - prevWrite;
  prevWrite = counterInclusiveValues[0][0];
  TAU_EVENT(wb, bytes/currentWrite);
  TAU_EVENT(byteswritten, bytes);
#ifdef DEBUG
  printf("write = %g usecs %d calls bytes = %d\n", currentWrite, numOfCalls[0], bytes);
#endif /* DEBUG */
/*
  free(inFuncs);
  free(counterExclusiveValues);
  free(counterInclusiveValues);
  free(numOfCalls);
  free(numOfChildCalls);
  free(numOfChildCalls);
  free(counterNames);
*/

  return 0;
}

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
  TAU_PROFILE_TIMER(t, "read()", " ", TAU_READ);
  TAU_PROFILE_START(t);

  ret = read(fd, buf, nbytes);

  
  TAU_PROFILE_STOP(t);
  CheckRead(nbytes);
  return ret;
}

size_t TauWrapperWrite(int fd, void *buf, size_t nbytes)
{
  int ret;
  TAU_PROFILE_TIMER(t, "write()", " ", TAU_WRITE);
  TAU_PROFILE_START(t);

  ret = write(fd, buf, nbytes);

  TAU_PROFILE_STOP(t);
  CheckWrite(nbytes);
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
