/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************/

#ifndef _TAU_UNISTD_H_
#define _TAU_UNISTD_H_

#define close(a) 	TauWrapperClose(a)
#define read(a,b,c) 	TauWrapperRead(a,b,c)
#define write(a,b,c) 	TauWrapperWrite(a,b,c)
#define fsync(a) 	TauWrapperFsync(a)
#define open(a,b,...)   TauWrapperOpen(a,b)

#include <sys/types.h>
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int TauWrapperFysnc( int fd);
int TauWrapperOpen(const char *pathname, int flags);
int TauWrapperClose(int fd);
ssize_t TauWrapperRead(int fd, void *buf, size_t nbytes);
ssize_t TauWrapperWrite(int fd, const void *buf, size_t nbytes);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#include "/usr/include/unistd.h"
#include "/usr/include/fcntl.h"
#endif /* _TAU_UNISTD_H_ */
