#ifndef _AIX_H_
#define _AIX_H_


//#define _XOPEN _SOURCE
//#define _XOPEN_SOURCE_EXTENDED 1
#define _POSIX_SOURCE
#define _XOPEN_SOURCE 

#ifndef _TIME_T
#define _TIME_T
typedef long            time_t;
#endif

#ifndef _SIZE_T
#define _SIZE_T
typedef unsigned long   size_t;
#endif

struct timeval {
        time_t          tv_sec;         /* seconds */
        long            tv_usec;        /* microseconds */
};

extern "C" {
int              getdtablesize(void);
int gettimeofday(struct timeval *, void *);
int  strncasecmp(const char *, const char *, size_t);
int  strcasecmp(const char *, const char *);
}
/*
extern "C" {
//#include <sys/types.h>
//#include <unistd.h>
//extern int     getopt(int, char* const*, const char*);
}
*/
#endif /* _AIX_H_ */
