#ifndef GETOPT_H
#define GETOPT_H

//#ifdef __cplusplus
//extern "C" {
//#endif

extern int optind, opterr;
extern char *optarg;

int getopt (int argc, char *argv[], char *optstring);

//#ifdef __cplusplus
//}
//#endif

#endif  /* GETOPT_H */

