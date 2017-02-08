#ifndef GETOPT_H
#define GETOPT_H

//#ifdef __cplusplus
//extern "C" {
//#endif

int optind, opterr;
char *optarg;

int getopt (int argc, char *argv[], char *optstring);

//#ifdef __cplusplus
//}
//#endif

#endif  /* GETOPT_H */

