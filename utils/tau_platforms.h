/**********************************************************************
 * tau_platforms.h
 *
 *A header file that contains many of the ifdef and ifndef that are 
 *used to determine different include calls for different platforms
 *and compilers.
 *
 *********************************************************************/

#ifdef TAU_WINDOWS
#pragma warning( disable : 4786 )
#endif

# include <stdio.h>
# include <stdlib.h>

# include <errno.h>
# include <string.h>
# ifdef SOL2CC
#define TAUERRNO ::errno
#define qsort(a, b, c, d) std::qsort(a, (unsigned) b, (unsigned) c, d)
# else // SOL2CC
#define TAUERRNO errno
# endif // SOL2CC

#ifdef TAU_DOT_H_LESS_HEADERS 
# include <iostream>
# include <map>
using namespace std;
#else 
# include <iostream.h>
# include <map.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
# include <stdlib.h>
#if (!defined(TAU_WINDOWS))
# include <unistd.h>
#endif //TAU_WINDOWS
# include <string.h>
# include <fcntl.h>
# include <limits.h>
# include <math.h>
#ifdef TAU_WINDOWS
# include <float.h>
#endif //TAU_WINDOWS

#ifdef COMPAQ_ALPHA 
# include <float.h>
#endif /* COMPAQ_ALPHA */
#ifdef KAI
# include <algobase>
using namespace std;
#endif
#ifdef POOMA_TFLOP 
extern "C" int getopt(int, char *const *, const char *);
extern char *optarg;
extern int optind, opterr, optopt;
#endif

#ifdef FUJITSU
#include <Profile/fujitsu.h>
#endif /* FUJITSU */

#ifndef DBL_MULTIN
#include <float.h>
#endif 

# ifndef TRUE
#   define FALSE 0
#   define TRUE  1
# endif

# if defined(ultrix) || defined(sequent) || defined(butterfly) || defined(GNU)
double fmod (double x, double y)
{
  return ( x - floor(x/y) * y );
}
# endif
