/*   $Source: /mnt/fast/tau2git/cvsroot/tau2/include/Profile/TauGASP.h,v $ */
/*      $Date: 2009/11/07 09:37:08 $ */
/*  $Revision: 1.1 $ */
/*  Description: upcalls from GASP profiling tool into UPC code */
/*  Copyright 2005, Dan Bonachea <bonachea@cs.berkeley.edu> */

typedef uint64_t gasp_tick_t;

void gaspu_init(int *pmythread, int *pthreads);
void gaspu_barrier();
void gaspu_ticks_now(gasp_tick_t *pval);
void gaspu_ticks_to_sec(gasp_tick_t ticks, double *pval);
void gaspu_dump_shared(void *ptr_to_ptr_to_shared, char *outputbuf, int bufsz);
void gaspu_getenv(const char *key, const char **val);
void gaspu_flags_to_string(int flags, char *str, int sz);
void gaspu_collop_to_string(int op, char *str, int sz);
