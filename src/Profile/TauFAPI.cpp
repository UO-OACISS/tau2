/* Fortran Wrapper layer for TAU Portable Profiling */
#include <stdio.h>
#include "Profile/ProfileGroups.h"

/* 
#define DEBUG_PROF
*/

extern "C" {
void * tau_get_profiler(char *, char *, unsigned int);
void tau_start_timer(void *);
void tau_stop_timer(void *);
void tau_exit(char *);
void tau_init(int, char **);
void tau_set_node(int);
void tau_set_context(int);
void tau_register_thread(void);

/*****************************************************************************
* The following routines are called by the Fortran program and they in turn
* invoke the corresponding C routines. 
*****************************************************************************/
void tau_profile_timer_(void **ptr, char *fname, int *flen, char *type, int *tlen, unsigned int *group)
{
#ifdef DEBUG_PROF
  printf("Inside tau_profile_timer_ fname=%s, type=%s *ptr = %x\n", 
    fname, type, *ptr);
#endif /* DEBUG_PROF */
  fname[*flen] = '\0';
  type[*tlen]  = '\0';

  if (*ptr == 0) 
  { 
    *ptr = tau_get_profiler(fname, type, *group);
  }

#ifdef DEBUG_PROF 
  printf("get_profiler returns %x\n", *ptr);
#endif /* DEBUG_PROF */

  return;
}


void tau_profile_start_(void **profiler)
{ 
#ifdef DEBUG_PROF
  printf("start_timer gets %x\n", *profiler);
#endif /* DEBUG_PROF */

  tau_start_timer(*profiler);
  return;
}

void tau_profile_stop_(void **profiler)
{
  tau_stop_timer(*profiler);
  return;
}

void tau_profile_exit_(char *msg)
{
  tau_exit(msg);
  return;
}

void tau_profile_init_(int *argc, char **argv)
{
  tau_init(*argc, argv);
  return;
}

void tau_profile_set_node_(int *node)
{
  tau_set_node(*node);
  return;
} 

void tau_profile_set_context_(int *context)
{
  tau_set_context(*context);
  return;
}

#if (defined (PTHREADS) || defined (TULIPTHREADS))
void tau_register_thread_(void)
{
  tau_register_thread();
  return;
}

/* Cray F90 specific extensions */
void TAU_REGISTER_THREAD(void)
{
  tau_register_thread();
}
#endif 
/* Cray F90 specific extensions */
void _main();
void TAU_PROFILE_TIMER(void **ptr, char *fname, int *flen, char *type, int *tlen, unsigned int *group)
{
#ifdef DEBUG_PROF
  printf("Inside tau_profile_timer_ fname=%s, type=%s *ptr = %x\n", 
    fname, type, *ptr);
#endif /* DEBUG_PROF */


  if (*ptr == 0) 
  { 
#ifdef DEBUG_PROF
    printf("tau_get_profiler() \n");
#endif /* DEBUG_PROF */
    *ptr = tau_get_profiler(fname, " ", TAU_DEFAULT);
  }

#ifdef DEBUG_PROF 
  printf("get_profiler returns %x\n", *ptr);
#endif /* DEBUG_PROF */

  return;
}

void TAU_PROFILE_START(void **profiler)
{
  tau_profile_start_(profiler);
}

void TAU_PROFILE_STOP(void **profiler)
{
  tau_profile_stop_(profiler);
}

void TAU_PROFILE_EXIT(char *msg)
{
  tau_profile_exit_(msg);
}

void TAU_PROFILE_INIT()
{
  _main();
  //tau_profile_init_(argc, argv);
}

void TAU_PROFILE_SET_NODE(int *node)
{
  tau_profile_set_node_(node);
}

void TAU_PROFILE_SET_CONTEXT(int *context)
{
  tau_profile_set_context_(context);
}
} /* extern "C" */
