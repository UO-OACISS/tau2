/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauFAPI.cpp					  **
**	Description 	: TAU Profiling Package wrapper for F77/F90	  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

/* Fortran Wrapper layer for TAU Portable Profiling */
#ifndef TAU_FAPI
#define TAU_FAPI
#endif

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include <Profile/Profiler.h>
#include <Profile/ProfileGroups.h>
#include <Profile/TauMemory.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void Tau_enable_tracking_mpi_t(void);
extern void Tau_disable_tracking_mpi_t(void);
extern void Tau_track_mpi_t(void);


/* Utility function to retrieve fortran strings */
static inline
void getFortranName(char const ** ocname, int * oclen, char const * fname, int flen)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  // Skip over leading whitespace
  while (isspace(*fname)) {
    ++fname;
    --flen;
  }

  // Copy the string and null terminate
  char * cname = (char *)malloc(flen + 1);
  strncpy(cname, fname, flen);
  cname[flen] = '\0';

  // Cut short at the first unprintable char
  int clen;
  for(clen=0; clen<flen; ++clen) {
    if (!isprint(cname[clen])) {
      cname[clen] = '\0';
      break;
    }
  }

  // Fix continuation lines
  char * p = cname;
  char * q = cname;
  char c;
  while((c = *p++)) {
    if (c == '&') {
      while (isspace(*p))
        ++p;
      continue;
    }
    *q++ = c;
  }
  *q = '\0';

  *ocname = cname;
  *oclen = clen;
}

/*****************************************************************************
* The following routines are called by the Fortran program and they in turn
* invoke the corresponding C routines. 
*****************************************************************************/

void tau_profile_timer_group_(void **ptr, char *infname, int *group, int slen)
{
  if (!*ptr) {
    char const * name;
    int len;
    getFortranName(&name, &len, infname, slen);
    *ptr = Tau_get_profiler(name, "", *group, name);
    free((void*)name);
  }
}

void tau_enable_group_name_local(char *& group_name, int slen)
{
  char const * name;
  int len;
  getFortranName(&name, &len, group_name, slen);
  Tau_enable_group_name(name);
  free((void*)name);
}

void tau_disable_group_name_local(char *& group_name, int slen)
{
  char const * name;
  int len;
  getFortranName(&name, &len, group_name, slen);
  Tau_disable_group_name(name);
  free((void*)name);
}

void tau_pure_start(char const * fname, int flen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, fname, flen);
  Tau_pure_start(localname);
  free((void*)localname);
}
void TAU_PURE_START(char *fname, int flen) {
  tau_pure_start(fname, flen);
}
void tau_pure_start_(char *fname, int flen) {
  tau_pure_start(fname, flen);
}
void tau_pure_start__(char *fname, int flen) {
  tau_pure_start(fname, flen);
}

void tau_pure_stop(char *fname, int flen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, fname, flen);
  Tau_pure_stop(localname);
  free((void*)localname);
}
void TAU_PURE_STOP(char *fname, int flen) {
  tau_pure_stop(fname, flen);
}
void tau_pure_stop_(char *fname, int flen) {
  tau_pure_stop(fname, flen);
}
void tau_pure_stop__(char *fname, int flen) {
  tau_pure_stop(fname, flen);
}

void tau_static_phase_start(char *fname, int flen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, fname, flen);
  Tau_static_phase_start(localname);
  free((void*)localname);
}

void tau_static_phase_start_(char *fname, int flen) {
  tau_static_phase_start(fname, flen);
}

void tau_static_phase_start__(char *fname, int flen) {
  tau_static_phase_start(fname, flen);
}

void TAU_STATIC_PHASE_START(char *fname, int flen) {
  tau_static_phase_start(fname, flen);
}

void tau_static_phase_stop(char *fname, int flen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, fname, flen);
  Tau_static_phase_stop(localname);
  free((void*)localname);
}

void tau_static_phase_stop_(char *fname, int flen) {
  tau_static_phase_stop(fname, flen);
}

void tau_static_phase_stop__(char *fname, int flen) {
  tau_static_phase_stop(fname, flen);
}

void TAU_STATIC_PHASE_STOP(char *fname, int flen) {
  tau_static_phase_stop(fname, flen);
}

void tau_dynamic_phase_start(void *iteration, char *fname, int flen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, fname, flen);
  Tau_dynamic_start(localname, 1);
  free((void*)localname);
}

void tau_dynamic_phase_start_(void *iteration, char *fname, int flen) {
  tau_dynamic_phase_start(iteration, fname, flen);
}

void tau_dynamic_phase_start__(void *iteration, char *fname, int flen) {
  tau_dynamic_phase_start(iteration, fname, flen);
}

void TAU_DYNAMIC_PHASE_START(void *iteration, char *fname, int flen) {
  tau_dynamic_phase_start(iteration, fname, flen);
}

void tau_dynamic_phase_stop(void *iteration, char *fname, int flen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, fname, flen);
  Tau_dynamic_stop(localname, 1);
  free((void*)localname);
}

void tau_dynamic_phase_stop_(void *iteration, char *fname, int flen) {
  tau_dynamic_phase_stop(iteration, fname, flen);
}
  
void tau_dynamic_phase_stop__(void *iteration, char *fname, int flen) {
  tau_dynamic_phase_stop(iteration, fname, flen);
}

void TAU_DYNAMIC_PHASE_STOP(void *iteration, char *fname, int flen) {
  tau_dynamic_phase_stop(iteration, fname, flen);
}

/* TAU_DYNAMIC_TIMER_START/STOP are similar to TAU_DYNAMIC_PHASE_START/STOP */
void tau_dynamic_timer_start(void *iteration, char *fname, int flen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, fname, flen);
  Tau_dynamic_start(localname, 0);
  free((void*)localname);
}

void tau_dynamic_timer_start_(void *iteration, char *fname, int flen) {
  tau_dynamic_timer_start(iteration, fname, flen);
}

void tau_dynamic_timer_start__(void *iteration, char *fname, int flen) {
  tau_dynamic_timer_start(iteration, fname, flen);
}

void TAU_DYNAMIC_TIMER_START(void *iteration, char *fname, int flen) {
  tau_dynamic_timer_start(iteration, fname, flen);
}


/* TAU_STATIC_TIMER_START is the same as TAU_START */
void tau_static_timer_start_(char *fname, int flen)
{
  tau_pure_start(fname, flen);
}

void tau_static_timer_start__(char *fname, int flen)
{
  tau_static_timer_start_(fname, flen);
}

void tau_static_timer_start(char *fname, int flen)
{
  tau_static_timer_start_(fname, flen);
}

void TAU_STATIC_TIMER_START(char *fname, int flen)
{
  tau_static_timer_start_(fname, flen);
}


/* TAU_DYNAMIC_TIMER_STOP */
void tau_dynamic_timer_stop(void *iteration, char *fname, int flen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, fname, flen);
  Tau_dynamic_stop(localname, 0);
  free((void*)localname);
}

void tau_dynamic_timer_stop_(void *iteration, char *fname, int flen) {
  tau_dynamic_timer_stop(iteration, fname, flen);
}

void tau_dynamic_timer_stop__(void *iteration, char *fname, int flen) {
  tau_dynamic_timer_stop(iteration, fname, flen);
}

void TAU_DYNAMIC_TIMER_STOP(void *iteration, char *fname, int flen) {
  tau_dynamic_timer_stop(iteration, fname, flen);
}


/* TAU_STATIC_TIMER_STOP is the same as TAU_STOP */
void tau_static_timer_stop_(char *fname, int flen)
{
  tau_pure_stop(fname, flen);
}

void tau_static_timer_stop__(char *fname, int flen)
{
  tau_static_timer_stop_(fname, flen);
}

void tau_static_timer_stop(char *fname, int flen)
{
  tau_static_timer_stop_(fname, flen);
}

void TAU_STATIC_TIMER_STOP(char *fname, int flen)
{
  tau_static_timer_stop_(fname, flen);
}

#include <vector>
using namespace std; 
/* C API */
void Tau_start(const char *name)
{
  Tau_pure_start(name);
}


void TAU_START(char *fname, int flen)
{
  tau_pure_start(fname, flen);
}

void tau_start(char *fname, int flen)
{
  tau_pure_start(fname, flen);
}

void tau_start_(char *fname, int flen)
{
  tau_pure_start(fname, flen);
}

void tau_start__(char *fname, int flen)
{
  tau_pure_start(fname, flen);
}

void TAU_STOP(char *fname, int flen)
{
  tau_pure_stop(fname, flen);
}

void tau_stop(char *fname, int flen)
{
  tau_pure_stop(fname, flen);
}

void tau_stop_(char *fname, int flen)
{
  tau_pure_stop(fname, flen);
}

void tau_stop__(char *fname, int flen)
{
  tau_pure_stop(fname, flen);
}

/* C API */
void Tau_stop(const char *name)
{
  Tau_pure_stop(name);
}

void tau_profile_timer_(void **ptr, char const * fname, int flen)
{
  if (!*ptr) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

#ifdef TAU_OPENMP
#pragma omp critical (crit_tau_profile_timer)
    {
      if (!*ptr) {
#endif /* TAU_OPENMP */

        char const * localname;
        int locallen;
        getFortranName(&localname, &locallen, fname, flen);

        /* See if a > appears in the function name. If it does, it separates the
         group name from the function name. Separate these two */
        TauGroup_t gr = TAU_USER;
        char const * gr_name = NULL;
        char *first, *second;

        first = strtok((char*)localname, ">");
        if (first) {
          second = strtok(NULL, ">");
          if (second) {
            gr = Tau_get_profile_group(first);
            gr_name = first;
            localname = second;
          } else {
            localname = first;
            gr_name = "TAU_DEFAULT";
          }
        }

        *ptr = Tau_get_profiler(localname, "", gr, gr_name);
        free((void*)localname);
#ifdef TAU_OPENMP
      }
    }
#endif /* TAU_OPENMP */
  }
}

void tau_phase_create_static_(void **ptr, char *infname, int slen)
{
  bool firsttime = (*ptr == 0);
  tau_profile_timer_(ptr, infname, slen);
  /* we know the FunctionInfo pointer in ptr. If its here the first time 
     set the group name to be | TAU_PHASE */
  if (firsttime)
    Tau_mark_group_as_phase(*ptr);
}

void tau_phase_create_dynamic_(void **ptr, char const * infname, int slen)
{
  *ptr = 0;  /* reset it each time so it creates a new timer */
  tau_profile_timer_(ptr, infname, slen);
  Tau_mark_group_as_phase(*ptr);
}

void tau_profile_timer_dynamic_(void **ptr, char const * infname, int slen)
{
  *ptr = 0;  /* reset it each time so it creates a new timer */
  tau_profile_timer_(ptr, infname, slen);
}

void tau_profile_start_(void **profiler)
{ 
#ifdef DEBUG_PROF
  TAU_VERBOSE("start_timer gets %lx\n", *profiler);
#endif /* DEBUG_PROF */

  Tau_lite_start_timer(*profiler, 0);
  return;
}

void tau_profile_stop_(void **profiler)
{
  Tau_lite_stop_timer(*profiler);
  return;
}

void tau_phase_start_(void **profiler)
{
  Tau_lite_start_timer(*profiler, 1); /* 1 indicates phase based profiling */
  return;
}

void tau_phase_stop_(void **profiler)
{
  Tau_lite_stop_timer(*profiler);
  return;
}

void tau_dynamic_iter(int *iteration, void **ptr, char *infname, int slen, int isPhase)
{
  /* we append the iteration number to the name and then call the 
     appropriate routine for creating dynamic phases or timers */  

  /* This routine creates dynamic timers and phases by embedding the
     iteration number in the name. isPhase argument tells whether we
     choose phases or timers. */

  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, infname, slen);
  char const * newName = Tau_append_iteration_to_name(*iteration, localname, locallen);
  int newLength = strlen(newName);
  if (isPhase)  {
    tau_phase_create_dynamic_(ptr, newName, newLength);
  } else {
    tau_profile_timer_dynamic_(ptr, newName, newLength);
  }

  free((void*)newName);
  free((void*)localname);
}

void tau_phase_dynamic_iter_(int *iteration, void **ptr, char *infname, int slen)
{ 
  tau_dynamic_iter(iteration, ptr, infname, slen, 1); /* 1 is for isPhase */
}

void tau_profile_dynamic_iter_(int *iteration, void **ptr, char *infname, int slen)
{ 
  tau_dynamic_iter(iteration, ptr, infname, slen, 0); 
  /* isPhase=0 is for specifying a timer */
}

void tau_phase_dynamic_iter__(int *iteration, void **ptr, char *infname, int slen)
{
  tau_phase_dynamic_iter_(iteration, ptr, infname, slen);
}

void tau_phase_dynamic_iter(int *iteration, void **ptr, char *infname, int slen)
{
  tau_phase_dynamic_iter_(iteration, ptr, infname, slen);
}

void TAU_PHASE_DYNAMIC_ITER(int *iteration, void **ptr, char *infname, int slen)
{
  tau_phase_dynamic_iter_(iteration, ptr, infname, slen);
}

void tau_profile_dynamic_iter__(int *iteration, void **ptr, char *infname, int slen)
{
  tau_profile_dynamic_iter_(iteration, ptr, infname, slen);
}

void tau_profile_dynamic_iter(int *iteration, void **ptr, char *infname, int slen)
{
  tau_profile_dynamic_iter_(iteration, ptr, infname, slen);
}

void TAU_PROFILE_DYNAMIC_ITER(int *iteration, void **ptr, char *infname, int slen)
{
  tau_profile_dynamic_iter_(iteration, ptr, infname, slen);
}

void tau_profile_exit_(char *msg)
{
  Tau_exit(msg);
}

void tau_db_dump_incr_(void)
{
  Tau_dump_incr();
}

void tau_db_dump_(void)
{
  Tau_dump();
}

void tau_db_purge_(void)
{
  Tau_purge();
}

void tau_db_dump_prefix_(char *prefix)
{
  Tau_dump_prefix(prefix);
}

void tau_profile_init_()
{
#ifndef TAU_MPI
#ifndef TAU_SHMEM
  Tau_set_node(0); 
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */
}

void tau_enable_instrumentation(void)
{
  Tau_enable_instrumentation();
}

void tau_disable_instrumentation(void)
{
  Tau_disable_instrumentation();
}

void tau_enable_group(TauGroup_t *group)
{
  Tau_enable_group(*group);
}

void tau_disable_group(TauGroup_t *group)
{
  Tau_disable_group(*group);
}

void tau_enable_all_groups(void)
{
  Tau_enable_all_groups();
}

void tau_disable_all_groups(void)
{
  Tau_disable_all_groups();
}

void tau_enable_group_name(char * group_name, int len)
{
  tau_enable_group_name_local(group_name, len);
}

void tau_disable_group_name(char * group_name, int len)
{
  tau_disable_group_name_local(group_name, len);
}

//////////////////////////////////////////////////////
// MEMORY API
//////////////////////////////////////////////////////

void tau_track_memory(void)
{
  Tau_track_memory();
} 

void tau_track_memory_here(void)
{
  Tau_track_memory_here();
} 

void tau_track_memory_headroom(void)
{
  Tau_track_memory_headroom();
} 

void tau_track_memory_headroom_here(void)
{
  Tau_track_memory_headroom_here();
}

void tau_track_memory_footprint(void)
{
  Tau_track_memory_rss_and_hwm();
} 

void tau_track_memory_footprint_here(void)
{
  Tau_track_memory_rss_and_hwm_here();
} 

void tau_enable_tracking_memory(void)
{
  Tau_enable_tracking_memory();
} 

void tau_disable_tracking_memory(void)
{
  Tau_disable_tracking_memory();
} 

void tau_enable_tracking_memory_headroom(void)
{
  Tau_enable_tracking_memory_headroom();
} 

void tau_disable_tracking_memory_headroom(void)
{
  Tau_disable_tracking_memory_headroom();
} 

void tau_memdbg_protect_above(int* value)
{
  TauEnv_set_memdbg_protect_above(*value);
}
void tau_memdbg_protect_below(int* value)
{
  TauEnv_set_memdbg_protect_below(*value);
}
void tau_memdbg_protect_free(int* value)
{
  TauEnv_set_memdbg_protect_free(*value);
}



void tau_set_interrupt_interval(int* value)
{
  Tau_set_interrupt_interval(*value);
} 

////////////////////////////////////////////////////
void tau_enable_group_(TauGroup_t *group)
{
  Tau_enable_group(*group);
}

void tau_disable_group_(TauGroup_t *group)
{
  Tau_disable_group(*group);
}

void tau_enable_all_groups_(void)
{
  Tau_enable_all_groups();
}

void tau_disable_all_groups_(void)
{
  Tau_disable_all_groups();
}

void tau_profile_set_node_(int *node)
{
  Tau_set_node(*node);
} 

void tau_profile_set_context_(int *context)
{
  Tau_set_context(*context);
}


void tau_enable_instrumentation_(void)
{
  Tau_enable_instrumentation();
}

void tau_disable_instrumentation_(void)
{
  Tau_disable_instrumentation();
}

void tau_enable_group_name_(char * group_name, int len)
{
  tau_enable_group_name_local(group_name, len);
}

void tau_disable_group_name_(char * group_name, int len)
{
  tau_disable_group_name_local(group_name, len);
}

#if (defined (PTHREADS) || defined (TULIPTHREADS))
void tau_register_thread_(void)
{
  Tau_register_thread();
}

void tau_register_thread__(void)
{
  Tau_register_thread();
}

void tau_register_thread(void)
{
  Tau_register_thread();
}

void TAU_REGISTER_THREAD(void)
{
  Tau_register_thread();
}
#endif /* PTHREADS || TULIPTHREADS */

void tau_trace_sendmsg_(int *type, int *destination, int *length)
{
  Tau_trace_sendmsg(*type, *destination, *length);
}

void tau_trace_recvmsg_(int *type, int *source, int *length)
{
  Tau_trace_recvmsg(*type, *source, *length);
}

void tau_register_event_(void **ptr, char *name, int slen)
{
  if (!*ptr) {
    char const * localname;
    int locallen;
    getFortranName(&localname, &locallen, name, slen);
    *ptr = Tau_get_userevent(localname);
    free((void*)localname);
  }
}

void tau_register_context_event_(void **ptr, char *name, int slen)
{
  if (!*ptr) {
    char const * localname;
    int locallen;
    getFortranName(&localname, &locallen, name, slen);
    Tau_get_context_userevent(ptr, localname);
    free((void*)localname);
  }
}

void tau_event_(void **ptr, double *data)
{
  Tau_userevent(*ptr, *data);
}

void tau_context_event_(void **ptr, double *data)
{
  Tau_context_userevent((void *)*ptr, *data);
}

void tau_report_statistics_(void)
{
  Tau_report_statistics();
}

void tau_report_thread_statistics_(void)
{
  Tau_report_thread_statistics();
}

//////////////////////////////////////////////////////
// MEMORY API
//////////////////////////////////////////////////////
void tau_track_memory_(void)
{
  Tau_track_memory();
} 

void tau_track_memory_here_(void)
{
  Tau_track_memory_here();
} 

void tau_track_memory_headroom_(void)
{
  Tau_track_memory_headroom();
} 

void tau_track_memory_headroom_here_(void)
{
  Tau_track_memory_headroom_here();
} 

void tau_track_memory_footprint_(void)
{
  Tau_track_memory_rss_and_hwm();
} 

void tau_track_memory_footprint_here_(void)
{
  Tau_track_memory_rss_and_hwm_here();
} 

void tau_enable_tracking_memory_(void)
{
  Tau_enable_tracking_memory();
} 

void tau_disable_tracking_memory_(void)
{
  Tau_disable_tracking_memory();
} 

void tau_enable_tracking_memory_headroom_(void)
{
  Tau_enable_tracking_memory_headroom();
} 

void tau_disable_tracking_memory_headroom_(void)
{
  Tau_disable_tracking_memory_headroom();
} 

void tau_memdbg_protect_above_(int* value)
{
  TauEnv_set_memdbg_protect_above(*value);
}
void tau_memdbg_protect_below_(int* value)
{
  TauEnv_set_memdbg_protect_below(*value);
}
void tau_memdbg_protect_free_(int* value)
{
  TauEnv_set_memdbg_protect_free(*value);
}


void tau_set_interrupt_interval_(int* value)
{
  Tau_set_interrupt_interval(*value);
} 

/* Cray F90 specific extensions */
#if (defined(CRAYKAI) || defined(HP_FORTRAN))
void _main();
#endif /* CRAYKAI || HP_FORTRAN */
void TAU_PROFILE_TIMER(void **ptr, char *fname, int flen)
{
  tau_profile_timer_(ptr, fname, flen);
}

void TAU_PROFILE_TIMER_(void **ptr, char *fname, int flen)
{
  tau_profile_timer_(ptr, fname, flen);
}

void TAU_PROFILE_START(void **profiler)
{
  tau_profile_start_(profiler);
}

void TAU_PROFILE_START_(void **profiler)
{
  tau_profile_start_(profiler);
}

void TAU_PROFILE_STOP(void **profiler)
{
  tau_profile_stop_(profiler);
}

void TAU_PROFILE_STOP_(void **profiler)
{
  tau_profile_stop_(profiler);
}

void TAU_PROFILE_EXIT(char *msg)
{
  tau_profile_exit_(msg);
}

void TAU_PROFILE_EXIT_(char *msg)
{
  tau_profile_exit_(msg);
}

void TAU_DB_DUMP(void)
{
  Tau_dump();
}

void TAU_DB_PURGE(void)
{
  Tau_purge();
}

void TAU_DB_DUMP_PREFIX(char *prefix)
{
  Tau_dump_prefix(prefix);
}

void TAU_PROFILE_INIT()
{
#ifdef CRAYKAI 
  _main();
#endif /* CRAYKAI */
  // tau_profile_init_(argc, argv);
#ifndef TAU_MPI
#ifndef TAU_SHMEM
  Tau_set_node(0); 
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */
}

void TAU_PROFILE_INIT_()
{
#ifndef TAU_MPI
#ifndef TAU_SHMEM
  Tau_set_node(0); 
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */
}

void TAU_PROFILE_SET_NODE(int *node)
{
  tau_profile_set_node_(node);
}

void TAU_PROFILE_SET_CONTEXT(int *context)
{
  tau_profile_set_context_(context);
}

void TAU_TRACE_SENDMSG(int *type, int *destination, int *length)
{
  Tau_trace_sendmsg(*type, *destination, *length);
}

void TAU_TRACE_RECVMSG(int *type, int *source, int *length)
{
  Tau_trace_recvmsg(*type, *source, *length);
}

void TAU_ENABLE_INSTRUMENTATION(void)
{
  Tau_enable_instrumentation();
}

void TAU_DISABLE_INSTRUMENTATION(void)
{
  Tau_disable_instrumentation();
}

void TAU_ENABLE_ALL_GROUPS(void)
{
  Tau_enable_all_groups();
}

void TAU_DISABLE_ALL_GROUPS(void)
{
  Tau_disable_all_groups();
}

void TAU_ENABLE_GROUP(TauGroup_t *group)
{
  Tau_enable_group(*group);
}

void TAU_DISABLE_GROUP(TauGroup_t *group)
{
  Tau_disable_group(*group);
}

void TAU_ENABLE_GROUP_NAME(char * group_name, int len)
{
  tau_enable_group_name_local(group_name, len);
}

void TAU_DISABLE_GROUP_NAME(char * group_name, int len)
{
  tau_disable_group_name_local(group_name, len);
}
void TAU_REGISTER_EVENT(void **ptr, char *event_name, int flen)
{
  if (!*ptr) {
    char const * localname;
    int locallen;
    getFortranName(&localname, &locallen, event_name, flen);
    *ptr = Tau_get_userevent(localname);
    free((void*)localname);
  }
}

void TAU_REGISTER_CONTEXT_EVENT(void **ptr, char *event_name, int flen)
{
  if (!*ptr) {
    tau_register_context_event_(ptr, event_name, flen);
  }
}


void TAU_EVENT(void **ptr, double *data)
{
  Tau_userevent(*ptr, *data);
}

void TAU_CONTEXT_EVENT(void **ptr, double *data)
{
  Tau_context_userevent(*ptr, *data);
}

void TAU_REPORT_STATISTICS(void)
{
  Tau_report_statistics();
}

void TAU_REPORT_THREAD_STATISTICS(void)
{
  Tau_report_thread_statistics();
}

void tau_profile_timer(void **ptr, char *fname, int flen)
{
  tau_profile_timer_(ptr, fname, flen);
}

//////////////////////////////////////////////////////
// MEMORY API
//////////////////////////////////////////////////////
void TAU_TRACK_MEMORY(void)
{
  Tau_track_memory();
} 

void TAU_TRACK_MEMORY_HERE(void)
{
  Tau_track_memory_here();
} 

void TAU_TRACK_MEMORY_HEADROOM(void)
{
  Tau_track_memory_headroom();
} 

void TAU_TRACK_MEMORY_HEADROOM_HERE(void)
{
  Tau_track_memory_headroom_here();
} 

void TAU_TRACK_MEMORY_FOOTPRINT(void)
{
  Tau_track_memory_rss_and_hwm();
} 

void TAU_TRACK_MEMORY_FOOTPRINT_HERE(void)
{
  Tau_track_memory_rss_and_hwm_here();
} 

void TAU_ENABLE_TRACKING_MEMORY(void)
{
  Tau_enable_tracking_memory();
} 


void TAU_DISABLE_TRACKING_MEMORY(void)
{
  Tau_disable_tracking_memory();
} 

void TAU_ENABLE_TRACKING_MEMORY_HEADROOM(void)
{
  Tau_enable_tracking_memory_headroom();
} 

void TAU_DISABLE_TRACKING_MEMORY_HEADROOM(void)
{
  Tau_disable_tracking_memory_headroom();
} 

void TAU_MEMDBG_PROTECT_ABOVE(int* value)
{
  TauEnv_set_memdbg_protect_above(*value);
}
void TAU_MEMDBG_PROTECT_BELOW(int* value)
{
  TauEnv_set_memdbg_protect_below(*value);
}
void TAU_MEMDBG_PROTECT_FREE(int* value)
{
  TauEnv_set_memdbg_protect_free(*value);
}


//////////////////////////////////////////////////////
// MPI_T API
//////////////////////////////////////////////////////
void TAU_ENABLE_TRACKING_MPI_T(void) 
{
  Tau_enable_tracking_mpi_t();
}

void tau_enable_tracking_mpi_t(void) 
{
  Tau_enable_tracking_mpi_t();
}

void tau_enable_tracking_mpi_t_(void) 
{
  Tau_enable_tracking_mpi_t();
}

void tau_enable_tracking_mpi_t__(void) 
{
  Tau_enable_tracking_mpi_t();
}

void TAU_DISABLE_TRACKING_MPI_T(void)
{
  Tau_disable_tracking_mpi_t();
}

void tau_disable_tracking_mpi_t(void)
{
  Tau_disable_tracking_mpi_t();
}

void tau_disable_tracking_mpi_t_(void)
{ 
  Tau_disable_tracking_mpi_t();
}

void tau_disable_tracking_mpi_t__(void) 
{ 
  Tau_disable_tracking_mpi_t();
}

void TAU_TRACK_MPI_T(void)
{
  Tau_track_mpi_t();
} 

void tau_track_mpi_t(void)
{
  Tau_track_mpi_t();
} 

void tau_track_mpi_t_(void)
{
  Tau_track_mpi_t();
} 

void tau_track_mpi_t__(void)
{
  Tau_track_mpi_t();
} 


//////////////////////////////////////////////////////
// POWER API
//////////////////////////////////////////////////////
void TAU_TRACK_POWER(void)
{
  Tau_track_power();
} 

void TAU_TRACK_POWER_HERE(void)
{
  Tau_track_power_here();
} 

void TAU_DISABLE_TRACKING_POWER(void)
{
  Tau_disable_tracking_power();
} 

void TAU_ENABLE_TRACKING_POWER(void)
{
  Tau_enable_tracking_power();
} 

//////////////////////////////////////////////////////
void tau_track_power(void)
{
  Tau_track_power();
} 

void tau_track_power_here(void)
{
  Tau_track_power_here();
} 

void tau_disable_tracking_power(void)
{
  Tau_disable_tracking_power();
} 

void tau_enable_tracking_power(void)
{
  Tau_enable_tracking_power();
} 

//////////////////////////////////////////////////////
void tau_track_power_(void)
{
  Tau_track_power();
} 

void tau_track_power_here_(void)
{
  Tau_track_power_here();
} 

void tau_disable_tracking_power_(void)
{
  Tau_disable_tracking_power();
} 

void tau_enable_tracking_power_(void)
{
  Tau_enable_tracking_power();
} 

//////////////////////////////////////////////////////
void tau_track_power__(void)
{
  Tau_track_power();
} 

void tau_track_power_here__(void)
{
  Tau_track_power_here();
} 

void tau_disable_tracking_power__(void)
{
  Tau_disable_tracking_power();
} 

void tau_enable_tracking_power__(void)
{
  Tau_enable_tracking_power();
} 


//////////////////////////////////////////////////////
// LOAD API
//////////////////////////////////////////////////////
void TAU_TRACK_LOAD(void)
{
  Tau_track_load();
} 

void TAU_TRACK_LOAD_HERE(void)
{
  Tau_track_load_here();
} 

void TAU_DISABLE_TRACKING_LOAD(void)
{
  Tau_disable_tracking_load();
} 

void TAU_ENABLE_TRACKING_LOAD(void)
{
  Tau_enable_tracking_load();
} 

//////////////////////////////////////////////////////
void tau_track_load(void)
{
  Tau_track_load();
} 

void tau_track_load_here(void)
{
  Tau_track_load_here();
} 

void tau_disable_tracking_load(void)
{
  Tau_disable_tracking_load();
} 

void tau_enable_tracking_load(void)
{
  Tau_enable_tracking_load();
} 

//////////////////////////////////////////////////////
void tau_track_load_(void)
{
  Tau_track_load();
} 

void tau_track_load_here_(void)
{
  Tau_track_load_here();
} 

void tau_disable_tracking_load_(void)
{
  Tau_disable_tracking_load();
} 

void tau_enable_tracking_load_(void)
{
  Tau_enable_tracking_load();
} 

//////////////////////////////////////////////////////
void tau_track_load__(void)
{
  Tau_track_load();
} 

void tau_track_load_here__(void)
{
  Tau_track_load_here();
} 

void tau_disable_tracking_load__(void)
{
  Tau_disable_tracking_load();
} 

void tau_enable_tracking_load__(void)
{
  Tau_enable_tracking_load();
} 

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////


void TAU_SET_INTERRUPT_INTERVAL(int* value)
{
  Tau_set_interrupt_interval(*value);
} 

void tau_profile_start(int **profiler)
{
  Tau_lite_start_timer((void *)*profiler, 0);
}

void tau_profile_stop(int **profiler)
{
  Tau_lite_stop_timer((void *)*profiler);
}

void tau_profile_init(void)
{
#ifdef HP_FORTRAN
  _main();
#endif /* HP_FORTRAN */

#ifndef TAU_MPI
#ifndef TAU_SHMEM
  Tau_set_node(0); 
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */
}

void tau_profile_set_node(int *node)
{
  Tau_set_node(*node);
}

void tau_profile_exit(char *msg)
{
  Tau_exit(msg);
}

void tau_db_dump_incr(void)
{
  Tau_dump_incr();
}

void tau_db_dump(void)
{
  Tau_dump();
}

void tau_db_purge(void)
{
  Tau_purge();
}

void tau_db_dump_prefix(char *prefix)
{
  Tau_dump_prefix(prefix);
}

void tau_profile_set_context(int *context)
{
  Tau_set_context(*context);
}

void tau_trace_sendmessage(int *type, int *destination, int *length)
/* FOR IBM use TAU_TRACE_SENDMESSAGE instead of TAU_TRACE_SENDMSG in Fortran*/
{ 
  Tau_trace_sendmsg(*type, *destination, *length);
}

void tau_trace_recvmessage(int *type, int *source, int *length)
/* FOR IBM use TAU_TRACE_RECVMESSAGE instead of TAU_TRACE_RECVMSG in Fortran*/
{
  Tau_trace_recvmsg(*type, *source, *length);
}

void tau_register_event(int **ptr, char *event_name, int flen)
{
  if (!*ptr)
  {
    char const * localname;
    int locallen;
    getFortranName(&localname, &locallen, event_name, flen);
    *ptr = (int *)Tau_get_userevent(localname);
    free((void*)localname);
  }
}

void tau_register_context_event(void **ptr, char *event_name, int flen)
{
  if (!*ptr) {
    tau_register_context_event_(ptr, event_name, flen);
  }
}


void tau_event(int **ptr, double *data)
{
  Tau_userevent((void *)*ptr, *data);
}

void tau_context_event(int **ptr, double *data)
{
  Tau_context_userevent((void *)*ptr, *data);
}


#if (defined (TAU_GNU) || defined (TAU_PATHSCALE) || defined (TAU_OPEN64ORC))
void tau_profile_timer__(void **ptr, char *fname, int flen)
{
  tau_profile_timer_(ptr, fname, flen);
}

void tau_profile_start__(void **profiler)
{
  tau_profile_start_(profiler);
}

void tau_profile_stop__(void **profiler)
{
  tau_profile_stop_(profiler);
}

void tau_profile_exit__(char *msg)
{
  tau_profile_exit_(msg);
}

void tau_db_dump__(void)
{
  Tau_dump();
}

void tau_db_purge__(void)
{
  Tau_purge();
}

void tau_db_dump_prefix__(char *prefix)
{
  Tau_dump_prefix(prefix);
}

void tau_profile_init__()
{
  //_main();
  // tau_profile_init_(argc, argv);
#ifndef TAU_MPI
#ifndef TAU_SHMEM
  Tau_set_node(0); 
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */
}

void tau_profile_set_node__(int *node)
{
  tau_profile_set_node_(node);
}

void tau_profile_set_context__(int *context)
{
  tau_profile_set_context_(context);
}

void tau_trace_sendmsg__(int *type, int *destination, int *length)
{
  Tau_trace_sendmsg(*type, *destination, *length);
}

void tau_trace_recvmsg__(int *type, int *source, int *length)
{
  Tau_trace_recvmsg(*type, *source, *length);
}

void tau_register_event__(void **ptr, char *event_name, int flen)
{
  if (*ptr == 0) {
    char const * localname;
    int locallen;
    getFortranName(&localname, &locallen, event_name, flen);
    *ptr = Tau_get_userevent(localname);
    free((void*)localname);
  }
}

void tau_register_context_event__(void **ptr, char *event_name, int flen)
{
  if (*ptr == 0) {
    tau_register_context_event_(ptr, event_name, flen);
  }
}

void tau_event__(void **ptr, double *data)
{
  Tau_userevent(*ptr, *data);
}

void tau_context_event__(int **ptr, double *data)
{
  Tau_context_userevent((void *)*ptr, *data);
}

void tau_report_statistics__(void)
{
  Tau_report_statistics();
}

void tau_report_thread_statistics__(void)
{
  Tau_report_thread_statistics();
}

void tau_enable_group__(TauGroup_t *group)
{
  Tau_enable_group(*group);
}

void tau_disable_group__(TauGroup_t *group)
{
  Tau_disable_group(*group);
}

void tau_enable_all_groups__(void)
{
  Tau_enable_all_groups();
}

void tau_disable_all_groups__(void)
{
  Tau_disable_all_groups();
}

void tau_enable_instrumentation__(void)
{
  Tau_enable_instrumentation();
}

void tau_disable_instrumentation__(void)
{
  Tau_disable_instrumentation();
}

void tau_enable_group_name__(char * group_name, int len)
{
  tau_enable_group_name_local(group_name, len);
}

void tau_disable_group_name__(char * group_name, int len)
{
  tau_disable_group_name_local(group_name, len);
}

//////////////////////////////////////////////////////
// MEMORY API
//////////////////////////////////////////////////////
void tau_track_memory__(void)
{
  Tau_track_memory();
} 

void tau_track_memory_here__(void)
{
  Tau_track_memory_here();
} 

void tau_track_memory_headroom__(void)
{
  Tau_track_memory_headroom();
} 

void tau_track_memory_headroom_here__(void)
{
  Tau_track_memory_headroom_here();
} 

void tau_track_memory_footprint__(void)
{
  Tau_track_memory_rss_and_hwm();
} 

void tau_track_memory_footprint_here__(void)
{
  Tau_track_memory_rss_and_hwm_here();
} 

void tau_enable_tracking_memory__(void)
{
  Tau_enable_tracking_memory();
} 

void tau_disable_tracking_memory__(void)
{
  Tau_disable_tracking_memory();
} 

void tau_enable_tracking_memory_headroom__(void)
{
  Tau_enable_tracking_memory_headroom();
} 

void tau_disable_tracking_memory_headroom__(void)
{
  Tau_disable_tracking_memory_headroom();
} 

void tau_memdbg_protect_above__(int* value)
{
  TauEnv_set_memdbg_protect_above(*value);
}
void tau_memdbg_protect_below__(int* value)
{
  TauEnv_set_memdbg_protect_below(*value);
}
void tau_memdbg_protect_free__(int* value)
{
  TauEnv_set_memdbg_protect_free(*value);
}


void tau_set_interrupt_interval__(int* value)
{
  Tau_set_interrupt_interval(*value);
} 

void tau_phase_create_static__(void **ptr, char *infname, int slen)
{
  tau_phase_create_static_(ptr, infname, slen);
}

void tau_phase_create_dynamic__(void **ptr, char *infname, int slen)
{
  tau_phase_create_dynamic_(ptr, infname, slen);
}

void tau_profile_timer_dynamic__(void **ptr, char *infname, int slen)
{
  tau_profile_timer_dynamic_(ptr, infname, slen);
}

void tau_phase_start__(void **profiler)
{
  tau_phase_start_(profiler);
}

void tau_phase_stop__(void **profiler)
{
  tau_phase_stop_(profiler);
}

#endif /* TAU_GNU || TAU_PATHSCALE */
//////////////////////////////////////////////////////
// PHASE BASED PROFILING
//////////////////////////////////////////////////////

void tau_phase_create_static(void **ptr, char *infname, int slen)
{
  tau_phase_create_static_(ptr, infname, slen);
}

void tau_phase_create_dynamic(void **ptr, char *infname, int slen)
{
  tau_phase_create_dynamic_(ptr, infname, slen);
}

void tau_profile_timer_dynamic(void **ptr, char *infname, int slen)
{
  tau_profile_timer_dynamic_(ptr, infname, slen);
}

void tau_phase_start(void **profiler)
{
  tau_phase_start_(profiler);
}

void tau_phase_stop(void **profiler)
{
  tau_phase_stop_(profiler);
}

void TAU_PHASE_CREATE_STATIC(void **ptr, char *infname, int slen)
{
  tau_phase_create_static_(ptr, infname, slen);
}

void TAU_PHASE_CREATE_DYNAMIC(void **ptr, char *infname, int slen)
{
  tau_phase_create_dynamic_(ptr, infname, slen);
}

void TAU_PROFILE_TIMER_DYNAMIC(void **ptr, char *infname, int slen)
{
  tau_profile_timer_dynamic_(ptr, infname, slen);
}

void TAU_PHASE_START(void **profiler)
{
  tau_phase_start_(profiler);
}

void TAU_PHASE_STOP(void **profiler)
{
  tau_phase_stop_(profiler);
}

//////////////////////////////////////////////////////////////////////
// Snapshot related routines
//////////////////////////////////////////////////////////////////////

void tau_profile_snapshot_1l_(char *name, int *number, int slen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, name, slen);
  Tau_profile_snapshot_1l(localname, *number);
  free((void*)localname);
}

void tau_profile_snapshot_1l(char *name, int *number, int slen) {
  tau_profile_snapshot_1l_(name, number, slen);
}

void tau_profile_snapshot_1l__(char *name, int *number, int slen) {
  tau_profile_snapshot_1l_(name, number, slen);
}

void TAU_PROFILE_SNAPSHOT_1L(char *name, int *number, int slen) {
  tau_profile_snapshot_1l_(name, number, slen);
}

void tau_profile_snapshot_(char *name, int slen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, name, slen);
  Tau_profile_snapshot(localname);
  free((void*)localname);
}

void tau_profile_snapshot(char *name, int slen) {
  tau_profile_snapshot_(name, slen);
}

void tau_profile_snapshot__(char *name, int slen) {
  tau_profile_snapshot_(name, slen);
}

void TAU_PROFILE_SNAPSHOT(char *name, int slen) {
  tau_profile_snapshot_(name, slen);
}

void tau_online_dump_() {
  Tau_mon_onlineDump();
}

void tau_online_dump() {
  Tau_mon_onlineDump();
}

void tau_online_dump__() {
  Tau_mon_onlineDump();
}

void TAU_ONLINE_DUMP() {
  Tau_mon_onlineDump();
}

//////////////////////////////////////////////////////////////////////
// Parameter Profiling
//////////////////////////////////////////////////////////////////////
void tau_profile_param_1l_(char *name, int *number, int slen) {
  char const * localname;
  int locallen;
  getFortranName(&localname, &locallen, name, slen);
  Tau_profile_param1l(*number, localname);
  free((void*)localname);
}

void tau_profile_param_1l(char *name, int *number, int slen) {
  tau_profile_param_1l_(name, number, slen);
}

void tau_profile_param_1l__(char *name, int *number, int slen) {
  tau_profile_param_1l_(name, number, slen);
}

void TAU_PROFILE_PARAM_1L(char *name, int *number, int slen) {
  tau_profile_param_1l_(name, number, slen);
}

void Tau_profile_param1l(long data, const char *dataname);


//////////////////////////////////////////////////////////////////////
// Metadata routines
//////////////////////////////////////////////////////////////////////
void tau_metadata_(char *name, char *value, int nlen, int vlen) {
  char const * fname;
  int fnlen;
  getFortranName(&fname, &fnlen, name, nlen);

  char const * fvalue;
  int fvlen;
  getFortranName(&fvalue, &fvlen, value, vlen);

  Tau_metadata(fname, fvalue);

  free((void*)fname);
  free((void*)fvalue);
}

void tau_metadata(char *name, char *value, int nlen, int vlen) {
  tau_metadata_(name, value, nlen, vlen);
}

void tau_metadata__(char *name, char *value, int nlen, int vlen) {
  tau_metadata_(name, value, nlen, vlen);
}

void TAU_METADATA(char *name, char *value, int nlen, int vlen) {
  tau_metadata_(name, value, nlen, vlen);
}

void tau_alloc_(void ** ptr, int* line, int *size, char *name, int slen)
{
  if (ptr) {
    char const * localname;
    int locallen;
    getFortranName(&localname, &locallen, name, slen);
    if (!Tau_memory_wrapper_is_registered()) {
      Tau_track_memory_allocation((void*)ptr, *size, localname, *line);
    }
    free((void*)localname);
  }
}

void tau_alloc(void ** ptr, int* line, int *size, char *name, int slen)
{
  tau_alloc_(ptr, line, size, name, slen);
}

void tau_alloc__(void ** ptr, int* line, int *size, char *name, int slen)
{
  tau_alloc_(ptr, line, size, name, slen);
}

void TAU_ALLOC(void ** ptr, int* line, int *size, char *name, int slen)
{
  tau_alloc_(ptr, line, size, name, slen);
}

void tau_dealloc_(void ** ptr, int* line, char *name, int slen)
{
  if (ptr) {
    char const * localname;
    int locallen;
    getFortranName(&localname, &locallen, name, slen);
    if (!Tau_memory_wrapper_is_registered()) {
      Tau_track_memory_deallocation((void*)ptr, localname, *line);
    }
    free((void*)localname);
  }
}

void tau_dealloc(void ** ptr, int* line, char *name, int slen)
{
  tau_dealloc_(ptr, line, name, slen);
}

void tau_dealloc__(void ** ptr, int* line, char *name, int slen)
{
  tau_dealloc_(ptr, line, name, slen);
}

void TAU_DEALLOC(void ** ptr, int* line, char *name, int slen)
{
  tau_dealloc_(ptr, line, name, slen);
}

} /* extern "C" */


/* Empty MUSE events for compatibility */
void tau_track_muse_events(void) {}
void tau_enable_tracking_muse_events(void) {}
void tau_disable_tracking_muse_events(void) {}
void tau_track_muse_events__(void) {}
void tau_enable_tracking_muse_events__(void) {}
void tau_disable_tracking_muse_events__(void) {}
void tau_track_muse_events_(void) {}
void tau_enable_tracking_muse_events_(void) {}
void tau_disable_tracking_muse_events_(void) {}
void TAU_TRACK_MUSE_EVENTS(void) {}
void TAU_ENABLE_TRACKING_MUSE_EVENTS(void) {}
void TAU_DISABLE_TRACKING_MUSE_EVENTS(void) {}



/***************************************************************************
 * $RCSfile: TauFAPI.cpp,v $   $Author: cheelee $
 * $Revision: 1.84 $   $Date: 2010/06/08 01:09:53 $
 * POOMA_VERSION_ID: $Id: TauFAPI.cpp,v 1.84 2010/06/08 01:09:53 cheelee Exp $ 
 ***************************************************************************/
