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
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

/* Fortran Wrapper layer for TAU Portable Profiling */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "Profile/ProfileGroups.h"
#include "Profile/TauMemory.h"




extern "C" void Tau_pure_start(char *name);
extern "C" void Tau_pure_stop(char *name);

/* 
#define DEBUG_PROF
*/
#define VALID_NAME_CHAR(x) (isprint(x))

extern "C" {
void * Tau_get_profiler(char *, char *, TauGroup_t, char *gr_name);
void Tau_start_timer(void * timer, int phase);
void Tau_stop_timer(void *);
void Tau_exit(char *);
void Tau_init(int, char **);
void Tau_enable_group(TauGroup_t group);
void Tau_disable_group(TauGroup_t group);
void Tau_set_node(int);
void Tau_set_context(int);
void Tau_register_thread(void);
void Tau_enable_instrumentation(void);
void Tau_disable_instrumentation(void);
void Tau_trace_sendmsg(int type, int destination, int length);
void Tau_trace_recvmsg(int type, int source, int length);
void * Tau_get_userevent(char *name);
void * Tau_get_context_userevent(char *name);
void Tau_userevent(void *ue, double data);
void Tau_context_userevent(void *ue, double data);
void Tau_report_statistics(void);
void Tau_report_thread_statistics(void);
void Tau_dump(void);
void Tau_dump_prefix(char *prefix);
void tau_extract_groupinfo(char *& fname, TauGroup_t & gr, char *& gr_name);
TauGroup_t Tau_get_profile_group(char * group) ; 
TauGroup_t Tau_enable_all_groups(void);
TauGroup_t Tau_disable_all_groups(void);
TauGroup_t Tau_enable_group_name(char *group_name);
TauGroup_t Tau_disable_group_name(char *group_name);
void Tau_track_memory(void);
void Tau_track_memory_here(void);
void Tau_track_muse_events(void);
void Tau_enable_tracking_memory(void);
void Tau_disable_tracking_memory(void);
void Tau_enable_tracking_muse_events(void);
void Tau_disable_tracking_muse_events(void);
void Tau_set_interrupt_interval(int value);
void Tau_track_memory_headroom(void);
void Tau_track_memory_headroom_here(void); 
void Tau_enable_tracking_memory_headroom(void);
void Tau_disable_tracking_memory_headroom(void);
void Tau_mark_group_as_phase(void **ptr);
void Tau_profile_callstack(void );
void Tau_profile_snapshot(char *name);
void Tau_profile_snapshot_1l(char *name, int number);
void Tau_metadata(char *name, char *value);
void Tau_static_phase_start(char *name);
void Tau_static_phase_stop(char *name);
void Tau_dynamic_start(char *name, int iteration, int isPhase);
void Tau_dynamic_stop(char *name, int iteration, int isPhase);
char * Tau_append_iteration_to_name(int iteration, char *name);
  


#define EXTRACT_GROUP(n, l, gr, gr_name) TauGroup_t gr; char *gr_name = NULL; tau_extract_groupinfo(n, gr, gr_name); 

/*****************************************************************************
* The following routines are called by the Fortran program and they in turn
* invoke the corresponding C routines. 
*****************************************************************************/

void tau_profile_timer_group_(void **ptr, char *infname, int *group, int slen)
{

  if (*ptr == 0) {
    char * fname = (char *) malloc((size_t) slen+1);
    strncpy(fname, infname, slen);
    fname[slen] = '\0';  
    
#ifdef DEBUG_PROF
    printf("Inside tau_profile_timer_group_ fname=%s\n", fname);
#endif /* DEBUG_PROF */
    
    *ptr = Tau_get_profiler(fname, (char *)" ", *group, fname);
  }
  
#ifdef DEBUG_PROF 
  printf("get_profiler returns %lx\n", *ptr);
#endif /* DEBUG_PROF */

  return;
}

void tau_enable_group_name_local(char *& group_name, int len)
{
  char *name = (char *) malloc(len+1);
  strncpy(name, group_name, len);
  name[len]='\0';

  Tau_enable_group_name(name);

}

void tau_disable_group_name_local(char *& group_name, int len)
{
  char *name = (char *) malloc(len+1);
  strncpy(name, group_name, len);
  name[len]='\0';

  Tau_disable_group_name(name);

}

void tau_extract_groupinfo(char *& fname, TauGroup_t & gr, char *& gr_name)
{
  /* See if a > appears in the function name. If it does, it separates the
     group name from the function name. Separate these two */
  char *first, *second;
  first = strtok(fname, ">"); 
  if (first != 0)
  {
    second = strtok(NULL,  ">");
    if (second == NULL) 
    {
      fname = first; 
      gr = TAU_USER;
      gr_name = fname; 
    }
    else
    {
      gr = Tau_get_profile_group(first); 
      gr_name = first;
      fname = second;
    }
  }
  

}


void tau_pure_start(char *fname, int flen) {
  // make a copy so that we can null terminate it
  char *localname = (char *) malloc((size_t)flen+1);
  strncpy(localname, fname, flen);
  localname[flen] = '\0';
  
  // check for unprintable characters
  for(int i=0; i<strlen(localname); i++) {
    if (!VALID_NAME_CHAR(localname[i])) { 
      localname[i] = '\0';
      break;
    }
  }

  Tau_pure_start(localname);
  free(localname);
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
  // make a copy so that we can null terminate it
  char *localname = (char *) malloc((size_t)flen+1);
  strncpy(localname, fname, flen);
  localname[flen] = '\0';
  
  // check for unprintable characters
  for(int i=0; i<strlen(localname); i++) {
    if (!VALID_NAME_CHAR(localname[i])) { 
      localname[i] = '\0';
      break;
    }
  }

  Tau_pure_stop(localname);
  free(localname);
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
  // make a copy so that we can null terminate it
  char *localname = (char *) malloc((size_t)flen+1);
  strncpy(localname, fname, flen);
  localname[flen] = '\0';

  // check for unprintable characters
  for(int i=0; i<strlen(localname); i++) {
    if (!VALID_NAME_CHAR(localname[i])) {
      localname[i] = '\0';
      break;
    }
  }

#ifdef DEBUG_PROF
  printf("tau_static_phase_start: %s\n", localname);
#endif /* DEBUG_PROF */
  Tau_static_phase_start(localname);
  free(localname);
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
  // make a copy so that we can null terminate it
  char *localname = (char *) malloc((size_t)flen+1);
  strncpy(localname, fname, flen);
  localname[flen] = '\0';

  // check for unprintable characters
  for(int i=0; i<strlen(localname); i++) {
    if (!VALID_NAME_CHAR(localname[i])) {
      localname[i] = '\0';
      break;
    }
  }

  printf("tau_static_phase_stop: %s\n", localname);
  Tau_static_phase_stop(localname);
  free(localname);
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

void tau_dynamic_phase_start(int *iteration, char *fname, int flen) {
  // make a copy so that we can null terminate it
  char *localname = (char *) malloc((size_t)flen+1);
  strncpy(localname, fname, flen);
  localname[flen] = '\0';

  // check for unprintable characters
  for(int i=0; i<strlen(localname); i++) {
    if (!VALID_NAME_CHAR(localname[i])) {
      localname[i] = '\0';
      break;
    }
  }

  Tau_dynamic_start(localname, *iteration, 1); /* 1 is isPhase */
  free(localname);
}

void tau_dynamic_phase_start_(int *iteration, char *fname, int flen) {
  tau_dynamic_phase_start(iteration, fname, flen);
}

void tau_dynamic_phase_start__(int *iteration, char *fname, int flen) {
  tau_dynamic_phase_start(iteration, fname, flen);
}

void TAU_DYNAMIC_PHASE_START(int *iteration, char *fname, int flen) {
  tau_dynamic_phase_start(iteration, fname, flen);
}

void tau_dynamic_phase_stop(int *iteration, char *fname, int flen) {
  // make a copy so that we can null terminate it
  char *localname = (char *) malloc((size_t)flen+1);
  strncpy(localname, fname, flen);
  localname[flen] = '\0';

  // check for unprintable characters
  for(int i=0; i<strlen(localname); i++) {
    if (!VALID_NAME_CHAR(localname[i])) {
      localname[i] = '\0';
      break;
    }
  }

  Tau_dynamic_stop(localname, *iteration, 1); /* 1 is isPhase */
  free(localname);
}

void tau_dynamic_phase_stop_(int *iteration, char *fname, int flen) {
  tau_dynamic_phase_stop(iteration, fname, flen);
}
  
void tau_dynamic_phase_stop__(int *iteration, char *fname, int flen) {
  tau_dynamic_phase_stop(iteration, fname, flen);
}

void TAU_DYNAMIC_PHASE_STOP(int *iteration, char *fname, int flen) {
  tau_dynamic_phase_stop(iteration, fname, flen);
}

/* TAU_DYNAMIC_TIMER_START/STOP are similar to TAU_DYNAMIC_PHASE_START/STOP */
void tau_dynamic_timer_start(int *iteration, char *fname, int flen) {
  // make a copy so that we can null terminate it
  char *localname = (char *) malloc((size_t)flen+1);
  strncpy(localname, fname, flen);
  localname[flen] = '\0';

  // check for unprintable characters
  for(int i=0; i<strlen(localname); i++) {
    if (!VALID_NAME_CHAR(localname[i])) {
      localname[i] = '\0';
      break;
    }
  }

  Tau_dynamic_start(localname, *iteration, 0); /* isPhase=0 implies a timer */
  free(localname);
}

void tau_dynamic_timer_start_(int *iteration, char *fname, int flen) {
  tau_dynamic_timer_start(iteration, fname, flen);
}

void tau_dynamic_timer_start__(int *iteration, char *fname, int flen) {
  tau_dynamic_timer_start(iteration, fname, flen);
}

void TAU_DYNAMIC_TIMER_START(int *iteration, char *fname, int flen) {
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
void tau_dynamic_timer_stop(int *iteration, char *fname, int flen) {
  // make a copy so that we can null terminate it
  char *localname = (char *) malloc((size_t)flen+1);
  strncpy(localname, fname, flen);
  localname[flen] = '\0';

  // check for unprintable characters
  for(int i=0; i<strlen(localname); i++) {
    if (!VALID_NAME_CHAR(localname[i])) {
      localname[i] = '\0';
      break;
    }
  }

  Tau_dynamic_stop(localname, *iteration, 0); /* isPhase = 0 implies timer */
  free(localname);
}

void tau_dynamic_timer_stop_(int *iteration, char *fname, int flen) {
  tau_dynamic_timer_stop(iteration, fname, flen);
}

void tau_dynamic_timer_stop__(int *iteration, char *fname, int flen) {
  tau_dynamic_timer_stop(iteration, fname, flen);
}

void TAU_DYNAMIC_TIMER_STOP(int *iteration, char *fname, int flen) {
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

/* C API */
void Tau_start(char *name)
{
  tau_pure_start(name, strlen(name));
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
void Tau_stop(char *name)
{
  tau_pure_stop(name, strlen(name));
}

void tau_profile_timer_(void **ptr, char *fname, int flen)
{
  

#ifdef DEBUG_PROF
  printf("Inside tau_profile_timer_ fname=%s\n", fname);
#endif /* DEBUG_PROF */
  if (*ptr == 0) 
  {

#ifdef TAU_OPENMP
#pragma omp critical
    {
      if (*ptr == 0) {
#endif /* TAU_OPENMP */

    // make a copy so that we can null terminate it
    char *localname = (char *) malloc((size_t)flen+1);
    char *modname = (char *) malloc((size_t)flen+1);
    // hold on to the original pointer to free it since EXTRACT_GROUP
    // might change fname
    char *tmp = localname;
    char *tmp2 = modname;
    int skipwhite = 1;
    int idx = 0;
    strncpy(localname, fname, flen);
    localname[flen] = '\0';

    // check for unprintable characters
    for(int i=0; i<strlen(localname); i++) {
      if (!VALID_NAME_CHAR(localname[i])) { 
        localname[i] = '\0';
        break;
      }
    }

    // fix continuation lines
    for(int j=0; j<strlen(localname); j++) {
      if (localname[j] == '&') {
	skipwhite = 1;
      } else {
	if (skipwhite && localname[j] == ' ') {
	  // nothing, skip over it
	} else {
	  modname[idx++] = localname[j];
	  skipwhite = 0;
	}
      }
    }
    modname[idx] = 0;
    localname = modname;

    EXTRACT_GROUP(localname, flen, gr, gr_name);

    *ptr = Tau_get_profiler(localname, (char *)" ", gr, gr_name);
    free(tmp); 
    free(tmp2);
#ifdef TAU_OPENMP
      }
    }
#endif /* TAU_OPENMP */

  }

#ifdef DEBUG_PROF 
  printf("get_profiler returns %lx\n", *ptr);
#endif /* DEBUG_PROF */

  return;
}

void tau_phase_create_static_(void **ptr, char *infname, int slen)
{
  bool firsttime = false;
  if (*ptr == 0) 
    firsttime = true;
  /* is it in here for the first time? */

  tau_profile_timer_(ptr, infname, slen);

  /* we know the FunctionInfo pointer in ptr. If its here the first time 
     set the group name to be | TAU_PHASE */
  if (firsttime)
    Tau_mark_group_as_phase(ptr);
}

void tau_phase_create_dynamic_(void **ptr, char *infname, int slen)
{
  *ptr = 0;  /* reset it each time so it creates a new timer */
  tau_profile_timer_(ptr, infname, slen);
  Tau_mark_group_as_phase(ptr);
}

void tau_profile_timer_dynamic_(void **ptr, char *infname, int slen)
{ /* This routine is identical to tau_phase_create_dynamic */
  *ptr = 0;  /* reset it each time so it creates a new timer */
  tau_profile_timer_(ptr, infname, slen);
}

void tau_profile_start_(void **profiler)
{ 
#ifdef DEBUG_PROF
  printf("start_timer gets %lx\n", *profiler);
#endif /* DEBUG_PROF */

  Tau_start_timer(*profiler, 0);
  return;
}

void tau_profile_stop_(void **profiler)
{
  Tau_stop_timer(*profiler);
  return;
}

void tau_phase_start_(void **profiler)
{
  Tau_start_timer(*profiler, 1); /* 1 indicates phase based profiling */
  return;
}

void tau_phase_stop_(void **profiler)
{
  Tau_stop_timer(*profiler);
  return;
}

void tau_dynamic_iter(int *iteration, void **ptr, char *infname, int slen, int isPhase)
{
  /* we append the iteration number to the name and then call the 
     appropriate routine for creating dynamic phases or timers */  

  /* This routine creates dynamic timers and phases by embedding the
     iteration number in the name. isPhase argument tells whether we
     choose phases or timers. */

  char *newName = Tau_append_iteration_to_name(*iteration, infname);
  int newLength = strlen(newName);
  if (isPhase) 
    tau_phase_create_dynamic_(ptr, newName, newLength);
  else
    tau_profile_timer_dynamic_(ptr, newName, newLength);

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
  return;
}

void tau_db_dump_(void)
{
  Tau_dump();
  return;
}

void tau_db_dump_prefix_(char *prefix)
{
  Tau_dump_prefix(prefix);
  return;
}

void tau_profile_init_()
{
#ifndef TAU_MPI
#ifndef TAU_SHMEM
  Tau_set_node(0); 
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */
  return;
}

void tau_enable_instrumentation(void)
{
  Tau_enable_instrumentation();
  return;
}

void tau_disable_instrumentation(void)
{
  Tau_disable_instrumentation();
  return;
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
// MEMORY, MUSE events API
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

void tau_track_muse_events(void)
{
  Tau_track_muse_events();
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

void tau_enable_tracking_muse_events(void)
{
  Tau_enable_tracking_muse_events();
} 

void tau_disable_tracking_muse_events(void)
{
  Tau_disable_tracking_muse_events();
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
  return;
} 

void tau_profile_set_context_(int *context)
{
  Tau_set_context(*context);
  return;
}


void tau_enable_instrumentation_(void)
{
  Tau_enable_instrumentation();
  return;
}

void tau_disable_instrumentation_(void)
{
  Tau_disable_instrumentation();
  return;
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
  return;
}

void tau_register_thread__(void)
{
  Tau_register_thread();
  return;
}

void tau_register_thread(void)
{
  Tau_register_thread();
  return;
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
  if (*ptr == 0) 
  {  // remove garbage characters from the end of name
    char *localname = (char *) malloc((size_t)slen+1);
    char *modname = (char *) malloc((size_t)slen+1);
    char *tmp = localname;
    char *tmp2 = modname;
    int skipwhite = 1;
    int idx = 0;
    strncpy(localname, name, slen);
    localname[slen] = '\0';

    // check for unprintable characters
    for(int i=0; i<strlen(localname); i++) {
      if (!VALID_NAME_CHAR(localname[i])) {
        localname[i] = '\0';
        break;
      }
    }

    // fix continuation lines
    for(int j=0; j<strlen(localname); j++) {
      if (localname[j] == '&') {
        skipwhite = 1;
      } else {
        if (skipwhite && localname[j] == ' ') {
          // nothing, skip over it
        } else {
          modname[idx++] = localname[j];
          skipwhite = 0;
        }
      }
    }
    modname[idx] = 0;
    localname = modname;



#ifdef DEBUG_PROF
    printf("Tau_get_userevent(%s) \n", localname);
#endif /* DEBUG_PROF */
    *ptr = Tau_get_userevent(localname);
     free(tmp);
     free(tmp2);
  }
  return;

}

void tau_register_context_event_(void **ptr, char *name, int slen)
{

  if (*ptr == 0) 
  {  // remove garbage characters from the end of name
    char *localname = (char *) malloc((size_t)slen+1);
    char *modname = (char *) malloc((size_t)slen+1);
    char *tmp = localname;
    char *tmp2 = modname;
    int skipwhite = 1;
    int idx = 0;
    strncpy(localname, name, slen);
    localname[slen] = '\0';

    // check for unprintable characters
    for(int i=0; i<strlen(localname); i++) {
      if (!VALID_NAME_CHAR(localname[i])) {
        localname[i] = '\0';
        break;
      }
    }

    // fix continuation lines
    for(int j=0; j<strlen(localname); j++) {
      if (localname[j] == '&') {
        skipwhite = 1;
      } else {
        if (skipwhite && localname[j] == ' ') {
          // nothing, skip over it
        } else {
          modname[idx++] = localname[j];
          skipwhite = 0;
        }
      }
    }
    modname[idx] = 0;
    localname = modname;


#ifdef DEBUG_PROF
    printf("Tau_get_context_userevent(%s) \n", localname);
#endif /* DEBUG_PROF */
    *ptr = Tau_get_context_userevent(localname);
    free(tmp);
    free(tmp2);
  }
  return;

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
// MEMORY, MUSE events API
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

void tau_track_muse_events_(void)
{
  Tau_track_muse_events();
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

void tau_enable_tracking_muse_events_(void)
{
  Tau_enable_tracking_muse_events();
} 

void tau_disable_tracking_muse_events_(void)
{
  Tau_disable_tracking_muse_events();
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
  if (*ptr == 0) {
    tau_profile_timer_(ptr, fname, flen);
  }
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

void TAU_DB_DUMP(void)
{
  Tau_dump();
  return;
}

void TAU_DB_DUMP_PREFIX(char *prefix)
{
  Tau_dump_prefix(prefix);
  return;
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

  if (*ptr == 0) 
  {  // remove garbage characters from the end of name
    if (flen < 1024) event_name[flen] = '\0';
    else
    for(int i=0; i<1024; i++)
    {
      if (!VALID_NAME_CHAR(event_name[i]))
      { 
        event_name[i] = '\0';
        break;
      }
    }
#ifdef DEBUG_PROF
    printf("Tau_get_userevent() \n");
#endif /* DEBUG_PROF */
    *ptr = Tau_get_userevent(event_name);
  }
  return;

}

void TAU_REGISTER_CONTEXT_EVENT(void **ptr, char *event_name, int flen)
{

  if (*ptr == 0) {
    tau_register_context_event_(ptr, event_name, flen);
  }
  return;

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

//#if (defined (TAU_XLC) || defined(TAU_AIX) || defined(HP_FORTRAN))
void tau_profile_timer(void **ptr, char *fname, int flen)
{
  if (*ptr == 0) {
    tau_profile_timer_(ptr, fname, flen);
  }
  return;
}

//////////////////////////////////////////////////////
// MEMORY, MUSE events API
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

void TAU_TRACK_MUSE_EVENTS(void)
{
  Tau_track_muse_events();
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

void TAU_ENABLE_TRACKING_MUSE_EVENTS(void)
{
  Tau_enable_tracking_muse_events();
} 

void TAU_DISABLE_TRACKING_MUSE_EVENTS(void)
{
  Tau_disable_tracking_muse_events();
} 

void TAU_SET_INTERRUPT_INTERVAL(int* value)
{
  Tau_set_interrupt_interval(*value);
} 

void tau_profile_start(int **profiler)
{
  Tau_start_timer((void *)*profiler, 0);
}

void tau_profile_stop(int **profiler)
{
  Tau_stop_timer((void *)*profiler);
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

void tau_db_dump(void)
{
  Tau_dump();
}

void tau_db_dump_prefix(char *prefix)
{
  Tau_dump_prefix(prefix);
  return;
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

  if (*ptr == 0)
  {  // remove garbage characters from the end of name
    for(int i=0; i<1024; i++)
    {
      if (!VALID_NAME_CHAR(event_name[i]))
      {
        event_name[i] = '\0';
        break;
      }
    }
#ifdef DEBUG_PROF
    printf("Tau_get_userevent() \n");
#endif /* DEBUG_PROF */
    *ptr = (int *)Tau_get_userevent(event_name);
  }
  return;

}

void tau_register_context_event(void **ptr, char *event_name, int flen)
{

  if (*ptr == 0) {
    tau_register_context_event_(ptr, event_name, flen);
  }
  return;

}


void tau_event(int **ptr, double *data)
{
  Tau_userevent((void *)*ptr, *data);
}

void tau_context_event(int **ptr, double *data)
{
  Tau_context_userevent((void *)*ptr, *data);
}
//#endif /* TAU_XLC || TAU_AIX || HP_FORTRAN */


#if (defined (TAU_GNU) || defined (TAU_PATHSCALE) || defined (TAU_OPEN64ORC))

void tau_profile_timer__(void **ptr, char *fname, int flen)
{
  if (*ptr == 0) {
    tau_profile_timer_(ptr, fname, flen);
  }
  return;
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
  return;
}

void tau_db_dump_prefix__(char *prefix)
{
  Tau_dump_prefix(prefix);
  return;
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

  if (*ptr == 0) 
  {  // remove garbage characters from the end of name
    char * newname=new char[flen+1] ;
    for (int j =0; j < flen; j++)
      newname[j] = event_name[j];

    newname[flen] = '\0';
    for(int i=0; i<strlen(newname); i++)
    {
      if (!VALID_NAME_CHAR(newname[i]))
      { 
        newname[i] = '\0';
        break;
      }
    }
#ifdef DEBUG_PROF
    printf("tau_get_userevent() \n");
#endif /* DEBUG_PROF */
    *ptr = Tau_get_userevent(newname);
  }
  return;

}

void tau_register_context_event__(void **ptr, char *event_name, int flen)
{

  if (*ptr == 0) {
    tau_register_context_event_(ptr, event_name, flen);
  }
  return;

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
  return;
}

void tau_disable_instrumentation__(void)
{
  Tau_disable_instrumentation();
  return;
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
// MEMORY, MUSE events API
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

void tau_track_muse_events__(void)
{
  Tau_track_muse_events();
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

void tau_enable_tracking_muse_events__(void)
{
  Tau_enable_tracking_muse_events();
} 

void tau_disable_tracking_muse_events__(void)
{
  Tau_disable_tracking_muse_events();
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

void tau_profile_callstack_(void)
{
  Tau_profile_callstack();
}

void tau_profile_callstack(void)
{
  tau_profile_callstack_();
}

void tau_profile_callstack__(void)
{
  tau_profile_callstack_();
}

void TAU_PROFILE_CALLSTACK(void)
{
  tau_profile_callstack_();
}

//////////////////////////////////////////////////////////////////////
// Snapshot related routines
//////////////////////////////////////////////////////////////////////

static char *getFortranName(char *name, int slen) {
    char *fname = (char *) malloc((size_t) slen+1);
    strncpy(fname, name, slen);
    fname[slen] = '\0';  
    return fname;
}

void tau_profile_snapshot_1l_(char *name, int *number, int slen) {
  char *fname = getFortranName(name, slen);
  Tau_profile_snapshot_1l(fname, *number);
  free (fname);
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
  char *fname = getFortranName(name, slen);
  Tau_profile_snapshot(fname);
  free (fname);
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


//////////////////////////////////////////////////////////////////////
// Metadata routines
//////////////////////////////////////////////////////////////////////
void tau_metadata_(char *name, char *value, int nlen, int vlen) {
  char *fname = getFortranName(name, nlen);
  char *fvalue = getFortranName(value, vlen);
  Tau_metadata(fname, fvalue);
  free (fname);
  free (fvalue);
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
    char *localname = (char *) malloc((size_t)slen+1);
    char *modname = (char *) malloc((size_t)slen+1);
    char *tmp = localname;
    char *tmp2 = modname;
    int skipwhite = 1;
    int idx = 0;
    strncpy(localname, name, slen);
    localname[slen] = '\0';

    // check for unprintable characters
    for(int i=0; i<strlen(localname); i++) {
      if (!VALID_NAME_CHAR(localname[i])) {
        localname[i] = '\0';
        break;
      }
    }

    // fix continuation lines
    for(int j=0; j<strlen(localname); j++) {
      if (localname[j] == '&') {
        skipwhite = 1;
      } else {
        if (skipwhite && localname[j] == ' ') {
          // nothing, skip over it
        } else {
          modname[idx++] = localname[j];
          skipwhite = 0;
        }
      }
    }
    modname[idx] = 0;
    localname = modname;

#ifdef DEBUG_PROF
  printf("ALLOCATE ptr %p *ptr %p line %d size %d\n", ptr, *ptr, *line, *size);
#endif /* DEBUG_PROF */
  Tau_track_memory_allocation(localname, *line, *size, ptr);
  free(tmp);
  free(tmp2);
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
    char *localname = (char *) malloc((size_t)slen+1);
    char *modname = (char *) malloc((size_t)slen+1);
    char *tmp = localname;
    char *tmp2 = modname;
    int skipwhite = 1;
    int idx = 0;
    strncpy(localname, name, slen);
    localname[slen] = '\0';

    // check for unprintable characters
    for(int i=0; i<strlen(localname); i++) {
      if (!VALID_NAME_CHAR(localname[i])) {
        localname[i] = '\0';
        break;
      }
    }

    // fix continuation lines
    for(int j=0; j<strlen(localname); j++) {
      if (localname[j] == '&') {
        skipwhite = 1;
      } else {
        if (skipwhite && localname[j] == ' ') {
          // nothing, skip over it
        } else {
          modname[idx++] = localname[j];
          skipwhite = 0;
        }
      }
    }
    modname[idx] = 0;
    localname = modname;

#ifdef DEBUG_PROF
  printf("DEALLOCATE ptr %p *ptr %p line %ld\n", ptr,  *ptr, *line);
#endif /* DEBUG_PROF */
  Tau_track_memory_deallocation(localname, *line, ptr);
  free(tmp);
  free(tmp2);
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


/***************************************************************************
 * $RCSfile: TauFAPI.cpp,v $   $Author: sameer $
 * $Revision: 1.64 $   $Date: 2007/09/16 21:59:38 $
 * POOMA_VERSION_ID: $Id: TauFAPI.cpp,v 1.64 2007/09/16 21:59:38 sameer Exp $ 
 ***************************************************************************/
