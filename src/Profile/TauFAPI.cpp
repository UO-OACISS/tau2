/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
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
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

/* Fortran Wrapper layer for TAU Portable Profiling */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "Profile/ProfileGroups.h"

/* 
#define DEBUG_PROF
*/
#define VALID_NAME_CHAR(x) (isprint(x))

extern "C" {
void * Tau_get_profiler(char *, char *, TauGroup_t, char *gr_name);
void Tau_start_timer(void *);
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
void Tau_userevent(void *ue, double data);
void Tau_report_statistics(void);
void Tau_report_thread_statistics(void);
void Tau_dump(void);
void tau_extract_groupinfo(char *& fname, TauGroup_t & gr, char *& gr_name);
TauGroup_t Tau_get_profile_group(char * group) ; 
TauGroup_t Tau_enable_all_groups(void);
TauGroup_t Tau_disable_all_groups(void);
TauGroup_t Tau_enable_group_name(char *group_name);
TauGroup_t Tau_disable_group_name(char *group_name);

#define EXTRACT_GROUP(n, l, gr, gr_name) TauGroup_t gr; char *gr_name = (char *) malloc(l+1); tau_extract_groupinfo(n, gr, gr_name); 

/*****************************************************************************
* The following routines are called by the Fortran program and they in turn
* invoke the corresponding C routines. 
*****************************************************************************/

void tau_profile_timer_group_(void **ptr, char *infname, int *group, int slen)
{

  char * fname = (char *) malloc((size_t) slen+1);
  strncpy(fname, infname, slen);
  fname[slen] = '\0';  
  
#ifdef DEBUG_PROF
  printf("Inside tau_profile_timer_group_ fname=%s\n", fname);
#endif /* DEBUG_PROF */
  
  if (*ptr == 0) 
  {
    *ptr = Tau_get_profiler(fname, (char *)" ", *group, fname);
  }

#ifdef DEBUG_PROF 
  printf("get_profiler returns %x\n", *ptr);
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


void tau_profile_timer_(void **ptr, char *infname, int slen)
{
  
  char * fname = (char *) malloc((size_t)slen+1);
  strncpy(fname, infname, slen);
  fname[slen] = '\0';

#ifdef DEBUG_PROF
  printf("Inside tau_profile_timer_ fname=%s\n", fname);
#endif /* DEBUG_PROF */
  if (*ptr == 0) 
  {  // remove garbage characters from the end of name
    
    EXTRACT_GROUP(fname, slen, gr, gr_name)
    *ptr = Tau_get_profiler(fname, (char *)" ", gr, gr_name);
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

  Tau_start_timer(*profiler);
  return;
}

void tau_profile_stop_(void **profiler)
{
  Tau_stop_timer(*profiler);
  return;
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

void tau_profile_init_()
{
#ifndef TAU_MPI
  Tau_set_node(0); 
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

void tau_register_event_(void **ptr, char *event_name, int *flen)
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
    *ptr = Tau_get_userevent(event_name);
  }
  return;

}

void tau_event_(void **ptr, double *data)
{
  Tau_userevent(*ptr, *data);
}

void tau_report_statistics_(void)
{
  Tau_report_statistics();
}

void tau_report_thread_statistics_(void)
{
  Tau_report_thread_statistics();
}

/* Cray F90 specific extensions */
#if (defined(CRAYKAI) || defined(HP_FORTRAN))
void _main();
#endif /* CRAYKAI || HP_FORTRAN */
void TAU_PROFILE_TIMER(void **ptr, char *fname, int flen)
{

#ifdef DEBUG_PROF
  printf("flen = %d\n", flen);
#endif /* DEBUG_PROF */
 

  if (*ptr == 0)
  {  // remove garbage characters from the end of name
    if (flen < 1024) fname[flen] = '\0';
    else
    {
      for(int i=0; i<1024; i++)
      {
        if (!VALID_NAME_CHAR(fname[i]))
        {
          fname[i] = '\0';
          break;
        }
      }
    }

#ifdef DEBUG_PROF
    printf("Tau_get_profiler() \n");
#endif /* DEBUG_PROF */
    EXTRACT_GROUP(fname, flen, gr, gr_name)
    *ptr = Tau_get_profiler(fname, (char *)" ", gr, gr_name);
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

void TAU_DB_DUMP(void)
{
  Tau_dump();
  return;
}

void TAU_PROFILE_INIT()
{
#ifdef CRAYKAI 
  _main();
#endif /* CRAYKAI */
  // tau_profile_init_(argc, argv);
#ifndef TAU_MPI
  Tau_set_node(0); 
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

void TAU_EVENT(void **ptr, double *data)
{
  Tau_userevent(*ptr, *data);
}

void TAU_REPORT_STATISTICS(void)
{
  Tau_report_statistics();
}

void TAU_REPORT_THREAD_STATISTICS(void)
{
  Tau_report_thread_statistics();
}

#if (defined (TAU_XLC) || defined(TAU_AIX) || defined(HP_FORTRAN))
void tau_profile_timer(int **profiler, char *fname, int len)
{
  if (*profiler == 0)
  {
    // remove garbage characters from the end of name
    if (len < 1024) {
#ifndef HP_FORTRAN
	fname[len] = '\0'; 
#endif /* HP_FORTRAN */
    }
    else 	
    {
      for(int i=0; i<1024; i++)
      {
        if (!VALID_NAME_CHAR(fname[i]))
        {
          fname[i] = '\0';
          break;
        }
      }
    }
#ifdef DEBUG_PROF
    printf("len = %d\n", len);
    printf("Tau_get_profiler() \n");
#endif /* DEBUG_PROF */
#ifdef HP_FORTRAN
    char *name = (char *) malloc(len+1); 
    strncpy(name, fname, len);
    name[len]='\0';
    EXTRACT_GROUP(name, len, gr, gr_name)
    *profiler = (int *) Tau_get_profiler(name, (char *)" ", gr, gr_name);
#else 
    EXTRACT_GROUP(fname, len, gr, gr_name)
    *profiler = (int *) Tau_get_profiler(fname, (char *)" ", gr, gr_name);
#endif /* HP_FORTRAN */
  }
}

void tau_profile_start(int **profiler)
{
  Tau_start_timer((void *)*profiler);
}

void tau_profile_stop(int **profiler)
{
  Tau_stop_timer((void *)*profiler);
}

void tau_profile_init()
{
#ifdef HP_FORTRAN
  _main();
#endif /* HP_FORTRAN */

#ifndef TAU_MPI
  Tau_set_node(0); 
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

void tau_event(int **ptr, double *data)
{
  Tau_userevent((void *)*ptr, *data);
}

#endif /* TAU_XLC || TAU_AIX || HP_FORTRAN */


#ifdef TAU_GNU

void tau_profile_timer__(void **ptr, char *fname, int flen)
{
  if (*ptr == 0) 
  {  // remove garbage characters from the end of name
    for(int i=0; i<strlen(fname); i++)
    {
      if (!VALID_NAME_CHAR(fname[i]))
      { 
        fname[i] = '\0';
        break;
      }
    }

#ifdef DEBUG_PROF
    printf("Tau_get_profiler() \n");
#endif /* DEBUG_PROF */
    EXTRACT_GROUP(fname, flen, gr, gr_name)
    *ptr = Tau_get_profiler(fname, " ", gr, gr_name);
  }

#ifdef DEBUG_PROF 
  printf("get_profiler returns %x\n", *ptr);
#endif /* DEBUG_PROF */

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

void tau_profile_init__()
{
  //_main();
  // tau_profile_init_(argc, argv);
#ifndef TAU_MPI
  Tau_set_node(0); 
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

void tau_register_event__(void **ptr, char *event_name, int *flen)
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
    printf("tau_get_userevent() \n");
#endif /* DEBUG_PROF */
    *ptr = Tau_get_userevent(event_name);
  }
  return;

}

void tau_event__(void **ptr, double *data)
{
  Tau_userevent(*ptr, *data);
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


#endif /* TAU_GNU */
} /* extern "C" */

/***************************************************************************
 * $RCSfile: TauFAPI.cpp,v $   $Author: sameer $
 * $Revision: 1.26 $   $Date: 2002/03/27 22:18:10 $
 * POOMA_VERSION_ID: $Id: TauFAPI.cpp,v 1.26 2002/03/27 22:18:10 sameer Exp $ 
 ***************************************************************************/
