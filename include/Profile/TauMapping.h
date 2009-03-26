/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1999-2009					   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: Profiler.cpp					   **
**	Description 	: TAU Mappings for relating profile data from one  **
**			  layer to another				   **
**	Author		: Sameer Shende					   **
**	Contact		: tau-bugs@cs.uoregon.edu                	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
****************************************************************************/

#ifndef _TAU_MAPPING_H_
#define _TAU_MAPPING_H_

#ifdef TAU_ENABLED

#include <unistd.h> // for exit() call

// For Mapping, global variables used between layers
void* &TheTauMapFI(TauGroup_t key=TAU_DEFAULT);

#define TAU_MAPPING(stmt, key) \
  { \
    static void *TauMapFI = NULL; \
    tauCreateFI(&TauMapFI, #stmt, " " , key, #key); \
    TheTauMapFI(key) = TauMapFI; \
    Tau_start_timer(TauMapFI, 0, Tau_get_tid()); \
    stmt; \
    Tau_stop_timer(TauMapFI, Tau_get_tid()); \
  } 

#define TAU_MAPPING_REGISTER(stmt, key)  { \
    static void *TauMapFI = NULL; \
    tauCreateFI(&TauMapFI,stmt, " " , key, #key); \
    TheTauMapFI(key) = TauMapFI; \
  } 

#define TAU_MAPPING_CREATE(name, type, key, groupname, tid)  { \
    void *TauMapFI = NULL; \
    tauCreateFI(&TauMapFI, name, type, key, groupname); \
    if (TauMapFI == 0) { \
	printf("ERROR: new returned NULL"); exit(1); \
    } \
    TheTauMapFI(key) = TauMapFI; \
  } 

#define TAU_MAPPING_CREATE1(name, type, key, groupid, groupname, tid)  { \
    void *TauMapFI = NULL; \
    tauCreateFI(&TauMapFI, name, type, groupid, groupname); \
    if (TauMapFI == 0) { \
	printf("ERROR: new returned NULL"); exit(1); \
    } \
    TheTauMapFI(key) = TauMapFI; \
  } 

/* TAU_MAPPING_TIMER_CREATE creates a functionInfo pointer with a specified 
   group name. */
#define TAU_MAPPING_TIMER_CREATE(t, name, type, gr, groupname) \
  tauCreateFI(&t, name, type, gr, groupname);

/* TAU_MAPPING_OBJECT creates a functionInfo pointer that may be stored in the 
   object that is used to relate a lower level layer with a higher level layer */
#define TAU_MAPPING_OBJECT(timer) void *timer;

/* TAU_MAPPING_LINK gets in a var the function info object associated with the given key (Group) */
#define TAU_MAPPING_LINK(timer, key) timer = TheTauMapFI(key); 

/* TAU_MAPPING_PROFILE profiles the entire routine by creating a profiler objeca
   and this behaves pretty much like TAU_PROFILE macro, except this gives in the
   FunctionInfo object pointer instead of name and type strings. */
#define TAU_MAPPING_PROFILE(timer) Tau_Profile_Wrapper timer##tauFP(timer);

/* TAU_MAPPING_PROFILE_TIMER acts like TAU_PROFILE_TIMER by creating a profiler
   object that can be subsequently used with TAU_PROFILE_START and TAU_PROFILE_STOP */
#define TAU_MAPPING_PROFILE_TIMER(timer, FuncInfoVar, tid) void *timer = FuncInfoVar;
   
/* TAU_MAPPING_PROFILE_START acts like TAU_PROFILE_START by starting the timer */
#define TAU_MAPPING_PROFILE_START(timer, tid) Tau_start_timer(timer, 0, Tau_get_tid());

/* TAU_MAPPING_PROFILE_STOP acts like TAU_PROFILE_STOP by stopping the timer */
#define TAU_MAPPING_PROFILE_STOP(tid) TAU_GLOBAL_TIMER_STOP();



/* These are all worthless macros which have non "MAPPING" versions, they should
   never have been created, but are left here for "legacy" purposes */
#define TAU_MAPPING_PROFILE_EXIT(msg, tid)  TAU_PROFILE_EXIT(msg); 
#define TAU_MAPPING_DB_DUMP(tid)  TAU_DB_DUMP(); 
#define TAU_MAPPING_DB_PURGE(tid)  TAU_DB_PURGE(); 
#define TAU_MAPPING_PROFILE_SET_NODE(node, tid)  TAU_PROFILE_SET_NODE(node); 
#define TAU_MAPPING_PROFILE_SET_NAME(timer, name) TAU_PROFILE_TIMER_SET_NAME(timer,name);
#define TAU_MAPPING_PROFILE_SET_TYPE(timer, name) TAU_PROFILE_TIMER_SET_TYPE(timer,name);
#define TAU_MAPPING_PROFILE_SET_GROUP(timer, id) TAU_PROFILE_TIMER_SET_GROUP(timer,id); 
#define TAU_MAPPING_PROFILE_SET_GROUP_NAME(timer, name) TAU_PROFILE_TIMER_SET_GROUP_NAME(timer,name);
#define TAU_MAPPING_PROFILE_GET_NAME(timer) TAU_PROFILE_TIMER_GET_NAME(timer)
#define TAU_MAPPING_PROFILE_GET_TYPE(timer) TAU_PROFILE_TIMER_GET_TYPE(timer)
#define TAU_MAPPING_PROFILE_GET_GROUP(timer) TAU_PROFILE_TIMER_GET_GROUP(timer)
#define TAU_MAPPING_PROFILE_GET_GROUP_NAME(timer) TAU_PROFILE_TIMER_GET_GROUP_NAME(timer)



#else
/* Create null , except the main statement which should be executed as it is*/

#endif /* TAU_ENABLED */
#endif /* _TAU_MAPPING_H_ */
