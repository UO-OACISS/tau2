/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.cpp					  **
**	Description 	: TAU Mappings for relating profile data from one **
**			  layer to another				  **
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
/* TAU Mappings */
#ifndef _TAU_MAPPING_H_
#define _TAU_MAPPING_H_

#if (PROFILING_ON || TRACING_ON)
// For Mapping, global variables used between layers
FunctionInfo *& TheTauMapFI(TauGroup_t ProfileGroup=TAU_DEFAULT);
#define TAU_MAPPING(stmt, group)   \
  { \
    static FunctionInfo TauMapFI(#stmt, " " , group, #group); \
    static Profiler *TauMapProf = new Profiler(&TauMapFI, group, true); \
    TheTauMapFI(group) = &TauMapFI; \
    TauMapProf->Start(); \
    stmt; \
    TauMapProf->Stop(); \
  } 

#define TAU_MAPPING_REGISTER(stmt, group)  { static FunctionInfo TauMapFI(stmt, " " , group, #group); \
    TheTauMapFI(group) = &TauMapFI; \
  } 

#define TAU_MAPPING_CREATE(name, type, key, groupname, tid)  { FunctionInfo *TauMapFI = new FunctionInfo(name, type, key, groupname, true, tid); \
    if (TauMapFI == (FunctionInfo *) NULL) { \
	printf("ERROR: new returns NULL"); exit(1); \
    } \
    TheTauMapFI(key) = TauMapFI; \
  } 
/* TAU_MAPPING_OBJECT creates a functionInfo pointer that may be stored in the 
   object that is used to relate a lower level layer with a higher level layer 
*/

#define TAU_MAPPING_OBJECT(FuncInfoVar) FunctionInfo * FuncInfoVar;

/* TAU_MAPPING_LINK gets in a var the function info object associated with the 
   given key (Group) 
*/
/*
This error should be reported when FuncInfoVar is NULL
	  //printf("ERROR: TAU_MAPPING_LINK map returns NULL FunctionInfo *\n"); \
*/
#define TAU_MAPPING_LINK(FuncInfoVar, Group) FuncInfoVar = TheTauMapFI(Group); \
	if (FuncInfoVar == (FunctionInfo *)NULL) { \
 	  return; \
        } 

/* TAU_MAPPING_PROFILE profiles the entire routine by creating a profiler objeca
   and this behaves pretty much like TAU_PROFILE macro, except this gives in the
   FunctionInfo object pointer instead of name and type strings. 
*/
#define TAU_MAPPING_PROFILE(FuncInfoVar) Profiler FuncInfoVar##Prof(FuncInfoVar, FuncInfoVar->GetProfileGroup(), false);

/* TAU_MAPPING_PROFILE_TIMER acts like TAU_PROFILE_TIMER by creating a profiler
   object that can be subsequently used with TAU_PROFILE_START and 
   TAU_PROFILE_STOP
*/
#define TAU_MAPPING_PROFILE_TIMER(Timer, FuncInfoVar, tid) Profiler *Timer; \
   Timer = new Profiler(FuncInfoVar, FuncInfoVar->GetProfileGroup(), true, tid); \
   if (Timer == (Profiler *) NULL) {\
     printf("ERROR: TAU_MAPPING_PROFILE_TIMER: new returns NULL Profiler *\n");\
   }
   

/* TAU_MAPPING_PROFILE_START acts like TAU_PROFILE_START by starting the timer 
*/
#define TAU_MAPPING_PROFILE_START(Timer, tid) Timer->Start(tid);

/* TAU_MAPPING_PROFILE_STOP acts like TAU_PROFILE_STOP by stopping the timer 
*/
#define TAU_MAPPING_PROFILE_STOP(tid) Profiler::CurrentProfiler[tid]->Stop(tid);
#define TAU_MAPPING_PROFILE_EXIT(msg, tid)  Profiler::ProfileExit(msg, tid); 
#define TAU_MAPPING_PROFILE_SET_NODE(node, tid)  RtsLayer::setMyNode(node, tid); 
#else
/* Create null , except the main statement which should be executed as it is*/
#define TAU_MAPPING(stmt, group) stmt
#define TAU_MAPPING_OBJECT(FuncInfoVar) 
#define TAU_MAPPING_LINK(FuncInfoVar, Group) 
#define TAU_MAPPING_PROFILE(FuncInfoVar) 
#define TAU_MAPPING_CREATE(name, type, key, groupname, tid) 
#define TAU_MAPPING_PROFILE_TIMER(Timer, FuncInfoVar, tid)
#define TAU_MAPPING_PROFILE_START(Timer, tid) 
#define TAU_MAPPING_PROFILE_STOP(tid) 
#define TAU_MAPPING_PROFILE_EXIT(msg, tid)  
#define TAU_MAPPING_PROFILE_SET_NODE(node, tid)  

#endif /* PROFILING_ON or TRACING_ON  */
#endif /* _TAU_MAPPING_H_ */
