/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauHooks.cpp					  **
**	Description 	: TAU hooks for DynInst (Dynamic Instrumentation) **
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

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <Profile/Profiler.h>

//int debugPrint = 0;
// control debug printf statements
//#define dprintf if (debugPrint) printf
#ifdef DEBUG_PROF
#define dprintf printf
#else // DEBUG_PROF 
#define dprintf if (0) printf
#endif



int TheFlag[TAU_MAX_THREADS] ;
#define MONITOR_ENTER(tid) if (TheFlag[tid] == 0) {TheFlag[tid] = 1;}  else {return; } 
#define MONITOR_EXIT(tid) TheFlag[tid] = 0
/* This doesn't work. Use TheFlag instead of TheFlag()
int& TheFlag(void)
{
  static int flag = 0;
  return flag;
}
*/


// Initialization procedure. Should be called before invoking 
// other TAU routines.
void TauInitCode(char *arg)
{
  char *name;
  int tid = 0;
  MONITOR_ENTER(0);
  int functionId = 0;
  name = strtok(arg, "|");
  while (name != (char *)NULL)
  { 
    functionId ++; 
    dprintf("Extracted : %s :id = %d\n", name, functionId);
    TAU_MAPPING_CREATE(name, " ", functionId, "TAU_DEFAULT", tid);
    name = strtok(NULL, "|");
  }
  dprintf("Inside TauInitCode Initializations to be done here!\n");
  TAU_MAPPING_PROFILE_SET_NODE(0, tid);
  dprintf("Node = %d\n", RtsLayer::myNode());
  MONITOR_EXIT(0);
}

// Hook for function entry.
void TauRoutineEntry(int id )
{
  int tid = 0;
  MONITOR_ENTER(tid);
  TAU_MAPPING_OBJECT(TauMethodName);
  TAU_MAPPING_LINK(TauMethodName, id);
  
  TAU_MAPPING_PROFILE_TIMER(TauTimer, TauMethodName, tid);
  TAU_MAPPING_PROFILE_START(TauTimer, tid);
  //dprintf("Entry into %s: id = %d\n", name, id);
  MONITOR_EXIT(tid);
}

// Hook for function exit.
void TauRoutineExit(int id)
{
  int tid = 0;
  MONITOR_ENTER(tid);
  TAU_MAPPING_PROFILE_STOP(tid);
  //dprintf("Exit from %s: id = %d\n", name, id);
  MONITOR_EXIT(tid);
}

// EOF TauHooks.cpp
