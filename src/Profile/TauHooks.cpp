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

//#define ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP 1


int TheFlag[TAU_MAX_THREADS] ;
#define TAU_MONITOR_ENTER(tid) if (TheFlag[tid] == 0) {TheFlag[tid] = 1;}  else {return; } 
#define TAU_MONITOR_EXIT(tid) TheFlag[tid] = 0

vector<FunctionInfo *> TauDynFI; /* global FI vector */
// Initialization procedure. Should be called before invoking 
// other TAU routines.
void TauInitCode(char *arg)
{
  char *name;
  int tid = 0;
  TAU_MONITOR_ENTER(0);
  int functionId = 0;
  name = strtok(arg, "|");
  while (name != (char *)NULL)
  { 
    functionId ++; 
#ifdef ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP
    dprintf("Extracted : %s :id = %d\n", name, functionId);
    TAU_MAPPING_CREATE(name, " ", functionId, "TAU_DEFAULT", tid);
#else
    dprintf("Extracted : %s :id = %d\n", name, functionId-1);
    /* Create a new FunctionInfo object with the given name and add it to 
       the global vector of FI pointers */
    FunctionInfo *taufi = new 
	FunctionInfo(name, " " , TAU_DEFAULT, "TAU_DEAULT", true, tid); 
    if (taufi == (FunctionInfo *) NULL) {
      printf("ERROR: new returns NULL in TauInitCode\n"); exit(1); 
    }
    TauDynFI.push_back(taufi); 
#endif
    
    name = strtok(NULL, "|");
  }
  dprintf("Inside TauInitCode Initializations to be done here!\n");
  TAU_MAPPING_PROFILE_SET_NODE(0, tid);
  dprintf("Node = %d\n", RtsLayer::myNode());
  TAU_MONITOR_EXIT(0);
}

// Hook for function entry.
void TauRoutineEntry(int id )
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  TAU_MAPPING_OBJECT(TauMethodName);
#ifdef ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP
  TAU_MAPPING_LINK(TauMethodName, id);
#else
  id--; /* to account for offset. Start from 0..n-1 instead of 1..n */
  TauMethodName = TauDynFI[id];
#endif /* retrieve it from the vector */
  
  dprintf("<tid %d> Entry <id %d> <<<<< name = %s\n", tid, id, TauMethodName->GetName());
  TAU_MAPPING_PROFILE_TIMER(TauTimer, TauMethodName, tid);
  TAU_MAPPING_PROFILE_START(TauTimer, tid);
  dprintf("Entry into %s: id = %d\n", TauMethodName->GetName(), id);
  TAU_MONITOR_EXIT(tid);
}

// Hook for function exit.
void TauRoutineExit(int id)
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  id --; 
  FunctionInfo *fi = TauDynFI[id];
  dprintf("<tid %d> Exit <id %d> >>>>>> name = %s\n", tid, id, fi->GetName());
  TAU_MAPPING_PROFILE_STOP(tid);
  TAU_MONITOR_EXIT(tid);
}

void TauRoutineEntryTest(int id )
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  id --; 
  dprintf("<tid %d> Entry <id %d>\n", tid, id);
  FunctionInfo *fi = TauDynFI[id];
  dprintf("<tid %d> Entry <id %d> <<<<< name = %s\n", tid, id, fi->GetName());

  TAU_MONITOR_EXIT(tid);
}
  
void TauRoutineExitTest(int id)
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  id --; 
  dprintf("<tid %d> Exit <id %d>\n", tid, id);
  FunctionInfo *fi = TauDynFI[id];
  printf("<tid %d> Exit <id %d> >>>>>> name = %s\n", tid, id, fi->GetName());
  TAU_MONITOR_EXIT(tid);
}

void HookEntry(char *name)
{
  dprintf("Entry ->: %s\n",name);
  return;
}

void HookExit(char *name)
{
  dprintf("Exit <-: %s\n",name);
  return;
}

// EOF TauHooks.cpp
/***************************************************************************
 * $RCSfile: TauHooks.cpp,v $   $Author: sameer $
 * $Revision: 1.5 $   $Date: 2000/02/22 23:29:57 $
 * TAU_VERSION_ID: $Id: TauHooks.cpp,v 1.5 2000/02/22 23:29:57 sameer Exp $ 
 ***************************************************************************/
