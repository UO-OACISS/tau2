/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
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
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <Profile/Profiler.h>

#ifdef TAU_WINDOWS
//#include <vector>
//#include <string>
using namespace std;
#endif

#include <stdlib.h>

//#define DEBUG_PROF
//int debugPrint = 0;
// control debug printf statements
//#define dprintf if (debugPrint) printf
#ifdef DEBUG_PROF
#define dprintf printf
#else // DEBUG_PROF 
#define dprintf if (0) printf
#endif

//#define ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP 1
#define TAUDYNVEC 1

#ifndef __ia64
int TheFlag[TAU_MAX_THREADS] ;
#define TAU_MONITOR_ENTER(tid) if (TheFlag[tid] == 0) {TheFlag[tid] = 1;}  else {return; } 
#define TAU_MONITOR_EXIT(tid) TheFlag[tid] = 0
#else /* FOR IA64 */
vector<int> TheFlag(TAU_MAX_THREADS); 
#define TAU_MONITOR_ENTER(tid)  if (TheFlag[tid] == 0) {TheFlag[tid] = 1;}  else {return; } 
#define TAU_MONITOR_EXIT(tid) TheFlag[tid] = 0
#endif /* IA64 */

vector<string> TauFuncNameVec; /* holds just names */
vector<FunctionInfo*>& TheTauDynFI(void)
{ // FunctionDB contains pointers to each FunctionInfo static object
  static vector<FunctionInfo*> FuncTauDynFI;

  return FuncTauDynFI;
}
/* global FI vector */
// Initialization procedure. Should be called before invoking 
// other TAU routines.
extern "C" {
void TauInitCode(char *arg, int isMPI)
{
  // Register that we are using dyninst so that the FIvector destructor will
  // perform cleanup for us
  TheUsingDyninst() = 1;
  char *name;
  int tid = 0;
  TAU_MONITOR_ENTER(0);
  int functionId = 0;

  /* iterate for each routine name */
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
#ifdef TAUDYNVEC
    FunctionInfo *taufi = new 
	FunctionInfo(name, " " , TAU_DEFAULT, "TAU_DEFAULT", true, tid); 
    if (taufi == (FunctionInfo *) NULL) {
      printf("ERROR: new returns NULL in TauInitCode\n"); exit(1); 
    }
    dprintf("TAU FI = %lx\n", taufi);
    TheTauDynFI().push_back(taufi); 
#else /* TAUDYNVEC */
    int id;
    id = functionId - 1; /* start from 0 */
    TauFuncNameVec.push_back(string(name));  /* Save the name with id */
    dprintf("TauFuncNameVec[%d] = %s\n", id, TauFuncNameVec[id].c_str()); 
    TheTauDynFI().push_back(NULL); /* create a null entry for this symbol */
#endif /* TAUDYNVEC */
#endif /* ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP */
    
    name = strtok(NULL, "|");
  }
  dprintf("Inside TauInitCode Initializations to be done here!\n");
  if (!isMPI)
    TAU_MAPPING_PROFILE_SET_NODE(0, tid);
  dprintf("Node = %d\n", RtsLayer::myNode());

#if (defined (__linux__) && defined(TAU_DYNINST41BUGFIX))
  Tau_create_top_level_timer_if_necessary();  
#endif /* DyninstAPI 4.1 bug appears only under Linux */
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
#elif TAUDYNVEC
  id--; /* to account for offset. Start from 0..n-1 instead of 1..n */
  vector<FunctionInfo *> vfi = TheTauDynFI();
  for (vector<FunctionInfo *>::iterator it = vfi.begin(); it != vfi.end(); it++)
  {
    TauMethodName = TheTauDynFI()[id];
    TAU_MAPPING_PROFILE_TIMER(TauTimer, TauMethodName, tid);
    TAU_MAPPING_PROFILE_START(TauTimer, tid);
    break;
  }
#else /* TAUDYNVEC */
  id--; /* to account for offset. Start from 0..n-1 instead of 1..n */ 
  if ((TauMethodName = TheTauDynFI()[id]) == NULL) 
  {	/* Function has not been called so far */
    TauMethodName = new   
	 FunctionInfo(TauFuncNameVec[id].c_str(), " " , TAU_DEFAULT, "TAU_DEFAULT", true, tid); 
    TheTauDynFI()[id] = TauMethodName;
  }
#endif /* retrieve it from the vector */
  
#ifndef TAUDYNVEC 
  dprintf("<tid %d> Entry <id %d> <<<<< name = %s\n", tid, id, TauMethodName->GetName());
  TAU_MAPPING_PROFILE_TIMER(TauTimer, TauMethodName, tid);
  TAU_MAPPING_PROFILE_START(TauTimer, tid);
  dprintf("Entry into %s: id = %d\n", TauMethodName->GetName(), id);
#endif /* TAUDYNVEC */
  TAU_MONITOR_EXIT(tid);
}

// Hook for function exit.
void TauRoutineExit(int id)
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  id --; 
  /*
  FunctionInfo *fi = TheTauDynFI()[id];
  dprintf("<tid %d> Exit <id %d> >>>>>> name = %s\n", tid, id, fi->GetName());
  */
  TAU_MAPPING_PROFILE_STOP(tid);
  TAU_MONITOR_EXIT(tid);
}

void TauRoutineEntryTest(int id )
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  id --; 
  dprintf("<tid %d> TAU Entry <id %d>\n", tid, id);
  // dprintf("At entry, Size = %d\n", TheTauDynFI().size());
  vector<FunctionInfo *> vfi = TheTauDynFI();
  FunctionInfo *fi = 0;
  for (vector<FunctionInfo *>::iterator it = vfi.begin(); it != vfi.end(); it++)
  {
    fi = TheTauDynFI()[id];
    TAU_MAPPING_PROFILE_TIMER(TauTimer, fi, tid);
    TAU_MAPPING_PROFILE_START(TauTimer, tid);
    break;
  }
  /*
  FunctionInfo *fi = TheTauDynFI()[0];
  dprintf("<tid %d> Entry <id %d> <<<<< name = %s\n", tid, id, fi->GetName());
  */
  

  TAU_MONITOR_EXIT(tid);
}
  

void TauRoutineExitTest(int id)
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  id --; 
  dprintf("<tid %d> TAU Exit <id %d>\n", tid, id);
  int val = TheTauDynFI().size();
  dprintf("Size = %d\n", val);
  TAU_MAPPING_PROFILE_STOP(tid);
  
  /*  
  FunctionInfo *fi = TheTauDynFI()[id];
  printf("<tid %d> Exit <id %d> >>>>>> name = %s\n", tid, id, fi->GetName());
  */ 
  TAU_MONITOR_EXIT(tid);
}

void TauProgramTermination(char *name)
{
  dprintf("TauProgramTermination %s\n", name);
  if (TheSafeToDumpData())
  {
    dprintf("Dumping data...\n");
    TAU_PROFILE_EXIT(name);
    TheSafeToDumpData() = 0;
  }
  return;
}

void HookEntry(int id)
{
  dprintf("Entry ->: %d\n",id);
  return;
}

void HookExit(int id)
{
  dprintf("Exit <-: %d\n",id);
  return;
}

void TauMPIInitStub(int *rank)
{
  dprintf("INSIDE TauMPIInitStub() rank = %d \n", *rank);

  TAU_PROFILE_SET_NODE(*rank);
  dprintf("Setting rank = %d\n", *rank);
}

int TauRenameTimer(char *oldName, char *newName)
{
  vector<FunctionInfo *>::iterator it;
  string *newfuncname = new string(newName);

  dprintf("Inside TauRenameTimer: Old = %s, New = %s\n", oldName, newName);
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
    //Check to see that it is one of the requested functions.
    dprintf("Comparing %s with %s\n", (*it)->GetName(), oldName);
    if (strcmp(oldName, (*it)->GetName()) == 0)
    {
      (*it)->SetName(*newfuncname);
      dprintf("Renaming %s to%s\n", oldName, newfuncname->c_str());
      return 1; /* found it! */
    }
  }
  dprintf("Didn't find the routine!\n");
  return 0; /* didn't find it! */
}
} // extern "C"

// EOF TauHooks.cpp
/***************************************************************************
 * $RCSfile: TauHooks.cpp,v $   $Author: amorris $
 * $Revision: 1.25 $   $Date: 2008/03/10 19:51:24 $
 * TAU_VERSION_ID: $Id: TauHooks.cpp,v 1.25 2008/03/10 19:51:24 amorris Exp $ 
 ***************************************************************************/
