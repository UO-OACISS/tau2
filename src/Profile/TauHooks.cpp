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
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
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
//#define dprintf printf
#ifdef DEBUG_PROF
#define dprintf printf
#else // DEBUG_PROF 
#define dprintf TAU_VERBOSE
#endif

//#define ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP 1
#define TAUDYNVEC 1
//extern "C" void Tau_get_func_name(long addr, char * fname, char *filename);


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
vector<void*>& TheTauBinDynFI(void)
{ // FunctionDB contains pointers to each FunctionInfo static object
  static vector<void*> FuncTauDynFI;

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
  // dprintf("TauInitCode: arg=%s, isMPI=%d\n", arg, isMPI);
  TheUsingDyninst() = 1;
  char *name;
  int tid = 0;
  TAU_MONITOR_ENTER(0);
  int functionId = 0;
  char funcname[1024];
  char *saveptr;

  int j;
  char *str1;
  /* iterate for each routine name */
/*
  for (j=1, str1 = arg; ; j++, str1 = NULL) {
    name = strtok_r(str1, "|", &saveptr);
    if (name == NULL) 
      break;
    printf("Extracted token = %s\n", name);
  }
*/
  for (j=1, str1 = arg; ; j++, str1 = NULL) 
  { 
    name = strtok_r(str1, "|", &saveptr);
    if (name == NULL) 
      break;
    dprintf("After loop: name = %s\n", name);

    functionId ++; 
#ifdef ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP
    dprintf("Extracted : %s :id = %d\n", name, functionId);
    TAU_MAPPING_CREATE(funcname, " ", functionId, "TAU_DEFAULT", tid);
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

void TauMPIInitStubInt (int rank)
{
  TauMPIInitStub(&rank);
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


static int tauFiniID; 
static int tauDyninstEnabled[TAU_MAX_THREADS];
void trace_register_func(char *func, int id)
{
  static int invocations = 0;
  int tid = RtsLayer::myThread();
  if (!tauDyninstEnabled[tid]) return;
  dprintf("trace_register_func: func = %s, id = %d\n", func, id);

  void *taufi;
  TAU_PROFILER_CREATE(taufi, func, " ", TAU_DEFAULT);

 
  if (strncmp(func, "_fini", 5) == 0) {
    dprintf("FOUND FINI id = %d\n", id);
    tauFiniID = id;
  } 
  if (func[0] == 't' && func[1] == 'a' && func[2] == 'r' && func[3] == 'g') {
    if (isdigit(func[4])) {
      long addr;
      dprintf("trace_register_func: Routine name is targN...\n");
      ((FunctionInfo *)taufi)->SetProfileGroup(TAU_GROUP_31);

    // TAU_GROUP_31 is special. It indicates that the routine is called targ...
    // This routine should be exited prior to the beginning of the next routine
    // Extract the name from the address:
/*
      sscanf(func, "targ%lx", &addr);
      dprintf("ADDR=%lx, name =%s\n", addr, func);
      char name[256];
      char filename[256];
      Tau_get_func_name(addr, (char *)name, (char *)filename); 
      printf("GOT: name = %s, filename = %s, addr = %lx\n", name, filename, addr);
*/
    }
  }
  dprintf("TAU FI = %lx\n", taufi);
  dprintf("id = %d, invocations = %d\n", id , invocations);
  if (id == invocations)
    TheTauBinDynFI().push_back(taufi);
  else {
    printf("WARNING: trace_register_func: id does not match invocations\n");
    TheTauBinDynFI()[id] = taufi;
  } 
  invocations ++;
}

void traceEntry(int id)
{
  int tid = RtsLayer::myThread();
  if (!tauDyninstEnabled[tid]) return;
  void *fi = TheTauBinDynFI()[id];

  // Additional sanity checks
  if (fi == NULL) { 
    dprintf("ERROR?: ENTRY: id = null!\n");
    return;
  }
  TAU_QUERY_DECLARE_EVENT(curr);
  TAU_QUERY_GET_CURRENT_EVENT(curr);

  if ( curr && ((Profiler *)curr)->ThisFunction && 
     ((Profiler *)curr)->ThisFunction->GetProfileGroup() == TAU_GROUP_31) {
    dprintf("TARG on the stack \n");
    TAU_PROFILER_STOP(((Profiler *)curr)->ThisFunction);
  }

  dprintf("Inside traceEntry: id = %d fi = %lx\n", id, fi);
  dprintf("Name = %s\n", ((FunctionInfo *)fi)->GetName());
  if (id == tauFiniID) { 
    TAU_DISABLE_INSTRUMENTATION();
    dprintf("Disabling instrumentation found id = %d\n", id);
  } 
  else {
    if (fi != NULL) {
      TAU_PROFILER_START(fi);
    } else {
      dprintf("ERROR?: traceEntry: fi = null!\n");
    }
  }
  

}

void traceExit(int id)
{
  const char *strcurr;
  const char *strbin;
  int tid = RtsLayer::myThread();
  if (!tauDyninstEnabled[tid]) return;
  void *fi = TheTauBinDynFI()[id];
  dprintf("traceExit: Name = %s, %lx\n", ((FunctionInfo *)fi)->GetName(), fi);

  TAU_QUERY_DECLARE_EVENT(curr);
  TAU_QUERY_GET_CURRENT_EVENT(curr);

  // Additional sanity checks: Stop profiling after main exits
  bool disableinstr = false;
  if ( curr && ((Profiler *)curr)->ParentProfiler == (Profiler *) NULL)
  {
    if (strncmp(((FunctionInfo *)fi)->GetName(), "main",4)== 0) {
      disableinstr = true;
      dprintf("Disabling instrumentation!\n");
    }
  }
  if (fi != NULL) {
    TAU_PROFILER_STOP(fi);
  } else {
    printf("ERROR: traceExit: fi = null!\n");
  }
  if(disableinstr) {
    tauDyninstEnabled[tid] = false;
  }

}

void my_otf_init(int isMPI)
{
  dprintf("Inside my otf_init\n");
  dprintf("isMPI = %d\n", isMPI);
  if (!isMPI)
  {
    dprintf("Calling SET NODE 0\n");
    TAU_PROFILE_SET_NODE(0);
  }
  int tid = RtsLayer::myThread();
  if (!tauDyninstEnabled[tid]) {
    tauDyninstEnabled[tid] = 1;
  }
}

void my_otf_cleanup()
{
  dprintf("Inside my otf_cleanup\n");
}

void tau_dyninst_init(int isMPI)
{
  dprintf("Inside tau_dyninst_init \n");
  dprintf("isMPI = %d\n", isMPI);
  if (!isMPI)
  {
    dprintf("Calling SET NODE 0\n");
    TAU_PROFILE_SET_NODE(0);
  }
  int tid = RtsLayer::myThread();
  if (!tauDyninstEnabled[tid]) {
    tauDyninstEnabled[tid] = 1;
  }
}

void tau_dyninst_cleanup()
{
  dprintf("Inside tau_dyninst_cleanup\n");
}


}
// extern "C"

// EOF TauHooks.cpp
/***************************************************************************
 * $RCSfile: TauHooks.cpp,v $   $Author: sameer $
 * $Revision: 1.34 $   $Date: 2010/04/30 23:22:24 $
 * TAU_VERSION_ID: $Id: TauHooks.cpp,v 1.34 2010/04/30 23:22:24 sameer Exp $ 
 ***************************************************************************/
