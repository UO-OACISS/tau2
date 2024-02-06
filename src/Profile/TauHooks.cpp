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
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <string>
#include <ctype.h>

#include <Profile/Profiler.h>
#include <Profile/TauPin.h>
#include <Profile/TauBfd.h>

using namespace std;
using namespace tau;


#define TAUDYNVEC 1


extern "C" void tau_dyninst_init(int isMPI);

/*#ifndef __ia64
int TheFlag[TAU_MAX_THREADS] ;
#define TAU_MONITOR_ENTER(tid) if (TheFlag[tid] == 0) {TheFlag[tid] = 1;}  else {return; }
#define TAU_MONITOR_EXIT(tid) TheFlag[tid] = 0
#else // FOR IA64 */
vector<int> TheFlag;//(TAU_MAX_THREADS)
#define TAU_MONITOR_ENTER(tid)  while(TheFlag.size()<=tid){TheFlag.push_back(0);} if (TheFlag[tid] == 0) {TheFlag[tid] = 1;}  else {return; }
#define TAU_MONITOR_EXIT(tid) TheFlag[tid] = 0
//#endif /* IA64 */


vector<string> TauLoopNames; /* holds just names of loops */
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
  // TAU_VERBOSE("TauInitCode: arg=%s, isMPI=%d\n", arg, isMPI);
  TheUsingDyninst() = 1;
  char *name;
  int tid = 0;
  TAU_MONITOR_ENTER(0);
  int functionId = 0;
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
#ifdef TAU_WINDOWS
    name = strtok(str1, "|");
#else
    name = strtok_r(str1, "|", &saveptr);
#endif
    if (name == NULL)
      break;
    TAU_VERBOSE("After loop: name = %s\n", name);

    functionId ++;
#ifdef ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP
    TAU_VERBOSE("Extracted : %s :id = %d\n", name, functionId);
    char funcname[1024];
    TAU_MAPPING_CREATE(funcname, " ", functionId, "TAU_DEFAULT", tid);
#else
    TAU_VERBOSE("Extracted : %s :id = %d\n", name, functionId-1);
    /* Create a new FunctionInfo object with the given name and add it to
       the global vector of FI pointers */
#ifdef TAUDYNVEC
    FunctionInfo *taufi = new
	FunctionInfo(name, " " , TAU_DEFAULT, "TAU_DEFAULT", true);
    if (taufi == (FunctionInfo *) NULL) {
      printf("ERROR: new returns NULL in TauInitCode\n"); exit(1);
    }
    TAU_VERBOSE("TAU FI = %lx\n", taufi);
    TheTauDynFI().push_back(taufi);
#else /* TAUDYNVEC */
    int id;
    id = functionId - 1; /* start from 0 */
    TauFuncNameVec.push_back(string(name));  /* Save the name with id */
    TAU_VERBOSE("TauFuncNameVec[%d] = %s\n", id, TauFuncNameVec[id].c_str());
    TheTauDynFI().push_back(NULL); /* create a null entry for this symbol */
#endif /* TAUDYNVEC */
#endif /* ORIGINAL_HEAVY_IMPLEMENTATION_USING_MAP */

  }
  TAU_VERBOSE("Inside TauInitCode Initializations to be done here!\n");
  if (!isMPI)
    TAU_MAPPING_PROFILE_SET_NODE(0, tid);
  TAU_VERBOSE("Node = %d\n", RtsLayer::myNode());

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
  TAU_VERBOSE("<tid %d> Entry <id %d> <<<<< name = %s\n", tid, id, TauMethodName->GetName());
  TAU_MAPPING_PROFILE_TIMER(TauTimer, TauMethodName, tid);
  TAU_MAPPING_PROFILE_START(TauTimer, tid);
  TAU_VERBOSE("Entry into %s: id = %d\n", TauMethodName->GetName(), id);
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
  TAU_VERBOSE("<tid %d> Exit <id %d> >>>>>> name = %s\n", tid, id, fi->GetName());
  */
  TAU_MAPPING_PROFILE_STOP(tid);
  TAU_MONITOR_EXIT(tid);
}

void TauRoutineEntryTest(int id )
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  id --;
  TAU_VERBOSE("<tid %d> TAU Entry <id %d>\n", tid, id);
  // TAU_VERBOSE("At entry, Size = %d\n", TheTauDynFI().size());
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
  TAU_VERBOSE("<tid %d> Entry <id %d> <<<<< name = %s\n", tid, id, fi->GetName());
  */


  TAU_MONITOR_EXIT(tid);
}


void TauRoutineExitTest(int id)
{
  int tid = RtsLayer::myThread();
  TAU_MONITOR_ENTER(tid);
  id --;
  TAU_VERBOSE("<tid %d> TAU Exit <id %d>\n", tid, id);
  int val = TheTauDynFI().size();
  TAU_VERBOSE("Size = %d\n", val);
  TAU_MAPPING_PROFILE_STOP(tid);

  /*
  FunctionInfo *fi = TheTauDynFI()[id];
  printf("<tid %d> Exit <id %d> >>>>>> name = %s\n", tid, id, fi->GetName());
  */
  TAU_MONITOR_EXIT(tid);
}

void TauProgramTermination(char *name)
{
  TAU_VERBOSE("TauProgramTermination %s\n", name);
  if (TheSafeToDumpData())
  {
    TAU_VERBOSE("Dumping data...\n");
    TAU_PROFILE_EXIT(name);
    TheSafeToDumpData() = 0;
  }
  return;
}

void HookEntry(int id)
{
  TAU_VERBOSE("Entry ->: %d\n",id);
  return;
}

void HookExit(int id)
{
  TAU_VERBOSE("Exit <-: %d\n",id);
  return;
}

void TauMPIInitStub(int *rank)
{
  TAU_VERBOSE("INSIDE TauMPIInitStub() rank = %d \n", *rank);

  TAU_PROFILE_SET_NODE(*rank);
  TAU_VERBOSE("Setting rank = %d\n", *rank);
}

void TauMPIInitStubInt (int rank)
{
  TauMPIInitStub(&rank);
}

int TauRenameTimer(char *oldName, char *newName)
{
  vector<FunctionInfo *>::iterator it;
  string *newfuncname = new string(newName);

  TAU_VERBOSE("Inside TauRenameTimer: Old = %s, New = %s\n", oldName, newName);
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
    //Check to see that it is one of the requested functions.
    TAU_VERBOSE("Comparing %s with %s\n", (*it)->GetName(), oldName);
    if (strcmp(oldName, (*it)->GetName()) == 0)
    {
      (*it)->SetName(*newfuncname);
      TAU_VERBOSE("Renaming %s to%s\n", oldName, newfuncname->c_str());
      return 1; /* found it! */
    }
  }
  TAU_VERBOSE("Didn't find the routine!\n");
  return 0; /* didn't find it! */
}


static int tauFiniID = -1; 
static vector <int> tauDyninstEnabled;//[TAU_MAX_THREADS];
static int defaultDyEn=0;
std::mutex DyEnVectorMutex;
static inline void checkDyEnVector(int tid){
    if(tauDyninstEnabled.size()<=tid){
      std::lock_guard<std::mutex> guard(DyEnVectorMutex);
      while(tauDyninstEnabled.size()<=tid){
        tauDyninstEnabled.push_back(defaultDyEn);
      }
    }
 }
static inline void setTauDyninstEnabled(int tid, int val){
    checkDyEnVector(tid);
    tauDyninstEnabled[tid]=val;
}

static inline int getTauDyninstEnabled(int tid){
    checkDyEnVector(tid);
    return tauDyninstEnabled[tid];
}

void trace_register_func(char *origname, int id)
{
  static int invocations = 0;
  int i;
  int tid = RtsLayer::myThread();
  int funclen;
  char *func = origname;
  //char *func;
  if ((func[0] == '_') && (func[1] == 'Z')) {
    funclen = strlen(func);
    char *mirror=strdup(func);
    for(i=0; i < funclen; i++) {
      if ((mirror[i] == '[') &&(mirror[i-1] == ' ')) {
        mirror[i-1] = '\0';
        break;
      }
    }
    char *dem = Tau_demangle_name(mirror);
    const int len = strlen(dem)+funclen-i+3;
    char *newname = (char *) malloc(len);
    snprintf(newname, len,  "%s %s", dem, &func[i-1]);
    TAU_VERBOSE("name=%s, newname = %s\n", func, newname);
    free(mirror);
    free(dem);
    func = newname;
  }
  TAU_VERBOSE("trace_register_func: func = %s, id = %d\n", func, id);
  if (invocations == 0) {
    if (!getTauDyninstEnabled(tid)) {
#ifdef TAU_MPI
      tau_dyninst_init(1);
#else
      tau_dyninst_init(0);
#endif /* TAU_MPI */
    }
  }

  int len = strlen(func);
  int startbracket = 0;
  int stopbracket = 0;
  for (i=0; i < len; i++) {
    if (func[i] == '[') startbracket = i;
    if (func[i] == ']') stopbracket = i;
    if (!isprint(func[i])) {
      TAU_VERBOSE("TauHooks.cpp: trace_register_func(): func=%s - isprint is false at i = %d\n", func, i);
      func[i] = '\0';
      if (i == 0) strcpy(func, "<unknown>");
    }
  }
  if (startbracket > 0 && stopbracket == 0) { /* didn't find stop bracket */
    TAU_VERBOSE("func=%s, before chopping off the bracket! \n", func);
    func[startbracket] = '\0'; /* chop it off - no need to show the name */
    TAU_VERBOSE("func=%s, after chopping off the bracket! \n", func);
  }

  if (!getTauDyninstEnabled(tid)) return;

  void *taufi;
  TAU_PROFILER_CREATE(taufi, func, " ", TAU_DEFAULT);


  if (strncmp(func, "_fini", 5) == 0) {
    TAU_VERBOSE("FOUND FINI id = %d\n", id);
    tauFiniID = id;
  }
  if (func[0] == 't' && func[1] == 'a' && func[2] == 'r' && func[3] == 'g') {
    if (isdigit(func[4])) {
      TAU_VERBOSE("trace_register_func: Routine name is targN...\n");
      ((FunctionInfo *)taufi)->SetProfileGroup(TAU_GROUP_31);

    // TAU_GROUP_31 is special. It indicates that the routine is called targ...
    // This routine should be exited prior to the beginning of the next routine
    // Extract the name from the address:
/*
      long addr;
      sscanf(func, "targ%lx", &addr);
      TAU_VERBOSE("ADDR=%lx, name =%s\n", addr, func);
      char name[256];
      char filename[256];
      Tau_get_func_name(addr, (char *)name, (char *)filename);
      printf("GOT: name = %s, filename = %s, addr = %lx\n", name, filename, addr);
*/
    }
  }
  TAU_VERBOSE("TAU FI = %lx\n", taufi);
  TAU_VERBOSE("id = %d, invocations = %d\n", id , invocations);
  if (id == invocations)
    TheTauBinDynFI().push_back(taufi);
  else {
    printf("WARNING: trace_register_func: id does not match invocations\n");
    TheTauBinDynFI().resize(id+1);
    TheTauBinDynFI()[id] = taufi;
  }
  invocations ++;
  TAU_VERBOSE("Exiting trace_register_func\n");
}

void traceEntry(int id)
{
  int tid = RtsLayer::myThread();
  if ( !RtsLayer::TheEnableInstrumentation()) return;
  if (!getTauDyninstEnabled(tid)) return;
  void *fi = TheTauBinDynFI()[id];

  // Additional sanity checks
  if (fi == NULL) {
    TAU_VERBOSE("ERROR?: ENTRY: id = null!\n");
    return;
  }

  // this event is throttled - exit!
  if (!(((FunctionInfo*)(fi))->GetProfileGroup() & RtsLayer::TheProfileMask())) {
    return;
  }

  TAU_QUERY_DECLARE_EVENT(curr);
  TAU_QUERY_GET_CURRENT_EVENT(curr);

  if ( curr && ((Profiler *)curr)->ThisFunction &&
     ((Profiler *)curr)->ThisFunction->GetProfileGroup() == TAU_GROUP_31) {
    TAU_VERBOSE("TARG on the stack \n");
    TAU_PROFILER_STOP(((Profiler *)curr)->ThisFunction);
  }

#if 0
  TAU_VERBOSE("Inside traceEntry: id = %d fi = %lx\n", id, fi);
  TAU_VERBOSE("Name = %s\n", ((FunctionInfo *)fi)->GetName());
#endif
  if (id == tauFiniID) {
    Tau_stop_top_level_timer_if_necessary();
	/* if there is .TAU application from tau_exec, write the files out */
    TAU_DISABLE_INSTRUMENTATION();
    TAU_VERBOSE("Disabling instrumentation found id = %d\n", id);
  }
  else {
    if (fi != NULL) {
      //TAU_PROFILER_START(fi);
      Tau_start_timer(fi, 0, tid);
    } else {
      TAU_VERBOSE("ERROR?: traceEntry: fi = null!\n");
    }
  }


}

void traceExit(int id)
{
  //const char *strcurr;
  //const char *strbin;
  //TAU_VERBOSE("Inside traceExit: id = %d\n", id);

  if ( !RtsLayer::TheEnableInstrumentation()) return;
  int tid = RtsLayer::myThread();

  if (!getTauDyninstEnabled(tid)) return;
  void *fi = TheTauBinDynFI()[id];
#if 0
  if (fi)
    TAU_VERBOSE("traceExit: Name = %s, %lx\n", ((FunctionInfo *)fi)->GetName(), fi);
  else
    return;
#endif

  // this event is throttled - exit!
  if (!(((FunctionInfo*)(fi))->GetProfileGroup() & RtsLayer::TheProfileMask())) {
    return;
  }

  TAU_QUERY_DECLARE_EVENT(curr);
  TAU_QUERY_GET_CURRENT_EVENT(curr);

  // Additional sanity checks: Stop profiling after main exits
  bool disableinstr = false;
  if ( curr && ((Profiler *)curr)->ParentProfiler == (Profiler *) NULL)
  {
    if (strncmp(((FunctionInfo *)fi)->GetName(), "main",4)== 0) {
      disableinstr = true;
      TAU_VERBOSE("Disabling instrumentation!\n");
    }
  }
  if (fi != NULL) {
    //TAU_PROFILER_STOP(fi);
     Tau_stop_timer(fi, tid);
  } else {
    printf("ERROR: traceExit: fi = null!\n");
  }
  if(disableinstr) {
    setTauDyninstEnabled(tid, false);
  }

}

void my_otf_init(int isMPI)
{
  TAU_VERBOSE("Inside my otf_init\n");
  TAU_VERBOSE("isMPI = %d\n", isMPI);
  if (!isMPI)
  {
    TAU_VERBOSE("Calling SET NODE 0\n");
    TAU_PROFILE_SET_NODE(0);
  }
  int tid = RtsLayer::myThread();
  if (!getTauDyninstEnabled(tid)) {
    setTauDyninstEnabled(tid, 1);
  }
}

void my_otf_cleanup()
{
  TAU_VERBOSE("Inside my otf_cleanup\n");
}

void tau_dyninst_init(int isMPI)
{
  TAU_VERBOSE("Inside tau_dyninst_init \n");
  TAU_VERBOSE("isMPI = %d\n", isMPI);
  if (!isMPI)
  {
    TAU_VERBOSE("Calling SET NODE 0\n");
    TAU_PROFILE_SET_NODE(0);
  }
  int tid = RtsLayer::myThread();
  if (!getTauDyninstEnabled(tid)) {
    defaultDyEn=1;
    RtsLayer::LockDB();
    int vecSize=tauDyninstEnabled.size();
    for (int i = 0; i < vecSize; i++)
      setTauDyninstEnabled(i,1);
    RtsLayer::UnLockDB();
  }
}

void tau_dyninst_cleanup()
{
  TAU_VERBOSE("Inside tau_dyninst_cleanup\n");
}

void  tau_register_func(char **func, char** file, int* lineno,
  int id) {
    char * tmpstr = Tau_demangle_name(*func);
    if (*file == NULL){
      TAU_VERBOSE("TAU: tau_register_func: name = %s, id = %d\n", *func, id);
      trace_register_func(tmpstr, id);
    } else {
      char funcname[2048];
      snprintf(funcname, sizeof(funcname),  "%s [{%s}{%d}]", tmpstr, *file, *lineno);
      trace_register_func(funcname, id);
      TAU_VERBOSE("TAU : tau_register_func: name = %s, id = %d\n", funcname, id);
    }
    free(tmpstr);
}

void tau_trace_entry(int id) {
  TAU_VERBOSE("TAU: tau_trace_entry: id = %d\n", id);
  traceEntry(id);
}

void tau_trace_exit(int id) {
  TAU_VERBOSE("TAU: tau_trace_exit : id = %d\n", id);
  traceExit(id);
}

void tau_trace_lib_entry(const char * func_name)
{
  if(!RtsLayer::TheEnableInstrumentation())
	  return;
  TAU_VERBOSE("TAU: tau_trace_lib_entry %s\n", func_name);
  TAU_START(func_name);
}

void tau_trace_lib_exit(const char * func_name)
{
  if(!RtsLayer::TheEnableInstrumentation())
	  return;
  TAU_VERBOSE("TAU: tau_trace_lib_exit %s\n", func_name);
  TAU_STOP(func_name);
}

void tau_loop_trace_entry(int id) {
  TAU_VERBOSE("TAU: tau_loop_trace_entry: id = %d\n", id);
  TAU_START(TauLoopNames[id].c_str());
}

void tau_loop_trace_exit(int id) {
  TAU_VERBOSE("TAU: tau_loop_trace_exit : id = %d\n", id);
  TAU_STOP(TauLoopNames[id].c_str());
}

#if !defined(TAU_PEBIL_DISABLE) && !defined(TAU_WINDOWS)
#include <pthread.h>
void* tool_thread_init(pthread_t args) {
  TAU_VERBOSE("TAU: initializing thread %#lx\n", args);
  Tau_create_top_level_timer_if_necessary();
  return NULL;
}

void* tool_thread_fini(pthread_t args) {
  TAU_VERBOSE("TAU: finalizing thread %#lx\n", args);
  Tau_stop_top_level_timer_if_necessary();
  return NULL;
}

void  tau_trace_register_loop(int id, char *loopname) {
  static int invocations = 0;
  TAU_VERBOSE("TAU: tau_trace_register_loop: id = %d, loopname = %s\n", id, loopname);
  if (invocations == id) {
    TauLoopNames.push_back(string(loopname));
    invocations++;
  } else {
    printf("WARNING: id = %d, invocations = %d, loopname = %s\n", id, invocations, loopname);
    TauLoopNames.resize(id+1);
    TauLoopNames[id] = string(loopname);
  }

}

void  tau_register_loop(char **func, char** file, int* lineno,
  int id) {

  char lname[2048];
  char *loopname;
  if (((*file) != (char *)NULL) && (*lineno != 0)) {
    snprintf(lname, sizeof(lname),  "Loop: %s [{%s}{%d}]", *func, *file, *lineno);
  } else {
    snprintf(lname, sizeof(lname),  "Loop: %s ",*func);
  }
  loopname = strdup(lname);
  tau_register_func(&loopname, file, lineno, id);

}
#endif /* TAU_PEBIL_DISABLE */


} /* extern "C" */

#ifdef __PIN__
/*
#include <iostream>
extern "C" int bar(int x) {
  printf("Inside bar: x = %d\n", x);
  string s("bar:");
  std::cout <<s<<"Returning "<<x+1<<endl;
  return x+1;
}
*/
#endif /* __PIN__ */


// EOF TauHooks.cpp
/***************************************************************************
 * $RCSfile: TauHooks.cpp,v $   $Author: sameer $
 * $Revision: 1.35 $   $Date: 2010/06/09 15:11:36 $
 * TAU_VERSION_ID: $Id: TauHooks.cpp,v 1.35 2010/06/09 15:11:36 sameer Exp $
 ***************************************************************************/
