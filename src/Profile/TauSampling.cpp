/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2009                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File            : TauSampling.cpp                                  **
**	Description     : TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains all the sampling related code **
**                                                                         **
****************************************************************************/

/****************************************************************************
*
*                      University of Illinois/NCSA
*                          Open Source License
*
*          Copyright(C) 2004-2006, The Board of Trustees of the
*              University of Illinois. All rights reserved.
*
*                             Developed by:
*
*                        The PerfSuite Project
*            National Center for Supercomputing Applications
*              University of Illinois at Urbana-Champaign
*
*                   http://perfsuite.ncsa.uiuc.edu/
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the "Software"),
* to deal with the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* + Redistributions of source code must retain the above copyright notice,
*   this list of conditions and the following disclaimers.
* + Redistributions in binary form must reproduce the above copyright
*   notice, this list of conditions and the following disclaimers in
*   the documentation and/or other materials provided with the distribution.
* + Neither the names of The PerfSuite Project, NCSA/University of Illinois
*   at Urbana-Champaign, nor the names of its contributors may be used to
*   endorse or promote products derived from this Software without specific
*   prior written permission.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS WITH THE SOFTWARE.
****************************************************************************/


#ifdef __APPLE__
#define _XOPEN_SOURCE 600 /* Single UNIX Specification, Version 3 */
#endif /* __APPLE__ */

#ifndef TAU_WINDOWS

#include <TAU.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauSampling.h>
#include <Profile/TauBfd.h>

#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <signal.h>
#include <stdlib.h>
#include <strings.h>

#include <ucontext.h>

// For STL string support
#include <string>

/*************************************
 * Shared Unwinder function prototypes.
 * These are internal to TAU and does
 *   not need to be extern "C"
 *************************************/
#ifdef TAU_UNWIND
extern void Tau_sampling_outputTraceCallstack(int tid, void *pc, 
					      void *context);
// *CWL* NOTE: This note applies to all implementations of the TAU context unwind -
//             The reason we unwind up to TAU_SAMP_NUM_ADDRESSES times is
//               because we cannot know, apriori, the exact number of function
//               calls made by TAU (eg. dependance on compilers) between the user
//               code representing that context to the point in TAU where we begin to
//               unwind the event context. All we know is we can safely drop
//               exactly 1 call layer, which explains the "skip" variable 
//               (see TauSampling_libunwind). This layer is invariably
//               "Tau_sampling_event_start"
//
//             The same is not true for sampling, where the signal handler itself
//             provides the originating context.
extern void Tau_sampling_unwindTauContext(int tid, void **address);
extern void Tau_sampling_unwind(int tid, Profiler *profiler,
				void *pc, void *context, unsigned long stack[]);

extern "C" bool unwind_cutoff(void **addresses, void *address) {
  bool found = false;
  for (int i=0; i<TAU_SAMP_NUM_ADDRESSES; i++) {
    if ((unsigned long)(addresses[i]) == (unsigned long)address) {
      //      printf("match found %p\n", address);
      found = true;
      break;
    }
  }
  return found; 
}

#endif /* TAU_UNWIND */

/*********************************************************************
 * Tau Sampling Record Definition
 ********************************************************************/
typedef struct {
  unsigned long pc;
  x_uint64 timestamp;
  double counters[TAU_MAX_COUNTERS];
  double counterDeltaStart[TAU_MAX_COUNTERS];
  double counterDeltaStop[TAU_MAX_COUNTERS];
  unsigned long deltaStart;
  unsigned long deltaStop;
} TauSamplingRecord;

typedef struct {
  unsigned long *pcStack;
  unsigned int sampleCount;
  FunctionInfo *tauContext;
} CallSiteCandidate;

typedef struct {
  unsigned long pc;
  int moduleIdx;
  char *name;
} CallSiteInfo;

// *CWL* - Keeping this structure in case we need extra fields
typedef struct {
  vector<CallSiteInfo *> *callSites;
} CallStackInfo;

/*********************************************************************
 * Global Variables
 ********************************************************************/

// Map for sample callsite/intermediate names to FunctionInfo objects.
//   We need this for two reasons:
//   1. because multiple sample addresses can map to the same source
//      line.
//   2. because multiple candidate samples can belong to the same
//      TAU context and we need to determine if an intermediate
//      FunctionInfo object has already been created for that context.
static map<string, FunctionInfo *> *name2FuncInfoMap[TAU_MAX_THREADS];
// Optimization: to find out if we need to resolve an address
static map<unsigned long, CallSiteInfo *> *pc2CallSiteMap[TAU_MAX_THREADS];

// For BFD-based name resolution
static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;

/* The trace for this node, mulithreaded execution currently not supported */
FILE *ebsTrace[TAU_MAX_THREADS];

/* Sample processing enabled/disabled */
int samplingEnabled[TAU_MAX_THREADS];
/* Sample processing suspended/resumed */
int suspendSampling[TAU_MAX_THREADS];
long long numSamples[TAU_MAX_THREADS];
long long samplesDroppedTau[TAU_MAX_THREADS];
long long samplesDroppedSuspended[TAU_MAX_THREADS];

// *CWL* This technique does NOT work when you have to rely on tau_exec for initialization
//   through the preload mechanism. Essentially, sampling initialization relies on
//   these thread variables initialized before its own operations. Unfortunately,
//   with tau_exec preload, sampling initialization will happen before static initializers
//   are invoked by the C++ runtime.
class initSamplingThreadStructs {
public:
  initSamplingThreadStructs() {
    TAU_VERBOSE("Initializing thread-specific variables\n");
    for (int i = 0; i < TAU_MAX_THREADS; i++) {
      samplingEnabled[i] = 0;
      suspendSampling[i] = 0;
      numSamples[i] = 0;
      samplesDroppedTau[i] = 0;
      samplesDroppedSuspended[i] = 0;
    }
  }
};
// initSamplingThreadStructs initializer = initSamplingThreadStructs();

/*********************************************************************
 * Get the architecture specific PC
 ********************************************************************/

#if __WORDSIZE == 32
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.uc_regs->gregs[reg])
#else
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.gp_regs[reg])
#endif

#define PPC_REG_PC 32

void issueUnavailableWarningIfNecessary(char *text) {
  static bool warningIssued = false;
  if (!warningIssued) {
    fprintf(stderr, text);
    warningIssued = true;
  }
}

unsigned long get_pc(void *p) {
  struct ucontext *uc = (struct ucontext *)p;
  unsigned long pc;

#ifdef sun
  issueUnavailableWarningIfNecessary("Warning, TAU Sampling does not work on solaris\n");
  return 0;
  //#elif defined(TAU_BGQ)
  // uc_mcontext->ss.srr0 *may* be the way forward.
  //  issueUnavailableWarningIfNecessary("Warning, TAU Sampling does not currently work on BGQ\n");
  //  return 0;
#elif __APPLE__
  issueUnavailableWarningIfNecessary("Warning, TAU Sampling does not work on apple\n");
  return 0;
#elif _AIX
  issueUnavailableWarningIfNecessary("Warning, TAU Sampling does not work on AIX\n");
  return 0;
#else
  struct sigcontext *sc;
  sc = (struct sigcontext *)&uc->uc_mcontext;
#ifdef TAU_BGP
  //  pc = (unsigned long)sc->uc_regs->gregs[PPC_REG_PC];
  pc = (unsigned long)UCONTEXT_REG(uc, PPC_REG_PC);
# elif TAU_BGQ
  //  201203 - Thanks to the Open|Speedshop team!
  pc = (unsigned long)((struct pt_regs *)(((&(uc->uc_mcontext))->regs))->nip);
# elif __x86_64__
  pc = (unsigned long)sc->rip;
# elif i386
  pc = (unsigned long)sc->eip;
# elif __ia64__
  pc = (unsigned long)sc->sc_ip;
# elif __powerpc64__
  // it could possibly be "link" - but that is supposed to be the return address.
  pc = (unsigned long)sc->regs->nip;
# elif __powerpc__
  // it could possibly be "link" - but that is supposed to be the return address.
  pc = (unsigned long)sc->regs->nip;
# elif __arm__
  pc = (unsigned long)sc->arm_pc;
# else
  issueUnavailableWarningIfNecessary("Warning, TAU Sampling does not work on unknown platform.\n");
  return 0;
# endif /* TAU_BGP */
  return pc;
#endif /* sun */
}

extern "C" void Tau_sampling_suspend(int tid) {
  //  int tid = RtsLayer::myThread();
  suspendSampling[tid] = 1;
  TAU_VERBOSE("Tau_sampling_suspend: on thread %d\n", tid);
}

extern "C" void Tau_sampling_resume(int tid) {
  //  int tid = RtsLayer::myThread();
  suspendSampling[tid] = 0;
  TAU_VERBOSE("Tau_sampling_resume: on thread %d\n", tid);
}

extern "C" void Tau_sampling_dlopen() {
  fprintf(stderr, "TAU: got a dlopen\n");
}

/*******************************************
 * EBS Tracing Input/Output Routines
 *******************************************/

void Tau_sampling_outputTraceHeader(int tid) {
  fprintf(ebsTrace[tid], "# Format version: 0.2\n");
  fprintf(
    ebsTrace[tid],
    "# $ | <timestamp> | <delta-begin> | <delta-end> | <metric 1> ... <metric N> | <tau callpath> | <location> [ PC callstack ]\n");
  fprintf(
    ebsTrace[tid],
    "# % | <delta-begin metric 1> ... <delta-begin metric N> | <delta-end metric 1> ... <delta-end metric N> | <tau callpath>\n");
  fprintf(ebsTrace[tid], "# Metrics:");
  for (int i = 0; i < Tau_Global_numCounters; i++) {
    const char *name = TauMetrics_getMetricName(i);
    fprintf(ebsTrace[tid], " %s", name);
  }
  fprintf(ebsTrace[tid], "\n");
}

void Tau_sampling_outputTraceCallpath(int tid) {
  Profiler *profiler = TauInternal_CurrentProfiler(tid);
  // *CWL* 2012/3/18 - EBS traces cannot handle callsites for now. Do not track.
  if ((profiler->CallPathFunction != NULL) && (TauEnv_get_callpath())) {
    fprintf(ebsTrace[tid], "%ld", profiler->CallPathFunction->GetFunctionId());
  } else if (profiler->ThisFunction != NULL) {
    fprintf(ebsTrace[tid], "%ld", profiler->ThisFunction->GetFunctionId());
  }
}

void Tau_sampling_flushTraceRecord(int tid, TauSamplingRecord *record, 
				   void *pc, ucontext_t *context) {
  fprintf(ebsTrace[tid], "$ | %lld | ", record->timestamp);

#ifdef TAU_EXP_DISABLE_DELTAS
  fprintf(ebsTrace[tid], "0 | 0 | ");
#else
  fprintf(ebsTrace[tid], "%lld | %lld | ", record->deltaStart, record->deltaStop);
#endif

  for (int i = 0; i < Tau_Global_numCounters; i++) {
    fprintf(ebsTrace[tid], "%.16G ", record->counters[i]);
  }

  fprintf(ebsTrace[tid], "| ");

  /* *CWL* - consider a check for TauEnv_get_callpath() here */
  Tau_sampling_outputTraceCallpath(tid);

  fprintf(ebsTrace[tid], " | %p", record->pc);

#ifdef TAU_UNWIND
  if (TauEnv_get_ebs_unwind() == 1) {
    Tau_sampling_outputTraceCallstack(tid, pc, context);
  }
#endif /* TAU_UNWIND */

  fprintf(ebsTrace[tid], "");
}

void Tau_sampling_outputTraceStop(int tid, Profiler *profiler, 
				  double *stopTime) {
  fprintf(ebsTrace[tid], "%% | ");

  for (int i = 0; i < Tau_Global_numCounters; i++) {
    double startTime = profiler->StartTime[i]; // gtod must be counter 0
    x_uint64 start = (x_uint64)startTime;
    fprintf(ebsTrace[tid], "%lld ", start);
  }
  fprintf(ebsTrace[tid], "| ");

  for (int i = 0; i < Tau_Global_numCounters; i++) {
    x_uint64 stop = (x_uint64)stopTime[i];
    fprintf(ebsTrace[tid], "%lld ", stop);
  }
  fprintf(ebsTrace[tid], "| ");

  Tau_sampling_outputTraceCallpath(tid);
  fprintf(ebsTrace[tid], "\n");
}

/*********************************************************************
 * Write Maps file for EBS Traces
 ********************************************************************/
int Tau_sampling_write_maps(int tid, int restart) {
  const char *profiledir = TauEnv_get_profiledir();

  int node = RtsLayer::myNode();
  node = 0;
  char filename[4096];
  sprintf(filename, "%s/ebstrace.map.%d.%d.%d.%d", profiledir, getpid(), node, RtsLayer::myContext(), tid);

  FILE *output = fopen(filename, "a");

  FILE *mapsfile = fopen("/proc/self/maps", "r");
  if (mapsfile == NULL) {
    return -1;
  }

  char line[4096];
  while (!feof(mapsfile)) {
    fgets(line, 4096, mapsfile);
    unsigned long start, end, offset;
    char module[4096];
    char perms[5];
    module[0] = 0;

    sscanf(line, "%lx-%lx %s %lx %*s %*u %[^\n]", &start, &end, perms, &offset, module);

    if (*module && ((strcmp(perms, "r-xp") == 0) || (strcmp(perms, "rwxp") == 0))) {
      fprintf(output, "%s %p %p %d\n", module, start, end, offset);
    }
  }
  fclose(output);

  return 0;
}

void Tau_sampling_outputTraceDefinitions(int tid) {
  const char *profiledir = TauEnv_get_profiledir();
  char filename[4096];
  int node = RtsLayer::myNode();
  node = 0;
  sprintf(filename, "%s/ebstrace.def.%d.%d.%d.%d", profiledir, getpid(), node, RtsLayer::myContext(), tid);

  FILE *def = fopen(filename, "w");

  fprintf(def, "# Format:\n");
  fprintf(def, "# <id> | <name>\n");

  for (vector<FunctionInfo *>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    if (strlen(fi->GetType()) > 0) {
      fprintf(def, "%ld | %s %s\n", fi->GetFunctionId(), fi->GetName(), fi->GetType());
    } else {
      fprintf(def, "%ld | %s\n", fi->GetFunctionId(), fi->GetName());
    }
  }
  fclose(def);

  /* write out the executable name at the end */
  char buffer[4096];
  bzero(buffer, 4096);
  int rc = readlink("/proc/self/exe", buffer, 4096);
  if (rc == -1) {
    fprintf(stderr, "TAU Sampling: Error, unable to read /proc/self/exe\n");
  } else {
    buffer[rc] = 0;
    fprintf(ebsTrace[tid], "# exe: %s\n", buffer);
  }

  /* write out the node number */
  fprintf(ebsTrace[tid], "# node: %d\n", RtsLayer::myNode());
  fprintf(ebsTrace[tid], "# thread: %d\n", tid);

  fclose(ebsTrace[tid]);

#if (defined (TAU_BGP) || (TAU_BGQ))
  /* do nothing */
#else
  Tau_sampling_write_maps(tid, 0);
#endif /* TAU_BGP || TAU_BGQ */

}

void Tau_sampling_handle_sampleTrace(void *pc, ucontext_t *context) {
  int tid = RtsLayer::myThread();
  Tau_global_incr_insideTAU_tid(tid);

#ifdef TAU_USE_HPCTOOLKIT
  // *CWL* - special case for HPCToolkit because it relies on 
  //         the runtime, or unwinding does not happen.
  if (hpctoolkit_process_started == 0) {
    printf("nope, quitting\n");
    return;
  }
#endif /* TAU_USE_HPCTOOLKIT */

  TauSamplingRecord theRecord;
  Profiler *profiler = TauInternal_CurrentProfiler(tid);

  // *CWL* - allow for debug only. Too much info.
  //  TAU_VERBOSE("[tid=%d] trace sample with pc %p\n", tid, pc);

  struct timeval tp;
  gettimeofday(&tp, 0);
  x_uint64 timestamp = ((x_uint64)tp.tv_sec * (x_uint64)1e6 + (x_uint64)tp.tv_usec);

  theRecord.timestamp = timestamp;
  theRecord.pc = (unsigned long)pc;
  theRecord.deltaStart = 0;
  theRecord.deltaStop = 0;

  double startTime = profiler->StartTime[0]; // gtod must be counter 0
  theRecord.deltaStart = (x_uint64)startTime;
  theRecord.deltaStop = 0;

  double values[TAU_MAX_COUNTERS];
  TauMetrics_getMetrics(tid, values);
  for (int i = 0; i < Tau_Global_numCounters; i++) {
    theRecord.counters[i] = values[i];
    startTime = profiler->StartTime[i];
    theRecord.counterDeltaStart[i] = (x_uint64)startTime;
    theRecord.counterDeltaStop[i] = 0;
  }

  Tau_sampling_flushTraceRecord(tid, &theRecord, pc, context);

  /* set this to get the stop event */
  profiler->needToRecordStop = 1;

  /* if we are doing EBS sampling, set whether we want inclusive samples */
  /* that is, main->foo->mpi_XXX is a sample for main, foo and mpi_xxx */
  if (TauEnv_get_ebs_inclusive() > 0) {
    profiler = (Profiler *)Tau_query_parent_event(profiler);
    while (profiler != NULL) {
      profiler->needToRecordStop = 1;
      profiler = (Profiler *)Tau_query_parent_event(profiler);
    }
  }

  Tau_global_decr_insideTAU_tid(tid);
}

/*********************************************************************
 * EBS Profiling Functions
 ********************************************************************/

void Tau_sampling_internal_initName2FuncInfoMapIfNecessary() {
  static bool name2FuncInfoMapInitialized = false;
  if (!name2FuncInfoMapInitialized) {
    RtsLayer::LockEnv();
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      name2FuncInfoMap[i] = NULL;
    }
    name2FuncInfoMapInitialized = true;
    RtsLayer::UnLockEnv();
  }
}

void Tau_sampling_internal_initPc2CallSiteMapIfNecessary() {
  static bool pc2CallSiteMapInitialized = false;
  if (!pc2CallSiteMapInitialized) {
    RtsLayer::LockEnv();
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      pc2CallSiteMap[i] = NULL;
    }
    pc2CallSiteMapInitialized = true;
    RtsLayer::UnLockEnv();
  }
}


char *Tau_sampling_getShortSampleName(const char *sampleName) {
  return NULL;
}

CallSiteInfo *Tau_sampling_resolveCallSite(unsigned long address,
					   const char *tag,
					   const char *childName,
					   char **newShortName,
					   bool addAddress) {
  CallSiteInfo *callsite;
  int bfdRet; // used only for an old interface

  unsigned long addr;
  if (!strcmp(tag,"UNWIND")) {
    // if we are dealing with callsites, adjust for the fact that the
    //   return address is the next instruction.
    addr = address-1;
  } else {
    addr = address;
  }

  char resolvedBuffer[4096];
  callsite = (CallSiteInfo *)malloc(sizeof(CallSiteInfo));

  callsite->pc = addr;
  // map current address to the corresponding module
  
  // resolved = Tau_sampling_resolveName(addr, &name, &resolvedModuleIdx);
  TauBfdInfo *resolvedInfo = NULL;
  // backup information in case we fail to resolve the address to specific
  //   line numbers.
  TauBfdAddrMap addressMap;
  sprintf(addressMap.name, "%s", "UNKNOWN");
#ifdef TAU_BFD
  // Attempt to use BFD to resolve names
  resolvedInfo = 
    Tau_bfd_resolveBfdInfo(bfdUnitHandle, (unsigned long)addr);
  // backup info
  bfdRet = Tau_bfd_getAddressMap(bfdUnitHandle, (unsigned long)addr,
				 &addressMap);
  if (resolvedInfo == NULL) {
      resolvedInfo = 
	  Tau_bfd_resolveBfdExecInfo(bfdUnitHandle, (unsigned long)addr);
      sprintf(addressMap.name, "%s", "EXEC");
  }
#endif /* TAU_BFD */
  if (resolvedInfo != NULL) {
    if (childName == NULL) {
      sprintf(resolvedBuffer, "[%s] %s [{%s} {%d}]",
	      tag,
	      resolvedInfo->funcname,
	      resolvedInfo->filename,
	      resolvedInfo->lineno);
    } else {
      sprintf(resolvedBuffer, "[%s] %s [@] %s [{%s} {%d}]",
	      tag,
	      childName,
	      resolvedInfo->funcname,
	      resolvedInfo->filename,
	      resolvedInfo->lineno);
    }
    // This will be reused later. Make sure to free after it is used.
    // strdup should not be used because we cannot guaranteed the allocation scheme.
    *newShortName = (char *)malloc(sizeof(char)*(strlen(resolvedInfo->funcname)+1));
    *newShortName = strcpy(*newShortName, resolvedInfo->funcname);
  } else {
    if (addAddress) {
      char tempAddrBuffer[32]; // not expecting more than 26 digits in addr
      if (childName == NULL) {
	sprintf(resolvedBuffer, "[%s] UNRESOLVED %s ADDR %p", 
		tag, addressMap.name, (void *)addr);
      } else {
	sprintf(resolvedBuffer, "[%s] [%s] [@] UNRESOLVED %s ADDR %p", 
		tag, childName, addressMap.name, (void *)addr);
      }
      sprintf(tempAddrBuffer, "ADDR %p", (void *)addr);
      *newShortName = (char *)malloc(sizeof(char)*(strlen(tempAddrBuffer)+1));
      *newShortName = strcpy(*newShortName, tempAddrBuffer);
    } else {
      if (childName == NULL) {
	sprintf(resolvedBuffer, "[%s] UNRESOLVED %s", 
		tag, addressMap.name);
      } else {
	sprintf(resolvedBuffer, "[%s] [%s] [@] UNRESOLVED %s", 
		tag, childName, addressMap.name);
      }
      *newShortName = (char *)malloc(sizeof(char)*(strlen("UNRESOLVED")+1));
      *newShortName = strcpy(*newShortName, "UNRESOLVED");
    }
  }
  //  printf("Address %p resolves to %s\n", address, resolvedBuffer);
  callsite->name = strdup(resolvedBuffer);
  return callsite;
}

char *Tau_sampling_getPathName(int index, CallStackInfo *callStack) {
  char buffer[4096];
  char *ret;
  vector<CallSiteInfo *> *sites = callStack->callSites;
  int startIdx;

  if (sites->size() <= 0) {
    fprintf(stderr, "ERROR: EBS attempted to access 0 length callstack\n");
    exit(-1);
  }
  if (index >= sites->size()) {
    fprintf(stderr, "ERROR: EBS attempted to access index %d of vector of length %d\n",
	    index, sites->size());
    exit(-1);
  }
  
  startIdx = sites->size()-1;
  strcpy(buffer, "");
  strcat(buffer, ((*sites)[startIdx])->name);
  for (int i=startIdx-1; i>=index; i--) {
    strcat(buffer, " => ");
    strcat(buffer, ((*sites)[i])->name);
  }
  ret = strdup(buffer);

  return ret;
}

CallStackInfo *Tau_sampling_resolveCallSites(const unsigned long *addresses) {
  CallStackInfo *callStack;
  bool addAddress = false;

  callStack = (CallStackInfo *)malloc(sizeof(CallStackInfo));
  callStack->callSites = new vector<CallSiteInfo *>();
    
  if (TauEnv_get_ebs_keep_unresolved_addr() == 1) {
    addAddress = true;
  }

  if (addresses == NULL) {
    return NULL;
  }
  int length = addresses[0];
  if (length < 1) {
    return NULL;
  }
  char *prevShortName = NULL;
  char *newShortName = NULL;
  callStack->callSites->push_back(Tau_sampling_resolveCallSite(addresses[1], 
							       "SAMPLE",
							       NULL,
							       &newShortName,
							       addAddress));
  // move the pointers
  if (newShortName != NULL) {
    prevShortName = newShortName;
    newShortName = NULL;
  }
  for (int i=1; i<length; i++) {
    unsigned long address = addresses[i+1];
    callStack->callSites->push_back(Tau_sampling_resolveCallSite(address, 
								 "UNWIND",
								 prevShortName,
								 &newShortName,
								 addAddress));
    // free the previous short name now.
    if (prevShortName != NULL) {
      free(prevShortName);
      if (newShortName != NULL) {
	prevShortName = newShortName;
      }
    }
    // move the pointers
    if (newShortName != NULL) {
      prevShortName = newShortName;
      newShortName = NULL;
    }
  }
  return callStack;
}

void Tau_sampling_eventStopProfile(int tid, Profiler *profiler,
				   double *stopTime) {
  // No activity required for Sampling Profiling at event stop for now.
}

char *Tau_sampling_internal_stripCallPath(const char *callpath) {
  char *pointer = NULL;
  char *temp = (char *)callpath;
  do {
    pointer = temp;
    temp = strstr(pointer,"=>");
    if (temp != NULL) {
      temp += 2;  // strip off the "=>"
      if (temp == NULL) {
	// takes care of case where string terminates with "=>"
	pointer = NULL;
      }
    }
  } while (temp != NULL);

  return strdup(pointer);
}

void Tau_sampling_finalizeProfile(int tid) {
  TAU_VERBOSE("Tau_sampling_finalizeProfile with tid=%d\n", tid);

  // Resolve all unresolved PC values.
  //
  // For resolution, each PC resolves to a unique CallSite tuple:
  //     filename X funcname X lineno
  // Each CallSite tuple maps to its own FunctionInfo object
  //

  // NOTE: This code ought to be at the start of a dlopen trap as well
  //       to take care of epoch changes.
  

#ifdef TAU_BFD
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_KEEP_GLOBALS);
  }
#endif /* TAU_BFD */

  // Iterate through all known FunctionInfo to acquire candidate callsites
  // for resolution.
  vector<CallSiteCandidate *> *candidates =
    new vector<CallSiteCandidate *>();
    
  RtsLayer::LockDB();
  // *CWL* NOTE: Cannot create intermediate FunctionInfo objects while
  //       we iterate TheFunctionDB()! Hence the candidates!
  for (vector<FunctionInfo *>::iterator fI_iter = TheFunctionDB().begin();
       fI_iter != TheFunctionDB().end(); fI_iter++) {
    FunctionInfo *parentTauContext = *fI_iter;
    if ((parentTauContext->pathHistogram[tid] == NULL) ||
	(parentTauContext->pathHistogram[tid]->size() == 0)) {
      // No samples encountered in this TAU context.
      //   Continue to next TAU context.
      TAU_VERBOSE("Tau Context %s has no samples.\n",
		  parentTauContext->GetName());
      continue;
    }

    /*
    printf("Sampled Parent %s has %d elements\n", parentTauContext->GetName(),
	   parentTauContext->pathHistogram[tid]->size());
    parentTauContext->pathHistogram[tid]->printTable();
    */
    pair<unsigned long *, unsigned long> *item = NULL;
    parentTauContext->pathHistogram[tid]->resetIter();
    item = parentTauContext->pathHistogram[tid]->nextIter();
    while (item != NULL) {
      // This is a placeholder for more generic pcStack extraction routines.
      CallSiteCandidate *candidate = new CallSiteCandidate();
      candidate->pcStack = item->first;
      /*
      for (int i=0; i<candidate->pcStack[0]; i++) {
	printf("%p ", candidate->pcStack[i+1]);
      }
      printf("\n");
      */
      candidate->sampleCount = item->second;
      candidate->tauContext = parentTauContext;
      //      printf("TESTING: context name [%s] has SAMPLES\n", candidate->tauContext->GetName());
      candidates->push_back(candidate);
      delete item;
      item = parentTauContext->pathHistogram[tid]->nextIter();
    }
  }
  RtsLayer::UnLockDB();

  // Initialization of maps for this thread if necessary.
  Tau_sampling_internal_initName2FuncInfoMapIfNecessary();
  if (name2FuncInfoMap[tid] == NULL) {
    name2FuncInfoMap[tid] = new map<string, FunctionInfo *>();
  }
  Tau_sampling_internal_initPc2CallSiteMapIfNecessary();
  if (pc2CallSiteMap[tid] == NULL) {
    pc2CallSiteMap[tid] = new map<unsigned long, CallSiteInfo *>();
  }
  
  // For each encountered sample PC in the non-empty TAU context,
  //
  //    resolve to the unique sample name as follows:
  //       <TAU Callpath Name> => <CallStack Path>
  //
  //    where <CallStack Path> is <CallSite> (=> <CallSite>)* and
  //       <CallSite> is:
  //       SAMPLE|UNWIND <funcname> [{filename} {lineno:colno}-{lineno:colno}]
  // 
  //    note that <CallStack Path> is the generalization of a sample
  //       whether or not stack unwinding is invoked.
  //
  vector<CallSiteCandidate *>::iterator cs_it;
  for (cs_it = candidates->begin(); cs_it != candidates->end(); cs_it++) {
    CallSiteCandidate *candidate = *cs_it;

    // STEP 0: Set up the metric values based on the candidate 
    //         to eventually be assigned to various FunctionInfo
    //         entities.
    double metricValue; 

    // Determine the EBS_SOURCE metric index and update the appropriate
    //   sample approximations.
    int ebsSourceMetricIndex = 
      TauMetrics_getMetricIndexFromName(TauEnv_get_ebs_source());
    if (ebsSourceMetricIndex == -1) {
      // *CWL* - Force it to be 0 and hope for the best.
      ebsSourceMetricIndex = 0;
    }
    unsigned int binFreq = candidate->sampleCount;
    metricValue = binFreq*TauEnv_get_ebs_period();

    // *CWL* - BFD is thread unsafe.
    RtsLayer::LockDB();
    // STEP 1: Resolve all addresses in a PC Stack.
    CallStackInfo *callStack =
      Tau_sampling_resolveCallSites(candidate->pcStack);
    RtsLayer::UnLockDB();

    // Name-to-function map iterator. To be shared for intermediate and callsite
    //   scenarios.
    map<string, FunctionInfo *>::iterator fi_it;

    // STEP 2: Find out if the Intermediate node for this candidate 
    //         has been created. Intermediate nodes need to be handled
    //         in a persistent mode across candidates.
    FunctionInfo *intermediateGlobalLeaf = NULL;
    FunctionInfo *intermediatePathLeaf = NULL;
    char intermediateGlobalLeafName[4096];
    char intermediatePathLeafName[4096];
    string *intermediateGlobalLeafString;
    string *intermediatePathLeafString;

    // STEP 2a: Locate or create Leaf Entry
    sprintf(intermediateGlobalLeafName, "[INTERMEDIATE] %s",
	    Tau_sampling_internal_stripCallPath(candidate->tauContext->GetName()));
    intermediateGlobalLeafString = new string(intermediateGlobalLeafName);
    fi_it = name2FuncInfoMap[tid]->find(*intermediateGlobalLeafString);
    if (fi_it == name2FuncInfoMap[tid]->end()) {
      string grname = string("TAU_SAMPLE | ") + string(candidate->tauContext->GetAllGroups());
      // Create the FunctionInfo object for the leaf Intermediate object.
      RtsLayer::LockDB();
      intermediateGlobalLeaf = 
	new FunctionInfo((const char*)intermediateGlobalLeafName,
			 candidate->tauContext->GetType(),
			 candidate->tauContext->GetProfileGroup(),
			 (const char*)grname.c_str(), true);
      RtsLayer::UnLockDB();
      name2FuncInfoMap[tid]->insert(std::pair<string,FunctionInfo*>(*intermediateGlobalLeafString, intermediateGlobalLeaf));
    } else {
      intermediateGlobalLeaf = ((FunctionInfo *)fi_it->second);
    }

    // Step 2b: Locate or create Full Path Entry. Requires name
    //   information about the Leaf Entry available.
    sprintf(intermediatePathLeafName, "%s %s => %s",
	    candidate->tauContext->GetName(),
	    candidate->tauContext->GetType(), intermediateGlobalLeafName);
    intermediatePathLeafString = new string(intermediatePathLeafName);
    fi_it = name2FuncInfoMap[tid]->find(*intermediatePathLeafString);
    if (fi_it == name2FuncInfoMap[tid]->end()) {
      string grname = string("TAU_SAMPLE  | ") + string(candidate->tauContext->GetAllGroups());
      // Create the FunctionInfo object for the leaf Intermediate object.
      RtsLayer::LockDB();
      intermediatePathLeaf = 
	new FunctionInfo((const char*)intermediatePathLeafName,
			 candidate->tauContext->GetType(),
			 candidate->tauContext->GetProfileGroup(),
			 (const char*)grname.c_str(), true);
      RtsLayer::UnLockDB();
      name2FuncInfoMap[tid]->insert(std::pair<string,FunctionInfo*>(*intermediatePathLeafString, intermediatePathLeaf));
    } else {
      intermediatePathLeaf = ((FunctionInfo *)fi_it->second);
    }
    // Accumulate the histogram into the Intermediate FunctionInfo objects.
    intermediatePathLeaf->SetCalls(tid, intermediatePathLeaf->GetCalls(tid)+binFreq);
    intermediateGlobalLeaf->SetCalls(tid, intermediateGlobalLeaf->GetCalls(tid)+binFreq);
    intermediatePathLeaf->AddInclTimeForCounter(metricValue, tid, ebsSourceMetricIndex);
    intermediateGlobalLeaf->AddInclTimeForCounter(metricValue, tid, ebsSourceMetricIndex);

    // STEP 3: For each sample, construct all FunctionInfo objects
    //    associated with the unwound addresses and the PC.
    //
    // Intermediate FunctionInfo objects must be found or created 
    //    at this time.
    //
    // For Each Address
    //   1. Check and Create Leaf Entry
    //   2. Check and Create Path Entry (Requires Intermediate)
    vector<CallSiteInfo *> *sites = callStack->callSites;
    // *CWL* - we need the index, which is why the iterator is not used.
    for (int i=0; i<sites->size(); i++) {
      string samplePathLeafString = Tau_sampling_getPathName(i, callStack);
      string sampleGlobalLeafString = ((*sites)[i])->name;
      FunctionInfo *samplePathLeaf = NULL;
      FunctionInfo *sampleGlobalLeaf = NULL;
      
      fi_it = name2FuncInfoMap[tid]->find(sampleGlobalLeafString);
      if (fi_it == name2FuncInfoMap[tid]->end()) {
	string grname = string("TAU_SAMPLE | ") + string(candidate->tauContext->GetAllGroups());
	RtsLayer::LockDB();
	sampleGlobalLeaf = 
	  new FunctionInfo((const char*)sampleGlobalLeafString.c_str(),
			   candidate->tauContext->GetType(),
			   candidate->tauContext->GetProfileGroup(),
			   (const char*)grname.c_str(), true);
	RtsLayer::UnLockDB();
	name2FuncInfoMap[tid]->insert(std::pair<string,FunctionInfo*>(sampleGlobalLeafString, sampleGlobalLeaf));
      } else {
	sampleGlobalLeaf = ((FunctionInfo *)fi_it->second);
      }
      
      char call_site_key[4096];
      sprintf(call_site_key,"%s %s => %s",
	      // *CWL* - ALREADY THERE in the intermediate nodes!
	      //	    candidate->tauContext->GetName(),
	      //	    candidate->tauContext->GetType(),
	      intermediatePathLeafString->c_str(),
	      candidate->tauContext->GetType(),
	      samplePathLeafString.c_str());
      // try to find the key
      string *callSiteKeyName = new string(call_site_key);
      fi_it = name2FuncInfoMap[tid]->find(*callSiteKeyName);
      if (fi_it == name2FuncInfoMap[tid]->end()) {
	string grname = string("TAU_SAMPLE | ") + string(candidate->tauContext->GetAllGroups()); 
	RtsLayer::LockDB();
	samplePathLeaf =
	  new FunctionInfo((const char*)callSiteKeyName->c_str(), "",
			   candidate->tauContext->GetProfileGroup(),
			   (const char*)grname.c_str(), true);
	RtsLayer::UnLockDB();
	name2FuncInfoMap[tid]->insert(std::pair<string,FunctionInfo*>(*callSiteKeyName, samplePathLeaf));
      } else {
      // found.
	samplePathLeaf = ((FunctionInfo *)fi_it->second);
      }
      
      // Update the count and time for the end of the path for sampled event.
      samplePathLeaf->SetCalls(tid, samplePathLeaf->GetCalls(tid)+binFreq);
      samplePathLeaf->AddInclTimeForCounter(metricValue, tid, ebsSourceMetricIndex);
      // Exclusive times are only incremented for actual sample data 
      //   and not unwound data
      if (i == 0) {
	samplePathLeaf->AddExclTimeForCounter(metricValue, tid, ebsSourceMetricIndex);
      }
      // Accumulate the count and time into the global leaf representative sampled event.
      sampleGlobalLeaf->SetCalls(tid, sampleGlobalLeaf->GetCalls(tid)+binFreq);
      sampleGlobalLeaf->AddInclTimeForCounter(metricValue, tid, ebsSourceMetricIndex);
      if (i == 0) {
	sampleGlobalLeaf->AddExclTimeForCounter(metricValue, tid, ebsSourceMetricIndex);
      }
    }
  }

  // Write out Metadata. 
  //
  // *CWL* - overload node numbers (not scalable in ParaProf display) in
  //         preparation for a more scalable way of displaying per-node
  //         metadata information.
  //
  char tmpstr[512];
  char tmpname[512];
  sprintf(tmpname, "TAU_EBS_SAMPLES_TAKEN_%d", tid);
  sprintf(tmpstr, "%lld", numSamples[tid]);
  TAU_METADATA(tmpname, tmpstr);

  sprintf(tmpname, "TAU_EBS_SAMPLES_DROPPED_TAU_%d", tid);
  sprintf(tmpstr, "%lld", samplesDroppedTau[tid]);
  TAU_METADATA(tmpname, tmpstr);

  sprintf(tmpname, "TAU_EBS_SAMPLES_DROPPED_SUSPENDED_%d", tid);
  sprintf(tmpstr, "%lld", samplesDroppedSuspended[tid]);
  TAU_METADATA(tmpname, tmpstr);
}

void Tau_sampling_handle_sampleProfile(void *pc, ucontext_t *context) {
  int tid = RtsLayer::myThread();
  Tau_global_incr_insideTAU_tid(tid);

  // *CWL* - Too "noisy" and useless a verbose output.
  //TAU_VERBOSE("[tid=%d] EBS profile sample with pc %p\n", tid, (unsigned long)pc);
  Profiler *profiler = TauInternal_CurrentProfiler(tid);
  FunctionInfo *samplingContext;

  // ok to be temporary. Hash table on the other end will copy the details.
  unsigned long pcStack[TAU_SAMP_NUM_ADDRESSES+1];
  for (int i=0; i<TAU_SAMP_NUM_ADDRESSES+1; i++) {
    pcStack[i] = 0;
  }
#ifdef TAU_UNWIND
  if (TauEnv_get_ebs_unwind() == 1) {
    Tau_sampling_unwind(tid, profiler, pc, context, pcStack);
  } else {
    pcStack[0] = 1;
    pcStack[1] = (unsigned long)pc;
  }
#else
  pcStack[0] = 1;
  pcStack[1] = (unsigned long)pc;
#endif /* TAU_UNWIND */

  if (TauEnv_get_callsite() && (profiler->CallSiteFunction != NULL)) {
    samplingContext = profiler->CallSiteFunction;
  } else if (TauEnv_get_callpath() && (profiler->CallPathFunction != NULL)) {
    samplingContext = profiler->CallPathFunction;
  } else {
    samplingContext = profiler->ThisFunction;
  }
  samplingContext->addPcSample(pcStack, tid);

  Tau_global_decr_insideTAU_tid(tid);
}

/*********************************************************************
 * Event triggers
 ********************************************************************/

/* Various unwinders might have their own implementation */
void Tau_sampling_event_start(int tid, void **addresses) {

  Tau_global_incr_insideTAU_tid(tid);

  TAU_VERBOSE("Tau_sampling_event_start: tid = %d address = %p\n", tid, addresses);

  //#ifdef TAU_USE_HPCTOOLKIT
  //  Tau_sampling_event_startHpctoolkit(tid, addresses);
  //#endif /* TAU_USE_HPCTOOLKIT */

  // This is undefined when no unwind capability has been linked into TAU
#ifdef TAU_UNWIND
  if (TauEnv_get_ebs_unwind() == 1) {
    Tau_sampling_unwindTauContext(tid, addresses);
  }
#endif /* TAU_UNWIND */

  if (TauEnv_get_profiling()) {
    // nothing for now
  }
  Tau_global_decr_insideTAU_tid(tid);
}

int Tau_sampling_event_stop(int tid, double *stopTime) {
#ifdef TAU_EXP_DISABLE_DELTAS
  return 0;
#endif

  Tau_global_incr_insideTAU_tid(tid);

  samplingEnabled[tid] = 0;

  Profiler *profiler = TauInternal_CurrentProfiler(tid);

  if (TauEnv_get_tracing()) {
    if (!profiler->needToRecordStop) {
      samplingEnabled[tid] = 1;
      return 0;
    }
    Tau_sampling_outputTraceStop(tid, profiler, stopTime);
  }

  if (TauEnv_get_profiling()) {
    Tau_sampling_eventStopProfile(tid, profiler, stopTime);
  }

  samplingEnabled[tid] = 1;
  Tau_global_decr_insideTAU_tid(tid);
  return 0;
}

/*********************************************************************
 * Sample Handling
 ********************************************************************/
void Tau_sampling_handle_sample(void *pc, ucontext_t *context) {
  int tid = RtsLayer::myThread();
  /* *CWL* too fine-grained for anything but debug.
  TAU_VERBOSE("Tau_sampling_handle_sample: tid=%d got sample [%p]\n",
  	      tid, (unsigned long)pc);
  */
  if (samplingEnabled[tid] == 0) {
    // Do not track counts when sampling is not enabled.
    TAU_VERBOSE("Tau_sampling_handle_sample: sampling not enabled\n");
    return;
  }
  numSamples[tid]++;

  /* Never sample anything internal to TAU */
  if (Tau_global_get_insideTAU_tid(tid) > 0) {
    samplesDroppedTau[tid]++;
    return;
  }

  if (suspendSampling[tid]) {
    samplesDroppedSuspended[tid]++;
    return;
  }

  Tau_global_incr_insideTAU_tid(tid);
  if (TauEnv_get_tracing()) {
    Tau_sampling_handle_sampleTrace(pc, context);
  }

  if (TauEnv_get_profiling()) {
    Tau_sampling_handle_sampleProfile(pc, context);
  }
  Tau_global_decr_insideTAU_tid(tid);
}

/*********************************************************************
 * Handler for itimer interrupt
 ********************************************************************/
void Tau_sampling_handler(int signum, siginfo_t *si, void *context) {
  unsigned long pc;
  pc = get_pc(context);

  //   TAU_VERBOSE("Tau_sampling_handler invoked\n");
  Tau_sampling_handle_sample((void *)pc, (ucontext_t *)context);
}

/*********************************************************************
 * PAPI Overflow handler
 ********************************************************************/
void Tau_sampling_papi_overflow_handler(int EventSet, void *address, x_int64 overflow_vector, void *context) {
  int tid = RtsLayer::myThread();
//   fprintf(stderr,"[%d] Overflow at %p! bit=0x%llx \n", tid, address,overflow_vector);

  x_int64 value = (x_int64)address;

  if ((value & 0xffffffffff000000ll) == 0xffffffffff000000ll) {
    return;
  }

  Tau_sampling_handle_sample(address, (ucontext_t *)context);
}

/*********************************************************************
 * Initialize the sampling trace system
 ********************************************************************/
int Tau_sampling_init(int tid) {
  int ret;

  static struct itimerval itval;


  Tau_global_incr_insideTAU_tid(tid);

  //int threshold = 1000;
  int threshold = TauEnv_get_ebs_period();
  TAU_VERBOSE("Tau_sampling_init: tid = %d with threshold %d\n", 
	      tid, threshold);

  samplingEnabled[tid] = 0;
  suspendSampling[tid] = 0;
  numSamples[tid] = 0;
  samplesDroppedTau[tid] = 0;
  samplesDroppedSuspended[tid] = 0;
  
  itval.it_interval.tv_usec = itval.it_value.tv_usec = threshold % 1000000;
  itval.it_interval.tv_sec =  itval.it_value.tv_sec = threshold / 1000000;

  const char *profiledir = TauEnv_get_profiledir();

  int node = RtsLayer::myNode();
  node = 0;
  char filename[4096];

  if (TauEnv_get_tracing()) {
    sprintf(filename, "%s/ebstrace.raw.%d.%d.%d.%d", profiledir, getpid(), node, RtsLayer::myContext(), tid);
    
    ebsTrace[tid] = fopen(filename, "w");
    if (ebsTrace[tid] == NULL) {
      fprintf(stderr, "Tau Sampling Error: Unable to open %s for writing\n", filename);
      exit(-1);
    }
    
    Tau_sampling_outputTraceHeader(tid);
  }

  // Nothing currently requires initialization work for sampling into
  //   profiles.
  /*
  if (TauEnv_get_profiling()) {
  }
  */
/*
   see:
   http://ftp.gnu.org/old-gnu/Manuals/glibc-2.2.3/html_node/libc_463.html#SEC473
   for details.  When using SIGALRM and ITIMER_REAL on MareNostrum (Linux on
   PPC970MP) the network barfs.  When using ITIMER_PROF and SIGPROF, everything
   was fine...
   //int which = ITIMER_REAL;
   //int alarmType = SIGALRM;
 */

  // int which = ITIMER_PROF;
  // int alarmType = SIGPROF;

  int which = ITIMER_REAL;
  int alarmType = SIGALRM;
  
  /*  *CWL* - NOTE: It is fine to establish the timer interrupts here
      (and the PAPI overflow interrupts elsewhere) only because we
      enable sample handling for each thread after init(tid) completes.
      See Tau_sampling_handle_sample().
   */
  // only thread 0 sets up the timer interrupts.
  if ((strcmp(TauEnv_get_ebs_source(), "itimer") == 0) && (tid == 0)) {
    struct sigaction act;

    // If TIME isn't on the list of TAU_METRICS, then do not sample.
    // Eventually, we could employ a best-effort attempt to add 
    //   TAU_EBS_SOURCE to TAU_METRICS if TAU_EBS_SOURCE is not a 
    //   a member of TAU_METRICS.
    int checkVal = TauMetrics_getMetricIndexFromName("TIME");
    if (checkVal == -1) {
      fprintf(stderr, "TAU Sampling Warning: TIME is not a member of TAU_METRICS. No sampling is enabled.\n");
      return -1;
    }

    memset(&act, 0, sizeof(struct sigaction));
    ret = sigemptyset(&act.sa_mask);
    if (ret != 0) {
      fprintf(stderr, "TAU: Sampling error: %s\n", strerror(ret));
      return -1;
    }
    ret = sigaddset(&act.sa_mask, alarmType);
    if (ret != 0) {
      fprintf(stderr, "TAU: Sampling error: %s\n", strerror(ret));
      return -1;
    }
    act.sa_sigaction = Tau_sampling_handler;
    act.sa_flags     = SA_SIGINFO|SA_RESTART;
    
    ret = sigaction(alarmType, &act, NULL);
    if (ret != 0) {
      fprintf(stderr, "TAU: Sampling error: %s\n", strerror(ret));
      return -1;
    }
    
    struct itimerval ovalue, pvalue;
    getitimer(which, &pvalue);
    
    ret = setitimer(which, &itval, &ovalue);
    if (ret != 0) {
      fprintf(stderr, "TAU: Sampling error: %s\n", strerror(ret));
      return -1;
    }
    
    if (ovalue.it_interval.tv_sec != pvalue.it_interval.tv_sec  ||
	ovalue.it_interval.tv_usec != pvalue.it_interval.tv_usec ||
	ovalue.it_value.tv_sec != pvalue.it_value.tv_sec ||
	ovalue.it_value.tv_usec != pvalue.it_value.tv_usec) {
      fprintf(stderr,"TAU [tid = %d]: Sampling error - Real time interval timer mismatch.\n", tid);
      return -1;
    }
    TAU_VERBOSE("Tau_sampling_init: pid = %d, tid = %d Signals set up.\n", getpid(), tid);
  }

  samplingEnabled[tid] = 1;
  Tau_global_decr_insideTAU_tid(tid);
  return 0;
}

/*********************************************************************
 * Finalize the sampling trace system
 ********************************************************************/
int Tau_sampling_finalize(int tid) {
  /* *CWL* - The reason for the following code is that we have multiple
     places in TAU from which finalization happens. We respect only the
     first instance. Right now, we should not have issues with the
     fact that this is not a per-thread construct.
  */
  TAU_VERBOSE("Tau_sampling_finalize tid=%d\n", tid);

  if (TauEnv_get_tracing()) {
    if (ebsTrace[tid] == 0) {
      return 0;
    }
  }

  Tau_global_incr_insideTAU_tid(tid);

  /* Disable sampling first */
  samplingEnabled[tid] = 0;

  struct itimerval itval;
  int ret;

  if (tid == 0) {
    // no timers to unset if on thread 0
    itval.it_interval.tv_usec = itval.it_value.tv_usec =
      itval.it_interval.tv_sec = itval.it_value.tv_sec = 0;
    
    ret = setitimer(ITIMER_REAL, &itval, 0);
    if (ret != 0) {
      /* ERROR */
    }
  }

  if (TauEnv_get_tracing()) {
    Tau_sampling_outputTraceDefinitions(tid);
  }

  if (TauEnv_get_profiling()) {
    Tau_sampling_finalizeProfile(tid);
  }

  Tau_global_decr_insideTAU_tid(tid);

  return 0;
}

/* *CWL* - This is workaround code for MPI where mvapich2 on Hera was
   found to conflict with EBS sampling operations if EBS was initialized
   before MPI_Init().

   Assume no threading in this debug version.
 */
extern "C" void Tau_sampling_init_if_necessary(void) {
  static bool initialized = false;
  static bool thrInitialized[TAU_MAX_THREADS];

  if (!initialized) {
    RtsLayer::LockEnv();
    // check again, someone else might already have initialized by now.
    if (!initialized) {
      //      printf("Sampling global initializing!\n");
      for (int i=0; i<TAU_MAX_THREADS; i++) {
	thrInitialized[i] = false;
      }
      initialized = true;
    }
    RtsLayer::UnLockEnv();
  }

  int myTid = RtsLayer::myThread();

  if (!thrInitialized[myTid]) {
    //    printf("Sampling thread %d initializing!\n", myTid);
    Tau_sampling_init(myTid);
    thrInitialized[myTid] = true;
  }
}

/* *CWL* - This is a preliminary attempt to allow MPI wrappers to invoke
   sampling finalization and name resolution for all threads through
   MPI_Finalize before the process of TAU event unification.
 */
extern "C" void Tau_sampling_finalize_if_necessary(void) {
  static bool finalized = false;
  static bool thrFinalized[TAU_MAX_THREADS];

  if (!finalized) {
    RtsLayer::LockEnv();
    // check again, someone else might already have finalized by now.
    if (!finalized) {
      //      printf("Sampling global finalizing!\n");
      for (int i=0; i<TAU_MAX_THREADS; i++) {
	thrFinalized[i] = false;
      }
      finalized = true;
    }
    RtsLayer::UnLockEnv();
  }

  int myTid = RtsLayer::myThread();

  if (!thrFinalized[myTid]) {
    //    printf("Sampling thread %d finalizing!\n", myTid);
    Tau_sampling_finalize(myTid);
    thrFinalized[myTid] = true;
  }
}


#endif //TAU_WINDOWS
