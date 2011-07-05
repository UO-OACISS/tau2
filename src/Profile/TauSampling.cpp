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
#ifndef TAU_WINDOWS
#include <ucontext.h>
#endif

/*********************************************************************
 * Tau Sampling Record Definition
 ********************************************************************/
typedef struct {
  caddr_t pc;
  x_uint64 timestamp;
  double counters[TAU_MAX_COUNTERS];
  double counterDeltaStart[TAU_MAX_COUNTERS];
  double counterDeltaStop[TAU_MAX_COUNTERS];
  x_uint64 deltaStart;
  x_uint64 deltaStop;
} TauSamplingRecord;

typedef struct {
  caddr_t pc; // should be a list for callsite paths
  unsigned int sampleCount;
  FunctionInfo *tauContext;
} CallSiteCandidate;

typedef struct {
  caddr_t pc; // should be a list for callsite paths
  caddr_t relative_pc;
  int moduleIdx;
  char *name;
} CallSiteInfo;

/*********************************************************************
 * Global Variables
 ********************************************************************/

// map for pc to FunctionInfo objects
static map<caddr_t, FunctionInfo *> *pc2FuncInfoMap[TAU_MAX_THREADS];
static map<string, FunctionInfo *> *callsite2FuncInfoMap[TAU_MAX_THREADS];

// For BFD-based name resolution
static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;

/* The trace for this node, mulithreaded execution currently not supported */
FILE *ebsTrace[TAU_MAX_THREADS];

/* Sample processing enabled/disabled */
int samplingEnabled[TAU_MAX_THREADS];
class initEnableFlags {
public:
initEnableFlags() {
  for (int i = 0; i < TAU_MAX_THREADS; i++) {
    samplingEnabled[i] = 0;
  }
}
};
initEnableFlags enableFlagsInitializer = initEnableFlags();

/* Sample processing suspended/resumed */
int suspendSampling[TAU_MAX_THREADS];
class initSuspendFlags {
public:
initSuspendFlags() {
  for (int i = 0; i < TAU_MAX_THREADS; i++) {
    suspendSampling[i] = 0;
  }
}
};
initSuspendFlags suspendFlagsInitializer = initSuspendFlags();


/*********************************************************************
 * Get the architecture specific PC
 ********************************************************************/

#if __WORDSIZE == 32
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.uc_regs->gregs[reg])
#else
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.gp_regs[reg])
#endif

#define PPC_REG_PC 32


static inline caddr_t get_pc(void *p) {
  struct ucontext *uc = (struct ucontext *)p;
  caddr_t pc;

#ifdef sun
  fprintf(stderr, "Warning, TAU Sampling does not work on solaris\n");
  return 0;
#elif __APPLE__
  fprintf(stderr, "Warning, TAU Sampling does not work on apple\n");
  return 0;
#elif _AIX
  fprintf(stderr, "Warning, TAU Sampling does not work on AIX\n");
  return 0;
#else
  struct sigcontext *sc;
  sc = (struct sigcontext *)&uc->uc_mcontext;
#ifdef TAU_BGP
  //  pc = (caddr_t)sc->uc_regs->gregs[PPC_REG_PC];
  pc = (caddr_t)UCONTEXT_REG(uc, PPC_REG_PC);
# elif __x86_64__
  pc = (caddr_t)sc->rip;
# elif i386
  pc = (caddr_t)sc->eip;
# elif __ia64__
  pc = (caddr_t)sc->sc_ip;
# elif __powerpc64__
  // it could possibly be "link" - but that is supposed to be the return address.
  pc = (caddr_t)sc->regs->nip;
# elif __powerpc__
  // it could possibly be "link" - but that is supposed to be the return address.
  pc = (caddr_t)sc->regs->nip;
# else
#  error "profile handler not defined for this architecture"
# endif /* TAU_BGP */
  return pc;
#endif /* sun */
}

extern "C" void Tau_sampling_suspend() {
  int tid = RtsLayer::myThread();
  suspendSampling[tid] = 1;
//   fprintf (stderr, "suspended sampling on thread %d\n", tid);
}

extern "C" void Tau_sampling_resume() {
  int tid = RtsLayer::myThread();
  suspendSampling[tid] = 0;
//   fprintf (stderr, "resumed sampling on thread %d\n", tid);
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
  if (profiler->CallPathFunction == NULL) {
    fprintf(ebsTrace[tid], "%ld", profiler->ThisFunction->GetFunctionId());
  } else {
    fprintf(ebsTrace[tid], "%ld", profiler->CallPathFunction->GetFunctionId());
  }
}
/*
void Tau_sampling_output_callpath_old(int tid) {
  TAU_QUERY_DECLARE_EVENT(event);
  const char *str;
  TAU_QUERY_GET_CURRENT_EVENT(event);
  TAU_QUERY_GET_EVENT_NAME(event, str);

  int depth = TauEnv_get_callpath_depth();
  if (depth < 1) {
    depth = 1;
  }

  while (str && depth > 0) {
    //    printf ("inside %s\n", str);

    Profiler *p = (Profiler *)event;
    fprintf(ebsTrace[tid], "%ld", p->ThisFunction->GetFunctionId());
    TAU_QUERY_GET_PARENT_EVENT(event);
    TAU_QUERY_GET_EVENT_NAME(event, str);
    if (str) {
      //fprintf (ebsTrace[tid], " : ", str);
      //fprintf(ebsTrace[tid], "  ", str);
      fprintf(ebsTrace[tid], " ");
    }
    depth--;
  }
}
*/

#if !defined(TAU_USE_LIBUNWIND) && !defined(TAU_USE_STACKWALKER) && !defined(TAU_USE_HPCTOOLKIT)
void Tau_sampling_outputTraceCallstack(int tid, void *pc, 
				       ucontext_t *context) {
  /* Default, do nothing */
}
#endif /* TAU_USE_LIBUNWIND */

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
    //fprintf(ebsTrace[tid], "%lld | ", record->counterDeltaStart[i]);
    //fprintf(ebsTrace[tid], "%lld | ", record->counterDeltaStop[i]);
  }

  fprintf(ebsTrace[tid], "| ");

  /* *CWL* - consider a check for TauEnv_get_callpath() here */
  Tau_sampling_outputTraceCallpath(tid);

  fprintf(ebsTrace[tid], " | %p", record->pc);

  Tau_sampling_outputTraceCallstack(tid, pc, context);

  fprintf(ebsTrace[tid], "\n");
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
    // printf ("=> %s", line);
    unsigned long start, end, offset;
    char module[4096];
    char perms[5];
    module[0] = 0;

    sscanf(line, "%lx-%lx %s %lx %*s %*u %[^\n]", &start, &end, perms, &offset, module);

    if (*module && ((strcmp(perms, "r-xp") == 0) || (strcmp(perms, "rwxp") == 0))) {
      // printf ("got %s, %p-%p (%d)\n", module, start, end, offset);
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

#ifndef TAU_BGP
  Tau_sampling_write_maps(tid, 0);
#endif

}

void Tau_sampling_handle_sampleTrace(void *pc, ucontext_t *context) {
  int tid = RtsLayer::myThread();
  Tau_global_incr_insideTAU_tid(tid);

#ifdef TAU_USE_HPCTOOLKIT
  if (hpctoolkit_process_started == 0) {
    printf("nope, quitting\n");
    return;
  }
#endif

  TauSamplingRecord theRecord;
  Profiler *profiler = TauInternal_CurrentProfiler(tid);

  TAU_VERBOSE("[tid=%d] trace sample with pc %p\n", tid, pc);

  struct timeval tp;
  gettimeofday(&tp, 0);
  x_uint64 timestamp = ((x_uint64)tp.tv_sec * (x_uint64)1e6 + (x_uint64)tp.tv_usec);

  theRecord.timestamp = timestamp;
  theRecord.pc = (caddr_t)pc;
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

void Tau_sampling_internal_initPc2FuncInfoMapIfNecessary() {
  static bool pc2FuncInfoMapInitialized = false;
  if (!pc2FuncInfoMapInitialized) {
    RtsLayer::LockEnv();
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      pc2FuncInfoMap[i] = NULL;
    }
    pc2FuncInfoMapInitialized = true;
    RtsLayer::UnLockEnv();
  }
}

void Tau_sampling_internal_initCallsite2FuncInfoMapIfNecessary() {
  static bool callsite2FuncInfoMapInitialized = false;
  if (!callsite2FuncInfoMapInitialized) {
    RtsLayer::LockEnv();
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      callsite2FuncInfoMap[i] = NULL;
    }
    callsite2FuncInfoMapInitialized = true;
    RtsLayer::UnLockEnv();
  }
}

CallSiteInfo *Tau_sampling_resolveCallSite(caddr_t addr) {
  CallSiteInfo *callsite;
  bool resolved = false;

  char resolvedBuffer[4096];

  callsite = (CallSiteInfo *)malloc(sizeof(CallSiteInfo));

  callsite->pc = addr;
  // map current address to the corresponding module
  
  // resolved = Tau_sampling_resolveName(addr, &name, &resolvedModuleIdx);
  TauBfdInfo *resolvedInfo = NULL;
#ifdef TAU_BFD
  resolvedInfo = 
    Tau_bfd_resolveBfdInfo(bfdUnitHandle, (unsigned long)addr);
  if (resolvedInfo == NULL) {
      resolvedInfo = 
	  Tau_bfd_resolveBfdExecInfo(bfdUnitHandle, (unsigned long)addr);
  }
#endif /* TAU_BFD */
  if (resolvedInfo != NULL) {
    sprintf(resolvedBuffer, "SAMPLE %s [{%s} {%d,%d}-{%d,%d}]",
	    resolvedInfo->funcname,
	    resolvedInfo->filename,
	    resolvedInfo->lineno, 0,
	    resolvedInfo->lineno, 0);
  } else {
    sprintf(resolvedBuffer, "SAMPLE UNRESOLVED ADDR %p", (unsigned long)addr);
  }
  callsite->name = strdup(resolvedBuffer);
  TAU_VERBOSE("Tau_sampling_resolveCallSite: Callsite name resolved to [%s]\n",
	      callsite->name);
  return callsite;
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
    if ((parentTauContext->pcHistogram == NULL) ||
	(parentTauContext->pcHistogram->size() == 0)) {
      // No samples encountered in this TAU context.
      //   Continue to next TAU context.
      TAU_VERBOSE("Tau Context %s has no samples.\n",
		  parentTauContext->GetName());
      continue;
    }
    map<caddr_t, unsigned int>::iterator it;
    for (it = parentTauContext->pcHistogram->begin();
	 it != parentTauContext->pcHistogram->end(); it++) {
      caddr_t addr = (caddr_t)it->first;
      CallSiteCandidate *candidate = new CallSiteCandidate();
      candidate->pc = addr;
      candidate->sampleCount = (unsigned int)it->second;
      candidate->tauContext = parentTauContext;
      candidates->push_back(candidate);
    }
  }
  RtsLayer::UnLockDB();

  vector<CallSiteCandidate *>::iterator cs_it;
  for (cs_it = candidates->begin(); cs_it != candidates->end(); cs_it++) {
    // For each encountered sample PC in the non-empty TAU context,
    //    resolve to the unique CallSite name as follows:
    //
    //       <TAU Callpath Name> => <CallSite Path>
    //
    //    where <CallSite Path> is <CallSite> (=> <CallSite>)* and
    //       <CallSite> is:
    //
    //       SAMPLE <funcname> [{filename} {lineno:colno}-{lineno:colno}]
    // 
    CallSiteCandidate *candidate = *cs_it;
    CallSiteInfo *callsite = 
      Tau_sampling_resolveCallSite(candidate->pc);
    char call_site_key[4096];

    // If there was a candidate, there is at least one sample.
    if (candidate->tauContext->ebsIntermediate == NULL) {
      // create the intermediate FunctionInfo object
      char intermediateName[4096];
      RtsLayer::LockDB();
      sprintf(intermediateName, "%s => INTERMEDIATE %s",
	      candidate->tauContext->GetName(),
	      Tau_sampling_internal_stripCallPath(candidate->tauContext->GetName()));
      TAU_VERBOSE("Tau_sampling_finalizeProfile: created intermediate node [%s]\n", intermediateName);
      string grname = string("SAMPLE | ") + 
	RtsLayer::PrimaryGroup(candidate->tauContext->GetAllGroups()); 
      candidate->tauContext->ebsIntermediate =
	new FunctionInfo((const char*)intermediateName, "",
			 candidate->tauContext->GetProfileGroup(),
			 (const char*)grname.c_str(), true);
      RtsLayer::UnLockDB();
    }
    sprintf(call_site_key,"%s => %s => %s",
	    candidate->tauContext->GetName(),
	    candidate->tauContext->ebsIntermediate->GetName(),
	    callsite->name);
    // try to find the key
    string *callSiteName = new string(call_site_key);
    // See if the callsite has been previously encountered.
    Tau_sampling_internal_initCallsite2FuncInfoMapIfNecessary();
    if (callsite2FuncInfoMap[tid] == NULL) {
      callsite2FuncInfoMap[tid] = new map<string, FunctionInfo *>();
    }
    map<string, FunctionInfo *>::iterator fi_it;
    FunctionInfo *sampledContextFuncInfo;
    fi_it = callsite2FuncInfoMap[tid]->find(*callSiteName);
    if (fi_it == callsite2FuncInfoMap[tid]->end()) {
      // not found - create new FunctionInfo object to be associated with
      //   newly resolved name.
      RtsLayer::LockDB();
      string grname = string("SAMPLE | ") + 
	RtsLayer::PrimaryGroup(candidate->tauContext->GetAllGroups()); 
      sampledContextFuncInfo =
	new FunctionInfo((const char*)callSiteName->c_str(), "",
			 candidate->tauContext->GetProfileGroup(),
			 (const char*)grname.c_str(), true);
      RtsLayer::UnLockDB();
    } else {
      // found.
      sampledContextFuncInfo = ((FunctionInfo *)fi_it->second);
    }
    // Accumulate the histogram into the located FunctionInfo object
    double *totalTime = 
      (double *)malloc(Tau_Global_numCounters*sizeof(double));
    for (int i=0; i<Tau_Global_numCounters; i++) {
      totalTime[i] = 0.0;
    }
    // work only with gtod for now
    unsigned int binFreq = candidate->sampleCount;
    totalTime[0] = binFreq*TauEnv_get_ebs_period();
    // Update the count and time for the group of sampled events.
    sampledContextFuncInfo->SetCalls(tid, binFreq);
    sampledContextFuncInfo->AddInclTime(totalTime, tid);
    sampledContextFuncInfo->AddExclTime(totalTime, tid);
    // Accumulate the count and time into the intermediate object
    FunctionInfo *intermediate = 
      candidate->tauContext->ebsIntermediate;
    intermediate->SetCalls(tid, intermediate->GetCalls(tid)+binFreq);
    intermediate->AddInclTime(totalTime, tid);
    intermediate->AddExclTime(totalTime, tid);
  }
}

void Tau_sampling_handle_sampleProfile(void *pc, ucontext_t *context) {
  int tid = RtsLayer::myThread();
  Tau_global_incr_insideTAU_tid(tid);

  TAU_VERBOSE("[tid=%d] EBS profile sample with pc %p\n", tid, (caddr_t)pc);
  Profiler *profiler = TauInternal_CurrentProfiler(tid);
  FunctionInfo *callSiteContext;

  if (TauEnv_get_callpath() && (profiler->CallPathFunction != NULL)) {
    callSiteContext = profiler->CallPathFunction;
  } else {
    callSiteContext = profiler->ThisFunction;
  }
  callSiteContext->addPcSample((caddr_t)pc);

  Tau_global_decr_insideTAU_tid(tid);
}

/*********************************************************************
 * Event triggers
 ********************************************************************/

/* Various unwinders might have their own implementation */
void Tau_sampling_event_start(int tid, void **addresses) {
#ifdef TAU_USE_HPCTOOLKIT
  Tau_sampling_event_startHpctoolkit(tid, addresses);
#endif /* TAU_USE_HPCTOOLKIT */

  if (TauEnv_get_profiling()) {
    // nothing for now
  }
}

int Tau_sampling_event_stop(int tid, double *stopTime) {
#ifdef TAU_EXP_DISABLE_DELTAS
  return 0;
#endif

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
  return 0;
}

/*********************************************************************
 * Sample Handling
 ********************************************************************/
void Tau_sampling_handle_sample(void *pc, ucontext_t *context) {
  int tid = RtsLayer::myThread();

  /* Never sample anything internal to TAU */
  if (Tau_global_get_insideTAU_tid(tid) > 0) {
    return;
  }

  if (suspendSampling[tid]) {
    return;
  }

  if (TauEnv_get_tracing()) {
    Tau_sampling_handle_sampleTrace(pc, context);
  }

  if (TauEnv_get_profiling()) {
    Tau_sampling_handle_sampleProfile(pc, context);
  }

}

/*********************************************************************
 * Handler for itimer interrupt
 ********************************************************************/
void Tau_sampling_handler(int signum, siginfo_t *si, void *context) {
  caddr_t pc;
  pc = get_pc(context);

  Tau_sampling_handle_sample(pc, (ucontext_t *)context);
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
  int i;

  //  printf ("init called! tid = %d\n", tid);
  static struct itimerval itval;

  //int threshold = 1000;
  int threshold = TauEnv_get_ebs_period();

  samplingEnabled[tid] = 0;

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

  if (TauEnv_get_profiling()) {
    // Do nothing for now.
  }

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

  /*  *CWL* - NOTE: It is fine to establish the timer interrupts here
      (and the PAPI overflow interrupts elsewhere) only because we
      enable sample handling for each thread after init(tid) completes.
      See Tau_sampling_handle_sample().
   */
  if (strcmp(TauEnv_get_ebs_source(), "itimer") == 0) {
    int which = ITIMER_REAL;
    int alarmType = SIGALRM;

    struct sigaction act;
    memset(&act, 0, sizeof(struct sigaction));
    ret = sigemptyset(&act.sa_mask);
    if (ret != 0) {
      printf("TAU: Sampling error: %s\n", strerror(ret));
      return -1;
    }
    ret = sigaddset(&act.sa_mask, alarmType);
    if (ret != 0) {
      printf("TAU: Sampling error: %s\n", strerror(ret));
      return -1;
    }
    act.sa_sigaction = Tau_sampling_handler;
    act.sa_flags     = SA_SIGINFO;

    ret = sigaction(alarmType, &act, NULL);
    if (ret != 0) {
      printf("TAU: Sampling error: %s\n", strerror(ret));
      return -1;
    }

    struct itimerval ovalue, pvalue;
    getitimer(which, &pvalue);

    ret = setitimer(which, &itval, &ovalue);
    if (ret != 0) {
      printf("TAU: Sampling error: %s\n", strerror(ret));
      return -1;
    }

    if (ovalue.it_interval.tv_sec != pvalue.it_interval.tv_sec  ||
        ovalue.it_interval.tv_usec != pvalue.it_interval.tv_usec ||
        ovalue.it_value.tv_sec != pvalue.it_value.tv_sec ||
        ovalue.it_value.tv_usec != pvalue.it_value.tv_usec) {
      printf("TAU: Sampling error: Real time interval timer mismatch\n");
      return -1;
    }
  }

  samplingEnabled[tid] = 1;
  return 0;
}

/*********************************************************************
 * Finalize the sampling trace system
 ********************************************************************/
int Tau_sampling_finalize(int tid) {
  TAU_VERBOSE("Tau_sampling_finalize tid=%d\n", tid);
  //  printf("Tau_sampling_finalize tid=%d\n", tid);

  if (TauEnv_get_tracing()) {
    if (ebsTrace[tid] == 0) {
      return 0;
    }
  }

  /* Disable sampling first */
  samplingEnabled[tid] = 0;

  struct itimerval itval;
  int ret;

  itval.it_interval.tv_usec = itval.it_value.tv_usec =
                                itval.it_interval.tv_sec = itval.it_value.tv_sec = 0;

  ret = setitimer(ITIMER_REAL, &itval, 0);
  if (ret != 0) {
    /* ERROR */
  }

  if (TauEnv_get_tracing()) {
    Tau_sampling_outputTraceDefinitions(tid);
  }

  if (TauEnv_get_profiling()) {
    Tau_sampling_finalizeProfile(tid);
  }

  return 0;
}

/* *CWL* - This is workaround code for MPI where mvapich2 on Hera was
   found to conflict with EBS sampling operations if EBS was initialized
   before MPI_Init()
 */
extern "C" void Tau_sampling_init_if_necessary(void ) {
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_init(RtsLayer::myThread());
  }
}

#endif //TAU_WINDOWS
