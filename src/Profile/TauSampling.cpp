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

#ifdef TAU_EXP_SAMPLING

#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <signal.h>

#include <TAU.h>
#include <Profile/TauMetrics.h>

/*********************************************************************
 * Tau Sampling Record Definition
 ********************************************************************/
typedef struct {
  caddr_t pc;
  x_uint64 timestamp;
  double counters[TAU_MAX_COUNTERS];
  x_uint64 deltaStart;
  x_uint64 deltaStop;
} TauSamplingRecord;

/*********************************************************************
 * Global Variables
 ********************************************************************/

/* The trace for this node, mulithreaded execution currently not supported */
FILE *ebsTrace;

/* Sample processing enabled/disabled */
int samplingEnabled;

/*********************************************************************
 * Get the architecture specific PC
 ********************************************************************/
static inline caddr_t get_pc(void *p) {
  struct ucontext *uc = (struct ucontext *)p;
  caddr_t pc;
  struct sigcontext *sc;
  sc = (struct sigcontext *)&uc->uc_mcontext;
# ifdef __x86_64__
  pc = (caddr_t)sc->rip;
# elif IA32
  pc = (caddr_t)sc->eip;
# elif __ia64__
  pc = (caddr_t)sc->sc_ip;
# else
#  error "profile handler not defined for this architecture"
# endif
  return pc;
}

/*********************************************************************
 * Write out the TAU callpath
 ********************************************************************/
void Tau_sampling_output_callpath() {
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
    fprintf(ebsTrace, "%d", p->ThisFunction->GetFunctionId());
    TAU_QUERY_GET_PARENT_EVENT(event);
    TAU_QUERY_GET_EVENT_NAME(event, str);
    if (str) {
      //fprintf (ebsTrace, " : ", str);
      fprintf(ebsTrace, " ", str);
    }
    depth--;
  }
}

/*********************************************************************
 * Write out a single event record
 ********************************************************************/
void Tau_sampling_flush_record(TauSamplingRecord *record) {
  fprintf(ebsTrace, "$ | %lld | ", record->timestamp);
  fprintf(ebsTrace, "%lld | ", record->deltaStart);
  fprintf(ebsTrace, "%lld | ", record->deltaStop);
  fprintf(ebsTrace, "%x | ", record->pc);

  for (int i = 0; i < Tau_Global_numCounters; i++) {
    fprintf(ebsTrace, "%.16G ", record->counters[i]);
  }

  fprintf(ebsTrace, "| ");

  Tau_sampling_output_callpath();
  fprintf(ebsTrace, "\n");
}

/*********************************************************************
 * Handler for event exit (stop)
 ********************************************************************/
int Tau_sampling_event_stop(double stopTime) {
  samplingEnabled = 0;

  int tid = RtsLayer::myThread();
  Profiler *profiler = TauInternal_CurrentProfiler(tid);

  if (!profiler->needToRecordStop) {
    samplingEnabled = 1;
    return 0;
  }

  double startTime = profiler->StartTime[0]; // gtod must be counter 0

  x_uint64 start = (x_uint64)startTime;
  x_uint64 stop = (x_uint64)stopTime;

  fprintf(ebsTrace, "%% | %lld | %lld | ", start, stop);

  Tau_sampling_output_callpath();
  fprintf(ebsTrace, "\n");

  samplingEnabled = 1;
  return 0;
}

/*********************************************************************
 * Handler for itimer interrupt
 ********************************************************************/
void Tau_sampling_handler(int signum, siginfo_t *si, void *p) {
  if (!samplingEnabled) {
    return;
  }

  TauSamplingRecord theRecord;
  int tid = RtsLayer::myThread();
  Profiler *profiler = TauInternal_CurrentProfiler(tid);

  caddr_t pc;
  pc = get_pc(p);
//   printf ("sample on %x\n", pc);

  struct timeval tp;
  gettimeofday(&tp, 0);
  x_uint64 timestamp = ((x_uint64)tp.tv_sec * (x_uint64)1e6 + (x_uint64)tp.tv_usec);

  theRecord.timestamp = timestamp;
  theRecord.pc = pc;
  theRecord.deltaStart = 0;
  theRecord.deltaStop = 0;

  double startTime = profiler->StartTime[0]; // gtod must be counter 0
  theRecord.deltaStart = (x_uint64)startTime;
  theRecord.deltaStop = 0;

  double values[TAU_MAX_COUNTERS];
  TauMetrics_getMetrics(tid, values);
  for (int i = 0; i < Tau_Global_numCounters; i++) {
    theRecord.counters[i] = values[i];
  }

  Tau_sampling_flush_record(&theRecord);

  /* set this to get the stop event */
  profiler->needToRecordStop = 1;
}

/*********************************************************************
 * Output Format Header
 ********************************************************************/
int Tau_sampling_outputHeader() {
  fprintf(ebsTrace, "# Format:\n");
  fprintf(ebsTrace, "# <timestamp> | <delta-begin> | <delta-end> | <location> | <metric 1> ... <metric N> | <tau callpath>\n");
  fprintf(ebsTrace, "# Metrics:");
  for (int i = 0; i < Tau_Global_numCounters; i++) {
    const char *name = TauMetrics_getMetricName(i);
    fprintf(ebsTrace, " %s", name);
  }
  fprintf(ebsTrace, "\n");
}

/*********************************************************************
 * Initialize the sampling trace system
 ********************************************************************/
int Tau_sampling_init() {
  int ret;

  static struct itimerval itval;

  int threshold = 1000;

  samplingEnabled = 0;

  itval.it_interval.tv_usec = itval.it_value.tv_usec = 1000 % 1000000;
  itval.it_interval.tv_sec =  itval.it_value.tv_sec = 1000 / 1000000;

  const char *profiledir = TauEnv_get_profiledir();

  char filename[4096];

  int tid = RtsLayer::myThread();
  int node = RtsLayer::myNode();
  node = 0;
  sprintf(filename, "%s/ebstrace.raw.%d.%d.%d.%d", profiledir, getpid(), node, RtsLayer::myContext(), tid);

  ebsTrace = fopen(filename, "w");

  Tau_sampling_outputHeader();

  struct sigaction act;
  memset(&act, 0, sizeof(struct sigaction));
  act.sa_sigaction = Tau_sampling_handler;
  act.sa_flags     = SA_SIGINFO;

  ret = sigaction(SIGALRM, &act, NULL);

  ret = setitimer(ITIMER_REAL, &itval, 0);
  if (ret != 0) {
    printf("TAU: Sampling error: %s\n", strerror(ret));
    return 0;
  }

  samplingEnabled = 1;
  return 0;
}

/*********************************************************************
 * Finalize the sampling trace system
 ********************************************************************/
int Tau_sampling_finalize() {

  /* Disable sampling first */
  samplingEnabled = 0;

  struct itimerval itval;
  int ret;

  itval.it_interval.tv_usec = itval.it_value.tv_usec =
                                itval.it_interval.tv_sec = itval.it_value.tv_sec = 0;

  ret = setitimer(ITIMER_REAL, &itval, 0);
  if (ret != 0) {
    /* ERROR */
  }

  const char *profiledir = TauEnv_get_profiledir();
  char filename[4096];
  int tid = RtsLayer::myThread();
  int node = RtsLayer::myNode();
  node = 0;
  sprintf(filename, "%s/ebstrace.def.%d.%d.%d.%d", profiledir, getpid(), node, RtsLayer::myContext(), tid);

  FILE *def = fopen(filename, "w");

  fprintf(def, "# Format:\n");
  fprintf(def, "# <id> | <name>\n");

  for (vector<FunctionInfo *>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    fprintf(def, "%d | %s %s\n", fi->GetFunctionId(), fi->GetName(), fi->GetType());
  }
  fclose(def);


  /* write out the executable name at the end */
  char buffer[4096];
  bzero(buffer, 4096);
  int rc = readlink("/proc/self/exe", buffer, 4096);
  fprintf(ebsTrace, "# exe: %s\n", buffer);

  /* write out the node number */
  fprintf(ebsTrace, "# node: %d\n", RtsLayer::myNode());

  fclose(ebsTrace);
}

#endif /* TAU_EXP_SAMPLING */
