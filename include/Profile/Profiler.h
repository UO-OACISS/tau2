/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.h					  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: tau-bugs@cs.uoregon.edu               	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/
#ifndef PROFILER_H
#define PROFILER_H

#if (defined(TAU_WINDOWS))
#pragma warning( disable : 4786 )
#define TAUDECL __cdecl
#else
#define TAUDECL
#endif /* TAU_WINDOWS */

#if (!defined(TAU_LIBRARY_SOURCE) && !defined(TAU_WINDOWS))

#ifdef TAU_INCLUDE_MPI_H_HEADER
#ifdef TAU_MPI
#include <mpi.h>
#endif 
#endif /* TAU_INCLUDE_MPI_H_HEADER */

#include <sys/types.h>
#include <unistd.h>

#undef fork
#define fork() tau_fork()

#ifdef __cplusplus
extern "C" 
#endif
pid_t tau_fork (void);

#ifdef PTHREADS
/* pthread_create wrapper */
#include <pthread.h>

#ifndef TAU_MPC

#undef pthread_create
#define pthread_create(thread, attr, function, arg) tau_pthread_create(thread, attr, function, arg)

#define pthread_join(thread, retval) tau_pthread_join(thread, retval)

#define pthread_exit(arg) tau_pthread_exit(arg)

#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
#define pthread_barrier_wait(barrier) tau_pthread_barrier_wait(barrier)
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */

#endif /* TAU_MPC */

#ifdef __cplusplus
extern "C" {
#endif
int tau_pthread_create (pthread_t *threadp, const pthread_attr_t *attr, void *(*start_routine) (void *), void *arg);

int tau_pthread_join(pthread_t thread, void ** retval);

void tau_pthread_exit (void *arg);

#ifdef TAU_PTHREAD_BARRIER_AVAILABLE 
int tau_pthread_barrier_wait(pthread_barrier_t *barrier);
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */

#ifdef __cplusplus
}
#endif 

#endif /* PTHREADS */
#endif /* TAU_LIBRARY_SOURCE */


/* This is to get TAU_START/TAU_STOP since some codes just #include <Profile/Profiler.h> */
#include <TAU.h>



#ifndef TAU_NAME_LENGTH
#define TAU_NAME_LENGTH 1024
#endif /* TAU_NAME_LENGTH: used in MPI_T */

#ifndef TAU_MAX_COUNTERS
#define TAU_MAX_COUNTERS 25
#endif

#ifndef TAU_MAX_CALLPATH_DEPTH
#define TAU_MAX_CALLPATH_DEPTH 200
#endif

#if (defined(PTHREADS) || defined(TAU_MPC) || defined(TULIPTHREADS) || defined(JAVA) || defined(TAU_WINDOWS) || defined (TAU_OPENMP) || defined (TAU_SPROC) || defined(TAU_PAPI_THREADS))


#ifndef TAU_MAX_THREADS
#if defined(TAU_CHARM) || defined(TAU_MIC_LINUX)
#define TAU_MAX_THREADS 512
#else /* TAU_CHARM || TAU_MIC_LINUX */
#define TAU_MAX_THREADS 128
#endif
#endif /* TAU_MAX_THREADS */

#else /* not using threads? */
#ifndef TAU_MAX_THREADS
/* *CWL* - If useropt is not specified, then GPUs need to override the non-threaded default of 1. 
         - If thread packages are used, their defaults (> 32) are used.
	 Ultimately, we would like some way of setting TAU_MAX_THREADS as a cumulative value of
         each component value (e.g., PTHREADS + GPU = 128 + 32 = 160).
*/
#ifdef TAU_GPU
#define TAU_MAX_THREADS 512 
#else /* TAU_GPU */
#define TAU_MAX_THREADS 1
#endif /* TAU_GPU */
#endif /* TAU_MAX_THREADS */
#endif /* PTHREADS || TULIPTHREADS || JAVA || TAU_WINDOWS || OPENMP || SPROC */


#ifdef TAU_OPENMP 
#define TAU_TRACK_IDLE_THREADS
#endif

#ifdef TAU_TRACK_PTHREAD_IDLE
#define TAU_TRACK_IDLE_THREADS
#endif


#include <Profile/ProfileGroups.h>
#include <Profile/TauAPI.h>

#if (defined (__cplusplus ) && !defined (TAU_USE_C_API))

#ifdef TAU_ENABLED

#include <Profile/ProfileHeaders.h>
#include <Profile/PthreadLayer.h>
#include <Profile/TulipThreadLayer.h>
#include <Profile/JNIThreadLayer.h>
#include <Profile/JVMTIThreadLayer.h>
#include <Profile/SprocLayer.h>
#include <Profile/PapiThreadLayer.h>
#include <Profile/RtsLayer.h>
#include <Profile/FunctionInfo.h>
#include <Profile/UserEvent.h>
#include <Profile/PapiLayer.h>
#include <Profile/LikwidLayer.h>
#include <Profile/WindowsThreadLayer.h>
#include <Profile/TauMemory.h>
#include <Profile/TauScalasca.h>
#include <Profile/TauCompensate.h>
#include <Profile/TauHandler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauMapping.h>
#include <Profile/TauSampling.h>

#ifndef TAU_WINDOWS
#include <sys/types.h>
#endif

#if defined(TAUKTAU)
class KtauProfiler;
#ifdef TAUKTAU_MERGE
#include <Profile/KtauMergeInfo.h>
#endif /* defined(TAUKTAU_MERGE) */
#endif /* defined(TAUKTAU) */


/*
//////////////////////////////////////////////////////////////////////
//
// class Profiler
//
// This class is intended to be instantiated once per function
// (or other code block to be timed) as an auto variable.
//
// It will be constructed each time the block is entered
// and destroyed when the block is exited.  The constructor
// turns on the timer, and the destructor turns it off.
//
//////////////////////////////////////////////////////////////////////
*/
namespace tau {
class Profiler
{
public:

  Profiler *ParentProfiler; 
  TauGroup_t MyProfileGroup_;
  bool StartStopUsed_;
  bool AddInclFlag; 
  bool PhaseFlag;
  bool AddInclCallPathFlag; 
  FunctionInfo *ThisFunction;
  FunctionInfo *CallPathFunction;
  FunctionInfo *CallSiteFunction;

  Profiler() : heapmem(0) {};
  ~Profiler() {};
  
  void Start(int tid = RtsLayer::myThread());
  void Stop(int tid = RtsLayer::myThread(), bool useLastTimeStamp = false);
  double *getStartValues();

  void CallPathStart(int tid);
  void CallPathStop(double* totaltime, int tid);

#ifdef TAU_PROFILEPARAM
  FunctionInfo *ProfileParamFunction; 
  bool 	       AddInclProfileParamFlag; 
  void ProfileParamStop(double* totaltime, int tid);
#endif /* TAU_PROFILEPARAM */
  
  double StartTime[TAU_MAX_COUNTERS];

  /* Compensate for instrumentation overhead based on total number of 
     child calls executed under the given timer */
  long NumChildren;
  void SetNumChildren(long n);
  long GetNumChildren(void);
  void AddNumChildren(long value);
  
  
#ifdef TAU_PROFILEPHASE
  bool GetPhase(void);
  void SetPhase(bool flag);
#endif /* TAU_PROFILEPHASE */

#if defined(TAUKTAU)
  KtauProfiler* ThisKtauProfiler;
#if defined(TAUKTAU_MERGE)
  KtauMergeInfo ThisKtauMergeInfo;
#endif /* TAUKTAU_MERGE */
#endif /* TAUKTAU */
  
#ifdef TAU_MPITRACE
  bool RecordEvent; /* true when an MPI call is in the callpath */
#endif /* TAU_MPITRACE */

  /* For EBS sampling */
  int needToRecordStop;
  void *address[TAU_SAMP_NUM_ADDRESSES];

  /* For tracking heap memory */
  double heapmem;

  // Callsite discovery
  unsigned long callsites[TAU_SAMP_NUM_ADDRESSES+1];
  unsigned long callsiteKeyId;
  long *path;
  void CallSiteStart(int tid, x_uint64 TraceTimeStamp);
  void CallSiteAddPath(long *comparison, int tid);
  void CallSiteStop(double *totalTime, int tid, x_uint64 TraceTimeStamp);
};
}
#ifdef TAU_LIBRARY_SOURCE
// This could be dangerous.  We need to phase this out.
using tau::Profiler;
#endif /* TAU_LIBRARY_SOURCE */

extern "C" tau::Profiler *TauInternal_CurrentProfiler(int tid);
extern "C" tau::Profiler *TauInternal_ParentProfiler(int tid);

int TauProfiler_updateIntermediateStatistics(int tid);
bool TauProfiler_createDirectories();
int TauProfiler_StoreData(int tid = RtsLayer::myThread()); 
int TauProfiler_DumpData(bool increment = false, int tid = RtsLayer::myThread(), const char *prefix = "dump"); 
int TauProfiler_writeData(int tid, const char *prefix = "profile", bool increment = false, 
		       const char **inFuncs = NULL, int numFuncs = 0);
void TauProfiler_PurgeData(int tid = RtsLayer::myThread());

void TauProfiler_theFunctionList(const char ***inPtr, int *numOfFunctions,
				 bool addName = false, const char *inString = NULL);
void TauProfiler_dumpFunctionNames();

void TauProfiler_theCounterList(const char ***inPtr, int *numOfCounters);
  
void TauProfiler_getFunctionValues(const char **inFuncs,
				   int numFuncs,
				   double ***counterExclusiveValues,
				   double ***counterInclusiveValues,
				   int **numOfCalls,
				   int **numOfSubRoutines,
				   const char ***counterNames,
				   int *numOfCounters,
				   int tid = RtsLayer::myThread());
int TauProfiler_dumpFunctionValues(const char **inFuncs,
				   int numFuncs,
				   bool increment = false,
				   int tid = RtsLayer::myThread(), 
				   const char *prefix = "dump");

void TauProfiler_getUserEventList(const char ***inPtr, int *numUserEvents);

void TauProfiler_getUserEventValues(const char **inUserEvents, int numUserEvents,
				    int **numEvents, double **max, double **min,
				    double **mean, double **sumSqr, 
				    int tid = RtsLayer::myThread());


void TauProfiler_AddProfileParamData(long key, const char *keyname);


#endif /* TAU_ENABLED */
/* included after class Profiler is defined. */
#endif /* __cplusplus && ! TAU_USE_C_API */

#ifdef TAU_APPLE_MACH_PORT_BUG
#include <sys/types.h>
typedef __darwin_mach_port_t mach_port_t;
mach_port_t pthread_mach_thread_np(pthread_t);
/* Ref: https://github.com/apache/arrow/pull/1139 */
#endif /* TAU_APPLE_MACH_PORT_BUG */


#endif /* PROFILER_H */
/***************************************************************************
 * $RCSfile: Profiler.h,v $   $Author: amorris $
 * $Revision: 1.120 $   $Date: 2010/03/18 17:31:12 $
 * POOMA_VERSION_ID: $Id: Profiler.h,v 1.120 2010/03/18 17:31:12 amorris Exp $ 
 ***************************************************************************/
