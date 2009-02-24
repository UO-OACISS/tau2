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

#include <sys/types.h>
#include <unistd.h>

#undef fork
#define fork() \
        tau_fork()

#ifdef __cplusplus
extern "C" 
#endif
pid_t tau_fork (void);

#ifdef PTHREADS
/* pthread_create wrapper */

#include <pthread.h>
#undef pthread_create
#define pthread_create(thread, attr, function, arg) \
        tau_pthread_create(thread, attr, function, arg)

#define pthread_exit(arg) \
        tau_pthread_exit(arg)

#ifdef __cplusplus
extern "C" 
#endif
int tau_pthread_create (pthread_t *threadp,
			const pthread_attr_t *attr,
			void *(*start_routine) (void *),
			void *arg);

int tau_track_pthread_create (pthread_t *threadp,
			const pthread_attr_t *attr,
			void *(*start_routine) (void *),
			      void *arg, int id);
void tau_pthread_exit (void *arg);

#endif /* PTHREADS */
#endif /* TAU_LIBRARY_SOURCE */


/* This is to get TAU_START/TAU_STOP since some codes just #include <Profile/Profiler.h> */
#include <TAU.h>



#ifdef TAU_MULTIPLE_COUNTERS
#define MAX_TAU_COUNTERS 25
#else
#define MAX_TAU_COUNTERS 1
#endif

#if (defined(PTHREADS) || defined(TULIPTHREADS) || defined(JAVA) || defined(TAU_WINDOWS) || defined (TAU_OPENMP) || defined (TAU_SPROC) || defined(TAU_PAPI_THREADS))


#ifndef TAU_MAX_THREADS
#ifdef TAU_CHARM
#define TAU_MAX_THREADS 512
#else /* TAU_CHARM */
#define TAU_MAX_THREADS 128
#endif
#endif /* TAU_MAX_THREADS */

#else
#define TAU_MAX_THREADS 1
#endif /* PTHREADS || TULIPTHREADS || JAVA || TAU_WINDOWS || OPENMP || SPROC */



#include <Profile/ProfileGroups.h>
#include <Profile/TauAPI.h>

#if (defined (__cplusplus ) && !defined (TAU_USE_C_API))


#ifdef TAU_ENABLED

#include <Profile/ProfileHeaders.h>
#include <Profile/PthreadLayer.h>
#include <Profile/TulipThreadLayer.h>
#include <Profile/JavaThreadLayer.h>
#include <Profile/SprocLayer.h>
#include <Profile/PapiThreadLayer.h>
#include <Profile/RtsLayer.h>
#include <Profile/FunctionInfo.h>
#include <Profile/UserEvent.h>
#include <Profile/PclLayer.h>
#include <Profile/PapiLayer.h>
#include <Profile/MultipleCounters.h>
#include <Profile/WindowsThreadLayer.h>
#include <Profile/TauMemory.h>
#include <Profile/TauScalasca.h>
#include <Profile/TauCompensate.h>
#include <Profile/TauHandler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauMapping.h>

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

  Profiler() {};
  
  void Start(int tid = RtsLayer::myThread());

  /* Clean up data from this invocation. */
  void Stop(int tid = RtsLayer::myThread(), bool useLastTimeStamp = false);
  ~Profiler();
  void CallPathStart(int tid);
#ifdef TAU_MULTIPLE_COUNTERS
  void CallPathStop(double* totaltime, int tid);
#else  /* TAU_MULTIPLE_COUNTERS  */
  void CallPathStop(double totaltime, int tid);
#endif /* TAU_MULTIPLE_COUNTERS  */
  
#ifdef TAU_PROFILEPARAM
#ifdef TAU_MULTIPLE_COUNTERS
  void ProfileParamStop(double* totaltime, int tid);
#else  /* TAU_MULTIPLE_COUNTERS  */
  void ProfileParamStop(double totaltime, int tid);
#endif /* TAU_MULTIPLE_COUNTERS */
#endif /* TAU_PROFILEPARAM */
  
  double *getStartValues();
  
#ifndef TAU_MULTIPLE_COUNTERS
  double StartTime;
#else /* TAU_MULTIPLE_COUNTERS */
  double StartTime[MAX_TAU_COUNTERS];
#endif /* TAU_MULTIPLE_COUNTERS */
  FunctionInfo * ThisFunction;
  FunctionInfo * CallPathFunction;
  bool 	       AddInclCallPathFlag; 
#ifdef TAU_PROFILEPARAM
  FunctionInfo * ProfileParamFunction; 
  bool 	       AddInclProfileParamFlag; 
#endif /* TAU_PROFILEPARAM */

#ifdef TAU_COMPENSATE
  /* Compensate for instrumentation overhead based on total number of 
     child calls executed under the given timer */
  long NumChildren;
  void SetNumChildren(long n);
  long GetNumChildren(void);
  void AddNumChildren(long value);
#endif /* TAU_COMPENSATE */
  
  
#ifdef TAU_PROFILEPHASE
  bool GetPhase(void);
  void SetPhase(bool flag);
#endif /* TAU_PROFILEPHASE */
#ifdef TAU_DEPTH_LIMIT
  int  GetDepthLimit(void);
  void SetDepthLimit(int value);
#endif /* TAU_DEPTH_LIMIT */ 
  
#if defined(TAUKTAU)
  KtauProfiler* ThisKtauProfiler;
#if defined(TAUKTAU_MERGE)
  KtauMergeInfo ThisKtauMergeInfo;
#endif /* TAUKTAU_MERGE */
#endif /* TAUKTAU */
  
public:
  Profiler *ParentProfiler; 
  TauGroup_t MyProfileGroup_;
  bool	StartStopUsed_;
  bool 	AddInclFlag; 
  bool 	PhaseFlag;

#ifdef TAU_DEPTH_LIMIT
  int  profiledepth; 
#endif /* TAU_DEPTH_LIMIT */
  
#ifdef TAU_MPITRACE
  bool 	RecordEvent; /* true when an MPI call is in the callpath */
#endif /* TAU_MPITRACE */

};
};
#ifdef TAU_LIBRARY_SOURCE
using tau::Profiler;
#endif /* TAU_LIBRARY_SOURCE */


extern "C" Profiler *TauInternal_CurrentProfiler(int tid);

int TauProfiler_updateIntermediateStatistics(int tid);
bool TauProfiler_createDirectories();
int TauProfiler_StoreData(int tid = RtsLayer::myThread()); 
int TauProfiler_DumpData(bool increment = false, int tid = RtsLayer::myThread(), const char *prefix = "dump"); 
int TauProfiler_writeData(int tid, const char *prefix = "profile", bool increment = false, 
		       const char **inFuncs = NULL, int numFuncs = 0);
void TauProfiler_PurgeData(int tid = RtsLayer::myThread());

int TauProfiler_Snapshot(const char *name, bool finalize = false,
		      int tid = RtsLayer::myThread()); 



void TauProfiler_theFunctionList(const char ***inPtr, int *numOfFunctions,
				 bool addName = false, const char * inString = NULL);
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
				   char *prefix = "dump");

void TauProfiler_getUserEventList(const char ***inPtr, int *numUserEvents);

void TauProfiler_getUserEventValues(const char **inUserEvents, int numUserEvents,
				    int **numEvents, double **max, double **min,
				    double **mean, double **sumSqr, 
				    int tid = RtsLayer::myThread());


void TauProfiler_AddProfileParamData(long key, const char *keyname);


#ifdef TAU_MPITRACE
void TauProfiler_EnableAllEventsOnCallStack(int tid, Profiler *current);
#endif /* TAU_MPITRACE */


#endif /* TAU_ENABLED */
/* included after class Profiler is defined. */
#endif /* __cplusplus && ! TAU_USE_C_API */


#endif /* PROFILER_H */
/***************************************************************************
 * $RCSfile: Profiler.h,v $   $Author: amorris $
 * $Revision: 1.99 $   $Date: 2009/02/24 22:30:42 $
 * POOMA_VERSION_ID: $Id: Profiler.h,v 1.99 2009/02/24 22:30:42 amorris Exp $ 
 ***************************************************************************/
