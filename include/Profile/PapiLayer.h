/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2006                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: PapiLayer.h                                      **
**	Description 	: TAU Profiling Package			           **
**	Contact		: tau-team@cs.uoregon.edu 		 	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
****************************************************************************/

#ifndef _PAPI_LAYER_H_
#define _PAPI_LAYER_H_

#ifdef TAU_PAPI


#define TAU_PAPI_MAX_COMPONENTS 4

#ifdef TAU_MULTIPLE_COUNTERS
#define MAX_PAPI_COUNTERS MAX_TAU_COUNTERS
#else
#define MAX_PAPI_COUNTERS 1
#endif

struct ThreadValue {
  int ThreadID;
  int EventSet[TAU_PAPI_MAX_COMPONENTS]; 
  int NumEvents[TAU_PAPI_MAX_COMPONENTS];
  long long *CounterValues;
  int Comp2Metric[TAU_PAPI_MAX_COMPONENTS][MAX_PAPI_COUNTERS];
};


class PapiLayer {
public:
  static int initializePapiLayer(bool lock = true);
  static long long getSingleCounter(int tid);
  static long long *getAllCounters(int tid, int *numValues);
  static long long getWallClockTime(void);
  static long long getVirtualTime(void);
  static int reinitializePAPI(void);
  static int addCounter(char *name);
private:
  static int initializeSingleCounter();
  static int initializeThread(int tid);
  static int initializePAPI(void);
  static void checkDomain(int domain, char* domainstr);
  static bool papiInitialized;
  static ThreadValue *ThreadList[TAU_MAX_THREADS];
  static int numCounters;
#ifdef TAU_MULTIPLE_COUNTERS
  static int counterList[MAX_TAU_COUNTERS];
#else
  static int counterList[1];
#endif


};

#endif /* TAU_PAPI */
#endif /* _PAPI_LAYER_H_ */
