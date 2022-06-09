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
#include <vector>
#ifndef _PAPI_LAYER_H_
#define _PAPI_LAYER_H_

#ifdef TAU_PAPI


#define TAU_PAPI_MAX_COMPONENTS 4

#define MAX_PAPI_COUNTERS TAU_MAX_COUNTERS
using namespace std;
struct ThreadValue {
  int ThreadID;
  int EventSet[TAU_PAPI_MAX_COMPONENTS]; 
  int NumEvents[TAU_PAPI_MAX_COMPONENTS];
  long long *CounterValues=0;
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
  static void setPapiInitialized(bool value); 
  static void triggerRAPLPowerEvents(bool in_signal_handler);
  static int numCounters;
  static int counterList[TAU_MAX_COUNTERS];
  static bool destroyed;
  inline static void setThreadValue(int tid, ThreadValue* tv){
	    checkPAPIVector(tid);
        ThePapiThreadList()[tid]=tv;
  }

  inline static ThreadValue* getThreadValue(int tid){
	  if(destroyed)return NULL;
		checkPAPIVector(tid);
        return ThePapiThreadList()[tid];
  }   

private:
  static int initializeSingleCounter();
  static int initializeThread(int tid);
  static int initializePAPI(void);
  static int initializeAndCheckRAPL(int tid);
  static int initializeRAPL(int tid);
  static int initializePerfRAPL(int tid);
  static void checkDomain(int domain, char* domainstr);
  static bool papiInitialized;
  static double scalingFactor;
  //static ThreadValue *ThreadList[TAU_MAX_THREADS];
  struct PapiThreadList : vector<ThreadValue*>{
     PapiThreadList(){
        //printf("Creating PapiThreadList at %p\n", this);
     }
    virtual ~PapiThreadList(){
        //printf("Destroying PapiThreadList at %p, with size %ld\n", this, this->size());
	destroyed=true;
        Tau_destructor_trigger();
    }
  };
  
  static PapiThreadList & ThePapiThreadList();
  
  static inline void checkPAPIVector(int tid){
	RtsLayer::LockDB();  
	while(ThePapiThreadList().size()<=tid){
	    ThePapiThreadList().push_back(NULL);
	}
	RtsLayer::UnLockDB();
}
};

#endif /* TAU_PAPI */
#endif /* _PAPI_LAYER_H_ */
