/////////////////////////////////////////////////
//Header file for the multiple counter layer.
//
//Author:   Robert Bell
//Created:  January 2002
//
/////////////////////////////////////////////////

#ifndef _MULTIPLE_COUNTER_LAYER_H_
#define _MULTIPLE_COUNTER_LAYER_H_

#ifdef TAU_MULTIPLE_COUNTERS


//If we are going to use Papi, include its headers.
#ifdef TAU_PAPI
extern "C" {
  /*#include "papiStdEventDefs.h" */
#include "papi.h"
  /*
#include "papi_internal.h"
  */
}
#endif /* TAU_PAPI */


#ifdef TAUKTAU_SHCTR
extern "C" {
#include "KtauCounters.h"
}
#endif /* TAUKTAU_SHCTR */

#define SIZE_OF_INIT_ARRAY 12 //!!Change this if functions are added to the system!!

extern int Tau_Global_numCounters;

//Some useful typedefs
typedef bool (*firstListType)(int);
typedef void (*secondListType)(int, double[]);

class MultipleCounterLayer
{
 public:
  static bool initializeMultiCounterLayer(void);
  static bool *getCounterUsedList();
  static bool getCounterUsed(int inPosition);
  static int  getNumberOfCountersUsed(void);
  static void setCounterUsed(bool inValue, int inPosition);
  static void getCounters(int tid, double values[]);
  static double getSingleCounter(int tid, int counter);
  static const char * getCounterNameAt(int position);
  static void theCounterList(const char ***inPtr, int *numOfCounters);
  static void theCounterListInternal(const char ***inPtr, int *numOfCounters, bool **tmpPtr);
  static void triggerCounterEvents(unsigned long long timestamp, double *values, int tid);
  //*********************
  //The list of counter functions, and their init. functions.
  //Please see the help files on multiple
  //counters to see our conventions.
  //*********************
  
  // private:
  static bool gettimeofdayMCLInit(int functionPosition);
  static void gettimeofdayMCL(int tid, double values[]);

  static bool linuxTimerMCLInit(int functionPosition);
  static void linuxTimerMCL(int tid, double values[]);

  static bool bglTimersMCLInit(int functionPosition);
  static void bglTimersMCL(int tid, double values[]);

  static bool sgiTimersMCLInit(int functionPosition);
  static void sgiTimersMCL(int tid, double values[]);

  static bool crayTimersMCLInit(int functionPosition);
  static void crayTimersMCL(int tid, double values[]);

  static bool cpuTimeMCLInit(int functionPosition);
  static void cpuTimeMCL(int tid, double values[]);

  static bool logicalClockMCLInit(int functionPosition);
  static void logicalClockMCL(int tid, double values[]);

  static bool tauMPIMessageSizeMCLInit(int functionPosition);
  static void tauMPIMessageSizeMCL(int tid, double values[]);
  
  static bool papiMCLInit(int functionPosition);
  static void papiMCL(int tid, double values[]);

  static bool papiWallClockMCLInit(int functionPosition);
  static void papiWallClockMCL(int tid, double values[]);

  static bool papiVirtualMCLInit(int functionPosition);
  static void papiVirtualMCL(int tid, double values[]);

  static bool ktauMCLInit(int functionPosition);
  static void ktauMCL(int tid, double values[]);
  //*********************
  //End - List of counter and init. functions.
  //*********************

  //Other class stuff.
  static char environment[25][10];

  static int gettimeofdayMCL_CP[1];
  static int gettimeofdayMCL_FP;

#ifdef TAU_LINUX_TIMERS  
  static int linuxTimerMCL_CP[1];
  static int linuxTimerMCL_FP;
#endif //TAU_LINUX_TIMERS

#if (defined(BGL_TIMERS) || defined(BGP_TIMERS))
  static int bglTimersMCL_CP[1];
  static int bglTimersMCL_FP;
#endif // BGL_TIMERS || BGP_TIMERS

#ifdef SGI_TIMERS
  static int sgiTimersMCL_CP[1];
  static int sgiTimersMCL_FP;
#endif // SGI_TIMERS

#ifdef CRAY_TIMERS
  static int crayTimersMCL_CP[1];
  static int crayTimersMCL_FP;
#endif // CRAY_TIMERS

#ifdef CPU_TIME
  static int cpuTimeMCL_CP[1];
  static int cpuTimeMCL_FP;
#endif // CPU_TIME

  static int logicalClockMCL_CP[1];
  static int logicalClockMCL_FP;

#ifdef TAU_MPI
  static int tauMPIMessageSizeMCL_CP[1];
  static int tauMPIMessageSizeMCL_FP;
#endif // TAU_MPI
  
#ifdef TAU_PAPI
  static int papiMCL_CP[MAX_TAU_COUNTERS];
  static int papiWallClockMCL_CP[1];
  static int papiVirtualMCL_CP[1];
  static int papiMCL_FP; 
  static int papiWallClockMCL_FP;
  static int papiVirtualMCL_FP;
#endif//TAU_PAPI

#ifdef TAUKTAU_SHCTR
  static int ktauMCL_CP[MAX_TAU_COUNTERS];
  static int ktauMCL_FP; 
#endif//TAUKTAU_SHCTR

  static firstListType initArray[SIZE_OF_INIT_ARRAY];
  static secondListType functionArray[MAX_TAU_COUNTERS];
  static int numberOfActiveFunctions;
  static char * names[MAX_TAU_COUNTERS];
  static bool counterUsed[MAX_TAU_COUNTERS];
  static int numberOfCounters[MAX_TAU_COUNTERS];
  static TauUserEvent **counterEvents;
};

#endif /* TAU_MULTIPLE_COUNTERS */
#endif /* _MULTIPLE_COUNTER_LAYER_H_ */

/////////////////////////////////////////////////
//
//End - Multiple counter layer.
//
/////////////////////////////////////////////////
