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


#define SIZE_OF_INIT_ARRAY 5 //!!Change this if functions are added to the system!!


//Some useful typedefs
typedef bool (*firstListType)(int);
typedef void (*secondListType)(int, double[]);

class MultipleCounterLayer
{
 public:
  static bool initializeMultiCounterLayer(void);
  static void getCounters(int tid, double values[]);
  static char * getCounterNameAt(int position);
  static bool getCounterUsed(int inPosition){return counterUsed[inPosition];}

  //*********************
  //The list of counter functions, and their init. functions.
  //Please see the help files on multiple
  //counters to see our conventions.
  //*********************
  
 private:
  static bool gettimeofdayMCLInit(int functionPosition);
  static void gettimeofdayMCL(int tid, double values[]);

  static bool linuxTimerMCLInit(int functionPosition);
  static void linuxTimerMCL(int tid, double values[]);

  static bool papiMCLInit(int functionPosition);
  static void papiMCL(int tid, double values[]);

  static bool papiWallClockMCLInit(int functionPosition);
  static void papiWallClockMCL(int tid, double values[]);

  static bool papiVirtualMCLInit(int functionPosition);
  static void papiVirtualMCL(int tid, double values[]);
  //*********************
  //End - List of counter and init. functions.
  //*********************

  //Other class stuff.
  static char environment[25][10];
  static int gettimeofdayMCL_CP[1];
  static int papiMCL_CP[MAX_TAU_COUNTERS];
  static int papiWallClockMCL_CP[1];
  static int papiVirtualMCL_CP[1];
  static int linuxTimerMCL_CP[1];

  //Data specific to the papiMCL function.
  static int numberOfPapiHWCounters;
  static int PAPI_CounterCodeList[MAX_TAU_COUNTERS];


  static firstListType initArray[SIZE_OF_INIT_ARRAY];
  static secondListType functionArray[MAX_TAU_COUNTERS];
  static int numberOfActiveFunctions;
  static char * names[MAX_TAU_COUNTERS];
  static bool counterUsed[MAX_TAU_COUNTERS];
};

#endif /* TAU_MULTIPLE_COUNTERS */
#endif /* _MULTIPLE_COUNTER_LAYER_H_ */

/////////////////////////////////////////////////
//
//End - Multiple counter layer.
//
/////////////////////////////////////////////////
