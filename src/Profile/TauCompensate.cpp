/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2004  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauCompensate.cpp				  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 			  **
**	Documentation	: http://www.cs.uoregon.edu/research/paracomp/tau **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////
#include "Profile/Profiler.h"
#include "Profile/TauAPI.h"
#ifdef DEBUG_PROF
#include <iostream>
using namespace std;
#endif /* DEBUG_PROF */
#include <stdlib.h>




extern "C" int Tau_compensate_initialization() {
  double *tover = TauGetTimerOverhead(TauFullTimerOverhead);
  double *tnull = TauGetTimerOverhead(TauNullTimerOverhead);
  return 0;
}



double*& TheTauNullTimerOverhead() {
  static double *over = new double[TAU_MAX_COUNTERS];
  static int flag = 0;
  
  if (flag == 0) {
    flag = 1;
    for (int i = 0; i < TAU_MAX_COUNTERS; i++) {
      over[i] = 0.0;
    }
  }

  return over;
}

double*& TheTauFullTimerOverhead() {
  static double *full = new double[TAU_MAX_COUNTERS];
  static int flag = 0;

  if (flag == 0) {
    flag = 1;
    for (int i = 0; i < TAU_MAX_COUNTERS; i++) {
      full[i] = 0.0;
    }
  }

  return full;
}



int TauCalibrateNullTimer(void) {
  TAU_PROFILE_TIMER(tnull, ".TAU null timer overhead", " ", TAU_DEFAULT);
  TAU_PROFILE_TIMER(tone,  ".TAU 1000 null timers overhead", " ", TAU_DEFAULT);
  int i, tid;
  char *iter;
  int iterations;

  if ((iter = getenv("TAU_COMPENSATE_ITERATIONS")) == 0) {
    iterations = 1000;
  } else {
    iterations = atoi(iter);
  }
    

#ifdef TAU_DEPTH_LIMIT
  int original = TauEnv_get_depth_limit();
  TauEnv_set_depth_limit(INT_MAX);
#endif /* TAU_DEPTH_LIMIT */

  bool oldSafeValue = TheSafeToDumpData();
  TheSafeToDumpData() = false;
  //Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_START(tone);
    /* nested */
  for(i=0; i< iterations; i++) {
    TAU_PROFILE_START(tnull);
    TAU_PROFILE_STOP(tnull);
  }
  TAU_PROFILE_STOP(tone);
  //Tau_stop_top_level_timer_if_necessary();
  TheSafeToDumpData() = oldSafeValue;

#ifdef TAU_DEPTH_LIMIT
  TauEnv_set_depth_limit(original); /* reset! */
#endif /* TAU_DEPTH_LIMIT */

  /* Get thread id */
  tid = RtsLayer::myThread();
  int n = ((FunctionInfo*)tnull)->GetCalls(tid);

  double *nullincltime = ((FunctionInfo*)tnull)->GetInclTime(tid);
  double *oneincltime  = ((FunctionInfo*)tone)->GetInclTime(tid);
  for (i=0; i < Tau_Global_numCounters; i++) {
    /* n*(a+b+c+d) + b+c = tone */
    TheTauNullTimerOverhead()[i] = nullincltime[i]/n;
    
    /* a+b+c+d = Toverhead = (tone - tnull) / n */
    TheTauFullTimerOverhead()[i] = (oneincltime[i] - TheTauNullTimerOverhead()[i]) / n; 
  }
#ifdef DEBUG_PROF
  cout <<"Calibrate: Tnull time "<< TheTauNullTimerOverhead() <<endl;
  cout <<"Calibrate: Toverhead time = "<<TheTauFullTimerOverhead() <<endl;
#endif /* DEBUG_PROF */
  return 0;
}

double* TauGetTimerOverhead(enum TauOverhead type) {
  static int flag = 0;
  if (flag == 0) {
    flag = 1; /* reset it */
    TauCalibrateNullTimer();
  }
  
  /* What kind of overhead are we looking for here? */
  if (type == TauNullTimerOverhead) {
    return TheTauNullTimerOverhead();
  } else {
    if (type == TauFullTimerOverhead) {
      return TheTauFullTimerOverhead();
    } else {
      return (double *) NULL;
    }
  }
}



