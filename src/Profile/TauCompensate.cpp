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
**	Flags		: Compile with				          **
**			  -DTAU_COMPENSATE for instrumentation overhead   **
**			   compensation to correct profiles               **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
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
#ifndef TAU_MULTIPLE_COUNTERS 
  double tover = TauGetTimerOverhead(TauFullTimerOverhead);
  double tnull = TauGetTimerOverhead(TauNullTimerOverhead);
#else
  double *tover = TauGetTimerOverhead(TauFullTimerOverhead);
  double *tnull = TauGetTimerOverhead(TauNullTimerOverhead);
#endif
}



#ifndef TAU_MULTIPLE_COUNTERS
double& TheTauNullTimerOverhead()
{
  static double over = 0.0;

  return over;
}

double& TheTauFullTimerOverhead()
{
  static double full = 0.0;

  return full;
}
#else /* TAU_MULTIPLE_COUNTERS */
double*& TheTauNullTimerOverhead()
{
  static double *over = new double[MAX_TAU_COUNTERS];
  static int flag = 0;
  
  if (flag == 0) {
    flag = 1;
    for (int i = 0; i < MAX_TAU_COUNTERS; i++) {
      over[i] = 0.0;
    }
  }

  return over;
}

double*& TheTauFullTimerOverhead()
{
  static double *full = new double[MAX_TAU_COUNTERS];
  static int flag = 0;

  if (flag == 0) {
    flag = 1;
    for (int i = 0; i < MAX_TAU_COUNTERS; i++) {
      full[i] = 0.0;
    }
  }

  return full;
}
#endif /* TAU_MULTIPLE_COUNTERS */

#ifdef TAU_DEPTH_LIMIT
int& TauGetDepthLimit(void);
#endif /* TAU_DEPTH_LIMIT */
int TauCalibrateNullTimer(void)
{
  TAU_PROFILE_TIMER(tnull, ".TAU null timer overhead", " ", TAU_DEFAULT);
  TAU_PROFILE_TIMER(tone,  ".TAU 1000 null timers overhead", " ", TAU_DEFAULT);
  int i, tid;
  char *iter;
  int iterations;

  if ((iter = getenv("TAU_COMPENSATE_ITERATIONS")) == 0)
  {
    iterations = 1000;
  }
  else
  {
    iterations = atoi(iter);
  }
    

#ifdef TAU_DEPTH_LIMIT
  int original = TauGetDepthLimit();
  TauGetDepthLimit() = INT_MAX;
#endif /* TAU_DEPTH_LIMIT */

  bool oldSafeValue = TheSafeToDumpData();
  TheSafeToDumpData() = false;
  //Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_START(tone);
    /* nested */
  for(i=0; i< iterations; i++)
  {
    TAU_PROFILE_START(tnull);
    TAU_PROFILE_STOP(tnull);
  }
  TAU_PROFILE_STOP(tone);
  //Tau_stop_top_level_timer_if_necessary();
  TheSafeToDumpData() = oldSafeValue;

#ifdef TAU_DEPTH_LIMIT
  TauGetDepthLimit() = original; /* reset! */
#endif /* TAU_DEPTH_LIMIT */

  /* Get thread id */
  tid = RtsLayer::myThread();
  int n = ((FunctionInfo*)tnull)->GetCalls(tid);

#ifndef TAU_MULTIPLE_COUNTERS 
  TheTauNullTimerOverhead() = ((FunctionInfo*)tnull)->GetInclTime(tid)/n;
  /* n*(a+b+c+d) + b+c = tone */
  /* a+b+c+d = Toverhead = (tone - tnull) / n */
  TheTauFullTimerOverhead() = (((FunctionInfo*)tone)->GetInclTime(tid) - TheTauNullTimerOverhead() ) / n; 
#else /* TAU_MULTIPLE_COUNTERS */
  double *nullincltime = ((FunctionInfo*)tnull)->GetInclTime(tid);
  double *oneincltime  = ((FunctionInfo*)tone)->GetInclTime(tid);
  for (i=0; i < MAX_TAU_COUNTERS; i++) 
  {
    /* n*(a+b+c+d) + b+c = tone */
    TheTauNullTimerOverhead()[i] = nullincltime[i]/n;

    /* a+b+c+d = Toverhead = (tone - tnull) / n */
    TheTauFullTimerOverhead()[i] = (oneincltime[i] - TheTauNullTimerOverhead()[i]) / n; 
  }
#endif /* TAU_MULTIPLE_COUNTERS */
#ifdef DEBUG_PROF
  cout <<"Calibrate: Tnull time "<< TheTauNullTimerOverhead() <<endl;
  cout <<"Calibrate: Toverhead time = "<<TheTauFullTimerOverhead() <<endl;
#endif /* DEBUG_PROF */
  return 0;
}

#ifndef TAU_MULTIPLE_COUNTERS 
double TauGetTimerOverhead(enum TauOverhead type)
#else /* TAU_MULTIPLE_COUNTERS */
double* TauGetTimerOverhead(enum TauOverhead type)
#endif /* TAU_MULTIPLE_COUNTERS */
{
  static int flag = 0;
  if (flag == 0)
  {
    flag = 1; /* reset it */
    TauCalibrateNullTimer();
  }
 
  /* What kind of overhead are we looking for here? */
  if (type == TauNullTimerOverhead)
  {
    return TheTauNullTimerOverhead();
  }
  else 
  {
    if (type == TauFullTimerOverhead)
      return TheTauFullTimerOverhead();
    else
#ifndef TAU_MULTIPLE_COUNTERS 
      return 0.0;
#else /* TAU_MULTIPLE_COUNTERS */
      return (double *) NULL;
#endif /* TAU_MULTIPLE_COUNTERS */
 
  }

}



