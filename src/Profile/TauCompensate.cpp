/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
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
int TauCalibrateNullTimer(void)
{
  TAU_PROFILE_TIMER(tnull, ".TAU null timer overhead", " ", TAU_DEFAULT);
  TAU_PROFILE_TIMER(tone,  ".TAU 100 null timers overhead", " ", TAU_DEFAULT);
  int i, tid;
  double tnull, toverhead; 

  TAU_PROFILE_START(tone);
    /* nested */
  for(i=0; i< 1000; i++)
  {
    TAU_PROFILE_START(tnull);
    TAU_PROFILE_STOP(tnull);
  }
  TAU_PROFILE_STOP(tone);
  /* Get thread id */
  tid = RtsLayer::myThread();
  int n = tnullfi.GetCalls(tid);
  TheTauNullTimerOverhead() = tnullfi.GetInclTime(tid)/n;
  /* n*(a+b+c+d) + b+c = tone */
  /* a+b+c+d = Toverhead = (tone - tnull) / n */
  TheTauFullTimerOverhead() = (tonefi.GetInclTime(tid) - TheTauNullTimerOverhead() ) / n; 
//#ifdef DEBUG_PROF
  cout <<"Calibrate: Tnull time "<< TheTauNullTimerOverhead() <<endl;
  cout <<"Calibrate: Toverhead time = "<<TheTauFullTimerOverhead() <<endl;
//#endif /* DEBUG_PROF */
  return 0;
}

double TauGetTimerOverhead(enum TauOverhead type)
{
  static int flag = 0;
  if (flag == 0)
  {
    flag = 1; /* reset it */
    TauCalibrateNullTimer();
  }
 
  /* What kind of overhead are we looking for here? */
  switch (type)
  { 
    case TauNullTimerOverhead: 
      return TheTauNullTimerOverhead();
      break;
    case TauFullTimerOverhead: 
      return TheTauFullTimerOverhead();
      break;
    default:
      return 0.0;
      break;
  }
	
  return 0.0;
}



