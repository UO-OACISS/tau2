/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2004  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauHandler.cpp				  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <Profile/Profiler.h>
#ifdef __linux__
#include <malloc.h> 
#endif /* __linux__ */

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

//////////////////////////////////////////////////////////////////////
// Routines
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Is TAU tracking memory events? Set to true/false.
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingMemory(void)
{
  static bool isit = false; /* TAU is not tracking memory */
  return isit;
}

//////////////////////////////////////////////////////////////////////
// Start tracking memory 
//////////////////////////////////////////////////////////////////////
void TauEnableTrackingMemory(void)
{
  // Set tracking to true
  TheIsTauTrackingMemory() = true;
}

//////////////////////////////////////////////////////////////////////
// Stop tracking memory 
//////////////////////////////////////////////////////////////////////
void TauDisableTrackingMemory(void)
{
  TheIsTauTrackingMemory() = false;
}

//////////////////////////////////////////////////////////////////////
// Get memory size (max resident set size) in KB 
//////////////////////////////////////////////////////////////////////
double TauGetMaxRSS(void)
{
  struct rusage res;
  getrusage(RUSAGE_SELF, &res);
  if (res.ru_maxrss == 0)
  { /* getrusage is not working */
#ifdef __linux__
    struct mallinfo minfo = mallinfo();
    /* compute the memory used */
    double used = (double) (minfo.hblkhd + minfo.usmblks + minfo.uordblks);
    /* This is in bytes, we need KB */
    return used/1024.0;
#endif /* __linux__ */
  }
     
  return (double) res.ru_maxrss; /* max resident set size */
}

//////////////////////////////////////////////////////////////////////
// Set interrupt interval
//////////////////////////////////////////////////////////////////////
int& TheTauInterruptInterval(void)
{ 
  static int interval = 10; /* interrupt every 10 seconds */
  return interval; 
}

//////////////////////////////////////////////////////////////////////
// Set interrupt interval
//////////////////////////////////////////////////////////////////////
void TauSetInterruptInterval(int interval)
{
  /* Set the interval */
  TheTauInterruptInterval() = interval;
}

//////////////////////////////////////////////////////////////////////
// Get user defined event
//////////////////////////////////////////////////////////////////////
TauUserEvent& TheTauMemoryEvent(void)
{
  static TauUserEvent mem("Memory Utilization (in KB)");
  return mem;
}

//////////////////////////////////////////////////////////////////////
// TAU's alarm signal handler
//////////////////////////////////////////////////////////////////////
void TauAlarmHandler(int signum)
{
   /* Check and see if we're tracking memory events */
  if (TheIsTauTrackingMemory())
  {
    /* trigger an event with the memory used */
    TheTauMemoryEvent().TriggerEvent(TauGetMaxRSS());
  }

  /* Set alarm for the next interrupt */
  alarm(TheTauInterruptInterval());
   
}
//////////////////////////////////////////////////////////////////////
// Track Memory
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryUtilization(void)
{
  struct sigaction new_action, old_action;

  // we're tracking memory
  TheIsTauTrackingMemory() = true; 

  // set signal handler 
  new_action.sa_handler = TauAlarmHandler; 
 
  new_action.sa_flags = 0;
  sigaction(SIGALRM, NULL, &old_action);
  if (old_action.sa_handler != SIG_IGN)
  { /* by default it is set to ignore */
    sigaction(SIGALRM, &new_action, NULL);
  }
  
  /* activate alarm */
  alarm(TheTauInterruptInterval());
}
  
/***************************************************************************
 * $RCSfile: TauHandler.cpp,v $   $Author: sameer $
 * $Revision: 1.2 $   $Date: 2004/03/02 01:46:35 $
 * POOMA_VERSION_ID: $Id: TauHandler.cpp,v 1.2 2004/03/02 01:46:35 sameer Exp $ 
 ***************************************************************************/

	





