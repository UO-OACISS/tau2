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

#ifndef TAU_WINDOWS
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

#include <signal.h>
#include <Profile/Profiler.h>

/* Which platforms support mallinfo? */
#ifndef TAU_HASMALLINFO
#if (defined (__linux__) || defined (_AIX) || defined(sgi) || \
     defined(sun) || defined (__alpha))
#define TAU_HASMALLINFO 1 
#endif /* platforms */
#endif 
/* TAU_HASMALLINFO: if your platform is not listed here and you know that
   it supports mallinfo system call, please configure with 
   -useropt=-DTAU_HASMALLINFO */
     

#ifdef TAU_HASMALLINFO
#include <malloc.h>
#endif /* TAU_HASMALLINFO */

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
// Is TAU using MUSE's user defined events? Set to true/false.
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingMuseEvents(void)
{
  static bool isit = false; /* TAU is not tracking MUSE events */
  return isit;
}

//////////////////////////////////////////////////////////////////////
// Start tracking memory 
//////////////////////////////////////////////////////////////////////
int TauEnableTrackingMemory(void)
{
  // Set tracking to true
  TheIsTauTrackingMemory() = true;
  return 1; 
}

//////////////////////////////////////////////////////////////////////
// Start tracking MUSE events 
//////////////////////////////////////////////////////////////////////
void TauEnableTrackingMuseEvents(void)
{
  // Set tracking to true
#ifdef TAU_MUSE_EVENT
  TheIsTauTrackingMuseEvents() = true;
#endif /* TAU_MUSE_EVENT */
}

//////////////////////////////////////////////////////////////////////
// Stop tracking memory 
//////////////////////////////////////////////////////////////////////
int TauDisableTrackingMemory(void)
{
  TheIsTauTrackingMemory() = false;
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Stop tracking MUSE events 
//////////////////////////////////////////////////////////////////////
void TauDisableTrackingMuseEvents(void)
{
  TheIsTauTrackingMuseEvents() = false;
}

//////////////////////////////////////////////////////////////////////
// Get memory size (max resident set size) in KB 
//////////////////////////////////////////////////////////////////////
double TauGetMaxRSS(void)
{
#ifdef TAU_HASMALLINFO
  struct mallinfo minfo = mallinfo();
  /* compute the memory used */
  double used = (double) (minfo.hblkhd + minfo.usmblks + minfo.uordblks);
  /* This is in bytes, we need KB */
  return used/1024.0;
#endif /* TAU_HASMALLINFO */

#ifndef TAU_WINDOWS
  /* if not, use getrusage */
  struct rusage res;
  getrusage(RUSAGE_SELF, &res);
  return (double) res.ru_maxrss; /* max resident set size */
#else
  return 0;
#endif
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
  static TauUserEvent mem("Memory Utilization (heap, in KB)");
  return mem;
}

//////////////////////////////////////////////////////////////////////
// TAU's alarm signal handler
//////////////////////////////////////////////////////////////////////
void TauAlarmHandler(int signum)
{
#ifdef TAU_MUSE_EVENT
  double musedata[MAXNUMOF_COUNTERS];
  static int flag = 0; 
  static TauUserEvent *e[MAXNUMOF_COUNTERS];
  int i, numevents;
  if (flag == 0)
  {
    char *eventnames[MAXNUMOF_COUNTERS]; 
    /* toggle flag. This code executes the first time the block is entered */
    flag = 1;
    for (i = 0; i < MAXNUMOF_COUNTERS; i++)
    { /* allocate memory for event names */
      eventnames[i] = new char[MAX_METRIC_LEN];
    }
    /* Fill the eventnames array inside the MUSE call. */
    numevents = TauMuseGetMetricsNonMono(eventnames, MAXNUMOF_COUNTERS);
    for(i=0; i<numevents; i++)
    { /* for the event names that we've received */
      e[i] = new TauUserEvent(eventnames[i]);
      /* we've created e, an array of numevents user defined events */
      /* delete the event name allocated */
      delete eventnames[i]; 
    }

  }

#endif /* TAU_MUSE_EVENT */
   /* Check and see if we're tracking memory events */
  if (TheIsTauTrackingMemory())
  {
    /* trigger an event with the memory used */
    TheTauMemoryEvent().TriggerEvent(TauGetMaxRSS());
  }

#ifdef TAU_MUSE_EVENT 
  if (TheIsTauTrackingMuseEvents())
  { /* get an array of doubles from MUSE */
    numevents = TauMuseEventQuery(musedata, MAXNUMOF_COUNTERS); 
    for (i = 0; i < numevents; i++)
    { /* iterate over numevents and trigger these user defined events */
      e[i]->TriggerEvent(musedata[i]);
    }
  }
#endif /* TAU_MUSE_EVENT */

  /* Set alarm for the next interrupt */
#ifndef TAU_WINDOWS
  alarm(TheTauInterruptInterval());
#endif   
}

//////////////////////////////////////////////////////////////////////
// Track Memory
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryUtilization(void)
{
#ifndef TAU_WINDOWS
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
#endif
}
//////////////////////////////////////////////////////////////////////
// Track Memory events at this location in the source code
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryHere(void)
{
  /* Enable tracking memory by default */
  static int flag = TauEnableTrackingMemory();
 
  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingMemory())
  {
    /* trigger an event with the memory used */
    TheTauMemoryEvent().TriggerEvent(TauGetMaxRSS());
  }
}


//////////////////////////////////////////////////////////////////////
// Track MUSE events
//////////////////////////////////////////////////////////////////////
void TauTrackMuseEvents(void)
{
#ifndef TAU_WINDOWS
  struct sigaction new_action, old_action;

  // we're tracking memory
  TheIsTauTrackingMuseEvents() = true; 

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
#endif
}
  
/***************************************************************************
 * $RCSfile: TauHandler.cpp,v $   $Author: amorris $
 * $Revision: 1.8 $   $Date: 2004/09/01 18:52:35 $
 * POOMA_VERSION_ID: $Id: TauHandler.cpp,v 1.8 2004/09/01 18:52:35 amorris Exp $ 
 ***************************************************************************/

	





