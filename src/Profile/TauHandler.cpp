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
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
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
#include <Profile/TauMemory.h>


//////////////////////////////////////////////////////////////////////
// Routines
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Is TAU tracking memory events? Set to true/false.
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingMemory(void) {
  static bool isit = false; /* TAU is not tracking memory */
  return isit;
}

//////////////////////////////////////////////////////////////////////
// Is TAU tracking memory headroom events? Set to true/false.
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingMemoryHeadroom(void) {
  static bool isit = false; /* TAU is not tracking memory headroom */
  return isit;
}

//////////////////////////////////////////////////////////////////////
// Start tracking memory 
//////////////////////////////////////////////////////////////////////
int TauEnableTrackingMemory(void) {
  // Set tracking to true
  TheIsTauTrackingMemory() = true;
  return 1; 
}

//////////////////////////////////////////////////////////////////////
// Start tracking memory 
//////////////////////////////////////////////////////////////////////
int TauEnableTrackingMemoryHeadroom(void) {
  // Set tracking to true
  TheIsTauTrackingMemoryHeadroom() = true;
  return 1; 
}

//////////////////////////////////////////////////////////////////////
// Stop tracking memory 
//////////////////////////////////////////////////////////////////////
int TauDisableTrackingMemory(void) {
  TheIsTauTrackingMemory() = false;
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Stop tracking memory headroom
//////////////////////////////////////////////////////////////////////
int TauDisableTrackingMemoryHeadroom(void) {
  TheIsTauTrackingMemoryHeadroom() = false;
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Set interrupt interval
//////////////////////////////////////////////////////////////////////
int& TheTauInterruptInterval(void) { 
  static int interval = 10; /* interrupt every 10 seconds */
  return interval; 
}

//////////////////////////////////////////////////////////////////////
// Set interrupt interval
//////////////////////////////////////////////////////////////////////
void TauSetInterruptInterval(int interval) {
  /* Set the interval */
  TheTauInterruptInterval() = interval;
}

//////////////////////////////////////////////////////////////////////
// TAU's alarm signal handler
//////////////////////////////////////////////////////////////////////
void TauAlarmHandler(int signum) {
   /* Check and see if we're tracking memory events */
  if (TheIsTauTrackingMemory()) {
    TauAllocation::TriggerHeapMemoryUsageEvent();
  }

  if (TheIsTauTrackingMemoryHeadroom()) {
    TauAllocation::TriggerMemoryHeadroomEvent();
  }

  /* Set alarm for the next interrupt */
#ifndef TAU_WINDOWS
  alarm(TheTauInterruptInterval());
#endif   
}

//////////////////////////////////////////////////////////////////////
// Track Memory
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryUtilization(bool allocated) {
//////////////////////////////////////////////////////////////////////
// Argument: allocated. TauTrackMemoryUtilization can keep track of memory
// allocated or memory free (headroom to grow). Accordingly, it is true
// for tracking memory allocated, and false to check the headroom 
//////////////////////////////////////////////////////////////////////

#ifndef TAU_WINDOWS
  struct sigaction new_action, old_action;

  // Are we tracking memory or headroom. Check the allocated argument. 
  if (allocated) {
    TheIsTauTrackingMemory() = true; 
  } else {
    TheIsTauTrackingMemoryHeadroom() = true; 
  }

  // set signal handler 
  new_action.sa_handler = TauAlarmHandler; 
 
  new_action.sa_flags = 0;
  sigaction(SIGALRM, NULL, &old_action);
  if (old_action.sa_handler != SIG_IGN) {
    /* by default it is set to ignore */
    sigaction(SIGALRM, &new_action, NULL);
  }
  
  /* activate alarm */
  alarm(TheTauInterruptInterval());
#endif
}

//////////////////////////////////////////////////////////////////////
// Track Memory events at this location in the source code
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryHere(void) {
  /* Enable tracking memory by default */
  static int flag = TauEnableTrackingMemory();
 
  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingMemory()) {
    TauAllocation::TriggerHeapMemoryUsageEvent();
  }
}

//////////////////////////////////////////////////////////////////////
// Track Memory headroom events at this location in the source code
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryHeadroomHere(void) {
  /* Enable tracking memory by default */
  static int flag = TauEnableTrackingMemoryHeadroom();
 
  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingMemoryHeadroom()) {
    TauAllocation::TriggerMemoryHeadroomEvent();
  }
}


  
/***************************************************************************
 * $RCSfile: TauHandler.cpp,v $   $Author: amorris $
 * $Revision: 1.24 $   $Date: 2010/05/14 22:21:04 $
 * POOMA_VERSION_ID: $Id: TauHandler.cpp,v 1.24 2010/05/14 22:21:04 amorris Exp $ 
 ***************************************************************************/

	





