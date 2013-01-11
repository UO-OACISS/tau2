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

#if (defined(__QK_USER__) || defined(__LIBCATAMOUNT__ ))
#define TAU_CATAMOUNT 
#endif /* __QK_USER__ || __LIBCATAMOUNT__ */
#ifdef TAU_CATAMOUNT
#include <catamount/catmalloc.h>
#endif /* TAU_CATAMOUNT */


#ifdef TAU_BGP
#include <kernel_interface.h>
#endif

/* Which platforms support mallinfo? */
#ifndef TAU_HASMALLINFO
#if (defined (__linux__) || defined (_AIX) || defined(sgi) || \
    defined (__alpha) || defined (CRAYCC) || defined(__blrts__))
#ifndef TAU_CATAMOUNT
#ifndef TAU_CRAYXMT
#define TAU_HASMALLINFO 1 
#endif /* TAU_CRAYXMT does not have mallinfo */
#endif /* TAU_CATAMOUNT does not have mallinfo */
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
// Get memory size (max resident set size) in KB 
//////////////////////////////////////////////////////////////////////
double TauGetMaxRSS(void)
{
  if (Tau_memory_enabled()) {
    size_t bytes = Tau_get_bytes_allocated();
    double kbytes = (double)bytes / 1024.0;
    return kbytes;
  }

#ifdef TAU_BGP
  uint32_t heap_size;
  Kernel_GetMemorySize( KERNEL_MEMSIZE_HEAP, &heap_size );
  return heap_size / 1024;
#endif

#ifdef TAU_HASMALLINFO
  struct mallinfo minfo = mallinfo();
  double used = (double)((unsigned int)minfo.hblkhd + 0.0 + (unsigned int)minfo.usmblks + (unsigned int)minfo.uordblks);
  /* This is in bytes, we need KB */
  return used / 1024.0;
#else 
#  ifdef TAU_CATAMOUNT
  size_t fragments;
  unsigned long total_free, largest_free, total_used;
  if (heap_info(&fragments, &total_free, &largest_free, &total_used) == 0) {
    return total_used/1024.0;
  }
#  endif /* TAU_CATAMOUNT */

#  if (! (defined (TAU_WINDOWS) || defined (CRAYCC)))
  /* if not, use getrusage */
  struct rusage res;
  getrusage(RUSAGE_SELF, &res);
  return (double) res.ru_maxrss; /* max resident set size */
#  else
  return 0;
#  endif

#endif /* TAU_HASMALLINFO */
}

//////////////////////////////////////////////////////////////////////
// The amount of memory available for use (in MB)
//////////////////////////////////////////////////////////////////////
int TauGetFreeMemory(void)
{

  /* Catamount has a heap_info call that returns the available memory headroom */
#if defined(TAU_CATAMOUNT)
  size_t fragments;
  unsigned long total_free, largest_free, total_used;
  if (heap_info(&fragments, &total_free, &largest_free, &total_used) == 0)
  {  /* return free memory in MB */
    return  (int) (total_free/(1024*1024));
  }
  return 0; /* if it didn't work */
#elif defined(TAU_BGP)
  uint32_t available_heap;
  Kernel_GetMemorySize( KERNEL_MEMSIZE_ESTHEAPAVAIL, &available_heap );
  return available_heap / (1024 * 1024);
#else
#define TAU_BLOCK_COUNT 1024

  char* blocks[TAU_BLOCK_COUNT];
  char* ptr;
  int i,j;
  int freemem = 0;
  int factor = 1;

  i = 0; /* initialize it */
  while (1)
  {
    ptr = (char *) malloc(factor*1024*1024); /* 1MB chunk */
    if (ptr && i < TAU_BLOCK_COUNT)
    { /* so we don't go over the size of the blocks */
      blocks[i] = ptr;
      i++; /* increment the no. of elements in the blocks array */
      freemem += factor; /* assign the MB allocated */
      factor *= 2;  /* try with twice as much the next time */
    }
    else
    {
      if (factor == 1) break; /* out of the loop */
      factor = 1; /* try with a smaller chunk size */
    }
  }

  for (j=0; j < i; j++)
    free(blocks[j]);

  return freemem;

#endif /* TAU_CATAMOUNT */
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
// Get user defined event
//////////////////////////////////////////////////////////////////////
TauUserEvent& TheTauMemoryEvent(void) {
  static TauUserEvent mem("Memory Utilization (heap, in KB)");
  return mem;
}

//////////////////////////////////////////////////////////////////////
// Get user defined event
//////////////////////////////////////////////////////////////////////
TauContextUserEvent& TheTauMemoryHeadroomEvent(void) {
  static TauContextUserEvent mem("Memory Headroom Left (in MB)");
  return mem;
}

//////////////////////////////////////////////////////////////////////
// TAU's alarm signal handler
//////////////////////////////////////////////////////////////////////
void TauAlarmHandler(int signum) {
   /* Check and see if we're tracking memory events */
  if (TheIsTauTrackingMemory()) {
    /* trigger an event with the memory used */
    TheTauMemoryEvent().TriggerEvent(TauGetMaxRSS());
  }

  if (TheIsTauTrackingMemoryHeadroom()) {
    /* trigger an event with the memory headroom available */
    TheTauMemoryHeadroomEvent().TriggerEvent((double)TauGetFreeMemory());
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
    /* trigger an event with the memory used */
    TheTauMemoryEvent().TriggerEvent(TauGetMaxRSS());
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
    /* trigger an event with the memory headroom available */
    TheTauMemoryHeadroomEvent().TriggerEvent((double)TauGetFreeMemory());
  }
}


  
/***************************************************************************
 * $RCSfile: TauHandler.cpp,v $   $Author: amorris $
 * $Revision: 1.24 $   $Date: 2010/05/14 22:21:04 $
 * POOMA_VERSION_ID: $Id: TauHandler.cpp,v 1.24 2010/05/14 22:21:04 amorris Exp $ 
 ***************************************************************************/

	





