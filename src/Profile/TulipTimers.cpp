/******************************************************************************                      TAU Portable Profiling Package                     ****                      http://www.acl.lanl.gov/tau                        *********************************************************************************    Copyright 1997                                                       ****    Department of Computer and Information Science, University of Oregon ****    Advanced Computing Laboratory, Los Alamos National Laboratory        ******************************************************************************/
/********************************************************************/
/* What used to be pcxx_timers.c 				    */
/********************************************************************/

/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1994                  */
/*  Indiana University  University of Oregon  University of Rennes   */
/*********************************************************************/

/*
 * pcxx_timers.c: timer routines for UNIX based multi-processors
 *
 * (c) 1994 Jerry Manic Saftware
 *
 * Version 3.0
 */

/* Written by Bernd Mohr & Pete Beckman */
/* Last overhauled Jun 1994             */

#define PCXX_EVENT_SRC
#ifdef TAU_AIX
#include "Profile/aix.h"
#endif /* TAU_AIX */
#include "Profile/TulipTimers.h"
#ifdef TAU_AIX
#include <sys/resource.h>
#endif /* TAU_AIX */

#include <sys/types.h>

/*********************************************************************/
/* tulip_InitClocks()                                                 */
/*********************************************************************/
static int pcxxT_notinit = 1;

#ifdef __PARAGON__
#  include <math.h>
#  include <i860paragon/rpm.h>

#  define _2_to_52d       (4503599627370496.0)
#  define OR_EXPONENT     (0x4330)
#  define MASK_EXPONENT   (0x000F)

   double pcxxT_start;

   static struct rpm *rpm = (struct rpm *) RPM_BASE_VADDR;

   double pcxxT_dclock() {
     union {
             unsigned short  wordwise[4];
             double  value;
     } t;
     t.value = rpm->rpm_time;
     t.wordwise[3] = (t.wordwise[3] & MASK_EXPONENT) | OR_EXPONENT;
     return (t.value - _2_to_52d);
   }

   void tulip_InitClocks() {
     if ( pcxxT_notinit ) {
       pcxxT_notinit = 0;
       pcxxT_start = floor (pcxxT_dclock()*1.0e-9)*1.0e9;
     }
   }
#else
#ifdef __ksr__
   void tulip_InitClocks() {}
#else
#if defined(_SEQUENT_) || defined(sequent)
   void tulip_InitClocks() {
     if ( pcxxT_notinit ) {
       pcxxT_notinit = 0;
       usclk_init();
     }
   }
#else
#ifdef butterfly
   void tulip_InitClocks() {}
#else
/* added SP1_T instead of SP1 by elj 3/26/97 -- tb0time not available on SP2 */
#ifdef __SP1_T__
#  include <math.h>
   double pcxxT_time;
   double pcxxT_start;

   void tulip_InitClocks() {
     if ( pcxxT_notinit ) {
       pcxxT_notinit = 0;
       tb0time(&pcxxT_time);
       pcxxT_start = floor(pcxxT_time*1.0e-5)*1.0e5;
     }
   }
#else 
#ifdef __SOLARIS2__
#  include <fcntl.h>
   static int pcxxT_fd;

   static void pcxx_InitUserClock() {
     char proc[64];
     pcxxT_notinit = 0;
     sprintf(proc,"/proc/%d", getpid());
     pcxxT_fd = open(proc,O_RDONLY);
   }
#endif
#if !defined( __T3D__) && !defined(_CRAYT3E)

#  include <sys/time.h>

   static long pcxxT_secs = 0L;

   unsigned long int pcxxT_GetClock() {
     struct timeval tp;
     gettimeofday (&tp, 0);
     return ( (tp.tv_sec - pcxxT_secs) * 1000000 + tp.tv_usec );
   }

   void tulip_InitClocks() {
     struct timeval tp;
     if ( pcxxT_notinit ) {
       pcxxT_notinit = 0;
       gettimeofday (&tp, 0);
       pcxxT_secs = tp.tv_sec;
#ifdef __SOLARIS2__
       pcxx_InitUserClock();
#endif
     }
   }
#endif
#endif
#endif
#endif
#endif
#endif

/*********************************************************************/
/* void tulip_UserClock                                               */
/*********************************************************************/
#include <sys/time.h>
#ifdef __SOLARIS2__
/* This is a problem.  Version 2.5 of Solaris (at least) does not have
 * rusage.h, preferring to include the definition of struct rusage in
 * resource.h.  Version 2.3 of Solaris (at least) has resource.h but
 * it does not include struct rusage, which is rather defined in rusage.h
 *
 * The _right_ fix
 */
#  ifdef USE_RUSAGE_H
#    include <sys/rusage.h>
#  else
#    include <sys/resource.h>
#  endif
#  include <sys/procfs.h>
#else
#  include <sys/resource.h>
#endif
 
#ifdef __hpux
#include <sys/syscall.h>
#define getrusage(a,b) syscall(SYS_getrusage,a,b)
#endif

#if defined(_SEQUENT_) || defined(cray) && !defined(__T3D__) && !defined(_CRAYT3E)
#  include <sys/types.h>
//#  include <sys/times.h>
#  include <sys/param.h>

#  if defined(cray)
#    define HZ CLK_TCK
#  endif

/*
   double tulip_UserClock() {
     struct tms timbuf;
     times(&timbuf);
     return ((double) timbuf.tms_utime + timbuf.tms_stime) / (double) HZ;
   }
*/ /* NOTE : times clashes with C++ times. */
#else
#if (defined (CRAYKAI) || defined(CRAYCC))
#else
    double tulip_UserClock() ; { return -1; /* Not implemented for now */ }
#endif /* CRAYKAI || CRAYCC */
#ifdef __SOLARIS2__
   double tulip_UserClock() {
     prusage_t myrusage;
     ioctl(pcxxT_fd, PIOCUSAGE, &myrusage);
     return (double)(myrusage.pr_utime.tv_sec+myrusage.pr_stime.tv_sec) +
       ((double)(myrusage.pr_utime.tv_nsec+myrusage.pr_stime.tv_nsec) * 1.0e-9);
   }
/*
#else
#if !defined(__ksr__) && !defined(__T3D__) && !defined(_CRAYT3E)
   double tulip_UserClock() {
     struct rusage myrusage;
     getrusage(RUSAGE_SELF,&myrusage);
     return (double)(myrusage.ru_utime.tv_sec+myrusage.ru_stime.tv_sec) +
       ((double)(myrusage.ru_utime.tv_usec+myrusage.ru_stime.tv_usec) * 1.0e-6);
   }
#endif
*/
#endif
#endif

/*********************************************************************/
/* tulip_WallClockTimers                                              */
/* tulip_UserClockTimers                                              */
/*********************************************************************/
#define MAXTIMERS 64

#if ! defined(__ksr__) || defined(UNIPROC)
#  define __private
#endif

/* Wall clock timers */
__private static double WTimerStart[MAXTIMERS];
__private static double WTimerStop[MAXTIMERS];
__private static double WTimerElapsed[MAXTIMERS];

/* User clock timers */
#ifndef CM5
__private static double UTimerStart[MAXTIMERS];
__private static double UTimerStop[MAXTIMERS];
__private static double UTimerElapsed[MAXTIMERS];
#endif

#ifdef CM5
#include <cm/cmmd.h>
#endif

int tulip_UserTimerClear(int i) {
  PCXX_EVENT (PCXX_EC_TIMER, PCXX_UTIMER_CLEAR, i);
 
#ifdef CM5
  CMMD_node_timer_clear(i);
#else
  UTimerStart[i]   = 0.0;
  UTimerStop[i]    = 0.0;
  UTimerElapsed[i] = 0.0;
#endif
  return 1;
}


int tulip_UserTimerStart(int i) {
  PCXX_EVENT (PCXX_EC_TIMER, PCXX_UTIMER_START + i, 0);
#ifdef CM5
  return CMMD_node_timer_start(i);
#else
  UTimerStart[i] = tulip_UserClock();
#endif
  return 1;
}

int tulip_UserTimerStop(int i) {
  PCXX_EVENT (PCXX_EC_TIMER, PCXX_UTIMER_STOP + i, 0);
#ifdef CM5
  CMMD_node_timer_stop(i);
#else
  UTimerStop[i]    = tulip_UserClock() - UTimerStart[i];
  UTimerElapsed[i] = UTimerElapsed[i] + UTimerStop[i];
#endif
  return 1;
}

double tulip_UserTimerElapsed(int i) {
#ifdef CM5
  return CMMD_node_timer_elapsed(i);
#else
  return UTimerElapsed[i];
#endif
}

/*******************************************************************/

int tulip_WallTimerClear(int i) {
  PCXX_EVENT (PCXX_EC_TIMER, PCXX_WTIMER_CLEAR, i);
 
  WTimerStart[i]   = 0.0;
  WTimerStop[i]    = 0.0;
  WTimerElapsed[i] = 0.0;
  return 1;
}
 
int tulip_WallTimerStart(int i) {
  PCXX_EVENT (PCXX_EC_TIMER, PCXX_WTIMER_START + i, 0);

  WTimerStart[i] = tulip_WallClock();
  return 1;
}

int tulip_WallTimerStop(int i) {
  PCXX_EVENT (PCXX_EC_TIMER, PCXX_WTIMER_STOP + i, 0);

  WTimerStop[i]    = tulip_WallClock() - WTimerStart[i];
  WTimerElapsed[i] = WTimerElapsed[i] + WTimerStop[i];
  return 1;
}
 
double tulip_WallTimerElapsed(int i) {
  return WTimerElapsed[i];
}
