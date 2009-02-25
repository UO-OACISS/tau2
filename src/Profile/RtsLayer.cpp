/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: RtsLayer.cpp					  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#include <tau_internal.h>

//#define DEBUG_PROF
#ifdef TAU_AIX
#include "Profile/aix.h" 
#endif /* TAU_AIX */
#ifdef FUJITSU
#include "Profile/fujitsu.h"
#endif /* FUJITSU */
#ifdef TAU_HITACHI
#include "Profile/hitachi.h"
#endif /* HITACHI */
#include "Profile/Profiler.h"


#if (defined(__QK_USER__) || defined(__LIBCATAMOUNT__ ))
#ifndef TAU_CATAMOUNT
#define TAU_CATAMOUNT 
#endif /* TAU_CATAMOUNT */
#include <catamount/dclock.h>
#endif /* __QK_USER__ || __LIBCATAMOUNT__ */

#ifdef CRAY_TIMERS
#ifndef TAU_CATAMOUNT
/* These header files are for Cray X1 */
#include <intrinsics.h>
#include <sys/param.h>
#endif /* TAU_CATAMOUNT */
#endif // CRAY_TIMERS

#ifdef BGL_TIMERS
/* header files for BlueGene/L */
#include <bglpersonality.h>
#include <rts.h>
#endif // BGL_TIMERS

#ifdef BGP_TIMERS
/* header files for BlueGene/P */
#include <bgp_personality.h>
#include <bgp_personality_inlines.h>
#include <kernel_interface.h>
#endif // BGP_TIMERS

#ifdef TAU_XLC
#define strcasecmp strcmp
#define strncasecmp strncmp 
#endif /* TAU_XLC */


#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <stdio.h> 
#include <fcntl.h>
#include <time.h>
#include <stdlib.h>
#ifdef CPU_TIME
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif // CPU_TIME

#ifdef JAVA_CPU_TIME
#include "Profile/JavaThreadLayer.h"
#endif // JAVA_CPU_TIME


#ifdef TAU_WINDOWS
//include the header for windows time functions.
#include <Windows.h>	//Various defines needed in Winbase.h.
#include <Winbase.h>	//For QueryPerformanceCounter/Frequency function (down to microsecond
                        //resolution depending on the platform. 
#include <sys/timeb.h>	//For _ftime function (millisecond resolution).
//Map strncasecmp and strcasecmp to strnicmp and stricmp.
#define strcasecmp stricmp
#define strncasecmp strnicmp  
#endif //TAU_WINDOWS

#if (!defined(TAU_WINDOWS))
#include <unistd.h>
#include <sys/time.h>
#endif //TAU_WINDOWS

#include <Profile/TauTrace.h>

#ifdef TAUKTAU
#include <Profile/ktau_timer.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <asm/unistd.h>
#endif /* TAUKTAU */


/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class RtsLayer
// Important for Porting to other platforms and frameworks.
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
TauGroup_t& RtsLayer::TheProfileMask(void) { 
  // to avoid initialization problems of non-local static variables
  static TauGroup_t ProfileMask = TAU_DEFAULT;

  return ProfileMask;
}

/////////////////////////////////////////////////////////////////////////
bool& RtsLayer::TheEnableInstrumentation(void) {
  // to avoid initialization problems of non-local static variables
  static bool EnableInstrumentation = true;

  return EnableInstrumentation;
}

/////////////////////////////////////////////////////////////////////////
long RtsLayer::GenerateUniqueId(void) {
  /* This routine is called in a locked region (RtsLayer::LockDB/UnLockDB)*/
  static long UniqueId = 0;
  return ++UniqueId;
}

/////////////////////////////////////////////////////////////////////////
int& RtsLayer::TheNode(void) {
#ifdef TAU_SETNODE0
  static int Node = 0;
#else /* TAU_SETNODE0  */
  static int Node =-1;
#endif /* TAU_SETNODE0 */
 
  return Node;
}

/////////////////////////////////////////////////////////////////////////
int& RtsLayer::TheContext(void) {
  static int Context = 0;
  return Context;
}

/////////////////////////////////////////////////////////////////////////

bool& RtsLayer::TheShutdown(void) {
  static bool shutdown = false;
  return shutdown;
}

/////////////////////////////////////////////////////////////////////////

ProfileMap_t& RtsLayer::TheProfileMap(void) {
  static ProfileMap_t *profilemap = new ProfileMap_t;
  return *profilemap;
}


/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::getProfileGroup(char * ProfileGroup) {
  ProfileMap_t::iterator it = TheProfileMap().find(string(ProfileGroup));
  TauGroup_t gr;
  if (it == TheProfileMap().end()) {
#ifdef DEBUG_PROF
    cout <<ProfileGroup << " not found, adding ... "<<endl;
#endif /* DEBUG_PROF */
    gr = generateProfileGroup();
    TheProfileMap()[string(ProfileGroup)] = gr; // Add
    return gr; 
  } else {
    return (*it).second; // The group that was found
  }

}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::disableProfileGroupName(char * ProfileGroup) {

  return disableProfileGroup(getProfileGroup(ProfileGroup)); 

}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::enableProfileGroupName(char * ProfileGroup) {

  return enableProfileGroup(getProfileGroup(ProfileGroup));

}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::generateProfileGroup(void) {
  static TauGroup_t key =  0x00000001;
  key = key << 1;
  if (key == 0x0) key = 0x1; // cycle
  return key;
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::enableProfileGroup(TauGroup_t ProfileGroup) {
  TheProfileMask() |= ProfileGroup; // Add it to the mask
  DEBUGPROFMSG("enableProfileGroup " << ProfileGroup <<" Mask = " 
	<< TheProfileMask() << endl;);
  return TheProfileMask();
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::enableAllGroups(void) {
  TheProfileMask() = TAU_DEFAULT; // make all bits 1 
  DEBUGPROFMSG("enableAllGroups " << " Mask = " << TheProfileMask() << endl;);
  return TheProfileMask();
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::disableAllGroups(void) {
  TheProfileMask() = 0; // make all bits 1 
  DEBUGPROFMSG("disableAllGroups " << " Mask = " << TheProfileMask() << endl;);
  return TheProfileMask();
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::disableProfileGroup(TauGroup_t ProfileGroup) {
  if (TheProfileMask() & ProfileGroup) { // if it is already set 
    TheProfileMask() ^= ProfileGroup; // Delete it from the mask
    DEBUGPROFMSG("disableProfileGroup " << ProfileGroup <<" Mask = " 
	<< TheProfileMask() << endl;);
  } // if it is not in the mask, disableProfileGroup does nothing 
  return TheProfileMask();
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::resetProfileGroup(void) {
  TheProfileMask() = 0;
  return TheProfileMask();
}

/////////////////////////////////////////////////////////////////////////
int RtsLayer::setMyNode(int NodeId, int tid) {
#if (defined(TRACING_ON) && (TAU_MAX_THREADS != 1))
  int oldid = TheNode();
  int newid = NodeId;
  if ((oldid != -1) && (oldid != newid)) {
    /* ie if SET_NODE macro was invoked twice for a threaded program : as 
       in MPI+JAVA where JAVA initializes it with pid and then MPI_INIT is 
       invoked several thousand events later, and TAU computes the process rank
       and invokes the SET_NODE with the correct rank. Handshaking between multiple
       levels of instrumentation. */
    
    TauTraceReinitialize(oldid, newid, tid); 
  } 
#endif // TRACING WITH THREADS
  TheNode() = NodeId;

  // At this stage, we should create the trace file because we know the node id
  if (TauEnv_get_tracing()) {
    TauTraceInit(tid);
  }

  return TheNode();
}

/////////////////////////////////////////////////////////////////////////
int RtsLayer::setMyContext(int ContextId) {
  TheContext() = ContextId;
  return TheContext();
}

/////////////////////////////////////////////////////////////////////////

bool RtsLayer::isEnabled(TauGroup_t ProfileGroup) {
TauGroup_t res =  ProfileGroup & TheProfileMask() ;

  if (res > 0)
    return true;
  else
    return false;
}

//////////////////////////////////////////////////////////////////////

#ifdef SGI_HW_COUNTERS 
extern "C" {
  int start_counters( int e0, int e1 );
  int read_counters( int e0, long long *c0, int e1, long long *c1);
};
#endif // SGI_HW_COUNTERS

//////////////////////////////////////////////////////////////////////
#ifdef SGI_HW_COUNTERS 
int RtsLayer::SetEventCounter() {
  int e0, e1;
  int start;


  e0 = 0;
  e1 = 0;


  int x0, x1;
  // 
  // DO NOT remove the following two lines. Otherwise start_counters 
  // crashes with "prioctl PIOCENEVCTRS returns error: Invalid argument"


  x0 = e0; 
  x1 = e1; 


  if((start = start_counters(e0,e1)) < 0) {
    perror("start_counters");
    exit(0);
  }
  return start;
}
#endif // SGI_HW_COUNTERS

/////////////////////////////////////////////////////////////////////////
#ifdef SGI_HW_COUNTERS 
double RtsLayer::GetEventCounter() {
  static int gen_start = SetEventCounter();
  int gen_read;
  int e0 = 0, e1 = 0;
  long long c0 , c1 ;
  static double accum = 0;

  if ((gen_read = read_counters(e0, &c0, e1, &c1)) < 0) {
    perror("read_counters");
  }

  if (gen_read != gen_start) {
    perror("lost counter! aborting...");
    exit(1);
  }

  accum += c0;
  DEBUGPROFMSG("Read counters e0 " << e0 <<" e1 "<< e1<<" gen_read " 
    << gen_read << " gen_start = " << gen_start << " accum "<< accum 
    << " c0 " << c0 << " c1 " << c1 << endl;);
  gen_start = SetEventCounter(); // Reset the counter

  return accum;
}
#endif //SGI_HW_COUNTERS

///////////////////////////////////////////////////////////////////////////
double getUserTimeInSec(void) {
  double current_time = 0;
#ifdef CPU_TIME
  
  struct rusage current_usage;

  getrusage (RUSAGE_SELF, &current_usage);
  
/* user time
  current_time = current_usage.ru_utime.tv_sec * 1e6 
	       + current_usage.ru_utime.tv_usec;
*/
  current_time = (current_usage.ru_utime.tv_sec + current_usage.ru_stime.tv_sec)* 1e6 
  + (current_usage.ru_utime.tv_usec + current_usage.ru_stime.tv_usec);
#endif // CPU_TIME
  return current_time; 
}


#if defined(TAUKTAU) || defined(TAUKTAU_MERGE) || defined(TAUKTAU_SHCTR)
///////////////////////////////////////////////////////////////////////////
extern "C" double ktau_get_tsc(void); //must move this into ktau headers. TODO
double KTauGetMHz(void) {
#ifdef KTAU_WALLCLOCK
  static double ktau_ratings = 1; //(microsec resolution from kernel)
#else
#ifdef TAU_mips
  //ON MIPS we have issues with cycles_per_sec - we dont know (yet) how to read cycles from uspace.
  static double ktau_ratings = ktau_get_tsc()/1000000;
#else
  static double ktau_ratings = cycles_per_sec()/1000000; //we need ratings per microsec to match tau's reporting
#endif //TAU_mips
#endif //KTAU_WALLCLOCK
  return ktau_ratings;
}
#endif /* TAUKTAU || TAUKTAU_MERGE */

///////////////////////////////////////////////////////////////////////////
double TauWindowsUsecD(void)
{
#ifdef TAU_WINDOWS
  
  //First need to find out whether we have performance
  //clock, and if so, the frequency.
  static bool PerfClockCheckedBefore = false;
  static bool PerformanceClock = false;
  static LARGE_INTEGER Frequency;
  LARGE_INTEGER ClockValue;
  double FinalClockValue = 0;
  static double Multiplier = 0;

  //Intializing!
  ClockValue.HighPart = 0;
  ClockValue.LowPart = 0;
  ClockValue.QuadPart = 0;

  //Testing clock.  This will only be done ONCE!
  if (!PerfClockCheckedBefore) {
    //Intializing!
    Frequency.HighPart = 0;
    Frequency.LowPart = 0;
    Frequency.QuadPart = 0;
    
    PerformanceClock = QueryPerformanceFrequency(&Frequency);
    PerfClockCheckedBefore = true;
    if (PerformanceClock) {
#ifdef DEBUG_PROF
      cout << "Frequency high part is: " << Frequency.HighPart << endl;
      cout << "Frequency low part is: " << Frequency.LowPart << endl;
      cout << "Frequency quad part is: " << (double) Frequency.QuadPart << endl;			
#endif /* DEBUG_PROF */
      //Shall be using Frequency.QuadPart and assuming a double as the main TAU
      //system does.
      
      //Checking for zero divide ... should not be one if the clock is working,
      //but need to be on the safe side!
      if (Frequency.QuadPart != 0) {
	Multiplier = (double) 1000000/Frequency.QuadPart;
	cout << "The value of the multiplier is: " << Multiplier << endl;
      } else {
	cout << "There was a problem with the counter ... should not have happened!!" << endl;
	return -1;
      }
    } else {
      cout << "No performace clock available ... using millisecond timers." << endl;
    }
  }

  //Getting clock value.
  if (PerformanceClock) {
    if (QueryPerformanceCounter(&ClockValue)) {
      //As mentioned above, assuming double value.
      return Multiplier * (double) ClockValue.QuadPart;
    } else {
      cout << "There was a problem with the counter ... should not have happened!!" << endl;
      return -1;
    }
  } else {
    struct _timeb tp;
    _ftime(&tp);
    return ( (double) tp.time * 1e6 + tp.millitm * 1e3);
  }
#else  /* TAU_WINDOWS */
  return 0; 
#endif /* TAU_WINDOWS */
}
///////////////////////////////////////////////////////////////////////////

#ifdef TAUKTAU_MERGE
  //declare the sys_ktau_gettimeofday syscall
  //#define __NR_ktau_gettimeofday ???
  //_syscall2(int,ktau_gettimeofday,struct timeval *,tv,struct timezone *,tz);
  extern "C" int ktau_gettimeofday(struct timeval *tv, struct timezone *tz);
#endif // TAUKTAU_MERGE 



#ifdef TAU_MULTIPLE_COUNTERS
void RtsLayer::getUSecD (int tid, double *values) {
#if ((defined(TAU_EPILOG) && !defined(PROFILING_ON)) || (defined(TAU_VAMPIRTRACE) && !defined(PROFILING_ON)))
  return;
#endif /* TAU_EPILOG/VAMPIRTRACE, PROFILING_ON */
  MultipleCounterLayer::getCounters(tid, values);
}
#else //TAU_MULTIPLE_COUNTERS


#ifdef TAU_LINUX_TIMERS
extern "C" unsigned long long getLinuxHighResolutionTscCounter(void);
extern "C" double TauGetMHz(void);
#endif

double RtsLayer::getUSecD (int tid) {

#if ((defined(TAU_EPILOG) && !defined(PROFILING_ON)) || (defined(TAU_VAMPIRTRACE) && !defined(PROFILING_ON)))
  return 0;
#endif /* TAU_EPILOG/VAMPIRTRACE, PROFILING_ON */

#ifdef TAU_PAPI
  static const char *papi_env = getenv("PAPI_EVENT");
  if (papi_env != NULL)
    return PapiLayer::getSingleCounter(tid);
#ifdef TAU_PAPI_WALLCLOCKTIME
  return PapiLayer::getWallClockTime();
#else /* TAU_PAPI_WALLCLOCKTIME */
#ifdef TAU_PAPI_VIRTUAL
  return PapiLayer::getVirtualTime();
#else  /* TAU_PAPI_VIRTUAL */
  return PapiLayer::getSingleCounter(tid);
#endif /* TAU_PAPI_VIRTUAL */
#endif /* TAU_PAPI_WALLCLOCKTIME */
#endif  // TAU_PAPI

#ifdef CPU_TIME
  return getUserTimeInSec();
#endif // CPU_TIME

#ifdef JAVA_CPU_TIME
  return JavaThreadLayer::getCurrentThreadCpuTime();
#endif // JAVA_CPU_TIME

#ifdef TAUKTAU_MERGE
  struct timeval tp;
  static double last_timestamp = 0.0;
  double timestamp;
  ktau_gettimeofday (&tp, 0);
  timestamp = (double) tp.tv_sec * 1e6 + tp.tv_usec;
  if (timestamp < last_timestamp) {
     DEBUGPROFMSG("RtsLayer::getUSecD(): ktau_gettimeofday() goes back in time. Fixing ...."<<endl;);
     timestamp = last_timestamp;
  }
  last_timestamp = timestamp;
  return timestamp;
#endif // TAUKTAU_MERGE

#ifdef BGL_TIMERS
  static double bgl_clockspeed = 0.0;

  if (bgl_clockspeed == 0.0) {
    BGLPersonality mybgl;
    rts_get_personality(&mybgl, sizeof(BGLPersonality));
    bgl_clockspeed = 1.0e6/(double)BGLPersonality_clockHz(&mybgl);
  }
  return (rts_get_timebase() * bgl_clockspeed);
#endif // BGL_TIMERS

#ifdef BGP_TIMERS
  static double bgp_clockspeed = 0.0;

  if (bgp_clockspeed == 0.0) {
    _BGP_Personality_t mybgp;
    Kernel_GetPersonality(&mybgp, sizeof(_BGP_Personality_t));
    bgp_clockspeed = 1.0/(double)BGP_Personality_clockMHz(&mybgp);
  }
  return (_bgp_GetTimeBase() * bgp_clockspeed);
#endif // BGP_TIMERS

#ifdef SGI_HW_COUNTERS
  return RtsLayer::GetEventCounter();
#endif  //SGI_HW_COUNTERS

#ifdef SGI_TIMERS
  struct timespec tp;
  clock_gettime(CLOCK_SGI_CYCLE,&tp);
  return (tp.tv_sec * 1e6 + (tp.tv_nsec * 1e-3)) ;
#endif  // SGI_TIMERS

#ifdef CRAY_TIMERS
#ifdef TAU_CATAMOUNT /* for Cray XT3 */
  return dclock()*1.0e6; 
#else /* for Cray X1 */
  long long tick = _rtc();
  return (double) tick/HZ;
#endif /* TAU_CATAMOUNT */
#endif // CRAY_TIMERS

#ifdef TAU_ALPHA_TIMERS
  struct timespec currenttime;
  clock_gettime(CLOCK_REALTIME, &currenttime);
  return (currenttime.tv_sec * 1e6 + (currenttime.tv_nsec * 1e-3));
#endif /* TAU_ALPHA_TIMERS */

#ifdef TAU_LINUX_TIMERS
  static double mhz = TauGetMHz();
  return (double) getLinuxHighResolutionTscCounter()/mhz;
#endif /* TAU_LINUX_TIMERS */

#ifdef TAU_LOGICAL_CLOCK
  static long long value = 0;
  return value++;
#endif

#ifdef TAU_WINDOWS
  return TauWindowsUsecD();

#else /* TAU_WINDOWS */
  // if none of those were defined (the default), we use gettimeofday

  struct timeval tp;
  static double last_timestamp = 0.0;
  double timestamp;
  gettimeofday (&tp, 0);
  timestamp = (double) tp.tv_sec * 1e6 + tp.tv_usec;
  if (timestamp < last_timestamp) {
    DEBUGPROFMSG("RtsLayer::getUSecD(): gettimeofday() goes back in time. Fixing ...."<<endl;);
    timestamp = last_timestamp;
  }
  last_timestamp = timestamp;
  return timestamp;

#endif /* TAU_WINDOWS */

}
#endif //TAU_MULTIPLE_COUNTERS


int RtsLayer::getPid() {
  #ifdef TAU_WINDOWS
  return 0;
  #else
  return getpid();
  #endif
}

// IA64 doesn't like this, commented out for now, it was only for debugging anyway
// #ifdef __linux
// #include <sys/types.h>
// #include <linux/unistd.h>
// _syscall0(pid_t,gettid)
// #endif

int RtsLayer::getTid() {
#ifdef __linux
  //  return gettid();
  return 0;
#else
  return 0;
#endif
}

bool RtsLayer::getCounterUsed(int i) {
#ifdef TAU_MULTIPLE_COUNTERS
  return MultipleCounterLayer::getCounterUsed(i);
#else
  return (i==0); // only 0 is active in single counter mode
#endif	
}

const char *RtsLayer::getCounterName(int i) {
#ifdef TAU_MULTIPLE_COUNTERS
  return MultipleCounterLayer::getCounterNameAt(i);
#else
  return getSingleCounterName();
#endif	
}

const char *RtsLayer::getSingleCounterName() {

#if ((defined(TAU_EPILOG) && !defined(PROFILING_ON)) || (defined(TAU_VAMPIRTRACE) && !defined(PROFILING_ON)))
  return "none";
#endif /* TAU_EPILOG/VAMPIRTRACE, PROFILING_ON */

#ifdef TAU_PAPI
#ifdef TAU_PAPI_WALLCLOCKTIME
  return "P_WALL_CLOCK_TIME";
#else /* TAU_PAPI_WALLCLOCKTIME */
#ifdef TAU_PAPI_VIRTUAL
  return "P_VIRTUAL_TIME";
#else  /* TAU_PAPI_VIRTUAL */
  return getenv("PAPI_EVENT");
#endif /* TAU_PAPI_VIRTUAL */
#endif /* TAU_PAPI_WALLCLOCKTIME */
#endif  // TAU_PAPI

#ifdef CPU_TIME
  return "CPU Time";
#endif // CPU_TIME

#ifdef JAVA_CPU_TIME
  return "Java CPU Time";
#endif // JAVA_CPU_TIME

#ifdef TAUKTAU_MERGE
  return "KTAU_TIME";
#endif // TAUKTAU_MERGE

#ifdef BGL_TIMERS
  return "BGL Timers";
#endif // BGL_TIMERS

#ifdef BGP_TIMERS
  return "BGP Timers";
#endif // BGP_TIMERS

#ifdef SGI_HW_COUNTERS
  return "SGI_HW_COUNTERS";
#endif  //SGI_HW_COUNTERS

#ifdef SGI_TIMERS
  return "SGI Timers";
#endif  // SGI_TIMERS

#ifdef CRAY_TIMERS
  return "Cray Timers";
#endif // CRAY_TIMERS

#ifdef TAU_ALPHA_TIMERS
  return "Alpha Timers";
#endif /* TAU_ALPHA_TIMERS */

#ifdef TAU_LINUX_TIMERS
  return "Linux Timers";
#endif /* TAU_LINUX_TIMERS */

#ifdef TAU_WINDOWS
  return "Time";
#endif // TAU_WINDOWS 

#ifdef TULIP_TIMERS
  return "Tulip Timers";
#endif

#ifdef TAU_LOGICAL_CLOCK
  return "Logical Clock";
#endif
  // if none of those were defined (the default), we use gettimeofday
  return "Time";
}


void RtsLayer::getCurrentValues (int tid, double *values) {

#ifdef TAU_MULTIPLE_COUNTERS
  for (int c=0; c<MAX_TAU_COUNTERS; c++) {
    values[c] = 0;
  }
  return getUSecD(tid, values);
#else
  values[0] = RtsLayer::getUSecD(tid);
#endif
}


///////////////////////////////////////////////////////////////////////////
//Note: This is similar to Tulip event classes during tracing
///////////////////////////////////////////////////////////////////////////
int RtsLayer::setAndParseProfileGroups(char *prog, char *str) {
  char *end;
  
  if ( str ) { 
    while (str && *str) {
      if ( ( end = strchr (str, '+')) != NULL) *end = '\0';
 
      switch ( str[0] ) {
 	case '0' :
	  RtsLayer::enableProfileGroup(TAU_GROUP_0);
	  printf("ENABLING 0!\n");
	  break;
	case '1' : // User1
	  switch (str[1]) {
	    case '0':
	      RtsLayer::enableProfileGroup(TAU_GROUP_10);
	      break; 
	    case '1':
	      RtsLayer::enableProfileGroup(TAU_GROUP_11);
	      break; 
	    case '2':
	      RtsLayer::enableProfileGroup(TAU_GROUP_12);
	      break; 
	    case '3':
	      RtsLayer::enableProfileGroup(TAU_GROUP_13);
	      break; 
	    case '4':
	      RtsLayer::enableProfileGroup(TAU_GROUP_14);
	      break; 
	    case '5':
	      RtsLayer::enableProfileGroup(TAU_GROUP_15);
	      break; 
	    case '6':
	      RtsLayer::enableProfileGroup(TAU_GROUP_16);
	      break; 
	    case '7':
	      RtsLayer::enableProfileGroup(TAU_GROUP_17);
	      break; 
	    case '8':
	      RtsLayer::enableProfileGroup(TAU_GROUP_18);
	      break; 
	    case '9':
	      RtsLayer::enableProfileGroup(TAU_GROUP_19);
	      break; 
	    default :
	      RtsLayer::enableProfileGroup(TAU_GROUP_1);
	      break; 
	  }
	  break;
	 
	case '2' : // User2
          switch (str[1]) {
            case '0':
              RtsLayer::enableProfileGroup(TAU_GROUP_20);
              break;
            case '1':
              RtsLayer::enableProfileGroup(TAU_GROUP_21);
              break;
            case '2':
              RtsLayer::enableProfileGroup(TAU_GROUP_22);
              break;
            case '3':
              RtsLayer::enableProfileGroup(TAU_GROUP_23);
              break;
            case '4':
              RtsLayer::enableProfileGroup(TAU_GROUP_24);
              break;
            case '5':
              RtsLayer::enableProfileGroup(TAU_GROUP_25);
              break;
            case '6':
              RtsLayer::enableProfileGroup(TAU_GROUP_26);
              break; 
            case '7':
              RtsLayer::enableProfileGroup(TAU_GROUP_27);
              break;
            case '8':
              RtsLayer::enableProfileGroup(TAU_GROUP_28);
              break;
            case '9':
              RtsLayer::enableProfileGroup(TAU_GROUP_29);
              break;
            default :
              RtsLayer::enableProfileGroup(TAU_GROUP_2);
              break;
	  }
	  break;
	case '3' : // User3
          switch (str[1]) {
            case '0':
              RtsLayer::enableProfileGroup(TAU_GROUP_30);
              break;
            case '1':
              RtsLayer::enableProfileGroup(TAU_GROUP_31);
              break;
            default :
              RtsLayer::enableProfileGroup(TAU_GROUP_3);
              break;
	  }
	  break;
	case '4' : // User4
	  RtsLayer::enableProfileGroup(TAU_GROUP_4);
	  break;
	case '5' : 
	  RtsLayer::enableProfileGroup(TAU_GROUP_5);
	  break;
	case '6' : 
	  RtsLayer::enableProfileGroup(TAU_GROUP_6);
	  break;
	case '7' : 
	  RtsLayer::enableProfileGroup(TAU_GROUP_7);
	  break;
	case '8' : 
	  RtsLayer::enableProfileGroup(TAU_GROUP_8);
	  break;
	case '9' : 
	  RtsLayer::enableProfileGroup(TAU_GROUP_9);
	  break;

	default  :
	  RtsLayer::enableProfileGroupName(str);
	  break; 
      } 
      if (( str = end) != NULL) *str++ = '+';
    }
  } else {
    enableProfileGroup(TAU_DEFAULT); // Enable everything 
  }
  return 1;
}

//////////////////////////////////////////////////////////////////////
void RtsLayer::ProfileInit(int& argc, char**& argv) {
  int i;
  int ret_argc;
  char **ret_argv;

#ifdef TAU_COMPENSATE
#ifndef TAU_MULTIPLE_COUNTERS
  double tover = TauGetTimerOverhead(TauNullTimerOverhead);
  if (tover < 0) tover = 0;
#else /* TAU_MULTIPLE_COUNTERS */
  double* tover = TauGetTimerOverhead(TauNullTimerOverhead);
  for (i = 0; i < MAX_TAU_COUNTERS; i++) { 
    /* iterate through all counters and reset null overhead to zero 
       if necessary */
    if (tover[i] < 0) tover[i] = 0;
  }
#endif /* TAU_MULTIPLE_COUNTERS */
#endif /* TAU_COMPENSATE */
  
  ret_argc = 1;
  ret_argv = new char *[argc];
  ret_argv[0] = argv[0]; // The program name 

  for(i=1; i < argc; i++) {
    if ( ( strcasecmp(argv[i], "--profile") == 0 ) ) {
        // Enable the profile groups
        if ( (i + 1) < argc && argv[i+1][0] != '-' )  { // options follow
           RtsLayer::resetProfileGroup(); // set it to blank
           RtsLayer::setAndParseProfileGroups(argv[0], argv[i+1]);
	   i++; // ignore the argv after --profile 
        }
    } else {
	ret_argv[ret_argc++] = argv[i];
    }
  }
  argc = ret_argc;
  argv = ret_argv;
  return;
}


//////////////////////////////////////////////////////////////////////
bool RtsLayer::isCtorDtor(const char *name) {
  /* other threads are not affected by this logic. Only on thread 0, do not
     call StoreData() when the name contains a :: and it is a top level routine */

  if ((RtsLayer::myThread() != 0) || (strstr(name, "::") == (char *)NULL)) {
    // if we're not thread 0, or there is no ::, this is definitely not a
    // pre-main ctor/dtor
    return false;
  }
  
  // RtsLayer::myThread() == 0 and there is a :: in the string

  if (strstr(name, "::~") != (char *)NULL) {
    // definitely a dtor
    return true;
  }

  // check the left and right side of the :: and see if they match
  const char *loc = strstr(name, "::");

  // ctor/dtors on thread 0 should return true;
  if (RtsLayer::myThread() == 0 && loc) return true;

  const char *pos1 = name;
  const char *pos2 = loc+2;
  while (pos1 != loc && *pos2 != 0 && *pos1 == *pos2) {
    pos1++;
    pos2++;
  }

  if (pos1 == loc) {
    // probably a ctor (xyz::xyz)
    return true;
  }
  
  return false;
}

//////////////////////////////////////////////////////////////////////
// PrimaryGroup returns the first group that the function belongs to.
// This is needed in tracing as Vampir can handle only one group per
// function. PrimaryGroup("TAU_FIELD | TAU_USER") should return "TAU_FIELD"
//////////////////////////////////////////////////////////////////////
string RtsLayer::PrimaryGroup(const char *ProfileGroupName) {
  string groups = ProfileGroupName;
  string primary; 
  string separators = " |"; 
  int start, stop, n;

  start = groups.find_first_not_of(separators, 0);
  n = groups.length();
  stop = groups.find_first_of(separators, start); 

  if ((stop < 0) || (stop > n)) stop = n;

  primary = groups.substr(start, stop - start) ;
  return primary;

}


//////////////////////////////////////////////////////////////////////



#ifdef __GNUC__
#ifndef NO_RTTI
#include <cxxabi.h>
#endif /* NO_RTTI */
#endif /* __GNUC__ */

/////////////////////////////////////////////////////////////////////////
std::string RtsLayer::GetRTTI(const char *name) {
#ifdef __GNUC__
#ifndef NO_RTTI
  std::size_t len;
  int stat;
  char *ptr = NULL;
  const std::string mangled = name;
  return abi::__cxa_demangle(mangled.c_str(), ptr, &len, &stat);
#else /* NO_RTTI */
  return string(name);
#endif /* NO_RTTI */
#else
  return string(CheckNotNull(name));
#endif /* GNUC */
}

/***************************************************************************
 * $RCSfile: RtsLayer.cpp,v $   $Author: amorris $
 * $Revision: 1.122 $   $Date: 2009/02/25 18:03:34 $
 * POOMA_VERSION_ID: $Id: RtsLayer.cpp,v 1.122 2009/02/25 18:03:34 amorris Exp $ 
 ***************************************************************************/
