/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: RtsLayer.cpp					  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**			  -DSGI_TIMERS  for SGI fast nanosecs timer	  **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF
#ifdef TAU_AIX
#include "Profile/aix.h" 
#endif /* TAU_AIX */
#ifdef FUJITSU
#include "Profile/fujitsu.h"
#endif /* FUJITSU */
#include "Profile/Profiler.h"

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

#if (defined(POOMA_TFLOP) || !defined(TULIP_TIMERS))
#include <sys/time.h>
#else
#ifdef TULIP_TIMERS 
#include "Profile/TulipTimers.h"
#endif //TULIP_TIMERS 
#endif //POOMA_TFLOP

#endif //TAU_WINDOWS

#ifdef TRACING_ON
#define PCXX_EVENT_SRC
#include "Profile/pcxx_events.h"
#endif // TRACING_ON 


/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class RtsLayer
// Important for Porting to other platforms and frameworks.
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
TauGroup_t& RtsLayer::TheProfileMask(void)
{ // to avoid initialization problems of non-local static variables
  static TauGroup_t ProfileMask = TAU_DEFAULT;

  return ProfileMask;
}

/////////////////////////////////////////////////////////////////////////
bool& RtsLayer::TheEnableInstrumentation(void)
{ // to avoid initialization problems of non-local static variables
  static bool EnableInstrumentation = true;

  return EnableInstrumentation;
}

/////////////////////////////////////////////////////////////////////////
int& RtsLayer::TheNode(void)
{
  static int Node = -1;
 
  return Node;
}

/////////////////////////////////////////////////////////////////////////
int& RtsLayer::TheContext(void)
{
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
  if (it == TheProfileMap().end())
  {
#ifdef DEBUG_PROF
    cout <<ProfileGroup << " not found, adding ... "<<endl;
#endif /* DEBUG_PROF */
    gr = generateProfileGroup();
    TheProfileMap()[string(ProfileGroup)] = gr; // Add
    return gr; 
  }
  else
    return (*it).second; // The group that was found

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
  if ((oldid != -1) && (oldid != newid))
  { /* ie if SET_NODE macro was invoked twice for a threaded program : as 
    in MPI+JAVA where JAVA initializes it with pid and then MPI_INIT is 
    invoked several thousand events later, and TAU computes the process rank
    and invokes the SET_NODE with the correct rank. Handshaking between multiple
    levels of instrumentation. */
    
    TraceReinitialize(oldid, newid, tid); 
  } 
#endif // TRACING WITH THREADS
  TheNode() = NodeId;
// At this stage, we should create the trace file because we know the node id
#ifdef TRACING_ON
  TraceEvInit(tid);
#endif // TRACING_ON
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
int RtsLayer::SetEventCounter()
{
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
double RtsLayer::GetEventCounter()
{
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
double getUserTimeInSec(void)
{
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

///////////////////////////////////////////////////////////////////////////
inline double TauGetMHzRatings(void)
{
  FILE *f;
  double rating;
  char *cmd = "cat /proc/cpuinfo | egrep -i '^cpu MHz' | head -1 | sed 's/^.*: //'";
  char buf[BUFSIZ];
  if ((f = popen(cmd,"r"))!=NULL)
  while (fgets(buf, BUFSIZ, f) != NULL)
  {
    rating = atof(buf);
  }
  pclose(f);
#ifdef DEBUG_PROF
  printf("Rating = %g Mhz\n", rating);
#endif /* DEBUG_PROF */
  return rating;
}

  
#ifdef TAU_LINUX_TIMERS
///////////////////////////////////////////////////////////////////////////
inline double TauGetMHz(void)
{
  static double ratings = TauGetMHzRatings();
  return ratings;
}
///////////////////////////////////////////////////////////////////////////
inline unsigned long long getLinuxHighResolutionTscCounter(void)
{
   unsigned long high, low;
   __asm__ __volatile__(".byte 0x0f,0x31" : "=a" (low), "=d" (high));
   return ((unsigned long long) high << 32) + low;
}

#endif /* TAU_LINUX_TIMERS */

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
  if(!PerfClockCheckedBefore)
  {
	  //Intializing!
	  Frequency.HighPart = 0;
	  Frequency.LowPart = 0;
	  Frequency.QuadPart = 0;
	  
	  PerformanceClock = QueryPerformanceFrequency(&Frequency);
	  PerfClockCheckedBefore = true;
	  if(PerformanceClock)
	  {
#ifdef DEBUG_PROF
		  cout << "Frequency high part is: " << Frequency.HighPart << endl;
		  cout << "Frequency low part is: " << Frequency.LowPart << endl;
		  cout << "Frequency quad part is: " << (double) Frequency.QuadPart << endl;			
#endif /* DEBUG_PROF */
		  //Shall be using Frequency.QuadPart and assuming a double as the main TAU
		  //system does.
		  
		  //Checking for zero divide ... should not be one if the clock is working,
		  //but need to be on the safe side!
		  if(Frequency.QuadPart != 0)
		  {
			  Multiplier = (double) 1000000/Frequency.QuadPart;
			  cout << "The value of the multiplier is: " << Multiplier << endl;
		  }
		  else
		  {
			  cout << "There was a problem with the counter ... should not have happened!!" << endl;
			  return -1;
		  }
	  }
	  else
		  cout << "No performace clock available ... using millisecond timers." << endl;
  }

  //Getting clock value.
  if(PerformanceClock)
  {
	  if(QueryPerformanceCounter(&ClockValue))
	  {
		  //As mentioned above, assuming double value.
		  return Multiplier * (double) ClockValue.QuadPart;
	  }
	  else
	  {
		  cout << "There was a problem with the counter ... should not have happened!!" << endl;
		  return -1;
	  }
  }
  else
  {
	  struct _timeb tp;
	  _ftime(&tp);
	  return ( (double) tp.time * 1e6 + tp.millitm * 1e3);
  }
#else  /* TAU_WINDOWS */
  return 0; 
#endif /* TAU_WINDOWS */
}
///////////////////////////////////////////////////////////////////////////

double RtsLayer::getUSecD (int tid) {

#ifdef TAU_PCL
  return PCL_Layer::getCounters(tid);
#else  // TAU_PCL 
#ifdef TAU_PAPI
#ifdef TAU_PAPI_WALLCLOCKTIME
  return PapiLayer::getWallClockTime();
#else /* TAU_PAPI_WALLCLOCKTIME */
#ifdef TAU_PAPI_VIRTUAL
  return PapiLayer::getVirtualTime();
#else  /* TAU_PAPI_VIRTUAL */
  return PapiLayer::getCounters(tid);
#endif /* TAU_PAPI_VIRTUAL */
#endif /* TAU_PAPI_WALLCLOCKTIME */
#else  // TAU_PAPI
#ifdef CPU_TIME
  return getUserTimeInSec();
#else // CPU_TIME
#ifdef SGI_HW_COUNTERS
  return RtsLayer::GetEventCounter();
#else  //SGI_HW_COUNTERS

#ifdef SGI_TIMERS
  struct timespec tp;
  clock_gettime(CLOCK_SGI_CYCLE,&tp);
  return (tp.tv_sec * 1e6 + (tp.tv_nsec * 1e-3)) ;

#else  // SGI_TIMERS
#ifdef TAU_LINUX_TIMERS
  return (double) getLinuxHighResolutionTscCounter()/TauGetMHz();
#else /* TAU_LINUX_TIMERS */
#if (defined(POOMA_TFLOP) || !defined(TULIP_TIMERS)) 
#if (defined(TAU_WINDOWS))
  return TauWindowsUsecD();
#else // TAU_WINDOWS 
  struct timeval tp;
  gettimeofday (&tp, 0);
  return ( (double) tp.tv_sec * 1e6 + tp.tv_usec );
#endif // TAU_WINDOWS
#else  // TULIP_TIMERS by default.  
  return pcxx_GetUSecD();
#endif  //POOMA_TFLOP
#endif /* TAU_LINUX_TIMERS */
#endif 	//SGI_TIMERS

#endif  // SGI_HW_COUNTERS
#endif // CPU_TIME
#endif // TAU_PAPI
#endif  // TAU_PCL
        }

///////////////////////////////////////////////////////////////////////////
//Note: This is similar to Tulip event classes during tracing
///////////////////////////////////////////////////////////////////////////
int RtsLayer::setAndParseProfileGroups(char *prog, char *str)
{
  char *end;
  
  if ( str )
  { 
    while (str && *str) 
    {
      if ( ( end = strchr (str, '+')) != NULL) *end = '\0';
 
      switch ( str[0] )
      {
	      /*
        case 'a' :
	case 'A' : // Assign Expression Evaluation Group
	  if (strncasecmp(str,"ac", 2) == 0) {
	    RtsLayer::enableProfileGroup(TAU_ACLMPL); 
	    // ACLMPL enabled 
	  } 
	  else 
	    RtsLayer::enableProfileGroup(TAU_ASSIGN);
	  break;
	case 'b' : 
	case 'B' : // Blitz++ profile group
	  RtsLayer::enableProfileGroup(TAU_BLITZ);
	  break; // Blitz++ enabled
        case 'f' :
	case 'F' : // Field Group
	  if (strncasecmp(str, "ff", 2) == 0) {
	    RtsLayer::enableProfileGroup(TAU_FFT);
	    // FFT enabled 
	  }
	  else 
	    RtsLayer::enableProfileGroup(TAU_FIELD);
	    // Field enabled 
	  break;
	case 'c' :
	case 'C' : 
	  RtsLayer::enableProfileGroup(TAU_COMMUNICATION);
	  break;
 	case 'h' :
	case 'H' :
	  RtsLayer::enableProfileGroup(TAU_HPCXX);
	  break;
        case 'i' :
	case 'I' : // DiskIO, Other IO 
	  RtsLayer::enableProfileGroup(TAU_IO);
	  break;
        case 'l' :
	case 'L' : // Field Layout Group
	  RtsLayer::enableProfileGroup(TAU_LAYOUT);
	  break;
	case 'm' : 
	case 'M' : 
          if (strncasecmp(str,"mesh", 4) == 0) {
  	    RtsLayer::enableProfileGroup(TAU_MESHES);
	    // Meshes enabled
 	  } 
 	  else 
	    RtsLayer::enableProfileGroup(TAU_MESSAGE);
	    // Message Profile Group enabled 
  	  break;
        case 'p' :
	case 'P' : 
          if (strncasecmp(str, "paws1", 5) == 0) {
	    RtsLayer::enableProfileGroup(TAU_PAWS1); 
	  } 
	  else {
	    if (strncasecmp(str, "paws2", 5) == 0) {
	      RtsLayer::enableProfileGroup(TAU_PAWS2); 
	    } 
	    else {
	      if (strncasecmp(str, "paws3", 5) == 0) {
	        RtsLayer::enableProfileGroup(TAU_PAWS3); 
	      } 
	      else {
	        if (strncasecmp(str,"pa",2) == 0) {
	          RtsLayer::enableProfileGroup(TAU_PARTICLE);
	          // Particle enabled 
	        } 
		else {
	          RtsLayer::enableProfileGroup(TAU_PETE);
	    	  // PETE Profile Group enabled 
	 	}
	      }
	    } 
 	  } 
	  
	  break;
  	case 'r' : 
	case 'R' : // Region Group 
	  RtsLayer::enableProfileGroup(TAU_REGION);
	  break;
        case 's' :
	case 'S' : 
	  if (strncasecmp(str,"su",2) == 0) {
	    RtsLayer::enableProfileGroup(TAU_SUBFIELD);
	    // SubField enabled 
	  } 
 	  else
	    RtsLayer::enableProfileGroup(TAU_SPARSE);
	    // Sparse Index Group
	  break;
        case 'd' :
	case 'D' : // Domainmap Group
	  if (strncasecmp(str,"de",2) == 0) {
	    RtsLayer::enableProfileGroup(TAU_DESCRIPTOR_OVERHEAD);
	  } else  
	     RtsLayer::enableProfileGroup(TAU_DOMAINMAP);
	  break;
 	case 'u' :
        case 'U' : // User or Utility 
          if (strncasecmp(str,"ut", 2) == 0) { 
	    RtsLayer::enableProfileGroup(TAU_UTILITY);
	  }
	  else // default - for u is USER 
 	    RtsLayer::enableProfileGroup(TAU_USER);
	  break;
        case 'v' :
	case 'V' : // ACLVIZ Group
	  RtsLayer::enableProfileGroup(TAU_VIZ);
	  break;
	  // Old stuff. Delete it!
	  */
 	case '0' :
	  RtsLayer::enableProfileGroup(TAU_GROUP_0);
	  break;
	case '1' : // User1
	  switch (str[1])
	  {
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
          switch (str[1])
          {
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
          switch (str[1])
          {
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
  }
  else 
    enableProfileGroup(TAU_DEFAULT); // Enable everything 
  return 1;
}

//////////////////////////////////////////////////////////////////////
void RtsLayer::ProfileInit(int& argc, char**& argv)
{
  int i;
  int ret_argc;
  char **ret_argv;
  
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
    }
    else
    {
	ret_argv[ret_argc++] = argv[i];
    }
  }
  argc = ret_argc;
  argv = ret_argv;
  return;
}


//////////////////////////////////////////////////////////////////////
bool RtsLayer::isCtorDtor(const char *name)
{

  // If the destructor a static object is called, it could have a null name
  // after main is over. Treat it like a Dtor and return true.
  if (name[0] == 0) {
    DEBUGPROFMSG("isCtorDtor name is NULL" << endl;);
    return true; 
  }
  DEBUGPROFMSG("RtsLayer::isCtorDtor("<< name <<")" <<endl;);
#ifdef JAVA
      /* In Java : is allowed, as the top level thread name could be something 
	 like THREAD=RMI ConnectionExpiration-[192.168.180.1:1871]; */
      DEBUGPROFMSG("JAVA RtsLayer::isCtorDtor("<<name<<") returns false! "<<endl;);
      return false;
#endif /* JAVA */
  if (myThread() == 0) 
  {
    if (strchr(name,'~') == NULL) // a destructor 
      if (strchr(name,':') == NULL) // could be a constructor 
        return false;
      else  
        return true; /* in C++ a : in a name could signify a ctor/dtor */
    else  
      return true;
  }
  else 
    return false; /* in C++, on other threads don't check for this */
}

//////////////////////////////////////////////////////////////////////
// PrimaryGroup returns the first group that the function belongs to.
// This is needed in tracing as Vampir can handle only one group per
// function. PrimaryGroup("TAU_FIELD | TAU_USER") should return "TAU_FIELD"
//////////////////////////////////////////////////////////////////////
string RtsLayer::PrimaryGroup(const char *ProfileGroupName) 
{
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
// TraceSendMsg traces the message send
//////////////////////////////////////////////////////////////////////
void RtsLayer::TraceSendMsg(int type, int destination, int length)
{
#ifdef TRACING_ON 
  long int parameter, othernode;

  if (RtsLayer::isEnabled(TAU_MESSAGE))
  {
    parameter = 0L;
    /* for send, othernode is receiver or destination */
    othernode = (long int) destination;
    /* Format for parameter is
       31 ..... 24 23 ......16 15..............0
          other       type          length       
     */
  
    parameter = (length & 0x0000FFFF) | ((type & 0x000000FF)  << 16) | 
  	      (othernode << 24);
    pcxx_Event(TAU_MESSAGE_SEND, parameter); 
#ifdef DEBUG_PROF
    printf("Node %d TraceSendMsg, type %x dest %x len %x par %lx \n", 
  	RtsLayer::myNode(), type, destination, length, parameter);
#endif //DEBUG_PROF
  } 
#endif //TRACING_ON
}

  
//////////////////////////////////////////////////////////////////////
// TraceRecvMsg traces the message recv
//////////////////////////////////////////////////////////////////////
void RtsLayer::TraceRecvMsg(int type, int source, int length)
{
#ifdef TRACING_ON
  long int parameter, othernode;

  if (RtsLayer::isEnabled(TAU_MESSAGE)) 
  {
    parameter = 0L;
    /* for recv, othernode is sender or source*/
    othernode = (long int) source;
    /* Format for parameter is
       31 ..... 24 23 ......16 15..............0
          other       type          length       
     */
  
    parameter = (length & 0x0000FFFF) | ((type & 0x000000FF)  << 16) | 
  	      (othernode << 24);
    pcxx_Event(TAU_MESSAGE_RECV, parameter); 
  
#ifdef DEBUG_PROF
    printf("Node %d TraceRecvMsg, type %x src %x len %x par %lx \n", 
  	RtsLayer::myNode(), type, source, length, parameter);
#endif //DEBUG_PROF
  }
#endif //TRACING_ON
}

//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// DumpEDF() writes the function information in the edf.<node> file
// The function info consists of functionId, group, name, type, parameters
//////////////////////////////////////////////////////////////////////
int RtsLayer::DumpEDF(int tid)
{
#ifdef TRACING_ON 
  	vector<FunctionInfo*>::iterator it;
	char filename[1024], errormsg[1024];
	char *dirname;
	FILE* fp;
	int  numEvents, numExtra;


	if (tid != 0) 
	  return 1; 
	// Only thread 0 on a node should write the edf files.
	if ((dirname = getenv("TRACEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}

	sprintf(filename,"%s/events.%d.edf",dirname, RtsLayer::myNode());
	DEBUGPROFMSG("Creating " << filename << endl;);
	if ((fp = fopen (filename, "w+")) == NULL) {
		sprintf(errormsg,"Error: Could not create %s",filename);
		perror(errormsg);
		return 0;
	}
	
	// Data Format 
	// <no.> events
	// # or \n ignored
	// %s %s %d "%s %s" %s 
	// id group tag "name type" parameters

	numExtra = 9; // Number of extra events
	numEvents = TheFunctionDB().size();

	numEvents += numExtra;

	fprintf(fp,"%d dynamic_trace_events\n", numEvents);

	fprintf(fp,"# FunctionId Group Tag \"Name Type\" Parameters\n");

 	for (it = TheFunctionDB().begin(); 
	  it != TheFunctionDB().end(); it++)
	{
  	  DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping EDF Id : " 
	    << (*it)->GetFunctionId() << " " << (*it)->GetPrimaryGroup() 
	    << " 0 " << (*it)->GetName() << " " << (*it)->GetType() 
	    << " EntryExit" << endl;); 
	
	  fprintf(fp, "%ld %s 0 \"%s %s\" EntryExit\n", (*it)->GetFunctionId(),
	    (*it)->GetPrimaryGroup(), (*it)->GetName(), (*it)->GetType() );
	}
	// Now add the nine extra events 
	fprintf(fp,"%ld TRACER 0 \"EV_INIT\" none\n", (long) PCXX_EV_INIT); 
	fprintf(fp,"%ld TRACER 0 \"FLUSH_ENTER\" none\n", (long) PCXX_EV_FLUSH_ENTER); 
	fprintf(fp,"%ld TRACER 0 \"FLUSH_EXIT\" none\n", (long) PCXX_EV_FLUSH_EXIT); 
	fprintf(fp,"%ld TRACER 0 \"FLUSH_CLOSE\" none\n", (long) PCXX_EV_CLOSE); 
	fprintf(fp,"%ld TRACER 0 \"FLUSH_INITM\" none\n", (long) PCXX_EV_INITM); 
	fprintf(fp,"%ld TRACER 0 \"WALL_CLOCK\" none\n", (long) PCXX_EV_WALL_CLOCK); 
	fprintf(fp,"%ld TRACER 0 \"CONT_EVENT\" none\n", (long) PCXX_EV_CONT_EVENT); 
	fprintf(fp,"%ld TAU_MESSAGE -7 \"MESSAGE_SEND\" par\n", (long) TAU_MESSAGE_SEND); 
	fprintf(fp,"%ld TAU_MESSAGE -8 \"MESSAGE_RECV\" par\n", (long) TAU_MESSAGE_RECV); 

  
	fclose(fp);
#endif //TRACING_ON
	return 1;
}

/***************************************************************************
 * $RCSfile: RtsLayer.cpp,v $   $Author: sameer $
 * $Revision: 1.35 $   $Date: 2002/01/24 23:44:53 $
 * POOMA_VERSION_ID: $Id: RtsLayer.cpp,v 1.35 2002/01/24 23:44:53 sameer Exp $ 
 ***************************************************************************/
