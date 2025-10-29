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
#include <Profile/TauMetrics.h>
#include <Profile/TauBfd.h>

//#define DEBUG_PROF
#ifdef TAU_AIX
#include "Profile/aix.h"
#endif /* TAU_AIX */
#ifdef TAU_FUJITSU
#include "Profile/fujitsu.h"
#endif /* TAU_FUJITSU */
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
#ifndef TAU_WINDOWS
#ifndef TAU_XLC
#ifndef TAU_AIX
#include <sys/syscall.h>
#endif /* TAU_AIX */
#endif /* TAU_XLC */
#endif /* TAU_WINDOWS */

#ifdef KTAU_NG
#ifdef __linux //To make getLinuxKernelTid work for ktau style file naming
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#endif
#endif /* KTAU_NG */

#ifdef TAU_WINDOWS
//include the header for windows time functions.
#include <windows.h>	//Various defines needed in Winbase.h.
#include <winbase.h>	//For QueryPerformanceCounter/Frequency function (down to microsecond
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

std::mutex RtsLayer::DBVectorMutex;

std::atomic<int> RtsLayer::maxLockTid(-1);
/////////////////////////////////////////////////////////////////////////
TauGroup_t& RtsLayer::TheProfileMask(void) {
  // to avoid initialization problems of non-local static variables
  static TauGroup_t ProfileMask = TAU_DEFAULT;

  return ProfileMask;
}

/////////////////////////////////////////////////////////////////////////
TauGroup_t& RtsLayer::TheProfileBlackMask(void) {
  // to avoid initialization problems of non-local static variables
  // This is set to TAU_EXCLUDE if an event is deactivated
  static TauGroup_t ProfileBlackMask = 0;

  return ProfileBlackMask;
}

/////////////////////////////////////////////////////////////////////////
std::atomic<bool>& RtsLayer::TheExcludeDefaultGroup(void) {
  static std::atomic<bool> is_default_group_excluded = { false };
  return is_default_group_excluded;
}

// --- For Thread Exclusion ---
//static std::unordered_set<int> gex_thread_set;

//static SpatialExclusion gex_thread_mode = SpatialExclusionMode::UNSET;

// --- For MPI Rank Exclusion ---
std::unordered_set<int>& RtsLayer::TheRankExclusionSet(void)
{
	static std::unordered_set<int> exclusion_rank_set;
	return exclusion_rank_set;
}

RtsLayer::SpatialExclusionMode& RtsLayer::TheRankExclusionMode(void){
	static SpatialExclusionMode rankex_mode=SpatialExclusionMode::UNSET;
	return rankex_mode;
}

std::atomic<uint64_t>& RtsLayer::TheRankExclusionVersion() {
    static std::atomic<uint64_t> version{0};
    return version;
}

/////////////////////////////////////////////////////////////////////////
bool& RtsLayer::TheEnableInstrumentation(void) {
  // to avoid initialization problems of non-local static variables
  static bool EnableInstrumentation = true;

  return EnableInstrumentation;
}

extern "C" int Tau_RtsLayer_TheEnableInstrumentation(void) {
  return (int)RtsLayer::TheEnableInstrumentation();
}

/////////////////////////////////////////////////////////////////////////
long RtsLayer::GenerateUniqueId(void) {
  /* This routine is called in a locked region (RtsLayer::LockDB/UnLockDB)*/
  static long UniqueId = 0;
  return ++UniqueId;
}

extern "C" void Tau_set_usesMPI(int value);

int Tau_test_for_MPI_comm_rank() {
#ifdef TAU_SETNODE0
    int commrank = 0;
#else /* TAU_SETNODE0  */
    int commrank = -1;
#endif /* TAU_SETNODE0 */
    /* Some configurations might use MPI without telling TAU - they can
       * call Tau_Init() even if
     * they are running in an MPI application.  For that reason, we double
     * check to make sure that we aren't in an MPI execution by checking
     * for some common environment variables. */
    // PMI, MPICH, Cray, Intel, MVAPICH2...
    const char * tmpvar = getenv("PMI_RANK");
	if (tmpvar != NULL) {
        commrank = atoi(tmpvar);
		// printf("Changing MPICH rank to %lu\n", commrank);
        Tau_set_usesMPI(1);
		return commrank;
    }
    // PMIx (Process Management Interface for Exascale)
    tmpvar = getenv("PMIX_RANK");
    if (tmpvar != NULL) {
        commrank = atoi(tmpvar);
	//printf("Found the rank! '%s', %d\n", tmpvar, commrank);
        Tau_set_usesMPI(1);
        return commrank;
    }
    // PALS (Cray Parallel Application Launch Service)
    tmpvar = getenv("PALS_RANKID");
    if (tmpvar != NULL) {
        commrank = atoi(tmpvar);
	//printf("Found the rank! '%s', %d\n", tmpvar, commrank);
        Tau_set_usesMPI(1);
        return commrank;
    }
    // ALPS on Cray
    tmpvar = getenv("ALPS_APP_PE");
    if (tmpvar != NULL) {
        commrank = atoi(tmpvar);
	//printf("Found the rank! '%s', %d\n", tmpvar, commrank);
        Tau_set_usesMPI(1);
        return commrank;
    }
	// OpenMPI, Spectrum
    tmpvar = getenv("OMPI_COMM_WORLD_RANK");
	if (tmpvar != NULL) {
        commrank = atoi(tmpvar);
		// printf("Changing openMPI rank to %lu\n", commrank);
        Tau_set_usesMPI(1);
		return commrank;
    }
	// ALPS on Cray
    tmpvar = getenv("ALPS_APP_PE");
	if (tmpvar != NULL) {
        commrank = atoi(tmpvar);
        Tau_set_usesMPI(1);
		return commrank;
    }
	// Slurm - last resort
    tmpvar = getenv("SLURM_PROCID");
	if (tmpvar != NULL) {
        commrank = atoi(tmpvar);
        Tau_set_usesMPI(1);
		return commrank;
    }
	return commrank;
}

/////////////////////////////////////////////////////////////////////////
int& RtsLayer::TheNode(void) {
  static int Node = Tau_test_for_MPI_comm_rank();
  return Node;
}

#ifdef JAVA
/////////////////////////////////////////////////////////////////////////
bool& RtsLayer::TheUsingJNI(void) {
  static bool isUsingJNI = false;
  return isUsingJNI;
}
#endif

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

TauGroup_t RtsLayer::getProfileGroup(char const * ProfileGroup) {
  TauInternalFunctionGuard protects_this_function;
  ProfileMap_t::iterator it = TheProfileMap().find(string(ProfileGroup));
  TauGroup_t gr;
  if (it == TheProfileMap().end()) {
#ifdef DEBUG_PROF
    cerr <<ProfileGroup << " not found, adding ... "<<endl;
#endif /* DEBUG_PROF */
    gr = generateProfileGroup();
    TheProfileMap()[string(ProfileGroup)] = gr; // Add
    return gr;
  } else {
    return (*it).second; // The group that was found
  }

}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::disableProfileGroupName(char const * ProfileGroup) {
  TauInternalFunctionGuard protects_this_function;
  return disableProfileGroup(getProfileGroup(ProfileGroup));
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::enableProfileGroupName(char const * ProfileGroup) {
  TauInternalFunctionGuard protects_this_function;
  return enableProfileGroup(getProfileGroup(ProfileGroup));
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::generateProfileGroup(void) {
  static TauGroup_t key =  0x00000001;
  key = key << 1;
  /*
   * TAU_EXCLUDE is reserved for shutting off events. Do not use it as a general id. 
   */
  if (key == TAU_EXCLUDE) {
    key = key << 1;
  }
  if (!key) key = 0x1; // cycle
  return key;
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::enableProfileGroup(TauGroup_t ProfileGroup) {
  TheProfileMask() |= ProfileGroup; // Add it to the mask
  DEBUGPROFMSG("enableProfileGroup " << ProfileGroup <<" Mask = " << TheProfileMask() << endl;);
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
    DEBUGPROFMSG("disableProfileGroup " << ProfileGroup <<" Mask = " << TheProfileMask() << endl;);
  } // if it is not in the mask, disableProfileGroup does nothing
  return TheProfileMask();
}

/////////////////////////////////////////////////////////////////////////

TauGroup_t RtsLayer::resetProfileGroup(void) {
  TheProfileMask() = 0;
  return TheProfileMask();
}

/////////////////////////////////////////////////////////////////////////
int RtsLayer::setMyNode(int NodeId, int tid)
{
  TauInternalFunctionGuard protects_this_function;

#if (TAU_MAX_THREADS != 1)
  int oldid = TheNode();
  int newid = NodeId;
  if ((oldid != -1) && (oldid != newid)) {
    /* ie if SET_NODE macro was invoked twice for a threaded program : as
     in MPI+JAVA where JAVA initializes it with pid and then MPI_INIT is
     invoked several thousand events later, and TAU computes the process rank
     and invokes the SET_NODE with the correct rank. Handshaking between multiple
     levels of instrumentation. */

    if (TauEnv_get_tracing()) {
      TauTraceReinitialize(oldid, newid, tid);
    }
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

 if (res > 0) {
   return true;
 } else {
   return false;
 }
}

//////////////////////////////////////////////////////////////////////




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
double TauWindowsUsecD(void) {
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
      cerr << "Frequency high part is: " << Frequency.HighPart << endl;
      cerr << "Frequency low part is: " << Frequency.LowPart << endl;
      cerr << "Frequency quad part is: " << (double) Frequency.QuadPart << endl;
#endif /* DEBUG_PROF */
      //Shall be using Frequency.QuadPart and assuming a double as the main TAU
      //system does.

      //Checking for zero divide ... should not be one if the clock is working,
      //but need to be on the safe side!
      if (Frequency.QuadPart != 0) {
	Multiplier = (double) 1000000/Frequency.QuadPart;
	DEBUGPROFMSG("The value of the multiplier is: " << Multiplier << endl;);
      } else {
	DEBUGPROFMSG("There was a problem with the counter ... should not have happened!!" << endl;);
	return -1;
      }
    } else {
      DEBUGPROFMSG("No performace clock available ... using millisecond timers." << endl;);
    }
  }

  //Getting clock value.
  if (PerformanceClock) {
    if (QueryPerformanceCounter(&ClockValue)) {
      //As mentioned above, assuming double value.
      return Multiplier * (double) ClockValue.QuadPart;
    } else {
      DEBUGPROFMSG("There was a problem with the counter ... should not have happened!!" << endl;);
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



void RtsLayer::getUSecD (int tid, double *values, int reversed) {
#if ((defined(TAU_EPILOG) && !defined(PROFILING_ON)) || (defined(TAU_VAMPIRTRACE) && !defined(PROFILING_ON)))
  return;
#endif /* TAU_EPILOG/VAMPIRTRACE, PROFILING_ON */
  TauMetrics_getMetrics(tid, values, reversed);
}


int RtsLayer::getPid()
{
#ifdef TAU_WINDOWS
  return 0;
#else
  return getpid();
#endif
}

//
// Returns the **system** thread ID.  DO NOT use this to index arrays!
//
int RtsLayer::getTid() {
#if defined(__linux) && !(defined(TAU_BGP) || defined(TAU_BGQ) || defined(_ARCH_PPC))
  return syscall(__NR_gettid);
#else
  return 0;
#endif
}

#ifdef KTAU_NG
int RtsLayer::getLinuxKernelTid(){
 pid_t tid;
 tid =  syscall(SYS_gettid);
 return tid;
}
#endif /* KTAU_NG */

const char *RtsLayer::getCounterName(int i) {
  return TauMetrics_getMetricName(i);
}


void RtsLayer::getCurrentValues (int tid, double *values) {
  for (int c=0; c<Tau_Global_numCounters; c++) {
    values[c] = 0;
  }
  return getUSecD(tid, values);
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
           printf("WARNING: TAU_GROUP_5 is unavailable. Reserved for TAU_EXCLUDE\n");
	  //RtsLayer::enableProfileGroup(TAU_GROUP_5);
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
void RtsLayer::ProfileInit(int& argc, char**& argv)
{
  int i;
  int ret_argc;
  char **ret_argv;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_COMPENSATE
  double* tover = TauGetTimerOverhead(TauNullTimerOverhead);
  for (i = 0; i < TAU_MAX_COUNTERS; i++) {
    /* iterate through all counters and reset null overhead to zero
     if necessary */
    if (tover[i] < 0) tover[i] = 0;
  }
#endif /* TAU_COMPENSATE */

  ret_argc = 1;
  ret_argv = new char *[argc];
  ret_argv[0] = argv[0];    // The program name

  for (i = 1; i < argc; i++) {
    if ((strcasecmp(argv[i], "--profile") == 0)) {
      // Enable the profile groups
      if ((i + 1) < argc && argv[i + 1][0] != '-') {    // options follow
        RtsLayer::resetProfileGroup();    // set it to blank
        RtsLayer::setAndParseProfileGroups(argv[0], argv[i + 1]);
        i++;    // ignore the argv after --profile
      }
    } else {
      ret_argv[ret_argc++] = argv[i];
    }
  }
  argc = ret_argc;
  argv = ret_argv;
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
string RtsLayer::PrimaryGroup(const char *ProfileGroupName)
{
  char c;

  char const * start = ProfileGroupName;
  c = *start;
  while (c) {
    switch (c) {
      case ' ':
      case '|':
        c = *(++start);
        break;
      default:
        c = 0;
        break;
    }
  }

  char const * stop = start;
  c = *stop;
  while (c) {
    switch (c) {
      case ' ':
      case '|':
        c = 0;
        break;
      default:
        c = *(++stop);
        break;
    }
  }

  return string(start, (size_t)(stop - start));
}


//////////////////////////////////////////////////////////////////////



#ifdef TAU_NEC_SX
#define NO_RTTI 1
#endif /* TAU_NEC_SX */

/////////////////////////////////////////////////////////////////////////
std::string RtsLayer::GetRTTI(const char *name) {
#ifdef __GNUC__
#ifndef NO_RTTI
  std::size_t len;
  int stat;
  char *ptr = NULL;
  const std::string mangled = name;
  char * demangled = Tau_demangle_name(mangled.c_str());
  std::string tmpstr(demangled);
  free(demangled);
  return tmpstr;
#else /* NO_RTTI */
  return string(name);
#endif /* NO_RTTI */
#else
  return string(CheckNotNull(name));
#endif /* GNUC */
}

/***************************************************************************
 * $RCSfile: RtsLayer.cpp,v $   $Author: amorris $
 * $Revision: 1.132 $   $Date: 2009/10/27 21:20:11 $
 * POOMA_VERSION_ID: $Id: RtsLayer.cpp,v 1.132 2009/10/27 21:20:11 amorris Exp $
 ***************************************************************************/
