/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2006                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: PapiLayer.cpp                                    **
**	Description 	: TAU Profiling Package			           **
**	Contact		: tau-team@cs.uoregon.edu 		 	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
****************************************************************************/


#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/UserEvent.h>

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#ifdef TAU_AT_FORK
#include <pthread.h>
#endif /* TAU_AT_FORK */

#ifdef TAU_BEACON
#include <Profile/TauBeacon.h>
#endif /* TAU_BEACON */

extern "C" {
#include "papi.h"
}

#ifdef PAPI_VERSION // 2.x doesn't have these macros
#if (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 3 && PAPI_VERSION_MINOR(PAPI_VERSION) >= 9)
#define TAU_COMPONENT_PAPI
#endif

#if (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 4)
#define TAU_COMPONENT_PAPI
#endif

#endif

#ifndef TAU_COMPONENT_PAPI
// if there is no component papi, then we pretend that there is just one component
#define PAPI_COMPONENT_INDEX(a) 0
/*
 *CWL* - 6/21/2012 - Thanks to Vince Weaver who has backported the old interface
                     for the use of PAPI_COMPONENT_INDEX, we should not need this
		     exception anymore. We should keep this code around just in
		     case, however.
#else
// *CWL* - Disable for the PAPI 5.0 pre-release (4.9) until we know what to do.
//         The interface for PAPI 5.0 where the acquisition of the PAPI Component
//         Index is concerned has also changed, so if it is appropriate (eg.
//         direct-mapping), some redefine of PAPI_COMPONENT_INDEX is probably
//         the best way forward.
#if (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 4 && PAPI_VERSION_MINOR(PAPI_VERSION) >= 9)
#define PAPI_COMPONENT_INDEX(a) 0
#endif
*/
#endif

// 0 will perform backward running counter checking and output critical errors
// 1 will perform output diagnostic information
// 10 will output all counter values, for each retrieval
//#define TAU_PAPI_DEBUG 1

bool PapiLayer::papiInitialized = false;
double PapiLayer::scalingFactor = 0.0;
int PapiLayer::numCounters = 0;
int PapiLayer::counterList[MAX_PAPI_COUNTERS];
bool PapiLayer::destroyed=false;
std::mutex PapiLayer::papiVectorMutex;
PapiLayer::PapiThreadList & PapiLayer::ThePapiThreadList() {
    static PapiLayer::PapiThreadList threadList;
    return threadList;
}


int tauSampEvent = 0;

extern "C" int Tau_is_thread_fake(int tid);
extern "C" int TauMetrics_init(void);


static int Tau_initialize_papi_library(void)
{
  int err = PAPI_library_init(PAPI_VER_CURRENT);
  switch (err) {
    case PAPI_VER_CURRENT:
      // Initialized successfully
      break;
    case PAPI_EINVAL:
      fprintf(stderr, "TAU: PAPI_library_init: papi.h is different from the version used to compile the PAPI library.\n");
      break;
    case PAPI_ENOMEM:
      fprintf(stderr, "TAU: PAPI_library_init: Insufficient memory to complete the operation.\n");
      break;
    case PAPI_ESBSTR:
      fprintf(stderr, "TAU: PAPI_library_init: This substrate does not support the underlying hardware.\n");
      break;
    case PAPI_ESYS:
      // Use perror to see the value of errno
      perror("TAU: PAPI_library_init: A system or C library call failed inside PAPI");
      break;
    default:
      if (err > 0) {
        fprintf(stderr, "TAU: PAPI_library_init: version mismatch: %d != %d\n", err, PAPI_VER_CURRENT);
      } else {
        fprintf(stderr, "TAU: PAPI_library_init: %s\n", PAPI_strerror(err));
      }
      break;
  }
  return err;
}

// Some versions of PAPI don't have these defined
// so we'll define them to 0 and if the user tries to use them
// we'll print out a warning
#ifndef PAPI_DOM_USER
#define PAPI_DOM_USER 0
#endif

#ifndef PAPI_DOM_KERNEL
#define PAPI_DOM_KERNEL 0
#endif

#ifndef PAPI_DOM_SUPERVISOR
#define PAPI_DOM_SUPERVISOR 0
#endif

#ifndef PAPI_DOM_OTHER
#define PAPI_DOM_OTHER 0
#endif

#ifndef PAPI_DOM_ALL
#define PAPI_DOM_ALL 0
#endif


#ifdef TAU_PAPI_DEBUG
#define dmesg(level, fmt, ...) if(level <= TAU_PAPI_DEBUG) { \
  fprintf(stderr, "[%d](%s) " fmt, RtsLayer::myThread(), __PRETTY_FUNCTION__, ##__VA_ARGS__); fflush(stderr); }
#else
#define dmesg(level, fmt, ...)
#endif

#ifdef TAU_AT_FORK
void Tau_prepare(void)
{
  TAU_VERBOSE("inside Tau_prepare: pid = %d\n", RtsLayer::getPid());
}

void Tau_parent(void)
{
  TAU_VERBOSE("inside Tau_parent: pid = %d\n", RtsLayer::getPid());
}


void Tau_child(void)
{
  TauInternalFunctionGuard protects_this_function;

  int i, rc, numCounters;
  int tid = Tau_get_thread();
  numCounters = PapiLayer::numCounters;
  TAU_VERBOSE("inside Tau_child: pid = %d\n", RtsLayer::getPid());
  TheSafeToDumpData() = 1;
  TAU_VERBOSE("--->[pid=%d, Rank=%d]: Setting TheSafeToDumpData=1\n", RtsLayer::getPid(), RtsLayer::myNode());

  if (Tau_initialize_papi_library() != PAPI_VER_CURRENT) {
    return;
  }

  rc = PAPI_thread_init((unsigned long (*)(void))(RtsLayer::unsafeThreadId));

  ThreadValue* localThreadValue=PapiLayer::getThreadValue(tid);
  /* Check ThreadList */
  if (localThreadValue == 0) {
    localThreadValue=new ThreadValue;
    PapiLayer::setThreadValue(tid, localThreadValue);
  }
  localThreadValue->ThreadID = tid;
  localThreadValue->CounterValues = new long long[MAX_PAPI_COUNTERS];
  for (i = 0; i < MAX_PAPI_COUNTERS; i++) {
    localThreadValue->CounterValues[i] = 0L;
  }

  for (i = 0; i < TAU_PAPI_MAX_COMPONENTS; i++) {
    localThreadValue->NumEvents[i] = 0;
    localThreadValue->EventSet[i] = PAPI_NULL;
    rc = PAPI_create_eventset(&(localThreadValue->EventSet[i]));
    if (rc != PAPI_OK) {
      fprintf(stderr, "TAU: Error creating PAPI event set: %s\n", PAPI_strerror(rc));
      return;
    }
    if(TauEnv_get_papi_multiplexing()) {
      rc = PAPI_assign_eventset_component( localThreadValue->EventSet[i], 0 );
      if ( PAPI_OK != rc ) {
        fprintf(stderr, "PAPI_assign_eventset_component failed (%s)\n", PAPI_strerror(rc));
        return;
      }
      rc = PAPI_set_multiplex(localThreadValue->EventSet[i]);
      if ( PAPI_OK != rc ) {
        fprintf(stderr, "PAPI_set_multiplex failed (%s)\n", PAPI_strerror(rc));
        return;
      }
    }
  }

  /* PAPI 3 support goes here */
  for (i = 0; i < numCounters; i++) {
    int comp = PAPI_COMPONENT_INDEX (PapiLayer::counterList[i]);
    rc = PAPI_add_event(localThreadValue->EventSet[comp], PapiLayer::counterList[i]);
    if (rc != PAPI_OK) {
      fprintf(stderr, "pid=%d, TAU (tau_child): Error adding PAPI events: %s\n", RtsLayer::getPid(), PAPI_strerror(rc));
      return;
    }

    // this creates a mapping from 'component', and index in that component back    // to the original index, since we return just a single array of values
    localThreadValue->Comp2Metric[comp][localThreadValue->NumEvents[comp]++] = i;
  }

  for (i = 0; i < TAU_PAPI_MAX_COMPONENTS; i++) {
    if (localThreadValue->NumEvents[i] >= 1) {    // if there were active counters for this component
      rc = PAPI_start(localThreadValue->EventSet[i]);
    }
  }

  if (rc != PAPI_OK) {
    fprintf(stderr, "pid=%d: TAU (tau_child2): Error calling PAPI_start: %s, tid = %d\n", RtsLayer::getPid(), PAPI_strerror(rc), tid);
    return;
  }

  // Before traversing the callstack. Reset any negative exclusive times for
  // papi counters
  Profiler *curr = TauInternal_CurrentProfiler(tid);
  while (curr != 0) {
    curr->ThisFunction->ResetExclTimeIfNegative(tid);
    curr = curr->ParentProfiler;
  }
}

#endif /* TAU_AT_FORK */


/////////////////////////////////////////////////
int PapiLayer::addCounter(char *name)
{
  int code;
  TAU_VERBOSE("TAU: PAPI: Adding counter %s\n", name);

  int rc = PAPI_event_name_to_code(name, &code);
#if (PAPI_VERSION_MAJOR(PAPI_VERSION) == 3 && PAPI_VERSION_MINOR(PAPI_VERSION) == 9)
#else
  // There is currently a bug in PAPI-C 3.9 that causes the return code to not
  // be PAPI_OK, even if it has succeeded, for now, we will just not check.
  if (rc != PAPI_OK) {
    fprintf (stderr, "TAU: Error: Couldn't Identify Counter '%s': %s\n", name, PAPI_strerror(rc));
    return -1;
  }
#endif

  if ((PAPI_query_event(code) != PAPI_OK)) {
    fprintf (stderr, "TAU: Error: Counter %s is not available!\n", name);
    return -1;
  }

  int counterID = numCounters++;
  counterList[counterID] = code;

  /* If this metrics matches the EBS source name, take note of the code */
  if (strcmp(name, TauEnv_get_ebs_source()) == 0) {
    tauSampEvent = code;
  }

  return counterID;
}



////////////////////////////////////////////////////
int PapiLayer::initializeThread(int tid)
{
  int rc;

  if (!getThreadValue(tid)){
    RtsLayer::LockDB();
    if (!getThreadValue(tid)){
      dmesg(1, "TAU: PAPI: Initializing Thread Data for TID = %d\n", tid);

      /* Task API does not have a real thread associated with it. It is fake */
      if (Tau_is_thread_fake(tid) == 1) tid = 0;
      ThreadValue* localThreadValue = new ThreadValue;
      setThreadValue(tid,localThreadValue);
      localThreadValue->ThreadID = tid;
      localThreadValue->CounterValues = new long long[MAX_PAPI_COUNTERS];
      memset(localThreadValue->CounterValues, 0, MAX_PAPI_COUNTERS*sizeof(long long));
      
      for (int i=0; i<TAU_PAPI_MAX_COMPONENTS; i++) {
        localThreadValue->NumEvents[i] = 0;
        localThreadValue->EventSet[i] = PAPI_NULL;
        rc = PAPI_create_eventset(&(localThreadValue->EventSet[i]));
        if (rc != PAPI_OK) {
          fprintf (stderr, "TAU: Error creating PAPI event set: %s\n", PAPI_strerror(rc));
          RtsLayer::UnLockDB();
          return -1;
        }
        if(TauEnv_get_papi_multiplexing()) {
          rc = PAPI_assign_eventset_component( localThreadValue->EventSet[i], 0 );
          if ( PAPI_OK != rc ) {
            fprintf(stderr, "PAPI_assign_eventset_component failed (%s)\n", PAPI_strerror(rc));
            exit(1);
          }
          rc = PAPI_set_multiplex(localThreadValue->EventSet[i]);
          if ( PAPI_OK != rc ) {
            fprintf(stderr, "PAPI_set_multiplex failed (%s)\n", PAPI_strerror(rc));
            return -1;
          }
        }
      }

#ifndef PAPI_VERSION
      /* PAPI 2 support goes here */
      rc = PAPI_add_events(&(localThreadValue->EventSet[0]), counterList, numCounters);
      if (rc != PAPI_OK) {
        fprintf (stderr, "TAU: Error adding PAPI events: %s\n", PAPI_strerror(rc));
        RtsLayer::UnLockDB();
        return -1;
      }
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 3)
      /* PAPI 3 support goes here */
      for (int i=0; i<numCounters; i++) {
        int comp = PAPI_COMPONENT_INDEX (counterList[i]);
        rc = PAPI_add_event(localThreadValue->EventSet[comp], counterList[i]);
        if (rc != PAPI_OK) {
          fprintf (stderr, "TAU: Error adding PAPI events: %s\n", PAPI_strerror(rc));
          RtsLayer::UnLockDB();
          return -1;
        }

        // this creates a mapping from 'component', and index in that component back
        // to the original index, since we return just a single array of values
        localThreadValue->Comp2Metric[comp][localThreadValue->NumEvents[comp]++] = i;
      }


#if ! (defined(TAU_CRAYXMT) || defined(TAU_BGL) || defined(TAU_DISABLE_SAMPLING) || defined(_AIX))
      if (TauEnv_get_ebs_enabled()) {
        if (tauSampEvent != 0) {
          int comp = PAPI_COMPONENT_INDEX (tauSampEvent);
          int threshold = TauEnv_get_ebs_period();
          TAU_VERBOSE("TAU: Setting PAPI overflow handler\n");
          rc = PAPI_overflow(localThreadValue->EventSet[comp], tauSampEvent, threshold, 0, Tau_sampling_papi_overflow_handler);
          if (rc != PAPI_OK) {
            fprintf (stderr, "TAU Sampling Warning: Error adding PAPI overflow handler: %s. Threshold=%d\n", PAPI_strerror(rc), threshold);
            tauSampEvent = 0; // Make sampling use itimer instead. We can disable it later.
          }
        }
      }
#endif /* ! (TAU_CRAYXMT || TAU_BGL || TAU_DISABLE_SAMPLING) */

#else
    /* PAPI future support goes here */
#error "TAU does not support this version of PAPI, please contact tau-bugs@cs.uoregon.edu"
#endif

      for (int i=0; i<TAU_PAPI_MAX_COMPONENTS; i++) {
        if (localThreadValue->NumEvents[i] > 0) { // if there were active counters for this component
          rc = PAPI_start(localThreadValue->EventSet[i]);
          if (rc != PAPI_OK) {
            fprintf (stderr, "pid=%d: TAU(initializeThread): Error calling PAPI_start: %s, tid = %d\n", RtsLayer::getPid(), PAPI_strerror(rc), tid);
            RtsLayer::UnLockDB();
            return -1;
          }
        }
      }
    } /*if (!ThreadList[tid]) */
    RtsLayer::UnLockDB();
  } /*if (!ThreadList[tid]) */

  dmesg(10, "ThreadList[%d] = %p\n", tid, localThreadValue);
  return 0;
}




/////////////////////////////////////////////////
long long *PapiLayer::getAllCounters(int tid, int *numValues) {
  int rc=0;
  long long tmpCounters[MAX_PAPI_COUNTERS] = {0};

  /* Task API does not have a real thread associated with it. It is fake */
  if (Tau_is_thread_fake(tid) == 1) { return NULL; }

  if (!papiInitialized) {
    if (initializePapiLayer()) {
      return NULL;
    }
  }

  if (numCounters == 0) {
    // adding must have failed, just return
    return NULL;
  }

  ThreadValue* localThreadValue=getThreadValue(tid);
  if (localThreadValue==NULL) {
    if(initializeThread(tid)) {
      return NULL;
    }
    localThreadValue=getThreadValue(tid);
  }

  *numValues = numCounters;
  //ThreadValue* localThreadValue=getThreadValue(tid);
  
#ifdef TAU_PAPI_DEBUG
  long long previousCounters[MAX_PAPI_COUNTERS];
  for (int i=0; i<numCounters; i++) {
    previousCounters[i] = localThreadValue->CounterValues[i];
  }
#endif

#ifdef PTHREADS
  if (tid != RtsLayer::myThread() ) {
    //printf("Returning values for %d instead of %d\n", tid, RtsLayer::myThread());
    return localThreadValue->CounterValues;
  }
#endif /* PTHREADS */

  for (int comp=0; comp<TAU_PAPI_MAX_COMPONENTS; comp++) {
    if (localThreadValue->NumEvents[comp] > 0) { // if there were active counters for this component
      // read eventset for this component and reset counters
      if (PAPI_read(localThreadValue->EventSet[comp], tmpCounters) != PAPI_OK)
        break;
      if (PAPI_reset(localThreadValue->EventSet[comp]) != PAPI_OK)
        break;
      // map back to original indices
      for (int j=0; j<localThreadValue->NumEvents[comp]; j++) {
	int index = localThreadValue->Comp2Metric[comp][j];
	localThreadValue->CounterValues[index] += tmpCounters[j];
        dmesg(10, "ThreadList[%d]->CounterValues[%d] = %lld\n", tid, index, localThreadValue->CounterValues[index]);
      }
    }
  }

#ifdef TAU_PAPI_DEBUG
  for (int i=0; i<numCounters; i++) {
    long long difference = localThreadValue->CounterValues[i] - previousCounters[i];
    dmesg(10, "TAU: PAPI: Difference[%d] = %lld\n", i, difference);
    if (difference < 0) {
      dmesg(0, "TAU: PAPI: Counter running backwards?\n");
      dmesg(0, "TAU: PAPI: Previous value[%d] = %lld\n", i, previousCounters[i]);
      dmesg(0, "TAU: PAPI: Current  value[%d] = %lld\n", i, localThreadValue->CounterValues[i]);
      dmesg(0, "TAU: PAPI: Difference    [%d] = %lld\n", i, difference);
    }
  }
#endif

  if (rc != PAPI_OK) {
    fprintf (stderr, "pid=%d, TAU: Error reading PAPI counters: %s\n", RtsLayer::getPid(), PAPI_strerror(rc));
    return NULL;
  }
  return localThreadValue->CounterValues;
}



/////////////////////////////////////////////////
int PapiLayer::reinitializePAPI() {
  dmesg(1, "%s", "TAU: PapiLayer::reinitializePAPI\n");

  // This function is called from the fork() handler
  // We need to clean up the ThreadList and then reinitialize PAPI

  int rc = 0;
  if (papiInitialized) {
    RtsLayer::LockDB();
    if (papiInitialized) {
      TAU_VERBOSE("Reinitializing papi...");
      for(int i=0; i<ThePapiThreadList().size(); i++){
	    ThreadValue* localThreadValue=getThreadValue(i);
        if (localThreadValue != NULL) {
          delete localThreadValue->CounterValues;
          delete localThreadValue;
        }
        setThreadValue(i,NULL);
      }
      TauMetrics_init();
      rc = initializePAPI();
    }
    RtsLayer::UnLockDB();
  }
  return rc;
}



#ifdef TAU_PAPI_THREADS
// note, this only works on linux
#include <sys/types.h>
#include <linux/unistd.h>
_syscall0(pid_t,gettid)
pid_t gettid(void);

unsigned long papi_thread_gettid(void) {
#ifdef SYS_gettid
  return(syscall(SYS_gettid));
#elif defined(__NR_gettid)
  return(syscall(__NR_gettid));
#else
  return(gettid());
#endif
}
#endif /* TAU_PAPI_THREADS */


/////////////////////////////////////////////////
void PapiLayer::checkDomain(int domain, char *domainstr) {
  if (domain == 0) {
    fprintf (stderr, "TAU: Warning: PAPI domain \"%s\" is not available with this version of PAPI\n", domainstr);
  }
}

void PapiLayer::setPapiInitialized(bool value) {
  papiInitialized = value;
  TAU_VERBOSE("setPapiInitialized: papiInitialized = %d\n", papiInitialized);
}

/////////////////////////////////////////////////
int PapiLayer::initializePAPI() {
  TAU_VERBOSE("inside TAU: PapiLayer::initializePAPI entry\n");

#ifdef TAU_AT_FORK
  TAU_VERBOSE("TAU: PapiLayer::initializePAPI: before pthread_at_fork()");
#ifdef TAU_MPI
  TheSafeToDumpData() = 0;
  TAU_VERBOSE("[pid=%d, Rank=%d]: Setting TheSafeToDumpData=0\n", RtsLayer::getPid(),RtsLayer::myNode());
  pthread_atfork(Tau_prepare, Tau_parent, Tau_child);
#endif /* TAU_MPI */
#endif /* TAU_AT_FORK */

  // Initialize PAPI
  if (Tau_initialize_papi_library() != PAPI_VER_CURRENT) {
    return -1;
  }

  int rc;

#ifdef TAU_PAPI_THREADS
  rc = PAPI_thread_init((unsigned long (*)(void))papi_thread_gettid);
#else /* TAU_PAPI_THREADS */

#ifndef __alpha
  // There must be some problem with PAPI_thread_init on alpha
#ifndef PAPI_VERSION
  /* PAPI 2 support goes here */
  rc = PAPI_thread_init((unsigned long (*)(void))(RtsLayer::unsafeThreadId),0);
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 3) || (PAPI_VERSION_MAJOR(PAPI_VERSION) <= 5)
  /* PAPI 3 support goes here */
  rc = PAPI_thread_init((unsigned long (*)(void))(RtsLayer::unsafeThreadId));
#else
  /* PAPI future support goes here */
#error "Unsupported PAPI Version, probably too new"
#endif

#endif

  if (rc != PAPI_OK) {
    fprintf(stderr, "TAU: Error Initializing PAPI: %s\n", PAPI_strerror(rc));
    return -1;
  }
#endif /* __alpha */

  // set the PAPI domain if desired
  static char *papi_domain = getenv("TAU_PAPI_DOMAIN");
  if (papi_domain != NULL) {
    TAU_METADATA("PAPI Domain", papi_domain);
    int domain = 0;
    char *token = strtok(papi_domain,":");
    while (token != NULL) {
      int thisDomain = 0;
      if (!strcmp(token,"PAPI_DOM_USER")) {
	thisDomain |= PAPI_DOM_USER;
      } else if (!strcmp(token,"PAPI_DOM_KERNEL")) {
	thisDomain |= PAPI_DOM_KERNEL;
      } else if (!strcmp(token,"PAPI_DOM_OTHER")) {
	thisDomain |= PAPI_DOM_OTHER;
      } else if (!strcmp(token,"PAPI_DOM_SUPERVISOR")) {
	thisDomain |= PAPI_DOM_SUPERVISOR;
      } else if (!strcmp(token,"PAPI_DOM_ALL")) {
	thisDomain |= PAPI_DOM_ALL;
      } else {
	fprintf (stderr, "TAU: Warning: Unknown PAPI domain, \"%s\"\n", token);
      }

      checkDomain(thisDomain, token);
      domain |= thisDomain;
      token = strtok(NULL,":");
    }

    if (domain == 0) {
      fprintf (stderr, "TAU: Warning, No valid PAPI domains specified\n");
    }
    rc = PAPI_set_domain(domain);
    if (rc != PAPI_OK) {
      fprintf(stderr, "TAU: Error setting PAPI domain: %s\n", PAPI_strerror(rc));
      return -1;
    }
  }

  papiInitialized = true;

  return 0;
}


/////////////////////////////////////////////////
int PapiLayer::initializePapiLayer(bool lock)
{
  static int rc = 0;

  TAU_VERBOSE("Inside TAU: PapiLayer::intializePapiLayer: papiInitialized = %d\n", papiInitialized);
  TAU_VERBOSE("[pid = %d] Inside TAU: Actually initializing PapiLayer::intializePapiLayer: papiInitialized = %d\n", RtsLayer::getPid(), papiInitialized);
  dmesg(1, "TAU: PAPI: Initializing PAPI Layer: lock=%d\n", lock);

  if (lock) {
    if (!papiInitialized) {
      RtsLayer::LockDB();
      if (!papiInitialized) {
        rc = initializePAPI();
      }
      RtsLayer::UnLockDB();
    }
  } else {
    rc = initializePAPI();
  }

  return rc;
}

/////////////////////////////////////////////////
long long PapiLayer::getWallClockTime(void) {
  // Returns the wall clock time from PAPI interface
  static int initflag = initializePapiLayer();
  // check the flag to avoid compiler warnings
  if (initflag != 0) { TAU_VERBOSE("Error when initilizing PAPI layer\n"); }
  static long long oldvalue = 0L;
  static long long offset = 0;
  long long newvalue = 0L;
#ifdef TAU_PAPI
  newvalue = PAPI_get_real_usec();
  if (newvalue < oldvalue) {
    offset += UINT_MAX;
    DEBUGPROFMSG("TAU: WARNING: papi counter overflow. Fixed in TAU! new = "
		 <<newvalue <<" old = " <<oldvalue<<" offset = "<<offset <<endl;);
    DEBUGPROFMSG("TAU: Returning "<<newvalue + offset<<endl;);
  }
  oldvalue = newvalue;
  return (newvalue + offset);
#endif // TAU_PAPI
}

/////////////////////////////////////////////////
long long PapiLayer::getVirtualTime(void) {
  // Returns the virtual (user) time from PAPI interface
  static int initflag = initializePapiLayer();
  if (initflag != 0) { TAU_VERBOSE("Error when initilizing PAPI layer\n"); }
  return PAPI_get_virt_usec();
}

#if  (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 5)

#define TAU_MAX_RAPL_EVENTS 64
char Tau_rapl_event_names[TAU_MAX_RAPL_EVENTS][PAPI_MAX_STR_LEN];
char Tau_rapl_units[TAU_MAX_RAPL_EVENTS][PAPI_MIN_STR_LEN];
#endif /* VERSION */

int PapiLayer::initializeAndCheckRAPL(int tid) {

  dmesg(1, "Inside PapiLayer::initializeAndCheckRAPL(), papiLayer::numCounters=%d\n", numCounters);
  if (!papiInitialized)
    initializePapiLayer();

  if (!getThreadValue(tid)) { 
    RtsLayer::LockDB();
    if (!getThreadValue(tid)) {
      dmesg(1, "TAU: PAPI: Initializing Thread Data for TID = %d\n", tid);

      /* Task API does not have a real thread associated with it. It is fake */
      if (Tau_is_thread_fake(tid) == 1) tid = 0;
      ThreadValue* localThreadValue = new ThreadValue;
      setThreadValue(tid,localThreadValue);
      localThreadValue->ThreadID = tid;
      localThreadValue->CounterValues = new long long[MAX_PAPI_COUNTERS];
      memset(localThreadValue->CounterValues, 0, MAX_PAPI_COUNTERS*sizeof(long long));
    }
    RtsLayer::UnLockDB();
  }


  if (numCounters > 0) {
    printf("WARNING: TAU: Disabling TAU_TRACK_POWER events\n");
    printf("WARNING: TAU is already using PAPI counters. Please unset the TAU_METRICS environment variable so PAPI events do no appear in it if you plan to use TAU_TRACK_POWER API. Currently, TAU does not support both at the same time due to the higer overhead of power events.\n");
    return -1;
  }
  return 1;
}
/////////////////////////////////////////////////
int PapiLayer::initializePerfRAPL(int tid) {
#if  (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 5)
  int ret = 0;
  PAPI_cpu_option_t opt;

  int rapl_cid  = 0;
  opt.cpu_num = 0;

  initializeAndCheckRAPL(tid);
/*
  ret = PAPI_library_init(PAPI_VER_CURRENT);
  if (PAPI_VER_CURRENT != ret) {
    fprintf(stderr, "PAPI library version mismatch with header used to compile the  TAU library\n");
    return -1;
  }
*/

  ret = PAPI_set_granularity(PAPI_GRN_SYS);
  if (PAPI_OK != ret) {
    fprintf(stderr,"PAPI_set_granularity\n");
    exit(1);
  }
  ThreadValue* localThreadValue=getThreadValue(tid);
  localThreadValue->EventSet[rapl_cid] = PAPI_NULL;
  ret = PAPI_create_eventset(&(localThreadValue->EventSet[rapl_cid]));
  if (PAPI_OK != ret) {
    fprintf(stderr,"PAPI_create_eventset.\n");
    exit(1);
  }
  opt.eventset = localThreadValue->EventSet[rapl_cid];
  ret = PAPI_assign_eventset_component( localThreadValue->EventSet[rapl_cid], 1 );

  if ( PAPI_OK != ret ) {
    fprintf(stderr, "PAPI_assign_eventset_component failed (%s)\n", PAPI_strerror(ret));
    exit(1);
  }

  ret = PAPI_set_opt(PAPI_CPU_ATTACH, (PAPI_option_t*)&opt);
  if ( PAPI_OK != ret ) {
    fprintf(stderr, "PAPI_set_opt failed (%s)\n", PAPI_strerror(ret));
    exit(1);
  }

  if(TauEnv_get_papi_multiplexing()) {
    ret = PAPI_set_multiplex(localThreadValue->EventSet[rapl_cid]);
    if ( PAPI_OK != ret ) {
      fprintf(stderr, "PAPI_set_multiplex failed (%s)\n", PAPI_strerror(ret));
      exit(1);
    }
  }

/* Check paranoid setting */
  FILE *para = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
  int para_val;
  int scanned = fscanf(para, "%d", &para_val);
  if (para_val != -1 || scanned == EOF) {
    TAU_VERBOSE("Error: To use TAU's PAPI RAPL Perf interface please ensure that /proc/sys/kernel/perf_event_paranoid has a -1 in it.\n");
    // don't return an error, there's nothing more to do here.
    return -1;
  }
  fclose(para);

  numCounters = 0;
  ret = PAPI_add_named_event(localThreadValue->EventSet[rapl_cid], (char*)"rapl::RAPL_ENERGY_CORES");
  if (PAPI_OK != ret) {
#ifdef DEBUG_PROF
    fprintf(stderr,"Error: PAPI_add_named_event(RAPL_ENERGY_CORES) because %s.\nPlease ensure that /proc/sys/kernel/perf_event_paranoid has a -1 and your system has /sys/devices/power/events/energy-pkg.scale.\n", PAPI_strerror(ret));
#endif /* DEBUG_PROF */
    // don't exit(1);
  } else {
    snprintf(Tau_rapl_event_names[numCounters], sizeof(Tau_rapl_event_names[numCounters]),  "rapl::RAPL_ENERGY_CORES");
    snprintf(Tau_rapl_units[numCounters], sizeof(Tau_rapl_units[numCounters]),  "Joules");
    numCounters++;
  }

  ret = PAPI_add_named_event(localThreadValue->EventSet[rapl_cid], (char*)"rapl::RAPL_ENERGY_PKG");
  if (PAPI_OK != ret) {

#ifdef DEBUG_PROF
    fprintf(stderr,"Error: PAPI_add_named_event(RAPL_ENERGY_PKG) because %s.\nPlease ensure that /proc/sys/kernel/perf_event_paranoid has a -1 and your system has /sys/devices/power/events/energy-pkg.scale.\n", PAPI_strerror(ret));
// don't    exit(1);
#endif /* DEBUG_PROF */
  } else {
    snprintf(Tau_rapl_event_names[numCounters], sizeof(Tau_rapl_event_names[numCounters]),  "rapl::RAPL_ENERGY_PKG");
    snprintf(Tau_rapl_units[numCounters], sizeof(Tau_rapl_units[numCounters]),  "Joules");
    numCounters++;
#ifdef TAU_BEACON
    TauBeaconInit();
#endif /* TAU_BEACON */
  }

  ret = PAPI_add_named_event(localThreadValue->EventSet[rapl_cid], (char*)"rapl::RAPL_ENERGY_GPU");
  if (PAPI_OK != ret) {
#ifdef DEBUG_PROF
    fprintf(stderr,"Error: PAPI_add_named_event(RAPL_ENERGY_GPU) because %s.\nPlease ensure that /proc/sys/kernel/perf_event_paranoid has a -1 and your system has /sys/devices/power/events/energy-pkg.scale.\n", PAPI_strerror(ret));
#endif /* DEBUG_PROF */
    // exit(1);
  } else {
    snprintf(Tau_rapl_event_names[numCounters], sizeof(Tau_rapl_event_names[numCounters]),  "rapl::RAPL_ENERGY_GPU");
    snprintf(Tau_rapl_units[numCounters], sizeof(Tau_rapl_units[numCounters]),  "Joules");
    numCounters++;
  }

  ret = PAPI_add_named_event(localThreadValue->EventSet[rapl_cid], (char*)"rapl::RAPL_ENERGY_DRAM");
  if (PAPI_OK != ret) {
   // OK: this event is only available on servers
  } else {
    snprintf(Tau_rapl_event_names[numCounters], sizeof(Tau_rapl_event_names[numCounters]),  "rapl::RAPL_ENERGY_DRAM");
    snprintf(Tau_rapl_units[numCounters], sizeof(Tau_rapl_units[numCounters]),  "Joules");
    numCounters++;
  }

  FILE *fp = fopen ("/sys/devices/power/events/energy-pkg.scale", "r");
  if (fp == NULL) {
    perror("Couldn't open file /sys/devices/power/events/energy-pkg.scale");
    exit(1);
  }

  char line[100];
  char *tmp = fgets( line,100,fp );
  if( tmp == NULL || sscanf(line,"%lf",&scalingFactor) != 1 ) {
     printf("%s: /sys/devices/power/events/energy-pkg.scale doesn't contain a double", line);
     exit(1);
  }

  localThreadValue->NumEvents[rapl_cid] = numCounters;

  if (PAPI_start(localThreadValue->EventSet[rapl_cid]) != PAPI_OK) {
    printf("TAU PERF: Error in PAPI_Start\n");
    return -1;
  }
  return rapl_cid; /* 0 in this case */
#else /* PAPI >= 5 */
  return -1;
#endif /* PAPI_VERSION_MAJOR >= 5 */
}
int PapiLayer::initializeRAPL(int tid) {
#if ((PAPI_VERSION_MAJOR(PAPI_VERSION) >= 5) && (defined(__x86_64) || defined(__x86_64__)))
  int ncomponents, i, code, ret, rapl_cid = -1;
  PAPI_event_info_t evinfo;
  const PAPI_component_info_t *cinfo = NULL;
  int num_events = 0;
 
  initializeAndCheckRAPL(tid);
  ThreadValue* localThreadValue=getThreadValue(tid);
  ncomponents = PAPI_num_components();

  for (i=0; i < ncomponents; i++) {
    if ((cinfo = PAPI_get_component_info(i)) == NULL) {
      printf("PAPI_get_component_info returns null. PAPI was not configured with --components=rapl and hence RAPL events for power cannot be measured.\n");
    return -1;
   }

    if (strstr(cinfo->name,"rapl")) {
      rapl_cid = i;
      if (cinfo->disabled) {
        printf("WARNING: TAU can't measure power events on your system using PAPI with RAPL. Please ensure that permissions on /dev/cpu/*/msr allow you to read it. You may need to run this code as root to read the power registers or enable superuser access to these registers for this executable.  Besides loading the MSR kernel module and setting the appropriate file permissions on the msr device file, one must grant the CAP_SYS_RAWIO capability to any user executable that needs access to the MSR driver, using the command below:\n");
        printf("# setcap cap_sys_rawio=ep <user_executable>\n");
        return -1;
      } /* rapl is disabled */
      /* create event set */
      localThreadValue->EventSet[rapl_cid] = PAPI_NULL;
      ret = PAPI_create_eventset(&(localThreadValue->EventSet[rapl_cid]));
      if (ret != PAPI_OK) {
        printf("WARNING: TAU couldn't create a PAPI eventset. Please check the LD_LIBRARY_PATH and ensure that there is no mismatch between the version of papi.h and the papi library that is loaded\n");
        return -1;
      }

      if(TauEnv_get_papi_multiplexing()) {
        ret = PAPI_assign_eventset_component( localThreadValue->EventSet[rapl_cid], 0 );
        if ( PAPI_OK != ret ) {
          fprintf(stderr, "PAPI_assign_eventset_component failed (%s)\n", PAPI_strerror(ret));
          return -1;
        }
        ret = PAPI_set_multiplex(localThreadValue->EventSet[rapl_cid]);
        if ( PAPI_OK != ret ) {
          fprintf(stderr, "PAPI_set_multiplex failed (%s)\n", PAPI_strerror(ret));
          return -1;
        }
      }

      /* Add RAPL events to the event set */
      code = PAPI_NATIVE_MASK;
      ret = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, rapl_cid );
      if (ret != PAPI_OK) {
        printf("WARNING: TAU: PAPI_enum_cmp_event returns %d. Power measurements will not be made.\n", ret);
        return -1;
      }
      while ( ret == PAPI_OK ) {
        ret = PAPI_event_code_to_name( code, Tau_rapl_event_names[num_events]);
        dmesg(1,"code = %d, event_name[%d]=%s\n", code, num_events, Tau_rapl_event_names[num_events]);
        if (ret != PAPI_OK) {
          printf("WARNING: TAU: PAPI_event_code_to_name returns an error. Can't add PAPI RAPL events for power measurement.\n");
          return -1;
        }

        ret = PAPI_get_event_info(code, &evinfo);
        if (ret != PAPI_OK) {
          printf("WARNING: TAU: PAPI_get_event_info returns an error. Can't add PAPI RAPL events for power measurement.\n");
          return -1;
        }

         /* Check for nano Joules or nJ in the units */
        if ((evinfo.units[0] == 'n') && (evinfo.units[1] == 'J')) {
          scalingFactor = 1.0e-9; /* nano Joules */
          strncpy(Tau_rapl_units[num_events], evinfo.units, PAPI_MIN_STR_LEN);
          ret = PAPI_add_event( (localThreadValue->EventSet[rapl_cid]), code);
          if (ret != PAPI_OK) {
            printf("PAPI_add_event is not OK!\n");
            break; /* hit an event limit */
          }
          dmesg(1,"Added PAPI event %s successfully, rapl_cid = %d, EventSet=%d\n", Tau_rapl_event_names[num_events], rapl_cid, localThreadValue->EventSet[rapl_cid]);
          localThreadValue->Comp2Metric[rapl_cid][localThreadValue->NumEvents[rapl_cid]++] = numCounters;
          localThreadValue->CounterValues[num_events] = 0;
          num_events++;
	  numCounters++;
          ret = PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, rapl_cid);
        } else { /* if units != nJ */
          ret = PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, rapl_cid);
          continue;
        }
      } /* while loop */
      numCounters += 1; /* ADD 1 for wallclock time! */
    } /* if rapl */
  } /* for ncomponents */

  if (PAPI_start(localThreadValue->EventSet[rapl_cid]) != PAPI_OK) {
    printf("PAPI RAPL: Error in PAPI_Start\n");
    return -1;
  }
  return rapl_cid;
#else
  return -1;
#endif /* PAPI_VERSION */
}


/////////////////////////////////////////////////
void PapiLayer::triggerRAPLPowerEvents(bool in_signal_handler) {
#if  (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 5)
  int tid = Tau_get_thread();
#ifdef TAU_PAPI_PERF_RAPL
  static int rapl_cid = PapiLayer::initializePerfRAPL(tid);
#else /*  TAU_PAPI_PERF_RAPL */
  static int rapl_cid = PapiLayer::initializeRAPL(tid);
#endif /* TAU_PAPI_PERF_RAPL */

  // if we don't have power, don't measure
  if (rapl_cid == -1) return;

  static bool firsttime = true;
  dmesg(1,"rapl_cid = %d\n", rapl_cid);
  int i;
  long long tmpCounters[MAX_PAPI_COUNTERS];
  double elapsedTimeInSecs = 0.0;
  long long curtime;
  char ename[1024];
  ThreadValue* localThreadValue=getThreadValue(tid);
  /* Have we initialized on this thread yet? */
  if (localThreadValue == 0) return;
  for (i=0; i<numCounters; i++) {
    tmpCounters[i] = 0;
    dmesg(1,"Tau_rapl_event_names = %s, Tau_rapl_units=%s, numCounters=%d\n",
	Tau_rapl_event_names[i], Tau_rapl_units[i], numCounters);
  }

  if (rapl_cid != -1) {
    dmesg(1, "%s", "Inside PapiLayer::triggerRAPLPowerEvents()\n");

    curtime = PAPI_get_real_nsec();
    if (firsttime) {
       firsttime = false;
       localThreadValue->CounterValues[numCounters - 1] = curtime;
       return;
    }
   
    // NOTE: We store the curtime in the numCounters index.
    dmesg(1,"curtime = %lld, EventSet=%d, numEvents=%d\n", curtime, localThreadValue->EventSet[rapl_cid], localThreadValue->NumEvents[rapl_cid]);
    if (localThreadValue->NumEvents[rapl_cid] > 0) { // active counters
      // read eventset for this component and reset counters
      if (PAPI_stop(localThreadValue->EventSet[rapl_cid], tmpCounters)
	!= PAPI_OK) {
        printf("Node %d, Thread %d:Error reading counters in PapiLayer::triggerRAPLPowerEvents\n", RtsLayer::myNode(), tid);
        return;
      }
      tmpCounters[numCounters - 1] = curtime;

      elapsedTimeInSecs = (curtime - localThreadValue->CounterValues[numCounters-1])/1.0e9;
      localThreadValue->CounterValues[numCounters - 1] = curtime;

      for (i = 0; i < numCounters-1; i++) {
	dmesg(1,"Before subtracting: Counter: %s: tmp value= %.8f, units = %s, old value=%.8f, time elapsed=%.4f seconds\n",
	Tau_rapl_event_names[i], (double) tmpCounters[i], Tau_rapl_units[i], (double) localThreadValue->CounterValues[i], elapsedTimeInSecs);
	
      }
      for(i=0; i < numCounters -1; i++) {
        double value = (((double) tmpCounters[i]) *scalingFactor)/elapsedTimeInSecs;
	dmesg(1,"Counter: %s: value %.9f, units = W\n", Tau_rapl_event_names[i], value);
	if (value > 1e-5) {
	  snprintf(ename, sizeof(ename), "%s (CPU Socket Power in Watts)", Tau_rapl_event_names[i]);
      if (in_signal_handler) {
          TAU_REGISTER_EVENT(ue, ename);
          Tau_userevent_thread(ue, value, 0);
      } else {
          TAU_TRIGGER_CONTEXT_EVENT(ename, value);
          }
#ifdef TAU_BEACON
          TauBeaconPublish(value, "Watts", "NODE_POWER", ename);
#endif /* TAU_BEACON */
        }
      }

      if (PAPI_start(localThreadValue->EventSet[rapl_cid]) != PAPI_OK) {
        printf("Node %d, Thread %d:Error starting counters in PapiLayer::triggerRAPLPowerEvents\n", RtsLayer::myNode(), tid);
        return;
      }
    }
  }

#endif /* VERSION */
}
