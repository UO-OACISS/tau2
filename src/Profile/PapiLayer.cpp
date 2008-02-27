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


#include "Profile/Profiler.h"
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

extern "C" {
#include "papi.h"
}

#if (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 3 && PAPI_VERSION_MINOR(PAPI_VERSION) >= 9)
#define TAU_COMPONENT_PAPI
#else
// if there is no component papi, then we pretend that there is just one component
#define PAPI_COMPONENT_INDEX(a) 0
#endif

//#define TAU_PAPI_DEBUG 
#define TAU_PAPI_DEBUG_LEVEL 1
// level 0 will perform backward running counter checking and output critical errors
// level 1 will perform output diagnostic information
// level 10 will output all counter values, for each retrieval

bool PapiLayer::papiInitialized = false;
ThreadValue *PapiLayer::ThreadList[TAU_MAX_THREADS];
int PapiLayer::numCounters = 0;
int PapiLayer::counterList[MAX_PAPI_COUNTERS];

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
#include <stdarg.h>
static void dmesg(int level, char* format, ...) {
#ifndef TAU_PAPI_DEBUG
  /* Empty body, so a good compiler will optimise calls
     to dmesg away */
#else
  va_list args;

  if (level > TAU_PAPI_DEBUG_LEVEL) {
    return;
  }

  fprintf (stderr, "[%d] ", getpid());
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
#endif /* TAU_PAPI_DEBUG */
}
#endif


/////////////////////////////////////////////////
int PapiLayer::addCounter(char *name) {
  int code, rc;

#ifdef TAU_PAPI_DEBUG
  dmesg(1, "PAPI: Adding counter %s\n", name);
#endif

  rc = PAPI_event_name_to_code(name, &code);
#ifndef TAU_COMPONENT_PAPI
  // There is currently a bug in PAPI-C 3.9 that causes the return code to not
  // be PAPI_OK, even if it has succeeded, for now, we will just not check.
  if (rc != PAPI_OK) {
    fprintf (stderr, "Error: Couldn't Identify Counter '%s': %s\n", name, PAPI_strerror(rc));
    return -1;
  }
#endif
  
  if ((PAPI_query_event(code) != PAPI_OK)) {
    fprintf (stderr, "Error: Counter %s is not available!\n", name);
    return -1;
  }

  int counterID = numCounters++;
  counterList[counterID] = code;
  return counterID;

}



////////////////////////////////////////////////////
int PapiLayer::initializeThread(int tid) {
  int rc;

#ifdef TAU_PAPI_DEBUG
  dmesg(1, "PAPI: Initializing Thread Data for TID = %d\n", tid);
#endif

  if (tid >= TAU_MAX_THREADS) {
    fprintf (stderr, "Exceeded max thread count of TAU_MAX_THREADS\n");
    return -1;
  }
  
  ThreadList[tid] = new ThreadValue;
  ThreadList[tid]->ThreadID = tid;
  ThreadList[tid]->CounterValues = new long long[MAX_PAPI_COUNTERS];

  for (int i=0; i<MAX_PAPI_COUNTERS; i++) {
    ThreadList[tid]->CounterValues[i] = 0L;
  }
  
  for (int i=0; i<TAU_PAPI_MAX_COMPONENTS; i++) {
    ThreadList[tid]->NumEvents[i] = 0;
    ThreadList[tid]->EventSet[i] = PAPI_NULL;
    rc = PAPI_create_eventset(&(ThreadList[tid]->EventSet[i]));
    if (rc != PAPI_OK) {
      fprintf (stderr, "Error creating PAPI event set: %s\n", PAPI_strerror(rc));
      return -1;
    }
  }

#ifndef PAPI_VERSION
  /* PAPI 2 support goes here */
  rc = PAPI_add_events(&(ThreadList[tid]->EventSet[0]), counterList, numCounters);
  if (rc != PAPI_OK) {
    fprintf (stderr, "Error adding PAPI events: %s\n", PAPI_strerror(rc));
    return -1;
  }
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) == 3)
  /* PAPI 3 support goes here */
  for (int i=0; i<numCounters; i++) {
    int comp = PAPI_COMPONENT_INDEX (counterList[i]);
    rc = PAPI_add_event(ThreadList[tid]->EventSet[comp], counterList[i]);
    if (rc != PAPI_OK) {
      fprintf (stderr, "Error adding PAPI events: %s\n", PAPI_strerror(rc));
      return -1;
    }
    // this creates a mapping from 'component', and index in that component back
    // to the original index, since we return just a single array of values
    ThreadList[tid]->Comp2Metric[comp][ThreadList[tid]->NumEvents[comp]++] = i;
  }
#else
/* PAPI future support goes here */
#error "TAU does not support this version of PAPI, please contact tau-bugs@cs.uoregon.edu"
#endif 
  
  for (int i=0; i<TAU_PAPI_MAX_COMPONENTS; i++) {
    if (ThreadList[tid]->NumEvents[i] >= 1) { // if there were active counters for this component
      rc = PAPI_start(ThreadList[tid]->EventSet[i]);
    }
  }
  
  if (rc != PAPI_OK) {
    fprintf (stderr, "Error calling PAPI_start: %s\n", PAPI_strerror(rc));
    return -1;
  }
  
  return 0;
}



////////////////////////////////////////////////////
long long PapiLayer::getSingleCounter(int tid) {

  int rc;
  if (!papiInitialized) {
    rc = initializePAPI();
    if (rc != 0) {
      return rc;
    }

    rc = initializeSingleCounter();
    if (rc != PAPI_OK) {
      return rc;
    }
  }

  if (numCounters == 0) {
    // adding must have failed, just return
    return 0;
  }

  if (ThreadList[tid] == NULL) {
    rc = initializeThread(tid);
    if (rc != 0) {
      return rc;
    }
  }


#ifdef TAU_PAPI_DEBUG
  long long oldValue = ThreadList[tid]->CounterValues[0];
#endif  
  
  int comp = PAPI_COMPONENT_INDEX (counterList[0]);

  rc = PAPI_read(ThreadList[tid]->EventSet[comp], ThreadList[tid]->CounterValues);

#ifdef TAU_PAPI_DEBUG
  dmesg(10, "PAPI: getSingleCounter<%d> = %lld\n", tid, ThreadList[tid]->CounterValues[0]);

  long long difference = ThreadList[tid]->CounterValues[0] - oldValue;
  dmesg(10, "PAPI: Difference = %lld\n", difference);
  if (difference < 0) dmesg (0, "PAPI: Counter running backwards?\n");
  dmesg(difference < 0 ? 0 : 10, "PAPI: Previous value = %lld\n", oldValue);
  dmesg(difference < 0 ? 0 : 10, "PAPI: Current  value = %lld\n", ThreadList[tid]->CounterValues[0]);
  dmesg(difference < 0 ? 0 : 10, "PAPI: Difference     = %lld\n", difference);
#endif  

  if (rc != PAPI_OK) {
    fprintf (stderr, "Error reading PAPI counters: %s\n", PAPI_strerror(rc));
    return -1;
  }

  return ThreadList[tid]->CounterValues[0];  
}

/////////////////////////////////////////////////
long long *PapiLayer::getAllCounters(int tid, int *numValues) {
  int rc;
  long long tmpCounters[MAX_PAPI_COUNTERS];

  if (!papiInitialized) {
    int rc = initializePAPI();
    if (rc != 0) {
      return NULL;
    }
  }

  if (numCounters == 0) {
    // adding must have failed, just return
    return NULL;
  }

  if (ThreadList[tid] == NULL) {
    rc = initializeThread(tid);
    if (rc != 0) {
      return NULL;
    }
  }
  
  *numValues = numCounters;

#ifdef TAU_PAPI_DEBUG
  long long previousCounters[MAX_PAPI_COUNTERS];
  for (int i=0; i<numCounters; i++) {
    previousCounters[i] = ThreadList[tid]->CounterValues[i];
  }
#endif

  for (int comp=0; comp<TAU_PAPI_MAX_COMPONENTS; comp++) {
    if (ThreadList[tid]->NumEvents[comp] >= 1) { // if there were active counters for this component
      // read eventset for this component
      rc = PAPI_read(ThreadList[tid]->EventSet[comp], tmpCounters);  
      // now map back to original indices
      for (int j=0; j<ThreadList[tid]->NumEvents[comp]; j++) {
	int index = ThreadList[tid]->Comp2Metric[comp][j];
	ThreadList[tid]->CounterValues[index] = tmpCounters[j];
      }
    }
  }

#ifdef TAU_PAPI_DEBUG
  for (int i=0; i<numCounters; i++) {
    long long difference = ThreadList[tid]->CounterValues[i] - previousCounters[i];
    dmesg(10, "PAPI: Difference[%d] = %lld\n", i, difference);
    if (difference < 0) {
      dmesg(0, "PAPI: Counter running backwards?\n");
      dmesg(0, "PAPI: Previous value[%d] = %lld\n", i, previousCounters[i]);
      dmesg(0, "PAPI: Current  value[%d] = %lld\n", i, ThreadList[tid]->CounterValues[i]);
      dmesg(0, "PAPI: Difference    [%d] = %lld\n", i, difference);
    }
  }
#endif

  if (rc != PAPI_OK) {
    fprintf (stderr, "Error reading PAPI counters: %s\n", PAPI_strerror(rc));
    return NULL;
  }
  
  return ThreadList[tid]->CounterValues;  
}



/////////////////////////////////////////////////
int PapiLayer::reinitializePAPI() {
#ifdef TAU_PAPI_DEBUG
  dmesg(1, "PapiLayer::reinitializePAPI\n");
#endif
  // This function is called from the fork() handler
  // We need to clean up the ThreadList and then reinitialize PAPI

  if (papiInitialized) {
    for(int i=0; i<TAU_MAX_THREADS; i++){
      if (ThreadList[i] != NULL) {
	delete ThreadList[i]->CounterValues;
	delete ThreadList[i];
      }
      ThreadList[i] = NULL;
    }
  }
  return initializePAPI();
}


/////////////////////////////////////////////////
int PapiLayer::initializeSingleCounter() {
  
  // This function may get called more than once if there is a fork
  if (numCounters != 0) { 
    return 0;
  }

  // Add the counter named by PAPI_EVENT
  char *papi_event = getenv("PAPI_EVENT");
  if (papi_event == NULL) {
    fprintf (stderr, "Error - You must define the PAPI_EVENT environment variable.\n");
    return -1;
  }

  int counterID = addCounter(papi_event);
  if (counterID < 0) {
    return -1;
  }

  return 0;
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
    fprintf (stderr, "Warning: PAPI domain \"%s\" is not available with this version of PAPI\n", domainstr);
  }
}


/////////////////////////////////////////////////
int PapiLayer::initializePAPI() {
#ifdef TAU_PAPI_DEBUG
  dmesg(1, "PapiLayer::initializePAPI\n");
#endif

  papiInitialized = true;

  for (int i=0; i<TAU_MAX_THREADS; i++) {
    ThreadList[i] = NULL;
  }

  // Initialize PAPI
  int papi_ver = PAPI_library_init(PAPI_VER_CURRENT);
  if (papi_ver != PAPI_VER_CURRENT) {
    if (papi_ver > 0) {
      fprintf(stderr, "Error initializing PAPI: version mismatch: %d\n", papi_ver);
    } else {
      fprintf(stderr, "Error initializing PAPI: %s\n", PAPI_strerror(papi_ver));
    }
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
  rc = PAPI_thread_init((unsigned long (*)(void))(RtsLayer::myThread),0);
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) == 3)
  /* PAPI 3 support goes here */
  rc = PAPI_thread_init((unsigned long (*)(void))(RtsLayer::myThread));
#else
  /* PAPI future support goes here */
#error "Unsupported PAPI Version, probably too new"
#endif 

#endif

  
  if (rc != PAPI_OK) {
    fprintf(stderr, "Error Initializing PAPI: %s\n", PAPI_strerror(rc));
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
	fprintf (stderr, "Warning: Unknown PAPI domain, \"%s\"\n", token);
      }

      checkDomain(thisDomain, token);
      domain |= thisDomain;
      token = strtok(NULL,":");
    }
    
    if (domain == 0) {
      fprintf (stderr, "Warning, No valid PAPI domains specified\n");
    }
    rc = PAPI_set_domain(domain);
    if (rc != PAPI_OK) {
      fprintf(stderr, "Error setting PAPI domain: %s\n", PAPI_strerror(rc));
      return -1;
    }
  }
  

  return 0;
}


/////////////////////////////////////////////////
int PapiLayer::initializePapiLayer(bool lock) { 
  static bool initialized = false;

  if (initialized) {
    return 0;
  }

  if (lock) RtsLayer::LockDB();
  int rc = initializePAPI();
  if (lock) RtsLayer::UnLockDB();

  return rc;
}

/////////////////////////////////////////////////
long long PapiLayer::getWallClockTime(void) { 
  // Returns the wall clock time from PAPI interface
  static int initflag = initializePapiLayer();
  static long long oldvalue = 0L;
  static long long offset = 0;
  long long newvalue = 0L;
#ifdef TAU_PAPI
  newvalue = PAPI_get_real_usec();
  if (newvalue < oldvalue) {
    offset += UINT_MAX;
    DEBUGPROFMSG("WARNING: papi counter overflow. Fixed in TAU! new = "
		 <<newvalue <<" old = " <<oldvalue<<" offset = "<<offset <<endl;);
    DEBUGPROFMSG("Returning "<<newvalue + offset<<endl;);
  }
  oldvalue = newvalue;
  return (newvalue + offset);
#endif // TAU_PAPI
}

/////////////////////////////////////////////////
long long PapiLayer::getVirtualTime(void) { 
  // Returns the virtual (user) time from PAPI interface
  static int initflag = initializePapiLayer();
  return PAPI_get_virt_usec();
}
