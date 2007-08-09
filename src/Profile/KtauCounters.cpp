/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2006                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: KtauCounters.cpp                                 **
**	Description 	: TAU Profiling Package			           **
**	Contact		: anataraj@cs.uoregon.edu 		 	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
****************************************************************************/

#ifdef TAUKTAU_SHCTR

//This code is based off the overall structure of the PapiLayer.cpp multiple-counter support, 
//instead of PAPI uses the KTAU facilities.

#include "Profile/Profiler.h"
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <ktau_proc_interface.h>
#include <linux/ktau/ktau_proc_data.h>
#include "Profile/KtauCounters.h"
#include "Profile/KtauSymbols.h"
#include "Profile/KtauProfiler.h"

//#define TAU_KTAUCTR_DEBUG 
#define TAU_KTAUCTR_DEBUG_LEVEL 0
// level 0 will perform backward running counter checking and output critical errors
// level 1 will perform output diagnostic information
// level 10 will output all counter values, for each retrieval

#define KTAU_SHCONT_SIZE 2000 //this is the size of the shared-container that is created to share mem between user/OS.

#ifdef TAU_MULTIPLE_COUNTERS
#define MAX_KTAU_COUNTERS MAX_TAU_COUNTERS
#else
#define MAX_KTAU_COUNTERS 1
#endif

bool KtauCounters::ktauInitialized = false;
KtauCtrThread KtauCounters::ThreadList[TAU_MAX_THREADS];
int KtauCounters::numCounters = 0;
unsigned long KtauCounters::counterList[MAX_KTAU_COUNTERS];
char KtauCounters::counterSyms[MAX_TAU_COUNTERS][KTAU_CTRSYM_MAXSZ];
int KtauCounters::counterType[MAX_KTAU_COUNTERS];

#ifdef TAU_KTAUCTR_DEBUG
#include <stdarg.h>
static void dmesg(int level, char* format, ...) {
#ifndef TAU_KTAUCTR_DEBUG
  /* Empty body, so a good compiler will optimise calls
     to dmesg away */
#else
  va_list args;

  if (level > TAU_KTAUCTR_DEBUG_LEVEL) {
    return;
  }

  fprintf (stderr, "[%d] ", getpid());
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
#endif /* TAU_KTAUCTR_DEBUG */
}
#endif

/////////////////////////////////////////////////
KtauCtrThread::~KtauCtrThread() {
//do nothing - let the kernel handle cleanup for now:
//we need to be able to trap death of a thread in TAU to
//be able to do this cleanup correctly in user-space -
//but the kernel knows when the death occurs and hence
//already implements this cleanup
/*
  printf("~KtauCtrThread(): ThreadID:%d\n", ThreadID);
  if(ThreadID != -1) {
    printf("~KtauCtrThread(): CounterValues:%p\n", CounterValues);
    free(CounterValues);
    CounterValues = NULL;
    //ktau_put_counter_user(ThreadList[i].shctr);
    printf("~KtauCtrThread(): shctr:%p\n", shctr);
    ktau_put_counter(shctr);
    shctr = NULL;
    //ktau_del_container_user(KTAU_TYPE_PROFILE, 1, NULL, 0, NULL, ThreadList[i].shcont);
    printf("~KtauCtrThread(): shcont:%p\n", shcont);
    ktau_del_container(KTAU_TYPE_PROFILE, 1, NULL, 0, NULL, shcont);
    shcont = NULL;
    ThreadID = -1;
  }
*/
}


/////////////////////////////////////////////////
int KtauCounters::addCounter(char *name, int cType) {
  int counter = 0, nOffset = 0;
#ifdef TAU_KTAUCTR_DEBUG
  dmesg(1, "KTAU: Adding counter %s\n", name);
#endif

  strncpy(counterSyms[numCounters], name, KTAU_CTRSYM_MAXSZ);
  counterSyms[numCounters][KTAU_CTRSYM_MAXSZ-1] = '\0';
  
  if(cType == KTAU_SHCTR_TYPE_INCL) {
    nOffset = 5;
  } else if(cType == KTAU_SHCTR_TYPE_NUM) {
    nOffset = 4;
  }
  while (counterSyms[numCounters][nOffset+counter]!='\0') {
    counterSyms[numCounters][counter]=counterSyms[numCounters][nOffset+counter];
    counter++;
  }
  counterSyms[numCounters][counter] = '\0';

  //get the address of this name
  unsigned long addr = KtauProfiler::getKtauSym().MapRevSym(string(counterSyms[numCounters]));
  //for now assume we are able to get the address
  //but assumptions are #$%@# so change it to check for failures.
  
  if(!addr) {
    //if it fails - maybe they gave the address directly instead of symbol name?
    fprintf(stderr, "KTAU CTR: getKtauSym FAILED for:%s. Assume they provided address instead of symbol-name in ENV variable.\n", counterSyms[numCounters]);
    counterList[numCounters] = strtoul(counterSyms[numCounters], NULL, 16);  
  } else {
    counterList[numCounters] = addr;
  }
  //remember the type - incl, excl or num
  counterType[numCounters] = cType;

  int counterID = numCounters++;
  return counterID;
}



////////////////////////////////////////////////////
int KtauCounters::initializeThread(int tid) {
  int ret;
  ktau_ushcont* ushcont = NULL;
  ktau_ush_ctr* ushctr = NULL;

#ifdef TAU_KTAUCTR_DEBUG
  dmesg(1, "KTAU CTR: Initializing Thread Data for TID = %d\n", tid);
#endif

  if (tid >= TAU_MAX_THREADS) {
    fprintf (stderr, "KTAU CTR: Exceeded max thread count of TAU_MAX_THREADS\n");
    return -1;
  }
  
  if((ThreadList[tid].shcont != NULL) || (ThreadList[tid].shctr != NULL)) {
    fprintf (stderr, "KTAU CTR: ThreadList[%d] being used? shcont:%p shctr:%p.\n", tid, ThreadList[tid].shcont, ThreadList[tid].shctr);
    return -1;
  }

  ThreadList[tid].CounterValues = (unsigned long long*) calloc(sizeof(unsigned long long)*numCounters, 1);
  if(!(ThreadList[tid].CounterValues)) {
    fprintf (stderr, "KTAU CTR: ThreadList[%d] calloc of CounterValues, size[%d] failed.\n", tid, sizeof(unsigned long long)*numCounters);
    return -1;
  }

  ushcont = ktau_add_container(KTAU_TYPE_PROFILE, 1, NULL, 0, NULL, KTAU_SHCONT_SIZE);
  if(!ushcont) {
	fprintf(stderr, "KTAU CTR: ktau_add_container FAILED. size:%ld\n", KTAU_SHCONT_SIZE);
	goto out_free_mem;
  }

  ret = ktau_get_counter(ushcont, "KTAU_MULTICTR", numCounters, counterList, &ushctr);
  if((ret) || (!ushctr)) {
    fprintf(stderr, "KTAU CTR: ktau_get_counter FAILED. ret:%d\n", ret);
    goto out_free_cont;
  }

  //alls well...save it
  ThreadList[tid].ThreadID = tid;
  ThreadList[tid].shcont = ushcont;
  ThreadList[tid].shctr = ushctr;

  return 0; //Success

out_free_cont:
  ret = ktau_del_container(KTAU_TYPE_PROFILE, 1, NULL, 0, NULL, ushcont);
  if(ret) {
    fprintf(stderr, "KTAU CTR: ktau_del_container also FAILED. size:%ld ret:%d\n", KTAU_SHCONT_SIZE, ret);
  }

out_free_mem: 
  free(ThreadList[tid].CounterValues);

  ThreadList[tid].CounterValues = NULL;
  ThreadList[tid].ThreadID = -1;
  ThreadList[tid].shcont = NULL;
  ThreadList[tid].shctr = NULL;
  return -1; //failure
}

////////////////////////////////////////////////////
long long KtauCounters::getSingleCounter(int tid) {

  int rc;
  if (!ktauInitialized) {
    rc = initializeKtauCtr();
    if (rc != 0) {
      return rc;
    }
  }

  if (numCounters == 0) {
    // adding must have failed, just return
    return 0;
  }

  if (ThreadList[tid].ThreadID == -1) {
    rc = initializeThread(tid);
    if (rc != 0) {
      return rc;
    }
  }


#ifdef TAU_KTAUCTR_DEBUG
/*
  long long oldValue = ThreadList[tid]->CounterValues[0];
*/
#endif  

  //for now KTAU counters in TAU look at excl time. Must have a way to choose incl/excl/ev-count
  //rc = ktau_copy_counter_excl(ThreadList[tid].shctr, ThreadList[tid].CounterValues, 1 /*Single Counter*/);
  rc = ktau_copy_counter_type(ThreadList[tid].shctr, ThreadList[tid].CounterValues, 1 /*Single Counter*/, counterType);
  if(rc) { //failed for some reason...
    fprintf (stderr, "KTAU CTR: Error reading KTAU counters.\n");
    return -1;
  }

#ifdef TAU_KTAUCTR_DEBUG
/*
  dmesg(10, "PAPI: getSingleCounter<%d> = %lld\n", tid, ThreadList[tid]->CounterValues[0]);

  long long difference = ThreadList[tid]->CounterValues[0] - oldValue;
  dmesg(10, "PAPI: Difference = %lld\n", difference);
  if (difference < 0) dmesg (0, "PAPI: Counter running backwards?\n");
  dmesg(difference < 0 ? 0 : 10, "PAPI: Previous value = %lld\n", oldValue);
  dmesg(difference < 0 ? 0 : 10, "PAPI: Current  value = %lld\n", ThreadList[tid]->CounterValues[0]);
  dmesg(difference < 0 ? 0 : 10, "PAPI: Difference     = %lld\n", difference);
*/
#endif  

  return ThreadList[tid].CounterValues[0];  
}

/////////////////////////////////////////////////
long long* KtauCounters::getAllCounters(int tid, int *numValues) {
  int rc;

  if (!ktauInitialized) {
    int rc = initializeKtauCtr();
    if (rc != 0) {
      return NULL;
    }
  }

  if (numCounters == 0) {
    // adding must have failed, just return
    return NULL;
  }


  if (ThreadList[tid].ThreadID == -1) {
    rc = initializeThread(tid);
    if (rc != 0) {
      return NULL;
    }
  }

  *numValues = numCounters;
  
#ifdef TAU_KTAUCTR_DEBUG
/*
  long long previousCounters[MAX_KTAU_COUNTERS];
  for (int i=0; i<numCounters; i++) {
    previousCounters[i] = ThreadList[tid]->CounterValues[i];
  }
*/
#endif

  //for now KTAU counters in TAU look at excl time. Must have a way to choose incl/excl/ev-count
  //rc = ktau_copy_counter_excl(ThreadList[tid].shctr, ThreadList[tid].CounterValues, numCounters);
  rc = ktau_copy_counter_type(ThreadList[tid].shctr, ThreadList[tid].CounterValues, numCounters, counterType);
  if(rc) { //failed for some reason...
    fprintf (stderr, "KTAU CTR: Error reading KTAU counters.\n");
    return NULL;
  }

#ifdef TAU_KTAUCTR_DEBUG
/*
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
*/
#endif

  return (long long*) ThreadList[tid].CounterValues;  
}



/////////////////////////////////////////////////
int KtauCounters::reinitializeKtauCtr() {
#ifdef TAU_KTAUCTR_DEBUG
  dmesg(1, "KtauCounters::reinitializeKtauCtr\n");
#endif
  // This function is called from the fork() handler
  // We need to clean up the ThreadList and then reinitialize KtauCtr

  if (ktauInitialized) {
    for(int i=0; i<TAU_MAX_THREADS; i++){
      if (ThreadList[i].ThreadID != -1) {
	free(ThreadList[i].CounterValues);
	ThreadList[i].CounterValues = NULL;
        ktau_put_counter(ThreadList[i].shctr);
	ThreadList[i].shctr = NULL;
	ktau_del_container(KTAU_TYPE_PROFILE, 1, NULL, 0, NULL, ThreadList[i].shcont);
	ThreadList[i].shcont = NULL;
	ThreadList[i].ThreadID = -1;
      }
    }
  }
  return initializeKtauCtr();
}


/////////////////////////////////////////////////
int KtauCounters::initializeSingleCounter() {
  
  // This function may get called more than once if there is a fork
  if (numCounters != 0) { 
    return 0;
  }

  // Add the counter named by KTAU_EVENT
  char *ktau_event = getenv("KTAU_EVENT");
  if (ktau_event == NULL) {
    fprintf (stderr, "KTAU CTR: Error - You must define the KTAU_EVENT environment variable.\n");
    return -1;
  }

  int counterID = addCounter(ktau_event, KTAU_SHCTR_TYPE_EXCL);
  if (counterID < 0) {
    return -1;
  }

  return 0;
}

// note, this only works on linux
// the below seems to not compile on certain distros...?
//======================================================
//#include <sys/types.h>
//#include <linux/unistd.h>
//_syscall0(pid_t,gettid)
//pid_t gettid(void);
//======================================================
//so replacing with this.........
#include <sys/types.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

unsigned long ktau_thread_gettid(void) {
#ifdef SYS_gettid  
  return(syscall(SYS_gettid));
#elif defined(__NR_gettid)
  return(syscall(__NR_gettid));
#else
  return(gettid());
#endif
}

/////////////////////////////////////////////////
int KtauCounters::initializeKtauCtr() {
#ifdef TAU_KTAUCTR_DEBUG
  dmesg(1, "KtauCounters::initializeKtauCounters\n");
#endif

  ktauInitialized = true;

  //do nothing for now

  return 0;
}


/////////////////////////////////////////////////
int KtauCounters::initializeKtauCounters(bool lock) { 
  static bool initialized = false;

  if (initialized) {
    return 0;
  }

  if (lock) RtsLayer::LockDB();
  int rc = initializeKtauCtr();
  if (lock) RtsLayer::UnLockDB();
  
  initialized = true;

  return rc;
}

/////////////////////////////////////////////////
int KtauCounters::RegisterFork(int type) {
  //ignore type for now
  //just call reinitialize?
  return reinitializeKtauCtr();
}

#endif //TAUKTAU_SHCTR
