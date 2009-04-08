/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2006                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: KtauCounters.h                                   **
**	Description 	: TAU Profiling Package			           **
**	Contact		: anataraj@cs.uoregon.edu 		 	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
****************************************************************************/

#ifndef _KTAU_COUNTERS_H_
#define _KTAU_COUNTERS_H_

#ifdef TAUKTAU_SHCTR
extern "C" {
#include <linux/ktau/ktau_cont_data.h>
#include <ktau_proc_interface.h>
}

/* This wraps the shared-ctr *
 * ptr - so as to dealloc is *
 * on program termination.   *
 * the obj. will be class-   *
 * -static. */
class KtauCtrThread {
  public:
  int ThreadID;
  ktau_ushcont* shcont;
  ktau_ush_ctr* shctr;
  unsigned long long* CounterValues;
  KtauCtrThread() {
    ThreadID = -1;
    shcont = NULL;
    shctr = NULL;
    CounterValues = NULL;
  }
  ~KtauCtrThread();

  private:
  //protect copy cons - dont allow it to be called
  //This is so that same shctr is not destructed twice!
  KtauCtrThread(KtauCtrThread& ref) {}
};

#define KTAU_CTRSYM_MAXSZ	32

#define KTAU_SHCTR_NOTYPE 0
#define KTAU_SHCTR_INCL 1
#define KTAU_SHCTR_CNT 2
#define KTAU_SHCTR_EXCL 3

class KtauCounters {
public:
  static int initializeKtauCounters(bool lock = true);
  static int initializeSingleCounter();
  static long long *getAllCounters(int tid, int *numValues);
  static long long getSingleCounter(int tid);
  static int reinitializeKtauCtr(void);
  static int addCounter(char *name, int cType);
  static int RegisterFork(int type);
  static int counterType[MAX_TAU_COUNTERS]; //hackaway - to let MultipleCounters.cpp have access place in public.
private:
  static int initializeThread(int tid);
  static int initializeKtauCtr(void);
  static bool ktauInitialized;
  static KtauCtrThread ThreadList[TAU_MAX_THREADS];
  static int numCounters;
  static unsigned long counterList[MAX_TAU_COUNTERS];
  static char counterSyms[MAX_TAU_COUNTERS][KTAU_CTRSYM_MAXSZ];
};

#endif /* TAUKTAU_SHCTR */
#endif /* _KTAU_COUNTERS_H_ */
