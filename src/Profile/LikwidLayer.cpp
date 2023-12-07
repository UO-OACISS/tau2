/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997-2006                                                  **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/****************************************************************************
 **	File 		: LikwidLayer.cpp                                    **
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
#include <likwid.h>
}

#define dmesg(level, fmt, ...)

bool LikwidLayer::likwidInitialized = false;
vector<ThreadValue *> & LikwidLayer::TheThreadList() {
    static vector<ThreadValue *> threadList;
    return threadList;
}
//string LikwidLayer::eventString;//[] = "L2_LINES_IN_ALL:PMC0,L2_TRANS_L2_WB:PMC1";
int* LikwidLayer::cpus;
int num_cpus = 0;
int LikwidLayer::gid;
int LikwidLayer::err;
int LikwidLayer::numCounters = 0;
//int LikwidLayer::counterList[MAX_PAPI_COUNTERS];

int tauSampEvent = 0;

int LikwidLayer::initializeLikwidLayer() //Tau_initialize_likwid_library(void)
{

	//int* cpus;

	int w;
	int y;
	int z;
	if (!LikwidLayer::likwidInitialized) // initialize only if not already initialized
	{
	    LikwidLayer::err = topology_init();
	    CpuInfo_t info = get_cpuInfo();
	    CpuTopology_t topo = get_cpuTopology();
	    numa_init(); // Should be done by affinity_init() also but we do it explicitly
	    affinity_init();
	    //printf("TAU: LIKWID: Initializing\n");

	    LikwidLayer::cpus = (int*) malloc(topo->activeHWThreads * sizeof(int)); //vs numHWThreads
	    if (!LikwidLayer::cpus)
		    return 1;
	    int w1 = 0;
	    for (w = 0; w < topo->numHWThreads; w++) {
		    if (topo->threadPool[w].inCpuSet == 1) {
			    LikwidLayer::cpus[w1] = topo->threadPool[w].apicId;
			    w1++;
		    }
	    }
	    //perfmon_setVerbosity(3);
            setenv("LIKWID_FORCE", "1", 1); // Overwrite already running counters because currently there are no stopCounters() or finalize() calls
            {
                char foo[100];
                int ret = snprintf(foo, 99, "%lu", getpid());
                if (ret > 0)
                {
                   foo[ret] = '\0';
                   setenv("LIKWID_PERF_PID", foo, 1);
                }
            }

	    LikwidLayer::err = perfmon_init(topo->activeHWThreads, LikwidLayer::cpus);
	    num_cpus = topo->activeHWThreads; // Store the number of CPUs to use it later in LikwidLayer::getAllCounters
	    LikwidLayer::likwidInitialized = true; // we initialized it, so set this to true
    }

}

extern "C" int Tau_is_thread_fake(int tid);
extern "C" int TauMetrics_init(void);

/////////////////////////////////////////////////
int LikwidLayer::addEvents(const char *estr) {
	int code;
	/*
	 if(firstString){
	 eventString = string(estr);
	 firstString=false;
	 }
	 else{
	 eventString=eventString+","+string(estr);
	 }
	 */
	if (!LikwidLayer::likwidInitialized) // Check flag and initialize if needed
	{
	    LikwidLayer::err = LikwidLayer::initializeLikwidLayer();
	}
	//printf("TAU: LIKWID: Adding events %s\n", estr);

	//LikwidLayer::err = perfmon_stopCounters();
	LikwidLayer::gid = perfmon_addEventSet(estr);
	LikwidLayer::err = perfmon_setupCounters(LikwidLayer::gid);
	//printf("TAU: LIKWID: SetupCounters error: %d\n",LikwidLayer::err);
	LikwidLayer::err = perfmon_startCounters();
	//printf("TAU: LIKWID: StartCounters error: %d\n",LikwidLayer::err);
	numCounters = perfmon_getNumberOfEvents(LikwidLayer::gid);
	//printf("TAU: LIKWID: NumberOfEvents: %d\n", numCounters);
	return LikwidLayer::gid;
}

////////////////////////////////////////////////////
int LikwidLayer::initializeThread(int tid) {
	int rc;

	//if (tid >= TAU_MAX_THREADS) {
	//	fprintf(stderr, "TAU: Exceeded max thread count of TAU_MAX_THREADS\n");
	//	return -1;
	//}

	if (!GetThreadList(tid)) {
		RtsLayer::LockDB();
		if (!GetThreadList(tid)) {
			dmesg(1, "TAU: LIKWID: Initializing Thread Data for TID = %d\n", tid);

			/* Task API does not have a real thread associated with it. It is fake */
			if (Tau_is_thread_fake(tid) == 1)
				tid = 0;

			SetThreadList(tid, new ThreadValue);
			GetThreadList(tid)->ThreadID = tid;

			GetThreadList(tid)->CounterValues = new long long[numCounters];
			memset(GetThreadList(tid)->CounterValues, 0,
					numCounters * sizeof(long long));

		} /*if (!ThreadList[tid]) */
		RtsLayer::UnLockDB();
	} /*if (!ThreadList[tid]) */

	dmesg(10, "ThreadList[%d] = %p\n", tid, ThreadList[tid]);
	return 0;
}

/////////////////////////////////////////////////
long long *LikwidLayer::getAllCounters(int tid, int *numValues) {
	int rc = 0;
	//long long tmpCounters[numCounters];
	//perfmon_setVerbosity(3);

	/* Task API does not have a real thread associated with it. It is fake */
	if (Tau_is_thread_fake(tid) == 1)
		tid = 0;

	if (!likwidInitialized) {
		if (initializeLikwidLayer()) {
			return NULL;
		}
	}

	if (numCounters == 0) {
		// adding must have failed, just return
		return NULL;
	}

	if (GetThreadList(tid) == NULL) {
		if (initializeThread(tid)) {
			return NULL;
		}
	}

	*numValues = numCounters; //TODO: Warning. Likwid adds two additional counters to the specified counter list. This does not seem to matter but could cause unexpected issues in the future. Consider adjusting the value array to contain only the user specified counters.
	//printf("About to read tid:%d, cpu:%d, gid:%d\n",tid,LikwidLayer::cpus[tid],LikwidLayer::gid);
	int readres = perfmon_readCounters(); // GroupThread(LikwidLayer::gid,tid);
	// If we know the CPU the application is running on, we could also use perfmon_readCountersCpu(int cpuid) to reduce overhead
	//printf("Read returned %d\n",readres);

    
	for (int comp = 0; comp < numCounters; comp++) {
		//int comp=0;
		// Sum up the results of all CPUs
		double dblsum = 0;
		for (int c = 0; c < num_cpus; c++) 
        {
            dblsum += perfmon_getLastResult(LikwidLayer::gid, comp, c);
		}
		//printf("Counter %d longsum %lld dblsum %f\n", comp, static_cast<long long>(dblsum), dblsum);
        //tmpCounters[comp] += static_cast<long long>(s);
		//for (int j=0; j<numCounters; j++) {
		GetThreadList(tid)->CounterValues[comp] += static_cast<long long>(dblsum);
		//printf("ThreadList[%d]->CounterValues[%d] = %lld\n", tid, comp, ThreadList[tid]->CounterValues[comp]);
		//}

	}

	return GetThreadList(tid)->CounterValues;
}


