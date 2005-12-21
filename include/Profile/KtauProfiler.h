/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: KtauProfiler.h				  **
**	Description 	: TAU Kernel Profiling Interface		  **
**	Author		: Aroon Nataraj					  **
**			: Suravee Suthikulpanit				  **
**	Contact		: {anataraj,suravee}@cs.uoregon.edu               **
**	Flags		: Compile with				          **
**			  -DTAU_KTAU to enable KTAU	                  **
**	Documentation	: 					          **
***************************************************************************/

#ifndef _KTAUPROFILER_H_
#define _KTAUPROFILER_H_

#ifdef TAUKTAU

#define KALLSYMS_PATH "/boot/System.map-tau2"

#include <Profile/KtauProfiler.h>
#include <Profile/Profiler.h>
#include <Profile/KtauSymbols.h>
#include <Profile/TauKtau.h>

extern double KTauGetMHz(void);

class KtauProfiler {

	public: //PUBLIC

	// managing thread-specific life-cycle using factory methods 
	static KtauProfiler* GetKtauProfiler(int tid = RtsLayer::myThread());
	static void PutKtauProfiler(int tid = RtsLayer::myThread());

	//Instrumentation Methods
	void Start(Profiler *profiler, int tid = RtsLayer::myThread());
	void Stop(Profiler *profiler, bool AddInclFlag, int tid = RtsLayer::myThread());

	//output functions
	static FILE* OpenOutStream(char* dirname, int node, int context, int tid);
	static void CloseOutStream(FILE* ktau_fp);

	int SetStartState(ktau_state* pstate, Profiler* pProfiler);
	int SetStopState(ktau_state* pstate, bool AddInclFlag, Profiler* pProfiler);

	//The actual profile state
	TauKtau KernProf;

	~KtauProfiler();

	private: //PRIVATE

	//cons
	KtauProfiler();
	KtauProfiler(int tid); /* private so that only factory-methods can be used */

	//data members
	int tid; //thread-id

#ifdef  TAUKTAU_MERGE 
	//merge related
	ktau_state* current_ktau_state;
	volatile int active_merge_index;
#endif /* TAUKTAU_MERGE */

	//statics
	//-------
	static KtauProfiler * CurrentKtauProfiler[TAU_MAX_THREADS];
	static long long refCount[TAU_MAX_THREADS];
	static KtauSymbols KtauSym; 
};

#endif /* TAUKTAU */

#endif /* _KTAUPROFILER_H_ */
/***************************************************************************
 * $RCSfile: KtauProfiler.h,v $   $Author: anataraj $
 * $Revision: 1.2 $   $Date: 2005/12/21 02:55:59 $
 * POOMA_VERSION_ID: $Id: KtauProfiler.h,v 1.2 2005/12/21 02:55:59 anataraj Exp $ 
 ***************************************************************************/

