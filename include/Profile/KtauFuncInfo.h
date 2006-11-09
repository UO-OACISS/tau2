/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: KtauFuncInfo.h				  **
**	Description 	: TAU Kernel Profiling Interface		  **
**	Author		: Aroon Nataraj					  **
**			: Suravee Suthikulpanit				  **
**	Contact		: {anataraj,suravee}@cs.uoregon.edu               **
**	Flags		: Compile with				          **
**			  -DTAU_KTAU to enable KTAU	                  **
**	Documentation	: 					          **
***************************************************************************/

#ifndef _KTAUFUNCINFO_H_
#define _KTAUFUNCINFO_H_

#include <linux/ktau/ktau_merge.h>

#if (defined(PTHREADS) || defined(TULIPTHREADS) || defined(JAVA) || defined(TAU_WINDOWS) || defined (TAU_OPENMP) || defined (TAU_SPROC))


#ifndef TAU_MAX_THREADS

#ifdef TAU_CHARM
#define TAU_MAX_THREADS 512
#else
#define TAU_MAX_THREADS 128
#endif

#endif //ifndef TAU_MAX_THREADS

#else
#define TAU_MAX_THREADS 1
#endif //defined(PTHREADS) || defined(TULIPTHREADS) || defined(JAVA) || defined(TAU_WINDOWS) || defined (TAU_OPENMP) || defined (TAU_SPROC)

class KtauFuncInfo {
	
	public:
	KtauFuncInfo();
	~KtauFuncInfo();

        unsigned long long GetInclTicks(int grp) { return inclticks[grp]; }
        void AddInclTicks(unsigned long long ticks, int grp) { inclticks[grp] += ticks; }
        unsigned long long GetExclTicks(int grp) { return exclticks[grp]; }
        void AddExclTicks(unsigned long long ticks, int grp) { exclticks[grp] += ticks; }

        unsigned long long GetInclKExcl(int grp) { return inclKExcl[grp]; }
        void AddInclKExcl(unsigned long long ticks, int grp) { inclKExcl[grp] += ticks; }
        unsigned long long GetExclKExcl(int grp) { return exclKExcl[grp]; }
        void AddExclKExcl(unsigned long long ticks, int grp) { exclKExcl[grp] += ticks; }

        unsigned long long GetInclCalls(int grp) { return inclcalls[grp]; }
        void AddInclCalls(unsigned long long calls, int grp) { inclcalls[grp] += calls; }
        unsigned long long GetExclCalls(int grp) { return exclcalls[grp]; }
        void AddExclCalls(unsigned long long calls, int grp) { exclcalls[grp] += calls; }

	unsigned long long GetKernelGrpCalls(int tid, int grp) { return kernelGrpCalls[tid][grp];}
	unsigned long long GetKernelGrpIncl(int tid, int grp) { return kernelGrpIncl[tid][grp];}
	unsigned long long GetKernelGrpExcl(int tid, int grp) { return kernelGrpExcl[tid][grp];}

	void ResetCounters(int tid, int grp) {
		inclticks[grp] = exclticks[grp] = inclKExcl[grp] = exclKExcl[grp] = inclcalls[grp] = exclcalls[grp] = 0;
	}

	void ResetAllCounters(int tid) {
		for(int i = 0; i<merge_max_grp; i++) {
			ResetCounters(tid, i);
		}
	}

	static void ResetGrpTotals(int tid, int grp) {
		kernelGrpCalls[tid][grp] = kernelGrpIncl[tid][grp] = kernelGrpExcl[tid][grp] = 0;
	}

	static void ResetAllGrpTotals(int tid) {
		for(int i = 0; i< merge_max_grp; i++) {
			ResetGrpTotals(tid, i);
		}
	}

	//for holding Kernel CallGroup totals
	static unsigned long long kernelGrpCalls[TAU_MAX_THREADS][merge_max_grp];
	static unsigned long long kernelGrpIncl[TAU_MAX_THREADS][merge_max_grp];
	static unsigned long long kernelGrpExcl[TAU_MAX_THREADS][merge_max_grp];

private:
        unsigned long long inclticks[merge_max_grp];
        unsigned long long exclticks[merge_max_grp];

        unsigned long long inclKExcl[merge_max_grp];
        unsigned long long exclKExcl[merge_max_grp];

        unsigned long long inclcalls[merge_max_grp];
        unsigned long long exclcalls[merge_max_grp];

};

#endif /* _KTAUFUNCINFO_H_ */
/***************************************************************************
 * $RCSfile: KtauFuncInfo.h,v $   $Author: anataraj $
 * $Revision: 1.2 $   $Date: 2006/11/09 05:41:33 $
 * POOMA_VERSION_ID: $Id: KtauFuncInfo.h,v 1.2 2006/11/09 05:41:33 anataraj Exp $ 
 ***************************************************************************/

