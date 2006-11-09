/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: KtauMergeInfo.h				  **
**	Description 	: TAU Kernel Profiling Interface		  **
**	Author		: Aroon Nataraj					  **
**			: Suravee Suthikulpanit				  **
**	Contact		: {anataraj,suravee}@cs.uoregon.edu               **
**	Flags		: Compile with				          **
**			  -DTAU_KTAU to enable KTAU	                  **
**	Documentation	: 					          **
***************************************************************************/

#ifndef _KTAUMERGEINFO_H_
#define _KTAUMERGEINFO_H_

#ifdef TAUKTAU

#ifdef   TAUKTAU_MERGE 
#include <linux/ktau/ktau_merge.h>

class KtauMergeInfo {

	public: //PUBLIC
	KtauMergeInfo() { 
		for(int i=0; i<merge_max_grp; i++) {
			child_ticks[i] = child_calls[i] = start_ticks[i] = start_calls[i] = start_KExcl[i] = child_KExcl[i] = 0; 
		}
	}
	~KtauMergeInfo() { 
		for(int i=0; i<merge_max_grp; i++) {
			child_ticks[i] = child_calls[i] = start_ticks[i] = start_calls[i] = start_KExcl[i] = child_KExcl[i] = 0; 
		}
	}

	void SetStartTicks(unsigned long long ticks, int grp) { start_ticks[grp] = ticks;  }
	void SetStartCalls(unsigned long long calls, int grp) { start_calls[grp] = calls;  }
	void SetStartKExcl(unsigned long long kexcl, int grp) { start_KExcl[grp] = kexcl;  }

        unsigned long long GetStartTicks(int grp) { return start_ticks[grp]; }
        unsigned long long GetStartCalls(int grp) { return start_calls[grp]; }
        unsigned long long GetStartKExcl(int grp) { return start_KExcl[grp]; }

        void AddChildTicks(unsigned long long ticks, int grp) { child_ticks[grp] += ticks; }
        void AddChildCalls(unsigned long long calls, int grp) { child_calls[grp] += calls; }
        void AddChildKExcl(unsigned long long calls, int grp) { child_KExcl[grp] += calls; }

        unsigned long long GetChildTicks(int grp) { return child_ticks[grp]; }
        unsigned long long GetChildCalls(int grp) { return child_calls[grp]; }
        unsigned long long GetChildKExcl(int grp) { return child_KExcl[grp]; }

	void ResetCounters(int i) { 
			child_KExcl[i] = child_ticks[i] = child_calls[i] = start_KExcl[i] = start_ticks[i] = start_calls[i] = 0;
	}


	private: //PRIVATE

	//ktau-function-info related
	unsigned long long start_ticks[merge_max_grp];
	unsigned long long start_calls[merge_max_grp];
	unsigned long long start_KExcl[merge_max_grp];

	unsigned long long child_ticks[merge_max_grp];
	unsigned long long child_calls[merge_max_grp];
	unsigned long long child_KExcl[merge_max_grp];

};

#endif /* TAUKTAU_MERGE */

#endif /* TAUKTAU */

#endif /* _KTAUMERGEINFO_H_ */
/***************************************************************************
 * $RCSfile: KtauMergeInfo.h,v $   $Author: anataraj $
 * $Revision: 1.2 $   $Date: 2006/11/09 05:41:33 $
 * POOMA_VERSION_ID: $Id: KtauMergeInfo.h,v 1.2 2006/11/09 05:41:33 anataraj Exp $ 
 ***************************************************************************/

