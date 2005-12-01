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

class KtauMergeInfo {

	public: //PUBLIC
	KtauMergeInfo() { child_ticks = child_calls = start_ticks = start_calls = 0; }
	~KtauMergeInfo() { child_ticks = child_calls = start_ticks = start_calls = 0; }

	void SetStartTicks(unsigned long long ticks) { start_ticks = ticks;  }
	void SetStartCalls(unsigned long long calls) { start_calls = calls;  }

        unsigned long long GetStartTicks() { return start_ticks; }
        unsigned long long GetStartCalls() { return start_calls; }

        void AddChildTicks(unsigned long long ticks) { child_ticks += ticks; }
        void AddChildCalls(unsigned long long calls) { child_calls += calls; }

        unsigned long long GetChildTicks() { return child_ticks; }
        unsigned long long GetChildCalls() { return child_calls; }

	void ResetCounters() { child_ticks = child_calls = start_ticks = start_calls = 0;}

	private: //PRIVATE

	//ktau-function-info related
	unsigned long long start_ticks;
	unsigned long long start_calls;
	unsigned long long child_ticks;
	unsigned long long child_calls;

};

#endif /* TAUKTAU_MERGE */

#endif /* TAUKTAU */

#endif /* _KTAUMERGEINFO_H_ */
/***************************************************************************
 * $RCSfile: KtauMergeInfo.h,v $   $Author: anataraj $
 * $Revision: 1.1 $   $Date: 2005/12/01 02:50:55 $
 * POOMA_VERSION_ID: $Id: KtauMergeInfo.h,v 1.1 2005/12/01 02:50:55 anataraj Exp $ 
 ***************************************************************************/

