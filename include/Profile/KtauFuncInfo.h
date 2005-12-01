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

class KtauFuncInfo {
	
	public:
	KtauFuncInfo();
	~KtauFuncInfo();

        unsigned long long GetInclTicks() { return inclticks; }
        void AddInclTicks(unsigned long long ticks) { inclticks += ticks; }
        unsigned long long GetExclTicks() { return exclticks; }
        void AddExclTicks(unsigned long long ticks) { exclticks += ticks; }

        unsigned long long GetInclCalls() { return inclcalls; }
        void AddInclCalls(unsigned long long calls) { inclcalls += calls; }
        unsigned long long GetExclCalls() { return exclcalls; }
        void AddExclCalls(unsigned long long calls) { exclcalls += calls; }

private:
        unsigned long long inclticks;
        unsigned long long exclticks;

        unsigned long long inclcalls;
        unsigned long long exclcalls;

};

#endif /* _KTAUFUNCINFO_H_ */
/***************************************************************************
 * $RCSfile: KtauFuncInfo.h,v $   $Author: anataraj $
 * $Revision: 1.1 $   $Date: 2005/12/01 02:50:55 $
 * POOMA_VERSION_ID: $Id: KtauFuncInfo.h,v 1.1 2005/12/01 02:50:55 anataraj Exp $ 
 ***************************************************************************/

