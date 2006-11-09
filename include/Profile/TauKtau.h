/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauKtau.h					  **
**	Description 	: TAU Kernel Profiling Interface		  **
**	Author		: Suravee Suthikulpanit				  **
**			: Aroon Nataraj					  **
**	Contact		: suravee@cs.uoregon.edu                    	  **
**			: anataraj@cs.uoregon.edu                    	  **
**	Flags		: Compile with				          **
**			  -DTAU_KTAU to enable KTAU	                  **
**	Documentation	: 					          **
***************************************************************************/

#ifndef _TAUKTAU_H_
#define _TAUKTAU_H_

#include "Profile/ktau_proc_interface.h"
#include <Profile/KtauSymbols.h>

#include <Profile/ProfileGroups.h> //for enum TauFork_t

#define NAME_SIZE	100
#define MAP_SIZE	10 * 1024


/////////////////////////////////////////////////////////////////////
//
// class TauKtau
//
//////////////////////////////////////////////////////////////////////

typedef struct _ktau_output_info{
	pid_t pid;
	unsigned int templ_fun_counter;
	unsigned int user_ev_counter;
}ktau_output_info;

class TauKtau 
{
public:
	char *startBuf;
	char *stopBuf;
	long startSize;
	long stopSize;
	long outSize;
	ktau_output *diffOutput;
	
	char *startBufTWO;
	char *stopBufTWO;
	long startSizeTWO;
	long stopSizeTWO;
	long outSizeTWO;
	ktau_output *diffOutputTWO;
	
	// Constructor
	TauKtau(KtauSymbols& sym);

	// Destructor
	~TauKtau();
	
	// APIs
	int StartKProfile();
	int StopKProfile();
	int DumpKProfile();
	int DumpKProfileOut();
	int StopKProfileTWO();
	int DumpKProfileTWO(int, ktau_output*, char*);
	int MergingKProfileFunc(FILE * fp);
	int MergingKProfileEvent(FILE * fp);
	int GetNumKProfileFunc();
	int GetNumKProfileEvent();
	int AggrKProfiles(char* start, int startSz, char* stop, int stopSz, ktau_output** aggrprofs);
	static int RegisterFork(TauKtau* pKernProf, enum TauFork_t opcode);
private:
	ktau_output_info ThisKtauOutputInfo;
	int GetKProfileInfo();
	int ReadKallsyms();
	int DiffKProfile();
	int DiffKProfileTWO(char* startB, char* stopB, int startSz, int stopSz, ktau_output** pdiffOut);
	KtauSymbols& KtauSym;
};


#endif /* _TAUKTAU_H_*/
/***************************************************************************
 * $RCSfile: TauKtau.h,v $   $Author: anataraj $
 * $Revision: 1.2 $   $Date: 2006/11/09 05:41:33 $
 * POOMA_VERSION_ID: $Id: TauKtau.h,v 1.2 2006/11/09 05:41:33 anataraj Exp $ 
 ***************************************************************************/
