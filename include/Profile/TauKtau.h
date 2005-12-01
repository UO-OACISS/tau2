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
	
	// Constructor
	TauKtau(KtauSymbols& sym);

	// Destructor
	~TauKtau();
	
	// APIs
	int StartKProfile();
	int StopKProfile();
	int DumpKProfile();
	int MergingKProfileFunc(FILE * fp);
	int MergingKProfileEvent(FILE * fp);
	int GetNumKProfileFunc();
	int GetNumKProfileEvent();
private:
	ktau_output_info ThisKtauOutputInfo;
	int GetKProfileInfo();
	int ReadKallsyms();
	int DiffKProfile();
	KtauSymbols& KtauSym;
};


#endif /* _TAUKTAU_H_*/
/***************************************************************************
 * $RCSfile: TauKtau.h,v $   $Author: anataraj $
 * $Revision: 1.1 $   $Date: 2005/12/01 02:50:56 $
 * POOMA_VERSION_ID: $Id: TauKtau.h,v 1.1 2005/12/01 02:50:56 anataraj Exp $ 
 ***************************************************************************/
