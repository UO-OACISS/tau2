/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: KtauFuncInfo.cpp				  **
**	Description 	: Kernel-space  FunctionInfo 			  **
**	Author		: Surave Suthikulpanit				  **
**			: Aroon Nataraj					  **
**	Contact		: suravee@cs.uoregon.edu		 	  **
**			: anataraj@cs.uoregon.edu		 	  **
**	Flags		: Compile with				          **
**			  -DTAUKTAU or -DTAUKTAU_MERGE			  **
**	Documentation	:                                                 **
***************************************************************************/

#include "Profile/KtauFuncInfo.h"

/* definition of statics need to be done outside the class */
unsigned long long KtauFuncInfo::kernelGrpCalls[TAU_MAX_THREADS][merge_max_grp] = {{0}};
unsigned long long KtauFuncInfo::kernelGrpIncl[TAU_MAX_THREADS][merge_max_grp] = {{0}};
unsigned long long KtauFuncInfo::kernelGrpExcl[TAU_MAX_THREADS][merge_max_grp] = {{0}};

/*-------------------------- CON/DESTRUCTOR ---------------------------*/
/* 
 * Function 		: KtauFuncInfo::KtauFuncInfo
 * Description		: Constructor
 */
KtauFuncInfo::KtauFuncInfo()
{
	for(int i =0; i<merge_max_grp; i++) {
		inclticks[i] = 0;
		exclticks[i] = 0;

		inclKExcl[i] = 0;
		exclKExcl[i] = 0;

		inclcalls[i] = 0;
		exclcalls[i] = 0;
	}
}

/* 
 * Function 		: KtauFuncInfo::KtauFuncInfo
 * Description		: Destructor
 */
KtauFuncInfo::~KtauFuncInfo()
{
	for(int i =0; i<merge_max_grp; i++) {
		inclticks[i] = 0;
		exclticks[i] = 0;

		inclKExcl[i] = 0;
		exclKExcl[i] = 0;

		inclcalls[i] = 0;
		exclcalls[i] = 0;
	}
}


/***************************************************************************
 * $RCSfile: KtauFuncInfo.cpp,v $   $Author: anataraj $
 * $Revision: 1.2 $   $Date: 2006/11/09 06:11:10 $
 ***************************************************************************/

