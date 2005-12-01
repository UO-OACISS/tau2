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

/*-------------------------- CON/DESTRUCTOR ---------------------------*/
/* 
 * Function 		: KtauFuncInfo::KtauFuncInfo
 * Description		: Constructor
 */
KtauFuncInfo::KtauFuncInfo()
{
	inclticks = 0;
	exclticks = 0;

	inclcalls = 0;
	exclcalls = 0;
}

/* 
 * Function 		: KtauFuncInfo::KtauFuncInfo
 * Description		: Destructor
 */
KtauFuncInfo::~KtauFuncInfo()
{
	inclticks = 0;
	exclticks = 0;

	inclcalls = 0;
	exclcalls = 0;
}


/***************************************************************************
 * $RCSfile: KtauFuncInfo.cpp,v $   $Author: anataraj $
 * $Revision: 1.1 $   $Date: 2005/12/01 02:55:08 $
 ***************************************************************************/

