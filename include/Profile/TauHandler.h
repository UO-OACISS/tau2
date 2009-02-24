/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2004  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauHandler.h					  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Routines
//////////////////////////////////////////////////////////////////////

#ifndef _TAU_HANDLER_H_
#define _TAU_HANDLER_H_
int  TauEnableTrackingMemory(void);
int  TauDisableTrackingMemory(void);
void TauSetInterruptInterval(int interval);
void TauTrackMemoryUtilization(bool allocated);
void TauTrackMemoryHere(void);
double TauGetMaxRSS(void);
int  TauGetFreeMemory(void);
void TauTrackMemoryHeadroomHere(void);
int TauEnableTrackingMemoryHeadroom(void);
int TauDisableTrackingMemoryHeadroom(void);

#endif /* _TAU_HANDLER_H_ */
  
/***************************************************************************
 * $RCSfile: TauHandler.h,v $   $Author: amorris $
 * $Revision: 1.8 $   $Date: 2009/02/24 20:22:03 $
 * POOMA_VERSION_ID: $Id: TauHandler.h,v 1.8 2009/02/24 20:22:03 amorris Exp $ 
 ***************************************************************************/

	





