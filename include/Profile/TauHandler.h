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
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Routines
//////////////////////////////////////////////////////////////////////

#ifndef _TAU_HANDLER_H_
#define _TAU_HANDLER_H_
int  TauEnableTrackingMemory(void);
int  TauDisableTrackingMemory(void);
void TauEnableTrackingMuseEvents(void);
void TauDisableTrackingMuseEvents(void);
void TauSetInterruptInterval(int interval);
void TauTrackMemoryUtilization(void);
void TauTrackMuseEvents(void);
void TauTrackMemoryHere(void);

#endif /* _TAU_HANDLER_H_ */
  
/***************************************************************************
 * $RCSfile: TauHandler.h,v $   $Author: sameer $
 * $Revision: 1.3 $   $Date: 2004/06/10 18:08:30 $
 * POOMA_VERSION_ID: $Id: TauHandler.h,v 1.3 2004/06/10 18:08:30 sameer Exp $ 
 ***************************************************************************/

	





