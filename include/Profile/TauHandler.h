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
void TauEnableTrackingMemory(void);
void TauDisableTrackingMemory(void);
void TauEnableTrackingMuseEvents(void);
void TauDisableTrackingMuseEvents(void);
void TauSetInterruptInterval(int interval);
void TauTrackMemoryUtilization(void);
void TauTrackMuseEvents(void);

#endif /* _TAU_HANDLER_H_ */
  
/***************************************************************************
 * $RCSfile: TauHandler.h,v $   $Author: sameer $
 * $Revision: 1.2 $   $Date: 2004/03/03 02:18:16 $
 * POOMA_VERSION_ID: $Id: TauHandler.h,v 1.2 2004/03/03 02:18:16 sameer Exp $ 
 ***************************************************************************/

	





