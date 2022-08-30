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

void TauSetupHandler(void);
void TauSetInterruptInterval(int interval);

int  TauEnableTrackingMemory(void);
int  TauDisableTrackingMemory(void);
void TauTrackMemoryUtilization(bool allocated);
void TauTrackMemoryHere(const char * prefix = nullptr);
void TauTrackPower(const char * prefix = nullptr);
void TauTrackPowerHere(void);
void TauTrackMemoryFootPrint(const char * prefix = nullptr);
void TauTrackMemoryFootPrintHere(void);
int TauEnableTrackingPower(void);
int TauDisableTrackingPower(void);
void TauTrackLoad(const char * prefix = nullptr);
void TauTrackLoadHere(void);
int TauEnableTrackingLoad(void);
int TauDisableTrackingLoad(void);

int TauEnableTrackingMemoryHeadroom(void);
int TauDisableTrackingMemoryHeadroom(void);
void TauTrackMemoryHeadroomHere(void);

int Tau_open_system_file(const char *filename);
int Tau_read_load_event(int fd, double *value);

#endif /* _TAU_HANDLER_H_ */

/***************************************************************************
 * $RCSfile: TauHandler.h,v $   $Author: amorris $
 * $Revision: 1.8 $   $Date: 2009/02/24 20:22:03 $
 * POOMA_VERSION_ID: $Id: TauHandler.h,v 1.8 2009/02/24 20:22:03 amorris Exp $
 ***************************************************************************/







