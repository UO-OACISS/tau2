/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2004  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauCompensate.h				  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 			  **
**	Documentation	: http://www.cs.uoregon.edu/research/paracomp/tau **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#ifndef _TAU_COMPENSATE_H_
#define _TAU_COMPENSATE_H_


enum TauOverhead { TauNullTimerOverhead, TauFullTimerOverhead };
void TauCalibrateOverhead(void);
double* TauGetSingleTimerOverhead(void);
double* TauGetTimerOverhead(enum TauOverhead);


#endif /* _TAU_COMPENSATE_H_ */



