/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2008  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Research Center Juelich, Germany, LANL                               **
****************************************************************************/
/***************************************************************************
**	File 		: TauScalasca.h   				  **
**	Description 	: TAU layer atop Scalasca                         **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu                           **
**	Flags		: Compile with				          **
**	Documentation	: See http://tau.uoregon.edu                      **
***************************************************************************/


#ifndef _TAU_SCALASCA_H_
#define _TAU_SCALASCA_H_

#ifdef TAU_EPILOG 

#ifndef TAU_SCALASCA 

#define esd_open   elg_open
#define esd_enter  elg_enter
#define esd_exit(a)   elg_exit()
#define esd_close  elg_close
#define esd_def_region elg_def_region

#endif /* TAU_SCALASCA */
#endif /* TAU_EPILOG */
#endif /* _TAU_SCALASCA_H_ */
/***************************************************************************
 * $RCSfile: TauScalasca.h,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2008/03/15 02:09:51 $
 * POOMA_VERSION_ID: $Id: TauScalasca.h,v 1.1 2008/03/15 02:09:51 sameer Exp $
 ***************************************************************************/


