/*****************************************************************************
 **			TAU Portable Profiling Package			    **
 **			http://www.cs.uoregon.edu/research/paracomp/tau     **
 *****************************************************************************
 **    Copyright 2005  						   	    **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 **    Research Center Juelich, Germany                                     **
 ****************************************************************************/
/*****************************************************************************
 **	File 		: TAU_tf_writer.cpp				    **
 **	Description 	: TAU trace format writer library C, C++ API	    **
 **	Author		: Alan Morris            			    **
 **	Contact		: amorris@cs.uoregon.edu 	                    **
 ****************************************************************************/
#ifndef _TAU_TF_WRITER_H_
#define _TAU_TF_WRITER_H_

#include <Profile/tau_types.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


  /* TAU file handler */
  typedef void* Ttf_FileHandleT;


  /* open a trace file for reading */
  Ttf_FileHandleT Ttf_OpenFileForOutput( const char *name, const char *edf);



  int Ttf_DefClkPeriod(Ttf_FileHandleT file, double clkPeriod);

  int Ttf_DefThread(Ttf_FileHandleT file, unsigned int nodeToken, unsigned int threadToken, 
		    const char *threadName);


  int Ttf_EnterState(Ttf_FileHandleT file, x_uint64 time, 
		     unsigned int nodeToken, unsigned int threadToken, 
		     unsigned int stateToken);

  int Ttf_LeaveState(Ttf_FileHandleT file, x_uint64 time, 
		     unsigned int nodeToken, unsigned int threadToken,
		     unsigned int stateToken);

  int Ttf_DefStateGroup(Ttf_FileHandleT file, const char *stateGroupName, unsigned int stateGroupToken);

  int Ttf_DefState(Ttf_FileHandleT file, unsigned int stateToken, const char *stateName, unsigned int stateGroupToken);

  int Ttf_SendMessage(Ttf_FileHandleT file, double time, unsigned int sourceNodeToken,
		      unsigned int sourceThreadToken,
		      unsigned int destinationNodeToken,
		      unsigned int destinationThreadToken,
		      unsigned int messageSize,
		      unsigned int messageTag,
		      unsigned int messageComm);

  int Ttf_RecvMessage(Ttf_FileHandleT file, double time, unsigned int sourceNodeToken,
		      unsigned int sourceThreadToken,
		      unsigned int destinationNodeToken,
		      unsigned int destinationThreadToken,
		      unsigned int messageSize,
		      unsigned int messageTag,
		      unsigned int messageComm);

  int Ttf_DefUserEvent(Ttf_FileHandleT file, unsigned int userEventToken, 
			  const char *userEventName, int monotonicallyIncreasing);

  int Ttf_EventTrigger(Ttf_FileHandleT file, double time, 
			   unsigned int nodeToken,
			   unsigned int threadToken,
			   unsigned int userEventToken,
			   double userEventValue
			   );

  int Ttf_FlushTrace(Ttf_FileHandleT file);


  int Ttf_CloseOutputFile(Ttf_FileHandleT file);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_TF_WRITERH_ */





/***************************************************************************
 * $RCSfile: TAU_tf_writer.h,v $   $Author: amorris $
 * $Revision: 1.3 $   $Date: 2005/10/11 16:18:07 $
 * TAU_VERSION_ID: $Id: TAU_tf_writer.h,v 1.3 2005/10/11 16:18:07 amorris Exp $ 
 ***************************************************************************/



