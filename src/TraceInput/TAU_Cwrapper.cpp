/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2003  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, Germany                                     **
****************************************************************************/
/***************************************************************************
**	File 		: TAU_Cwrapper.cpp				  **
**	Description 	: TAU trace format reader library's C API	  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 	                  **
***************************************************************************/
#include <TAU_tf.h>
/* C API */
/* open a trace file for reading */
extern "C" 
Ttf_FileHandleT CTtf_OpenFileForInput( const char *name, const char *edf)
{
  return Ttf_OpenFileForInput(name, edf);
}

/* Seek to an absolute event position. 
 * A negative position indicates to start from the tail of the event stream. 
 * Returns the position if successful or 0 if an error occured */
extern "C"
int  CTtf_AbsSeek( Ttf_FileHandleT handle, int eventPosition )
{
  return Ttf_AbsSeek(handle, eventPosition);
}

/* seek to a event position relative to the current position (just for completeness!) 
 * Returns the position if successful or 0 if an error occured */
extern "C"
int  CTtf_RelSeek( Ttf_FileHandleT handle, int plusMinusNumEvents )
{
  return Ttf_RelSeek(handle, plusMinusNumEvents);
}

/* read n events and call appropriate handlers 
 * Returns the number of records read (can be 0).
 * Returns a -1 value when an error takes place. Check errno */
extern "C"
int  CTtf_ReadNumEvents( Ttf_FileHandleT fileHandle,
                                   Ttf_CallbacksT callbacks,
                                   int numberOfEvents )
{
  return Ttf_ReadNumEvents(fileHandle, callbacks, numberOfEvents);
}

/* close a trace file */
extern "C"
Ttf_FileHandleT CTtf_CloseFile( Ttf_FileHandleT fileHandle )
{
  return Ttf_CloseFile(fileHandle);
}


/***************************************************************************
 * $RCSfile: TAU_Cwrapper.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2003/11/13 00:09:27 $
 * TAU_VERSION_ID: $Id: TAU_Cwrapper.cpp,v 1.1 2003/11/13 00:09:27 sameer Exp $ 
 ***************************************************************************/




