/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2006  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauMemory.h					  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@paratools.com      **
**	Flags		: Compile with				          **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#ifndef _TAU_MEMORY_H_
#define _TAU_MEMORY_H_

//////////////////////////////////////////////////////////////////////
// This class allows us to convert void * to the desired type in malloc
//////////////////////////////////////////////////////////////////////

class TauVoidPointer {
  void *p;
  public:
    TauVoidPointer (void *pp) : p (pp) { }
    template <class T> operator T *() { return (T *) p; }
};
int TauDetectMemoryLeaks(void);
void Tau_track_memory_allocation(const char *file, int line, size_t size, TauVoidPointer ptr);
void Tau_track_memory_deallocation(const char *file, int line, TauVoidPointer ptr);

#endif /* _TAU_MEMORY_H_ */

/***************************************************************************
 * $RCSfile: TauMemory.h,v $   $Author: sameer $
 * $Revision: 1.2 $   $Date: 2006/06/18 02:45:36 $
 * TAU_VERSION_ID: $Id: TauMemory.h,v 1.2 2006/06/18 02:45:36 sameer Exp $ 
 ***************************************************************************/
