/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://tau.uoregon.edu                             **
*****************************************************************************
**    Copyright 2009                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**      File            : TauMemory.h                                      **
**      Contact         : tau-bugs@cs.uoregon.edu                          **
**      Documentation   : See http://tau.uoregon.edu                       **
**      Description     : Support for memory tracking                      **
**                                                                         **
****************************************************************************/

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
TauVoidPointer Tau_new(const char *file, int line, size_t size, TauVoidPointer ptr);

#endif /* _TAU_MEMORY_H_ */

/***************************************************************************
 * $RCSfile: TauMemory.h,v $   $Author: amorris $
 * $Revision: 1.4 $   $Date: 2010/02/03 06:09:44 $
 * TAU_VERSION_ID: $Id: TauMemory.h,v 1.4 2010/02/03 06:09:44 amorris Exp $ 
 ***************************************************************************/
