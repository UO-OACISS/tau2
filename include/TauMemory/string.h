/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2004                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : TauHandler.cpp                                  **
**      Description     : TAU Profiling Package                           **
**      Author          : Sameer Shende                                   **
**      Contact         : sameer@cs.uoregon.edu sameer@acl.lanl.gov       **
**      Documentation   : See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


#ifndef _TAU_MALLOC_H_
#define _TAU_MALLOC_H_
#define _MALLOC_H 1 

#include <stdlib.h>
#include <TauMemory.h>

/* needed for Linux stdlib.h */
#define __malloc_and_calloc_defined 
#define __need_malloc_and_calloc

#define strdup(STR)                 Tau_strdup(STR, __FILE__, __LINE__)
#define memcpy(DEST, SRC, SIZE)     Tau_memcpy(DEST, SRC, SIZE, __FILE__, __LINE__)
#define strcpy(DEST, SRC)           Tau_strcpy(DEST, SRC, __FILE__, __LINE__)
#define strncpy(DEST, SRC, SIZE)    Tau_strncpy(DEST, SRC, SIZE, __FILE__, __LINE__)
#define strcat(DEST, SRC)           Tau_strcat(DEST, SRC, __FILE__, __LINE__)
#define strncat(DEST, SRC, SIZE)    Tau_strncat(DEST, SRC, SIZE, __FILE__, __LINE__)

#endif /* _TAU_MALLOC_H_ */
