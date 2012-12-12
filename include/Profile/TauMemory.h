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


#if defined(__darwin__) || defined(__APPLE__)
#undef HAVE_MEMALIGN
#else
#define HAVE_MEMALIGN 1
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void TauDetectMemoryLeaks(void);

void Tau_track_memory_allocation(const char *file, int line, size_t size, void* ptr);
void Tau_track_memory_deallocation(const char *file, int line, void* ptr);

void * Tau_new(const char *file, int line, size_t size, void* ptr);

void * Tau_malloc(size_t size, const char * filename, int lineno);
void * Tau_calloc(size_t elemCount, size_t elemSize, const char * filename, int lineno);
void   Tau_free(void * baseAdr, const char * filename, int lineno);
#ifdef HAVE_MEMALIGN
void * Tau_memalign(size_t alignment, size_t userSize, const char * filename, int lineno);
#endif
int    Tau_posix_memalign(void **memptr, size_t alignment, size_t userSize, const char * filename, int lineno);
void * Tau_realloc(void * baseAdr, size_t newSize, const char * filename, int lineno);
void * Tau_valloc(size_t size, const char * filename, int lineno);

char * Tau_strdup(const char *str, const char * filename, int lineno);
void * Tau_memcpy(void *dest, const void *src, size_t size, const char * filename, int lineno);
char * Tau_strcpy(char *dest, const char *src, const char * filename, int lineno);
char * Tau_strncpy(char *dest, const char *src, size_t size, const char * filename, int lineno);
char * Tau_strcat(char *dest, const char *src, const char * filename, int lineno);
char * Tau_strncat(char *dest, const char *src, size_t size, const char * filename, int lineno);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_MEMORY_H_ */

/***************************************************************************
 * $RCSfile: TauMemory.h,v $   $Author: amorris $
 * $Revision: 1.4 $   $Date: 2010/02/03 06:09:44 $
 * TAU_VERSION_ID: $Id: TauMemory.h,v 1.4 2010/02/03 06:09:44 amorris Exp $ 
 ***************************************************************************/
