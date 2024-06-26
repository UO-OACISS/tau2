/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauUtil.h      				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains TAU utility routines          **
**                                                                         **
****************************************************************************/

#ifndef _TAU_UTIL_H_
#define _TAU_UTIL_H_

#include <stdlib.h> 
#include <stdio.h>

typedef struct Tau_util_outputDevice_ {
  FILE *fp;
  int type; // 0 = file, 1 = buffer
  char *buffer;
  int bufidx;
  int buflen;
} Tau_util_outputDevice;

#define TAU_UTIL_OUTPUT_FILE 0
#define TAU_UTIL_OUTPUT_BUFFER 1
#define TAU_UTIL_INITIAL_BUFFER 5000000
#define TAU_UTIL_OUTPUT_THRESHOLD 100000

Tau_util_outputDevice *Tau_util_createBufferOutputDevice();
char *Tau_util_getOutputBuffer(Tau_util_outputDevice *out);
void Tau_util_destroyOutputDevice(Tau_util_outputDevice *out);
int Tau_util_getOutputBufferLength(Tau_util_outputDevice *out);
int Tau_util_output(Tau_util_outputDevice *out, const char *format, ...);
int Tau_util_readFullLine(char *line, FILE *fp);
char const * Tau_util_removeRuns(char const * str);

void TAU_ABORT(const char *format, ...);

void *Tau_util_malloc(size_t size, const char *file, int line);
#define TAU_UTIL_MALLOC(size) Tau_util_malloc(size, __FILE__, __LINE__);
void *Tau_util_calloc(size_t size, const char *file, int line);
#define TAU_UTIL_CALLOC(size) Tau_util_calloc(size, __FILE__, __LINE__);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void Tau_plugin_sendmsg(long unsigned int type, long unsigned int destination, long unsigned int length, long unsigned int remoteid);
void Tau_plugin_recvmsg(long unsigned int type, long unsigned int source, long unsigned int length, long unsigned int remoteid);
#ifdef __cplusplus
}
#endif /* __cplusplus */

// Use a macro so we can compile it out if too much overhead.

#define TAU_PLUGIN_SENDMSG(t, d, l, r) if (Tau_plugins_enabled.send) { Tau_plugin_sendmsg((long unsigned int)t, (long unsigned int)d, (long unsigned int)l, (long unsigned int)r); }
#define TAU_PLUGIN_RECVMSG(t, s, l, r) if (Tau_plugins_enabled.recv) { Tau_plugin_recvmsg((long unsigned int)t, (long unsigned int)s, (long unsigned int)l, (long unsigned int)r); }

/* The following macros help create a local array and assign to elements of 
   the local C array, values from Fortran array after conversion using f2c 
   MPI macros. Need to optimize the implementation. Use static array instead
   of malloc */
   

#if (defined(sgi)  || defined (TAU_WRAPPER_BLANK) || defined(__blrts__))
#define TAU_DECL_LOCAL(mtype, l) MPI_Fint * l
#define TAU_ALLOC_LOCAL(mtype, l, size) 
#define TAU_DECL_ALLOC_LOCAL(mtype, l, size) MPI_Fint * l
#define TAU_ASSIGN_VALUES(dest, src, size, func) dest = src 
#define TAU_ASSIGN_STATUS_F2C(dest, src, size, func) dest = src
#define TAU_ASSIGN_STATUS_C2F(dest, src, size, func) dest = src
#define TAU_FREE_LOCAL(l) 
#else
#define TAU_DECL_LOCAL(mtype, l) mtype *l
#define TAU_ALLOC_LOCAL(mtype, l, size) l = (mtype *) malloc(sizeof(mtype) * size)
#define TAU_DECL_ALLOC_LOCAL(mtype, l, size) TAU_DECL_LOCAL(mtype, l) = TAU_ALLOC_LOCAL(mtype, l, size) 
#define TAU_ASSIGN_VALUES(dest, src, size, func) { int i; for (i = 0; i < size; i++) dest[i] = func(src[i]); }

#define TAU_ASSIGN_STATUS_F2C(dest, src, size, func) { int i; for (i = 0; i < size; i++) func((MPI_Fint*)&((MPI_Status*)src)[i], &((MPI_Status*)dest)[i]); }
#define TAU_ASSIGN_STATUS_C2F(dest, src, size, func) { int i; for (i = 0; i < size; i++) func(&((MPI_Status*)src)[i], (MPI_Fint*)&((MPI_Status*)dest)[i]); }



#define TAU_FREE_LOCAL(l) free(l)
#endif /* sgi || TAU_MPI_NEEDS_STATUS */

/******************************************************/
#if (defined(sgi) || defined(TAU_MPI_NEEDS_STATUS))
#ifdef TAU_USE_PMPI_F2C
#define MPI_Status_f2c PMPI_Status_f2c
#define MPI_Status_c2f PMPI_Status_c2f
#else /* TAU_USE_PMPI_F2C */
#define MPI_Status_c2f(c,f) *(MPI_Status *)f=*(MPI_Status *)c 
#define MPI_Status_f2c(f,c) *(MPI_Status *)c=*(MPI_Status *)f
#endif /* TAU_USE_PMPI_F2C */
#endif /* sgi || TAU_MPI_NEEDS_STATUS */

#endif /* _TAU_UTIL_H_ */
