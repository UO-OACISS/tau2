/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauCollate.cpp  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Profile merging code                             **
**                                                                         **
****************************************************************************/

#ifdef TAU_MPI
#ifdef TAU_EXP_UNIFY

#include <mpi.h>
#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <Profile/tau_types.h>
#include <Profile/TauEnv.h>
#include <Profile/TauSnapshot.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauUnify.h>


#define DEBUG

#ifdef DEBUG

void TAU_MPI_DEBUG0(const char *format, ...) {
  int rank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}

#else

void TAU_MPI_DEBUG0(const char *format, ...) {
  return;
}

#endif



extern "C" int Tau_collate_writeProfile() {
  int rank, size;


  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

  FunctionEventLister *functionEventLister = new FunctionEventLister();
  unify_object_t *functionUnifier = Tau_unify_unifyEvents(functionEventLister);

  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  unify_object_t *atomicUnifier = Tau_unify_unifyEvents(atomicEventLister);

  int numThreads = RtsLayer::getNumThreads();

  TAU_MPI_DEBUG0 ("Found %d total regions\n", functionUnifier->globalNumItems);

  


  return 0;
}

#endif /* TAU_EXP_UNIFY */
#endif /* TAU_MPI */