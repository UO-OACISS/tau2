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
#include <tau_types.h>
#include <TauEnv.h>
#include <TauSnapshot.h>
#include <TauMetrics.h>
#include <TauUnify.h>
#include <TauUtil.h>
#include <float.h>

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



static void stat_reduce (void *i, void *o, int *len,  MPI_Datatype *type) {
  double * in = (double *) i;
  double * inout = (double *) o;
  // assert (*len == 5);

  if (in[0] < inout[0]) { /* min */
    inout[0] = in[0];
  }
  if (in[1] > inout[1]) { /* max */
    inout[1] = in[1];
  }
  inout[2] += in[2];    /* sum */
  inout[3] += in[3];    /* sum^2 */
  inout[4] += in[4];    /* n */
}



extern "C" int Tau_collate_writeProfile() {
  int rank, size;

  int tid = 0;

  TauProfiler_updateIntermediateStatistics(tid);

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

  FunctionEventLister *functionEventLister = new FunctionEventLister();
  Tau_unify_object_t *functionUnifier = Tau_unify_unifyEvents(functionEventLister);

  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  Tau_unify_object_t *atomicUnifier = Tau_unify_unifyEvents(atomicEventLister);

  int numThreads = RtsLayer::getNumThreads();

  TAU_MPI_DEBUG0 ("Found %d total regions\n", functionUnifier->globalNumItems);

  /* create a reverse mapping, not strictly necessary, but it makes things easier */
  int *globalmap = (int*)TAU_UTIL_MALLOC(functionUnifier->globalNumItems * sizeof(int));
  for (int i=0; i<functionUnifier->globalNumItems; i++) {
    globalmap[i] = -1; /* -1 indicates that the event did not occur for this rank */
  }
  for (int i=0; i<functionUnifier->localNumItems; i++) {
    globalmap[functionUnifier->mapping[i]] = i;
  }


  MPI_Op op; 
  MPI_Op_create (stat_reduce, 1, &op);

  for (int i=0; i<functionUnifier->globalNumItems; i++) {
    int local_index = -1;
    if (globalmap[i] != -1) {
      local_index = functionUnifier->sortMap[globalmap[i]];
    }


    double *incltime;
    double *excltime;

    double in[5] = { 0., 0., 0., 0., 0. }; 
    double out[5] = { DBL_MAX, - DBL_MAX, 0., 0., 0. }; 

    if (local_index != -1) {
      FunctionInfo *fi = TheFunctionDB()[local_index];
      incltime = fi->getDumpInclusiveValues(tid);
      excltime = fi->getDumpExclusiveValues(tid);
      in[0] = excltime[0];
      in[1] = excltime[0];
      in[2] = excltime[0];
      in[3] = excltime[0]*excltime[0];
      in[4] = 1;
      // fprintf (stderr, "%i : %s\n", i, fi->GetName());
    } else {
      // fprintf (stderr, "%i : N/A\n", i);
    }
    

    //in[0] = s->min; in[1] = s->max; in[2] = s->sum; in[3] = s->sum2;
    // in[4] = s->n;
    MPI_Reduce (in, out, 5, MPI_DOUBLE, op, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      TAU_MPI_DEBUG0 ("[id=%2d] min=%9.16G max=%9.16G sum=%9.16G avg=%12.16G count=%4.16G : %s\n", i, out[0], out[1], out[2], out[2] / out[4], out[4], functionUnifier->globalStrings[i]);
    }

  }
  MPI_Op_free (&op); 

  
  TauProfiler_DumpData(false, tid, "profile");



  return 0;
}

#endif /* TAU_EXP_UNIFY */
#endif /* TAU_MPI */
