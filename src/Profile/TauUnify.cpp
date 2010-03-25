/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File            : TauUnify.cpp                                     **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
**	Documentation	: See http://tau.uoregon.edu                       **
**                                                                         **
**      Description     : Event unification                                **
**                                                                         **
****************************************************************************/


// int *local_id_map;
// int *


#include <TauUtil.h>
#include <TauMetrics.h>
#include <Profiler.h>
#include <mpi.h>

Tau_util_outputDevice *Tau_unify_generateLocalDefinitionBuffer() {

  Tau_util_outputDevice *out = Tau_util_createBufferOutputDevice();
  if (out == NULL) {
    TAU_ABORT("TAU: Abort: Unable to generate create buffer for local definitions\n");
  }

  int numFuncs = TheFunctionDB().size();

  Tau_util_output(out,"%d\n", numFuncs);
  for(int i=0;i<numFuncs;i++) {
    FunctionInfo *fi = TheFunctionDB()[i];
    Tau_util_output(out,"%d %s\n", i, fi->GetName());
  }

  return out;
}


extern "C" int Tau_unify_unifyDefinitions() {
  int rank, numRanks, i;
  MPI_Status status;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  x_uint64 start, end;

  if (rank == 0) {
    TAU_VERBOSE("TAU: Unifying...\n");
    start = TauMetrics_getInitialTimeStamp();
  }


  Tau_util_outputDevice *out = Tau_unify_generateLocalDefinitionBuffer();
  if (!out) {
    TAU_ABORT("TAU: Abort: Unable to generate local definitions\n");
  }

  char *defBuf = Tau_util_getOutputBuffer(out);
  int defBufSize = Tau_util_getOutputBufferLength(out);

  TAU_VERBOSE("UNIFY: [%d] - My def buf size = %d\n", rank, defBufSize);

  int maxDefBufSize;
  PMPI_Reduce(&defBufSize, &maxDefBufSize, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);


  if (rank == 0) {
    TAU_VERBOSE("UNIFY: Maximum def buf size = %d\n", maxDefBufSize);
    char *recv_buf = (char *) malloc (maxDefBufSize);
    if (recv_buf == NULL) {
      TAU_ABORT("TAU: Abort: Unable to allocate recieve buffer for unification\n");
    }
    int recv_buflen;

    for (i=1; i<numRanks; i++) {
      /* send ok-to-go */
      PMPI_Send(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD);
      
      /* receive buffer length */
      PMPI_Recv(&recv_buflen, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

      /* receive buffer */
      PMPI_Recv(recv_buf, recv_buflen, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
    }

    /* Do something with the data */

    free (recv_buf);
    end = TauMetrics_getInitialTimeStamp();
    TAU_VERBOSE("TAU: Unifying Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
  } else {

    /* recieve ok to go */
    PMPI_Recv(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    /* send length */
    PMPI_Send(&defBufSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    /* send data */
    PMPI_Send(defBuf, defBufSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

    /* Get back a mapping table... */

  }

}




