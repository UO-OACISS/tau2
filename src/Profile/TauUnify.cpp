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

#ifdef TAU_MPI

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

  // determine maximum buffer size
  int maxDefBufSize;
  PMPI_Allreduce(&defBufSize, &maxDefBufSize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  
  // allocate receive buffer
  char *recv_buf = (char *) malloc (maxDefBufSize);
  if (recv_buf == NULL) {
    TAU_ABORT("TAU: Abort: Unable to allocate recieve buffer for unification\n");
  }

  // use binomial heap algorithm (like MPI_Reduce)
  int mask = 0x1;

  while (mask < numRanks) {
    if ((mask & rank) == 0) {
      int source = (rank | mask);
      if (source < numRanks) {
	
	/* send ok-to-go */
	PMPI_Send(NULL, 0, MPI_INT, source, 0, MPI_COMM_WORLD);
	
	/* receive buffer length */
	int recv_buflen;
	PMPI_Recv(&recv_buflen, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
	
	/* receive buffer */
	PMPI_Recv(recv_buf, recv_buflen, MPI_CHAR, source, 0, MPI_COMM_WORLD, &status);

	printf ("[%d] received from %d\n", rank, source);

	/* Do something with the data <HERE> */
      }

    } else {
      /* I've received all that I'm going to.  Send my result to my parent */
      int target = (rank & (~ mask));

      /* recieve ok to go */
      PMPI_Recv(NULL, 0, MPI_INT, target, 0, MPI_COMM_WORLD, &status);
      
      /* send length */
      PMPI_Send(&defBufSize, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
      
      /* send data */
      PMPI_Send(defBuf, defBufSize, MPI_CHAR, target, 0, MPI_COMM_WORLD);

      printf ("[%d] sent to %d\n", rank, target);
      break;
    }
    mask <<= 1;
  }



  free (recv_buf);

  if (rank == 0) {
    end = TauMetrics_getInitialTimeStamp();
    TAU_VERBOSE("TAU: Unifying Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
  }

  return 0;
}



#endif /* TAU_MPI */
