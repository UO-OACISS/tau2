#include <mpc_mpi.h>
#include <Profile/Profiler.h>
#include <stdio.h>

#include <Profile/TauEnv.h>

#ifdef TAU_MPI
#include <mpi.h>
#endif /* TAU_MPI */
#include <TauMetaData.h>
#include <TauMetrics.h>

//#include "gen_prof.h"


extern "C" int genProfileFakeExtern()
{
  int rank = 0;
  int numRanks;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  TAU_VERBOSE("TAU - genProfileFake Extern C: rank=%d, numRanks=%d\n", rank, numRanks);

 return 0;
}


#if 1
extern "C" int genProfile()
{

  //int numRanks;
  //PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  //TAU_VERBOSE("TAU - inside genProfile(): rank=%d, numRanks=%d\n", rank, numRanks);

  Tau_metadata_fillMetaData();

#if 0
  static int merged = 0;
  if (merged == 1) {
    return 0;
  }
  merged = 1;
#endif

  int rank = 0;

#ifdef TAU_MPI
  int numRanks;
  //if (TAU_MPI_Finalized()) {
  //  fprintf(stdout, "TAU_MPI_Finalized() called\n");
  //  return 0;
  //}

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  TAU_VERBOSE("TAU genProfile extern C: rank=%d, numRanks=%d\n", rank, numRanks);
#endif /* TAU_MPI */

  x_uint64 start, end;

#if 1
  if (rank == 0) {

    TAU_VERBOSE("TAU: Merging MetaData...\n");
    start = TauMetrics_getTimeOfDay();

#if 1
#ifdef TAU_MPI
    Tau_util_outputDevice *out = Tau_metadata_generateMergeBuffer();
    char *defBuf = Tau_util_getOutputBuffer(out);
    int defBufSize = Tau_util_getOutputBufferLength(out);

    PMPI_Bcast(&defBufSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    PMPI_Bcast(defBuf, defBufSize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */
#endif

    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: MetaData Merging Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
    char tmpstr[256];
    snprintf(tmpstr, sizeof(tmpstr),  "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU MetaData Merge Time", tmpstr);

#if 1
#ifdef TAU_MPI
        Tau_util_destroyOutputDevice(out);
#endif /* TAU_MPI */
#endif

  } else {

#if 1
#ifdef TAU_MPI
    TAU_VERBOSE("TAU: Metadata, rank different from 0\n");
    int BufferSize;
    PMPI_Bcast(&BufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *Buffer = (char*) TAU_UTIL_MALLOC(BufferSize);
    PMPI_Bcast(Buffer, BufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    Tau_metadata_removeDuplicates(Buffer, BufferSize);
        free(Buffer);
#endif /* TAU_MPI */
#endif
  }
#endif

  return 0;
}
#endif


