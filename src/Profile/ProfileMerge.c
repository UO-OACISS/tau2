

#ifdef TAU_MPI

#include <TAU.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <Profile/tau_types.h>
#include <Profile/TauEnv.h>

int TAUDECL Tau_RtsLayer_myThread();
char* TAUDECL getSnapshotBuffer();
int TAUDECL getSnapshotBufferLength();


int Tau_mergeProfiles() {
  int rank, size, tid, i, buflen;
  FILE *f;
  char *buf;
  MPI_Status status;

  tid = Tau_RtsLayer_myThread();

  if (tid != 0) {
    fprintf (stderr, "TAU: Merged file format does not support threads yet!\n");
    return 0;
  }

  
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

  buf = getSnapshotBuffer();
  buflen = getSnapshotBufferLength();

  TAU_VERBOSE("TAU: Merging Profiles\n");

  if (rank == 0) {
    if ((f = fopen ("tauprofile.xml", "w+")) == NULL) {
      char errormsg[4096];
      sprintf(errormsg,"Error: Could not create tauprofile.xml");
      perror(errormsg);
    }
    
    fwrite (buf, buflen, 1, f);

    for (i=1; i<size; i++) {
      PMPI_Recv(&buflen, 1, MPI_INT, MPI_ANY_SOURCE, 42, MPI_COMM_WORLD, &status);
      buf = (char*) malloc (buflen);
      PMPI_Recv(buf, buflen, MPI_CHAR, status.MPI_SOURCE, 42, MPI_COMM_WORLD, &status);
      fwrite (buf, buflen, 1, f);
      free (buf);
    }

  } else {

    /* send length */
    MPI_Send(&buflen, 1, MPI_INT, 0, 42, MPI_COMM_WORLD);

    /* send data */
    MPI_Send(buf, buflen, MPI_CHAR, 0, 42, MPI_COMM_WORLD);

  }

  return 0;
}


#else
int Tau_mergeProfiles() {
  return 0;
}

#endif
