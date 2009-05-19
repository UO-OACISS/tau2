

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

x_uint64 Tau_getTimeStamp();

int Tau_mergeProfiles() {
  int rank, size, tid, i, buflen, trash = 0;
  FILE *f;
  char *buf;
  MPI_Status status;
  x_uint64 start, end;

  tid = Tau_RtsLayer_myThread();

  if (tid != 0) {
    fprintf (stderr, "TAU: Merged file format does not support threads yet!\n");
    return 0;
  }

  
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

  buf = getSnapshotBuffer();
  buflen = getSnapshotBufferLength();


  if (rank == 0) {
    TAU_VERBOSE("TAU: Merging Profiles\n");
    start = Tau_getTimeStamp();
    
    if ((f = fopen ("tauprofile.xml", "w+")) == NULL) {
      char errormsg[4096];
      sprintf(errormsg,"Error: Could not create tauprofile.xml");
      perror(errormsg);
    }
    
    fwrite (buf, buflen, 1, f);

    for (i=1; i<size; i++) {
      /* send ok-to-go */
      PMPI_Send(&trash, 1, MPI_INT, i, 42, MPI_COMM_WORLD);
      
      /* receive buffer length */
      PMPI_Recv(&buflen, 1, MPI_INT, i, 42, MPI_COMM_WORLD, &status);
      buf = (char*) malloc (buflen);

      /* receive buffer */
      PMPI_Recv(buf, buflen, MPI_CHAR, i, 42, MPI_COMM_WORLD, &status);
      fwrite (buf, buflen, 1, f);
      free (buf);
    }

    end = Tau_getTimeStamp();
    TAU_VERBOSE("TAU: Merging Profiles Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
  
  } else {

    /* recieve ok to go */
    PMPI_Recv(&trash, 1, MPI_INT, 0, 42, MPI_COMM_WORLD, &status);

    /* send length */
    PMPI_Send(&buflen, 1, MPI_INT, 0, 42, MPI_COMM_WORLD);

    /* send data */
    PMPI_Send(buf, buflen, MPI_CHAR, 0, 42, MPI_COMM_WORLD);

  }
  TAU_VERBOSE("TAU: Profile Merging Complete\n");

  return 0;
}


#else
int Tau_mergeProfiles() {
  return 0;
}

#endif
