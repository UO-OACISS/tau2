/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: ProfileMerge.c  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Profile merging code                             **
**                                                                         **
****************************************************************************/


#ifdef TAU_MPI

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
#include <TauXML.h>

extern "C" int TAUDECL Tau_RtsLayer_myThread();


#ifdef TAU_EXP_UNIFY
void Tau_profileMerge_writeDefinitions(FILE *f) {

  Tau_unify_object_t *functionUnifier, *atomicUnifier;
  functionUnifier = Tau_unify_getFunctionUnifier();
  atomicUnifier = Tau_unify_getAtomicUnifier();

  Tau_util_outputDevice out;
  out.type = TAU_UTIL_OUTPUT_FILE;
  out.fp = f;

  // start of a profile block
  Tau_util_output (&out, "<profile_xml>\n");

  Tau_util_output (&out, "\n<definitions thread=\"*\">\n");

  for (int i=0; i<functionUnifier->globalNumItems; i++) {
    Tau_util_output (&out, "<event id=\"%d\"><name>", i);
    Tau_XML_writeString(&out, functionUnifier->globalStrings[i]);
    Tau_util_output (&out, "</name><group>");
    Tau_XML_writeString(&out, ":)");
    Tau_util_output (&out, "</group></event>\n");
  }

  for (int i=0; i<atomicUnifier->globalNumItems; i++) {
    Tau_util_output (&out, "<userevent id=\"%d\"><name>", i);
    Tau_XML_writeString(&out, atomicUnifier->globalStrings[i]);
    Tau_util_output (&out, "</name></userevent>\n");
  }

  Tau_util_output (&out, "\n</definitions>\n");

  Tau_util_output (&out, "</profile_xml>\n");

}
#endif


int Tau_mergeProfiles() {
  int rank, size, tid, i, buflen;
  FILE *f;
  char *buf;
  MPI_Status status;
  x_uint64 start, end;
  const char *profiledir = TauEnv_get_profiledir();

#ifdef TAU_EXP_UNIFY
  Tau_unify_unifyDefinitions();
  Tau_snapshot_writeUnifiedBuffer();
#else

  Tau_snapshot_writeToBuffer("merge");
#endif

  tid = Tau_RtsLayer_myThread();

  if (tid != 0) {
    fprintf (stderr, "TAU: Merged file format does not support threads yet!\n");
    return 0;
  }


  // temp: write regular profiles too, for comparison
  TauProfiler_DumpData(false, 0, "profile");

  
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

  buf = Tau_snapshot_getBuffer();
  buflen = Tau_snapshot_getBufferLength();

  int maxBuflen;
  MPI_Reduce(&buflen, &maxBuflen, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);


  if (rank == 0) {
    char *recv_buf = (char *) malloc (maxBuflen);

    TAU_VERBOSE("TAU: Merging Profiles\n");
    start = TauMetrics_getTimeOfDay();


    char filename[4096];
    sprintf (filename,"%s/tauprofile.xml", profiledir);

    if ((f = fopen (filename, "w+")) == NULL) {
      char errormsg[4096];
      sprintf(errormsg,"Error: Could not create tauprofile.xml");
      perror(errormsg);
    }

#ifdef TAU_EXP_UNIFY
    Tau_profileMerge_writeDefinitions(f);
#endif

    for (i=1; i<size; i++) {
      /* send ok-to-go */
      PMPI_Send(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD);
      
      /* receive buffer length */
      PMPI_Recv(&buflen, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

      /* receive buffer */
      PMPI_Recv(recv_buf, buflen, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      fwrite (recv_buf, buflen, 1, f);
    }
    free (recv_buf);

    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: Merging Profiles Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);

    char tmpstr[256];
    sprintf(tmpstr, "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU Profile Merge Time", tmpstr);
    Tau_snapshot_writeMetaDataBlock();

    buf = Tau_snapshot_getBuffer();
    buflen = Tau_snapshot_getBufferLength();
    fwrite (buf, buflen, 1, f);
    fclose(f);
  
  } else {

    /* recieve ok to go */
    PMPI_Recv(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    /* send length */
    PMPI_Send(&buflen, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    /* send data */
    PMPI_Send(buf, buflen, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  return 0;
}


#else /* TAU_MPI */
int Tau_mergeProfiles() {
  return 0;
}
#endif /* TAU_MPI */
