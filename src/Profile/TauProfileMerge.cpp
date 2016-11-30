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
#endif  /* TAU_MPI */

#ifdef TAU_SHMEM
#include <shmem.h>
#endif /* TAU_SHMEM */

#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <tau_types.h>
#include <TauEnv.h>
#include <TauSnapshot.h>
#include <TauMetrics.h>
#include <TauUnify.h>
#include <TauCollate.h>
#include <TauUtil.h>
#include <TauXML.h>

// Moved from header file
#ifdef __cplusplus
using namespace std;
#endif


extern "C" int TAUDECL Tau_RtsLayer_myThread();


#ifdef TAU_UNIFY
void Tau_profileMerge_writeDefinitions(int *globalEventMap, int
*globalAtomicEventMap, FILE *f) {

  Tau_unify_object_t *functionUnifier, *atomicUnifier;
  functionUnifier = Tau_unify_getFunctionUnifier();
  atomicUnifier = Tau_unify_getAtomicUnifier();

  Tau_util_outputDevice out;
  out.type = TAU_UTIL_OUTPUT_FILE;
  out.fp = f;

  // start of a profile block
  Tau_util_output (&out, "<profile_xml>\n");

  Tau_util_output (&out, "\n<definitions thread=\"*\">\n");

	for (int i=0; i<Tau_Global_numCounters; i++) {
      const char *tmpChar = RtsLayer::getCounterName(i);
      Tau_util_output (&out, "<metric id=\"%d\">", i);
      Tau_XML_writeTag(&out, "name", tmpChar, true);
      Tau_XML_writeTag(&out, "units", "unknown", true);
      Tau_util_output (&out, "</metric>\n");
  }

  for (int i=0; i<functionUnifier->globalNumItems; i++) {
    Tau_util_output (&out, "<event id=\"%d\"><name>", i);

    char *name = functionUnifier->globalStrings[i];
    char *group = strstr(name,":GROUP:");
    if (group == NULL) {
      fprintf (stderr, "TAU: Error extracting groups for %s!\n",name);
    } else {
      char *target = group;
      group+=strlen(":GROUP:");
      *target=0;
    }

    Tau_XML_writeString(&out, name);
    Tau_util_output (&out, "</name><group>");
    Tau_XML_writeString(&out, group);
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


int Tau_mergeProfiles()
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  int rank, size, i, buflen;
  FILE *f;
  char *buf;
#ifdef TAU_MPI
  MPI_Status status;
#endif  /* TAU_MPI */
  x_uint64 start, end;
  const char *profiledir = TauEnv_get_profiledir();
  const char *profile_prefix = TauEnv_get_profile_prefix();

  Tau_detect_memory_leaks();

#ifdef TAU_UNIFY
  Tau_unify_unifyDefinitions();

	for (int tid = 0; tid<RtsLayer::getTotalThreads(); tid++) {
		Tau_snapshot_writeUnifiedBuffer(tid);
	}
#else
  Tau_snapshot_writeToBuffer("merge");
#endif

  // temp: write regular profiles too, for comparison
  //TauProfiler_DumpData(false, 0, "profile");
  
  rank = 0;
  size = 1;
#ifdef TAU_MPI
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);
#endif  /* TAU_MPI */
#ifdef TAU_SHMEM
  size = shmem_n_pes();
  rank = shmem_my_pe();
#endif /* TAU_SHMEM */

	buflen = Tau_snapshot_getBufferLength()+1;
	buf = (char *) malloc(buflen);
	Tau_snapshot_getBuffer(buf);

  int maxBuflen = buflen;
#ifdef TAU_MPI
  PMPI_Reduce(&buflen, &maxBuflen, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
#endif  /* TAU_MPI */
#ifdef TAU_SHMEM
  int *shbuflen = (int*)shmem_malloc(sizeof(int));
  *shbuflen = buflen;
  int *shmaxBuflen = (int*)shmem_malloc(sizeof(int));

  int *maxBuflens = (int*)shmem_malloc(size*sizeof(int));
  shmem_int_put(&maxBuflens[rank], &maxBuflen, 1, 0);
  shmem_barrier_all();
  if(rank == 0)
    for(i =0; i < size; i++)
      if(maxBuflen < maxBuflens[i]) maxBuflen = maxBuflens[i];
  shmem_barrier_all();
  *shmaxBuflen = maxBuflen;
  shmem_int_get(shmaxBuflen, shmaxBuflen, 1, 0);
  shmem_barrier_all();
  maxBuflen = *shmaxBuflen;
  shmem_free(shmaxBuflen);
  shmem_free(maxBuflens);
  char *shbuf = (char*)shmem_malloc(maxBuflen);
  strncpy(shbuf, buf, maxBuflen);
#endif /* TAU_SHMEM */

#ifdef TAU_UNIFY
  Tau_unify_object_t *functionUnifier;
  int numEvents = 0;
  int globalNumThreads;
  int *numEventThreads;
  int *globalEventMap = 0;

  double ***gExcl, ***gIncl;
  double **gNumCalls, **gNumSubr;
  double ***sExcl, ***sIncl;
  double **sNumCalls, **sNumSubr;

  Tau_unify_object_t *atomicUnifier;
  int numAtomicEvents = 0;
  int *numAtomicEventThreads;
  int *globalAtomicEventMap = 0;
  
  double **gAtomicMin, **gAtomicMax;
  double **gAtomicCalls, **gAtomicMean;
  double **gAtomicSumSqr;
  double **sAtomicMin, **sAtomicMax;
  double **sAtomicCalls, **sAtomicMean;
  double **sAtomicSumSqr;

  if (TauEnv_get_stat_precompute() == 1) {
    // Unification must already be called.
    functionUnifier = Tau_unify_getFunctionUnifier();
    numEvents = functionUnifier->globalNumItems;
    numEventThreads = (int*)TAU_UTIL_MALLOC(numEvents*sizeof(int));
    globalEventMap = (int*)TAU_UTIL_MALLOC(numEvents*sizeof(int));
    // initialize all to -1
    for (int i=0; i<functionUnifier->globalNumItems; i++) { 
      // -1 indicates that the event did not occur for this rank
      globalEventMap[i] = -1; 
    }
    for (int i=0; i<functionUnifier->localNumItems; i++) {
      globalEventMap[functionUnifier->mapping[i]] = i; // set reverse mapping
    }
    Tau_collate_get_total_threads(functionUnifier, &globalNumThreads, &numEventThreads,
				  numEvents, globalEventMap,false);
    
    Tau_collate_allocateFunctionBuffers(&gExcl, &gIncl,
					&gNumCalls, &gNumSubr,
					numEvents,
					Tau_Global_numCounters,
					COLLATE_OP_BASIC);
    if (rank == 0) {
      Tau_collate_allocateFunctionBuffers(&sExcl, &sIncl,
					  &sNumCalls, &sNumSubr,
					  numEvents,
					  Tau_Global_numCounters,
					  COLLATE_OP_DERIVED);
    }
    Tau_collate_compute_statistics(functionUnifier, globalEventMap, 
				   numEvents, 
				   globalNumThreads, numEventThreads,
				   &gExcl, &gIncl, &gNumCalls, &gNumSubr,
				   &sExcl, &sIncl, &sNumCalls, &sNumSubr);

    atomicUnifier = Tau_unify_getAtomicUnifier();
    numAtomicEvents = atomicUnifier->globalNumItems;
    numAtomicEventThreads = 
      (int*)TAU_UTIL_MALLOC(numAtomicEvents*sizeof(int));
    globalAtomicEventMap = (int*)TAU_UTIL_MALLOC(numAtomicEvents*sizeof(int));
    // initialize all to -1
    for (int i=0; i<numAtomicEvents; i++) { 
      // -1 indicates that the event did not occur for this rank
      globalAtomicEventMap[i] = -1; 
    }
    for (int i=0; i<atomicUnifier->localNumItems; i++) {
      // set reverse mapping
      globalAtomicEventMap[atomicUnifier->mapping[i]] = i;
    }
    Tau_collate_get_total_threads(atomicUnifier, &globalNumThreads, &numAtomicEventThreads,
				  numAtomicEvents, globalAtomicEventMap,true);
    
    Tau_collate_allocateAtomicBuffers(&gAtomicMin, &gAtomicMax,
				      &gAtomicCalls, &gAtomicMean,
				      &gAtomicSumSqr,
				      numAtomicEvents,
				      COLLATE_OP_BASIC);
    if (rank == 0) {
      Tau_collate_allocateAtomicBuffers(&sAtomicMin, &sAtomicMax,
					&sAtomicCalls, &sAtomicMean,
					&sAtomicSumSqr,
					numAtomicEvents,
					COLLATE_OP_DERIVED);
    }
    Tau_collate_compute_atomicStatistics(atomicUnifier, globalAtomicEventMap, 
					 numAtomicEvents, 
					 globalNumThreads, 
					 numAtomicEventThreads,
					 &gAtomicMin, &gAtomicMax, 
					 &gAtomicCalls, &gAtomicMean,
					 &gAtomicSumSqr,
					 &sAtomicMin, &sAtomicMax, 
					 &sAtomicCalls, &sAtomicMean,
					 &sAtomicSumSqr);

  } /* TauEnv_get_stat_precompute() == 1 */
#endif /* TAU_UNIFY */
      
  if (rank == 0) {
    char *recv_buf = (char *) malloc (maxBuflen);

    TAU_VERBOSE("Before Merging Profiles: Tau_check_dirname()");
    profiledir=Tau_check_dirname(profiledir);

    TAU_VERBOSE("TAU: Merging Profiles\n");
    start = TauMetrics_getTimeOfDay();


    char filename[4096];
    if (profile_prefix != NULL) {
      sprintf (filename,"%s/%s-tauprofile.xml", profiledir, profile_prefix);
    } else {
      sprintf (filename,"%s/tauprofile.xml", profiledir);
    }

    if ((f = fopen (filename, "w+")) == NULL) {
      char errormsg[4096];
      sprintf(errormsg,"Error: Could not create tauprofile.xml");
      perror(errormsg);
    }

#ifdef TAU_UNIFY
    Tau_profileMerge_writeDefinitions(globalEventMap, globalAtomicEventMap, f);
#endif

    for (i=1; i<size; i++) {

#ifdef TAU_MPI
      /* send ok-to-go */
      PMPI_Send(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD);
      
      /* receive buffer length */
      PMPI_Recv(&buflen, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

      /* receive buffer */
      PMPI_Recv(recv_buf, buflen, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
#endif  /* TAU_MPI */
#ifdef TAU_SHMEM
      /* receive buffer length */
      shmem_int_get(&buflen, shbuflen, 1, i);

      /* receive buffer */
      shmem_getmem(recv_buf, shbuf, buflen, i);
#endif /* TAU_SHMEM */

      if (!TauEnv_get_summary_only()) { /* write each rank? */
        fwrite (recv_buf, buflen, 1, f);
      } else {
	// If Summary is desired, write only rank 1's data along with rank 0.
	// *CWL* NOTE: This is done so as to trick paraprof into displaying
	//             statistics-based data that we generate using
	//             pre-compute. This hack should be safe to remove after
	//             paraprof has been fixed to handle pure summary-only
	//             data.
	if (i == 1) {
	  fwrite (recv_buf, buflen, 1, f);
	}
      }
    }
    free (recv_buf);

    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: Merging Profiles Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);

    char tmpstr[256];
    sprintf(tmpstr, "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU Profile Merge Time", tmpstr);
    if (TauEnv_get_stat_precompute() == 1) {
      TAU_METADATA("TAU_PRECOMPUTE", "on");
    } else {
      TAU_METADATA("TAU_PRECOMPUTE", "off");
    }
    if (TauEnv_get_summary_only()) { /* write only rank one metadata for summary
		profile */
			if (rank == 0) {
    		Tau_snapshot_writeMetaDataBlock();
			}
	  }
		else {
    	Tau_snapshot_writeMetaDataBlock();
		}
    
		buflen = Tau_snapshot_getBufferLength()+1;
		buf = (char *) malloc(buflen);
    Tau_snapshot_getBuffer(buf);
    fwrite (buf, buflen, 1, f);

#ifdef TAU_UNIFY
    if (TauEnv_get_stat_precompute() == 1) {
      if (rank == 0) {
	// *CWL* Now write the computed statistics out in their own special
	//   profile and definition blocks.
	
	char metricList[4096];
	char *loc = metricList;
	for (int m=0; m<Tau_Global_numCounters; m++) {
	  loc += sprintf(loc,"%d ", m);
	}

	// write profile blocks for total value
	fprintf(f,"<profile_xml>\n");
	fprintf(f,"<derivedentity id=\"%s\">\n", "total");
	fprintf(f,"</derivedentity>\n");
	fprintf(f,"<derivedprofile derivedentity=\"%s\">\n", "total");
	
	fprintf(f,"<derivedinterval_data metrics=\"%s\">\n", metricList);
	for (int i=0; i<numEvents; i++) {
	  fprintf(f, "%d %lld %lld ", i, (long long)gNumCalls[step_sum][i], 
		  (long long)gNumSubr[step_sum][i]);
	  for (int m=0; m<Tau_Global_numCounters; m++) {
	    fprintf(f, "%.16G %.16G ", gExcl[step_sum][m][i], 
		    gIncl[step_sum][m][i]);
	  }	  
	  fprintf(f,"\n");
	}
	fprintf(f, "</derivedinterval_data>\n");
	  
	
	// close
	fprintf(f,"</derivedprofile>\n");
	fprintf(f,"\n</profile_xml>\n");

	// write profile blocks for each stat
	// *CWL* Tentatively not writing out min_all and max_all
	for (int s=0; s<NUM_STAT_TYPES; s++) {
	  fprintf(f,"<profile_xml>\n");
          fprintf(f,"<derivedentity id=\"%s\">\n", stat_names[s]);
	  fprintf(f,"</derivedentity>\n");
          if (s > NUM_STAT_TYPES-3) {
            fprintf(f,"<%s_derivedprofile derivedentity=\"%s\">\n", stat_names[s], stat_names[s]);
          } else {
            fprintf(f,"<derivedprofile derivedentity=\"%s\">\n", stat_names[s]);
          }
	  fprintf(f,"<derivedinterval_data metrics=\"%s\">\n", metricList);
	  for (int i=0; i<numEvents; i++) {
	    fprintf(f, "%d %.16G %.16G ", i, sNumCalls[s][i], sNumSubr[s][i]);
	    for (int m=0; m<Tau_Global_numCounters; m++) {
	      fprintf(f, "%.16G %.16G ", sExcl[s][m][i], sIncl[s][m][i]);
	    }	  
	    fprintf(f,"\n");
	  }

	  fprintf(f, "</derivedinterval_data>\n");
	  fprintf(f,"<derivedatomic_data>\n");
	  for (int i=0; i<numAtomicEvents; i++) {
	    // output order = num calls, max, min, mean, sumsqr
	    fprintf(f,"%d %.16G %.16G %.16G %.16G %.16G\n", i,
		   sAtomicCalls[s][i], 
		   sAtomicMax[s][i],
		   sAtomicMin[s][i], 
		   sAtomicMean[s][i],
		   sAtomicSumSqr[s][i]);
	  }
	  fprintf(f,"</derivedatomic_data>\n");
	  

	  // close
          if (s > NUM_STAT_TYPES -3) { // min & max 
	    fprintf(f,"</%s_derivedprofile>\n",stat_names[s]);
          } else {
	    fprintf(f,"</derivedprofile>\n");
          }
	  fprintf(f,"\n</profile_xml>\n");
	}
	// *CWL* Free allocated structures.
	free(globalEventMap);
	Tau_collate_freeFunctionBuffers(&sExcl, &sIncl,
					&sNumCalls, &sNumSubr,
					Tau_Global_numCounters,
					COLLATE_OP_DERIVED);
      }  /* rank == 0 */
      Tau_collate_freeFunctionBuffers(&gExcl, &gIncl,
				      &gNumCalls, &gNumSubr,
				      Tau_Global_numCounters,
				      COLLATE_OP_BASIC);
    }
#endif /* TAU_UNIFY */

    fflush(f);
    
#ifdef TAU_FCLOSE_MERGE
    fclose(f);
#endif
  } else {

#ifdef TAU_MPI
    /* recieve ok to go */
    PMPI_Recv(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    /* send length */
    PMPI_Send(&buflen, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    /* send data */
    PMPI_Send(buf, buflen, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
#endif  /* TAU_MPI */

  }
	free(buf);
#ifdef TAU_SHMEM
        shmem_free(shbuf);
        shmem_free(shbuflen);
#endif /* TAU_SHMEM */
  return 0;
}

