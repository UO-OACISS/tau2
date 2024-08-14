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
extern "C" void  __real_shmem_int_put(int * a1, const int * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_int_get(int * a1, const int * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_putmem(void * a1, const void * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_getmem(void * a1, const void * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_barrier_all() ;
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
extern "C" int   __real__num_pes() ;
extern "C" int   __real__my_pe() ;
extern "C" void* __real_shmalloc(size_t a1) ;
extern "C" void  __real_shfree(void * a1) ;
#else
extern "C" int   __real_shmem_n_pes() ;
extern "C" int   __real_shmem_my_pe() ;
extern "C" void* __real_shmem_malloc(size_t a1) ;
extern "C" void  __real_shmem_free(void * a1) ;
#endif
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
*globalAtomicEventMap, FILE *f, bool anonymize=false) {

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
  
  char anonymous_name[64*1024]; 
  char *group; 
  char anonymous_group[64];
  char MPI_group[64];
  char anonymous_event_name[64];
  if (anonymize) {
    snprintf(anonymous_group, sizeof(anonymous_group),  "TAU_ANONYMOUS_GROUP");
    snprintf(MPI_group, sizeof(MPI_group),  "MPI");
  }
  for (int i=0; i<functionUnifier->globalNumItems; i++) {
    Tau_util_output (&out, "<event id=\"%d\"><name>", i);
    char *name = functionUnifier->globalStrings[i];

    if (anonymize) { // fast character string operations to extract name. 
      if (name [0] == 'M' && name[1] == 'P' && name[2]== 'I' && name[3] == '_') {
	for (int j=0; j < strlen(name); j++) {
	  if (name[j] != ':') {
            anonymous_name[j] = name[j];
	  } else {
	    anonymous_name[j] = '\0';
	    break;
	  }
	}
	group = MPI_group;  // MPI
      } else { 
        snprintf(anonymous_name, sizeof(anonymous_name),  "FUNCTION_%d", i); 
        group = anonymous_group;
      }
      TAU_VERBOSE("writing: anonymous_name = %s\n", anonymous_name);
      Tau_XML_writeString(&out, anonymous_name);

    } else {
      group = strstr(name,":GROUP:");
      if (group == NULL) {
        fprintf (stderr, "TAU: Error extracting groups for %s!\n",name);
      } else {
        char *target = group;
        group+=strlen(":GROUP:");
        *target=0;
      }
      Tau_XML_writeString(&out, name);
    }

    Tau_util_output (&out, "</name><group>");
    Tau_XML_writeString(&out, group);
    Tau_util_output (&out, "</group></event>\n");
  }

  for (int i=0; i<atomicUnifier->globalNumItems; i++) {
    Tau_util_output (&out, "<userevent id=\"%d\"><name>", i);
    if (anonymize) {
      snprintf(anonymous_event_name, sizeof(anonymous_event_name),  "EVENT_%d", i); 
      Tau_XML_writeString(&out, anonymous_event_name);
    } else {
      Tau_XML_writeString(&out, atomicUnifier->globalStrings[i]);
    }
    Tau_util_output (&out, "</name></userevent>\n");
  }

  Tau_util_output (&out, "\n</definitions>\n");

  Tau_util_output (&out, "</profile_xml>\n");

}
#endif



FILE *Tau_create_merged_profile(const char *profiledir, const char *profile_prefix, const char *fname) {
  char filename[4096]; 
  FILE *f; 
  if (profile_prefix != NULL) {
    snprintf (filename, sizeof(filename), "%s/%s-%s", profiledir, profile_prefix, fname);
  } else {
    snprintf (filename, sizeof(filename), "%s/%s", profiledir, fname);
  }
  if ((f = fopen (filename, "w+")) == NULL) {
    char errormsg[4096];
    snprintf(errormsg, sizeof(errormsg),  "TAU Error: Could not create %s/%s-%s", profiledir, profile_prefix, fname);
    perror(errormsg);
  }
  return f; 
}

int Tau_mergeProfiles_MPI()
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  FILE *f;
  FILE *fa = (FILE *)NULL; /* for TAU_ANONYMIZE=1 */
  int anonymize = 0; /* don't anonymize event names by default */
#ifdef TAU_MPI
  MPI_Status status;
#endif  /* TAU_MPI */
  x_uint64 start, end;
  const char *profiledir = TauEnv_get_profiledir();
  const char *profile_prefix = TauEnv_get_profile_prefix();

  Tau_detect_memory_leaks();

#ifdef TAU_UNIFY
  Tau_unify_unifyDefinitions_MPI();

	for (int tid = 0; tid<RtsLayer::getTotalThreads(); tid++) {
		Tau_snapshot_writeUnifiedBuffer(tid);
	}
#else
  Tau_snapshot_writeToBuffer("merge");
#endif

  int rank = 0;
  int size = 1;

#ifdef TAU_MPI

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

#endif /* TAU_MPI */

	int buflen = Tau_snapshot_getBufferLength()+1;
  int maxBuflen = buflen;

#ifdef TAU_MPI

  PMPI_Reduce(&buflen, &maxBuflen, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	char * buf = (char *) malloc(buflen);
	Tau_snapshot_getBuffer(buf);

#endif  /* TAU_MPI */

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
    Tau_collate_get_total_threads_MPI(functionUnifier, &globalNumThreads, &numEventThreads,
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
    Tau_collate_compute_statistics_MPI(functionUnifier, globalEventMap, 
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
    Tau_collate_get_total_threads_MPI(atomicUnifier, &globalNumThreads, &numAtomicEventThreads,
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
    Tau_collate_compute_atomicStatistics_MPI(atomicUnifier, globalAtomicEventMap, 
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

    TAU_VERBOSE("Before Merging Profiles: Tau_check_dirname()\n");
    profiledir=Tau_check_dirname(profiledir);

    TAU_VERBOSE("TAU: Merging Profiles\n");
    start = TauMetrics_getTimeOfDay();
    anonymize = TauEnv_get_anonymize_enabled(); 

    f=Tau_create_merged_profile(profiledir, profile_prefix, "tauprofile.xml"); 
    if (f == (FILE *) NULL) {
      return -1; 
    }

    if (anonymize) {
      fa=Tau_create_merged_profile(profiledir, profile_prefix, "tau_anonymized_key.xml"); 
      if (fa == (FILE *) NULL) {
        return -1; 
      }
    }

#ifdef TAU_UNIFY
    Tau_profileMerge_writeDefinitions(globalEventMap, globalAtomicEventMap, f, anonymize);
    if (anonymize) {
      Tau_profileMerge_writeDefinitions(globalEventMap, globalAtomicEventMap, fa, false); 
      /* write the true names in the tau_anonymized_key file. anonymize = false */ 
    }
#endif

    for (int i=1; i<size; i++) {

#ifdef TAU_MPI
      /* send ok-to-go */
      PMPI_Send(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD);

      /* receive buffer length */
      PMPI_Recv(&buflen, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

      /* receive buffer */
      PMPI_Recv(recv_buf, buflen, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);

#endif  /* TAU_MPI */

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
    snprintf(tmpstr, sizeof(tmpstr),  "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU Profile Merge Time", tmpstr);
    if (TauEnv_get_stat_precompute() == 1) {
      TAU_METADATA("TAU_PRECOMPUTE", "on");
    } else {
      TAU_METADATA("TAU_PRECOMPUTE", "off");
    }
    if (TauEnv_get_summary_only()) { /* write only rank one metadata for summary profile */
			if (rank == 0) {
                TAU_VERBOSE("Tau Profile merge - rank = 0: write meta data block\n");
    		Tau_snapshot_writeMetaDataBlock();
			}
	  } else {
    	Tau_snapshot_writeMetaDataBlock();
		}

		buflen = Tau_snapshot_getBufferLength()+1;
		char * local_buf = (char *) malloc(buflen);
    Tau_snapshot_getBuffer(local_buf);
    fwrite (local_buf, buflen, 1, f);
    free(local_buf);

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
    if (anonymize) {
      fclose(fa);
    }
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

#ifdef TAU_MPI
	free(buf);
#endif /* TAU_MPI */

  return 0;
}

int Tau_mergeProfiles_SHMEM()
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  FILE *f;
  FILE *fa = (FILE *)NULL; /* for TAU_ANONYMIZE=1 */
  int anonymize = 0; /* don't anonymize event names by default */
  x_uint64 start, end;
  const char *profiledir = TauEnv_get_profiledir();
  const char *profile_prefix = TauEnv_get_profile_prefix();

  Tau_detect_memory_leaks();

#ifdef TAU_UNIFY
  Tau_unify_unifyDefinitions_SHMEM();

	for (int tid = 0; tid<RtsLayer::getTotalThreads(); tid++) {
		Tau_snapshot_writeUnifiedBuffer(tid);
	}
#else
  Tau_snapshot_writeToBuffer("merge");
#endif

  int rank = 0;
  int size = 1;

#ifdef TAU_SHMEM

#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  size = __real__num_pes();
  rank = __real__my_pe();
#else
  size = __real_shmem_n_pes();
  rank = __real_shmem_my_pe();
#endif /* SHMEM_1_1 || SHMEM_1_2 */

#endif /* TAU_SHMEM */

	int buflen = Tau_snapshot_getBufferLength()+1;
  int maxBuflen = buflen;

#ifdef TAU_SHMEM

  int * allBuflen = (int*)malloc(size*sizeof(int));

#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  int *shbuflen = (int*)__real_shmalloc(sizeof(int));
#else
  int *shbuflen = (int*)__real_shmem_malloc(sizeof(int));
#endif /* SHMEM_1_1 || SHMEM_1_2 */

  *shbuflen = buflen;
  __real_shmem_barrier_all();
  if (rank == 0) {
    allBuflen[0] = buflen;
    for (int i=1; i<size; ++i) {
      __real_shmem_int_get(allBuflen+i, shbuflen, 1, i);
      maxBuflen = max(maxBuflen, allBuflen[i]);
    }
    for (int i=1; i<size; ++i) {
      __real_shmem_int_put(shbuflen, &maxBuflen, 1, i);
    }
  }
  __real_shmem_barrier_all();
  maxBuflen = *shbuflen;

#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  __real_shfree(shbuflen);
  char *shbuf = (char*)__real_shmalloc(maxBuflen);
#else
  __real_shmem_free(shbuflen);
  char *shbuf = (char*)__real_shmem_malloc(maxBuflen);
#endif /* SHMEM_1_1 || SHMEM_1_2 */

	Tau_snapshot_getBuffer(shbuf);
  __real_shmem_barrier_all();

#endif  /* TAU_SHMEM */

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
    Tau_collate_get_total_threads_SHMEM(functionUnifier, &globalNumThreads, &numEventThreads,
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
    Tau_collate_compute_statistics_SHMEM(functionUnifier, globalEventMap, 
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
    Tau_collate_get_total_threads_SHMEM(atomicUnifier, &globalNumThreads, &numAtomicEventThreads,
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
    Tau_collate_compute_atomicStatistics_SHMEM(atomicUnifier, globalAtomicEventMap, 
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

    TAU_VERBOSE("Before Merging Profiles: Tau_check_dirname()\n");
    profiledir=Tau_check_dirname(profiledir);

    TAU_VERBOSE("TAU: Merging Profiles\n");
    start = TauMetrics_getTimeOfDay();

    anonymize = TauEnv_get_anonymize_enabled();

    f=Tau_create_merged_profile(profiledir, profile_prefix, "tauprofile.xml");
    if (f == (FILE *) NULL) {
      return -1;
    }

    if (anonymize) {
      fa=Tau_create_merged_profile(profiledir, profile_prefix, "tau_anonymized_key.xml");
      if (fa == (FILE *) NULL) {
        return -1;
      }
    }



#ifdef TAU_UNIFY
    Tau_profileMerge_writeDefinitions(globalEventMap, globalAtomicEventMap, f, anonymize);
    if (anonymize) {
      Tau_profileMerge_writeDefinitions(globalEventMap, globalAtomicEventMap, fa, false);
      /* write the true names in the tau_anonymized_key file. anonymize = false */
    }

#endif

    for (int i=1; i<size; i++) {

#ifdef TAU_SHMEM

      /* receive buffer */
      buflen = allBuflen[i];
      __real_shmem_getmem(recv_buf, shbuf, buflen, i);

#endif  /* TAU_SHMEM */

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
    snprintf(tmpstr, sizeof(tmpstr),  "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU Profile Merge Time", tmpstr);
    if (TauEnv_get_stat_precompute() == 1) {
      TAU_METADATA("TAU_PRECOMPUTE", "on");
    } else {
      TAU_METADATA("TAU_PRECOMPUTE", "off");
    }
    if (TauEnv_get_summary_only()) { /* write only rank one metadata for summary profile */
			if (rank == 0) {
    		Tau_snapshot_writeMetaDataBlock();
			}
	  } else {
    	Tau_snapshot_writeMetaDataBlock();
		}

		buflen = Tau_snapshot_getBufferLength()+1;
		char * local_buf = (char *) malloc(buflen);
    Tau_snapshot_getBuffer(local_buf);
    fwrite (local_buf, buflen, 1, f);
    free(local_buf);

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
  } 

#ifdef TAU_SHMEM
  __real_shmem_barrier_all();
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  __real_shfree(shbuf);
#else
  __real_shmem_free(shbuf);
#endif
#endif /* TAU_SHMEM */

  return 0;
}

