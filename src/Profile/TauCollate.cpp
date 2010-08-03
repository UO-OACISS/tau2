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

#include <stdarg.h>
#include <assert.h>

// #include <sstream>

//#define DEBUG

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


typedef enum {
  step_min,
  step_max,
  step_sum,
  step_sumsqr
} collate_step;

const int num_collate_steps = 4;
char *collate_step_name[4] = {"min", "max", "sum", "sumsqr"};

MPI_Op collate_op[4] = {MPI_MIN, MPI_MAX, MPI_SUM, MPI_SUM};


/*********************************************************************
 * getStepValue returns the incremental thread-combined value, similar 
 * to how MPI_Reduce will combine values across ranks, we need to do this for threads.
 ********************************************************************/
static double getStepValue(collate_step step, double prevValue, double nextValue) {
  if (step == step_sumsqr) {
    nextValue = nextValue * nextValue;
  }
  if (prevValue == -1) {
    return nextValue;
  }
  if (nextValue == -1) {
    return prevValue;
  }

  if (step == step_min) {
    if (nextValue < prevValue) {
      return nextValue;
    }
  } else if (step == step_max) {
    if (nextValue > prevValue) {
      return nextValue;
    }
  } else if (step == step_sum) {
    prevValue += nextValue;
  } else if (step == step_sumsqr) {
    prevValue += nextValue;
  }
  return prevValue;
}

/*********************************************************************
 * getStepValue returns the incremental thread-combined value, similar 
 * to how MPI_Reduce will combine values across ranks, we need to do this for threads.
 ********************************************************************/
static int getStepValue(collate_step step, int prevValue, int nextValue) {
  if (step == step_sumsqr) {
    nextValue = nextValue * nextValue;
  }
  if (prevValue == -1) {
    return nextValue;
  }
  if (nextValue == -1) {
    return prevValue;
  }

  if (step == step_min) {
    if (nextValue < prevValue) {
      return nextValue;
    }
  } else if (step == step_max) {
    if (nextValue > prevValue) {
      return nextValue;
    }
  } else if (step == step_sum) {
    prevValue += nextValue;
  } else if (step == step_sumsqr) {
    prevValue += nextValue;
  }
  return prevValue;
}



/*********************************************************************
 * An MPI_Reduce operator similar to MPI_MIN, but it allows for -1 values
 * to represent "non-existent"
 ********************************************************************/
static void stat_min (void *i, void *o, int *len,  MPI_Datatype *type) {
  if (*type == MPI_INT) {
    int *in = (int *) i;
    int *inout = (int *) o;
    for (int i=0; i<*len; i++) {
      if (inout[i] == -1) {
	inout[i] = in[i];
      } else if (in[i] != -1) {
	if (in[i] < inout[i]) {
	  inout[i] = in[i];
	}
      }
    }
  } else {
    double *in = (double *) i;
    double *inout = (double *) o;
    for (int i=0; i<*len; i++) {
      if (inout[i] == -1) {
	inout[i] = in[i];
      } else if (in[i] != -1) {
	if (in[i] < inout[i]) {
	  inout[i] = in[i];
	}
      }
    }
  }
}

/*********************************************************************
 * An MPI_Reduce operator similar to MPI_SUM, but it allows for -1 values
 * to represent "non-existent"
 ********************************************************************/
static void stat_sum (void *i, void *o, int *len,  MPI_Datatype *type) {
  if (*type == MPI_INT) {
    int *in = (int *) i;
    int *inout = (int *) o;
    for (int i=0; i<*len; i++) {
      if (inout[i] == -1) {
	inout[i] = in[i];
      } else if (in[i] != -1) {
	inout[i] += in[i];
      }
    }
  } else {
    double *in = (double *) i;
    double *inout = (double *) o;
    for (int i=0; i<*len; i++) {
      if (inout[i] == -1) {
	inout[i] = in[i];
      } else if (in[i] != -1) {
	inout[i] += in[i];
      }
    }
  }
}


static void Tau_collate_allocateBuffers(double ***excl, double ***incl, int **numCalls, int **numSubr, int numItems) {
  *excl = (double **) TAU_UTIL_MALLOC(sizeof(double *) * Tau_Global_numCounters);
  *incl = (double **) TAU_UTIL_MALLOC(sizeof(double *) * Tau_Global_numCounters);
  for (int m=0; m<Tau_Global_numCounters; m++) {
    (*excl)[m] = (double *) TAU_UTIL_CALLOC(sizeof(double) * numItems);
    (*incl)[m] = (double *) TAU_UTIL_CALLOC(sizeof(double) * numItems);
  }
  *numCalls = (int *) TAU_UTIL_CALLOC(sizeof(int) * numItems);
  *numSubr = (int *) TAU_UTIL_CALLOC(sizeof(int) * numItems);
}


static void Tau_collate_incrementHistogram(int *histogram, double min, double max, double value, int numBins) {
  double range = max-min;
  double binWidth = range / (numBins-1);
  int mybin = (int)((value - min) / binWidth);
  if (binWidth == 0) {
    mybin = 0;
  }
  
  if (mybin < 0 || mybin >= numBins) {
    TAU_ABORT("TAU: Error computing histogram, non-existent bin=%d\n", mybin);
  }

  histogram[mybin]++;
}




/*********************************************************************
 * Write a profile with data from all nodes/threads
 ********************************************************************/
extern "C" int Tau_collate_writeProfile() {
  static int invocationIndex = -1;
  invocationIndex++;

  int rank, size;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

  // timing info
  x_uint64 start, end;
  if (rank == 0) {
    TAU_VERBOSE("TAU: Collating...\n");
    start = TauMetrics_getTimeOfDay();
  }

  // create and assign specialized min and sum operators
  MPI_Op min_op, sum_op;
  PMPI_Op_create (stat_min, 1, &min_op);
  PMPI_Op_create (stat_sum, 1, &sum_op);
  collate_op[step_min] = min_op;
  collate_op[step_sum] = sum_op;
  collate_op[step_sumsqr] = sum_op;

  // Dump out all thread data with present values
  int numThreads = RtsLayer::getNumThreads();
  for (int tid = 0; tid<numThreads; tid++) {
    TauProfiler_updateIntermediateStatistics(tid);
  }

  x_uint64 start_unify, end_unify;
  if (rank == 0) {
    start_unify = TauMetrics_getTimeOfDay();
  }

  // Unify events
  FunctionEventLister *functionEventLister = new FunctionEventLister();
  Tau_unify_object_t *functionUnifier = Tau_unify_unifyEvents(functionEventLister);
  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  Tau_unify_object_t *atomicUnifier = Tau_unify_unifyEvents(atomicEventLister);

  TAU_MPI_DEBUG0 ("Found %d total regions\n", functionUnifier->globalNumItems);
  if (rank == 0) {
    end_unify = TauMetrics_getTimeOfDay();
  }

  x_uint64 start_aggregate, end_aggregate;

  if (rank == 0) {
    start_aggregate = TauMetrics_getTimeOfDay();
  }

  int globalNumThreads;
  PMPI_Reduce(&numThreads, &globalNumThreads, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // create a reverse mapping, not strictly necessary, but it makes things easier
  int *globalmap = (int*)TAU_UTIL_MALLOC(functionUnifier->globalNumItems * sizeof(int));
  for (int i=0; i<functionUnifier->globalNumItems; i++) { // initialize all to -1
    globalmap[i] = -1; // -1 indicates that the event did not occur for this rank
  }
  for (int i=0; i<functionUnifier->localNumItems; i++) {
    globalmap[functionUnifier->mapping[i]] = i; // set reserse mapping
  }

  // the global number of events
  int numItems = functionUnifier->globalNumItems;

  // allocate memory for values to compute/send
  double **excl, **incl;
  int *numCalls, *numSubr;
  Tau_collate_allocateBuffers(&excl, &incl, &numCalls, &numSubr, numItems);


  double ***gExcl, ***gIncl;
  int **gNumCalls, **gNumSubr;
  gExcl = (double ***) TAU_UTIL_MALLOC(sizeof(double **) * num_collate_steps);
  gIncl = (double ***) TAU_UTIL_MALLOC(sizeof(double **) * num_collate_steps);
  gNumCalls = (int **) TAU_UTIL_MALLOC(sizeof(int *) * num_collate_steps);
  gNumSubr = (int **) TAU_UTIL_MALLOC(sizeof(int *) * num_collate_steps);


  // we generate statistics in 4 steps
  for (int s=0; s < num_collate_steps; s++) {

    Tau_collate_allocateBuffers(&(gExcl[s]), &(gIncl[s]), &(gNumCalls[s]), &(gNumSubr[s]), numItems);

    // reset intermediate data
    for (int m=0; m<Tau_Global_numCounters; m++) {
      for (int i=0; i<numItems; i++) {
	excl[m][i] = -1;
	incl[m][i] = -1;
      }
    }
    for (int i=0; i<numItems; i++) {
      numCalls[i] = -1;
      numSubr[i] = -1;
    }

    for (int tid = 0; tid<numThreads; tid++) { // for each thread
      for (int i=0; i<numItems; i++) { // for each event
	if (globalmap[i] != -1) { // if it occurred in our rank
	  int local_index = functionUnifier->sortMap[globalmap[i]];
	  FunctionInfo *fi = TheFunctionDB()[local_index];
	  for (int m=0; m<Tau_Global_numCounters; m++) {
	    incl[m][i] = getStepValue((collate_step)s, incl[m][i], fi->getDumpInclusiveValues(tid)[m]);
	    excl[m][i] = getStepValue((collate_step)s, excl[m][i], fi->getDumpExclusiveValues(tid)[m]);
	  }
	  numCalls[i] = getStepValue((collate_step)s, numCalls[i], fi->GetCalls(tid));
	  numSubr[i] = getStepValue((collate_step)s, numSubr[i], fi->GetSubrs(tid));
	}
      }
    }

    // reduce data to rank 0
    for (int m=0; m<Tau_Global_numCounters; m++) {
      PMPI_Allreduce (excl[m], gExcl[s][m], numItems, MPI_DOUBLE, collate_op[s], MPI_COMM_WORLD);
      PMPI_Allreduce (incl[m], gIncl[s][m], numItems, MPI_DOUBLE, collate_op[s], MPI_COMM_WORLD);
    }
    PMPI_Allreduce (numCalls, gNumCalls[s], numItems, MPI_INT, collate_op[s], MPI_COMM_WORLD);
    PMPI_Allreduce (numSubr, gNumSubr[s], numItems, MPI_INT, collate_op[s], MPI_COMM_WORLD);

    // if (rank == 0) {
    //   fprintf (stderr, "\n----- data for statistic: %s\n", collate_step_name[s]);
    //   for (int i=0; i<numItems; i++) {
    // 	fprintf (stderr, "[id=%2d] incl=%9.16G excl=%9.16G numcalls=%9d numsubr=%9d : %s\n", i, 
    // 		gIncl[s][0][i], gExcl[s][0][i], gNumCalls[s][i], gNumSubr[s][i], functionUnifier->globalStrings[i]);

    //   }
    // }
  }

  if (rank == 0) {
    end_aggregate = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: Collate: Aggregation Complete, duration = %.4G seconds\n", ((double)((double)end_aggregate-start_aggregate))/1000000.0f);
  }

  // fflush(stderr);
  // PMPI_Barrier(MPI_COMM_WORLD);


  // now compute histograms
  x_uint64 start_hist, end_hist;

  int numBins = 20;

  const char *profiledir = TauEnv_get_profiledir();

  FILE *histoFile;
  char histFileNameTmp[512];
  char histFileName[512];
  if (rank == 0) {
    sprintf (histFileName, "%s/tau.histograms.%d", profiledir, 
	     invocationIndex);
    sprintf (histFileNameTmp, "%s/.temp.tau.histograms.%d", profiledir,
	     invocationIndex);
    histoFile = fopen(histFileNameTmp, "w");
    fprintf (histoFile, "%d\n", numItems);
    fprintf (histoFile, "%d\n", (Tau_Global_numCounters*2)+2);
    fprintf (histoFile, "%d\n", numBins);
    for (int m=0; m<Tau_Global_numCounters; m++) {
      fprintf (histoFile, "Exclusive %s\n", TauMetrics_getMetricName(m));
      fprintf (histoFile, "Inclusive %s\n", TauMetrics_getMetricName(m));
    }
    fprintf (histoFile, "Number of calls\n");
    fprintf (histoFile, "Child calls\n");
  }

  if (rank == 0) {
    // must not let file IO get in the way of a proper measure of this.
    start_hist = TauMetrics_getTimeOfDay();
  }

  int numHistoGrams = (Tau_Global_numCounters * 2) + 2; // two for each metric (excl, incl) and numCalls/numSubr;
  int histogramBufSize = sizeof(int) * numBins * numHistoGrams;
  int *histogram = (int *) TAU_UTIL_MALLOC(histogramBufSize);
  int *outHistogram = (int *) TAU_UTIL_MALLOC(histogramBufSize);
  for (int e=0; e<numItems; e++) { // for each event
    bzero (histogram, histogramBufSize);
    if (globalmap[e] != -1) { // if it occurred in our rank

      int local_index = functionUnifier->sortMap[globalmap[e]];
      FunctionInfo *fi = TheFunctionDB()[local_index];
      
      double min, max;
      for (int tid = 0; tid<numThreads; tid++) { // for each thread
  	for (int m=0; m<Tau_Global_numCounters; m++) {
	  Tau_collate_incrementHistogram(&(histogram[(m*2)*numBins]), 
					 gExcl[step_min][m][e], gExcl[step_max][m][e], fi->getDumpExclusiveValues(tid)[m], numBins);
	  Tau_collate_incrementHistogram(&(histogram[(m*2+1)*numBins]), 
					 gIncl[step_min][m][e], gIncl[step_max][m][e], fi->getDumpInclusiveValues(tid)[m], numBins);

	  
	  // // debugging the histogram
	  // if (e == 0) {
	  //   int *ptr = &(histogram[(m*2)*numBins]);

	  //   std::stringstream stream;
	  //   stream << "rank[" << rank << "] = ";

	  //   for (int j=0;j<numBins;j++) {
	  //     stream << ptr[j] << " ";
	  //   }

	  //   fprintf (stderr, "%s\n", stream.str().c_str());
	  // }
  	}
	Tau_collate_incrementHistogram(&(histogram[(Tau_Global_numCounters*2)*numBins]), 
				       gNumCalls[step_min][e], gNumCalls[step_max][e], fi->GetCalls(tid), numBins);
	Tau_collate_incrementHistogram(&(histogram[(Tau_Global_numCounters*2+1)*numBins]), 
				       gNumSubr[step_min][e], gNumSubr[step_max][e], fi->GetSubrs(tid), numBins);
      }
    }

    PMPI_Reduce (histogram, outHistogram, numBins * numHistoGrams, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      end_hist = TauMetrics_getTimeOfDay();
    }

    if (rank == 0) {
      fprintf (histoFile, "%s\n", functionUnifier->globalStrings[e]);

      for (int m=0; m<Tau_Global_numCounters; m++) {
	fprintf (histoFile, "%.16G %.16G ", gExcl[step_min][m][e], gExcl[step_max][m][e]);
	for (int j=0;j<numBins;j++) {
	  fprintf (histoFile, "%d ", outHistogram[(m*2)*numBins+j]);
	}
	fprintf (histoFile, "\n");
	fprintf (histoFile, "%.16G %.16G ", gIncl[step_min][m][e], gIncl[step_max][m][e]);
	for (int j=0;j<numBins;j++) {
	  fprintf (histoFile, "%d ", outHistogram[(m*2)*numBins+j]);
	}
	fprintf (histoFile, "\n");
      }
      
      fprintf (histoFile, "%.16G %.16G ", gNumCalls[step_min][e], gNumCalls[step_max][e]);
      for (int j=0;j<numBins;j++) {
	fprintf (histoFile, "%d ", outHistogram[(Tau_Global_numCounters*2)*numBins+j]);
      }
      fprintf (histoFile, "\n");
      
      fprintf (histoFile, "%.16G %.16G ", gNumSubr[step_min][e], gNumSubr[step_max][e]);
      for (int j=0;j<numBins;j++) {
	fprintf (histoFile, "%d ", outHistogram[(Tau_Global_numCounters*2+1)*numBins+j]);
      }
      fprintf (histoFile, "\n");
    }
  }

  if (rank == 0) {
    fclose (histoFile);
    rename (histFileNameTmp, histFileName);
    TAU_VERBOSE("TAU: Collate: Histogramming Complete, duration = %.4G seconds\n", ((double)((double)end_hist-start_hist))/1000000.0f);
  }

  if (rank == 0) {
    // using histogram output to approximate total output.
    end = TauMetrics_getTimeOfDay();
  }

  // *CWL* Delaying writing of the fake profile until after histograms are
  //   completed and written, so metadata can be tagged.
  if (rank == 0) {
    char profileName[512], profileNameTmp[512];
    char unifyMeta[512];
    char aggregateMeta[512];
    char histogramMeta[512];
    char monitoringMeta[512];
    sprintf (profileNameTmp, "%s/.temp.mean.%d.0.0", profiledir,
	     invocationIndex);
    sprintf (profileName, "%s/mean.%d.0.0", profiledir, invocationIndex);
    sprintf(unifyMeta,"<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Unification Time", ((double)((double)end_unify-start_unify))/1000000.0f);
    sprintf(aggregateMeta,"<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Mean Aggregation Time", ((double)((double)end_aggregate-start_aggregate))/1000000.0f);
    sprintf(histogramMeta,"<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Histogramming Time", ((double)((double)end_hist-start_hist))/1000000.0f);
    sprintf(monitoringMeta,"<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Total Monitoring Time", ((double)((double)end-start))/1000000.0f);
    FILE *profile = fopen(profileNameTmp, "w");
    fprintf (profile, "%d templated_functions_MULTI_TIME\n", numItems);
    fprintf (profile, "# Name Calls Subrs Excl Incl ProfileCalls % <metadata><attribute><name>TAU Monitoring Transport</name><value>MPI</value></attribute>%s%s%s%s</metadata>\n", 
	     unifyMeta, aggregateMeta, histogramMeta, monitoringMeta);
    for (int i=0; i<numItems; i++) {
      double exclusive = gExcl[step_sum][0][i] / globalNumThreads;
      double inclusive = gIncl[step_sum][0][i] / globalNumThreads;
      double numCalls = (double)gNumCalls[step_sum][i] / globalNumThreads;
      double numSubr = (double)gNumSubr[step_sum][i] / globalNumThreads;

      fprintf (profile, "\"%s\" %.16G %.16G %.16G %.16G 0 GROUP=\"TAU_DEFAULT\"\n", functionUnifier->globalStrings[i], 
	       numCalls, numSubr, exclusive, inclusive);
      
    }
    fprintf (profile, "0 aggregates\n");
    fclose (profile);
    rename (profileNameTmp, profileName);
  }

  if (rank == 0) {
    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: Collating Complete, duration = %.4G seconds\n", ((double)((double)end-start))/1000000.0f);
  }

  PMPI_Op_free(&min_op);
  PMPI_Op_free(&sum_op);


  // fflush(stderr);
  // PMPI_Barrier(MPI_COMM_WORLD);

  //temp: write regular profiles too, for comparison
  // int tid = 0;
  // TauProfiler_DumpData(false, tid, "compare");

  return 0;
}

/*********************************************************************
 * For Dagstuhl demo 2010
 ********************************************************************/
extern "C" void Tau_collate_onlineDump() {
  TAU_VERBOSE("collate online dump called\n");
  Tau_collate_writeProfile();
}

#endif /* TAU_EXP_UNIFY */
#endif /* TAU_MPI */
