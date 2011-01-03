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
// The subsequent guards are for existing dependencies. These may go away as we
//   expand TAUmon MPI capabilities.
#ifdef TAU_UNIFY

#include <mpi.h>
#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <tau_types.h>
#include <TauEnv.h>
#include <TauCollate.h>
#include <TauSnapshot.h>
#include <TauMetrics.h>
#include <TauUnify.h>
#include <TauUtil.h>
#include <float.h>

#include <math.h>

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

// Default reduction operations. These will be set appropriately later.
MPI_Op collate_op[NUM_COLLATE_STEPS] = {MPI_MIN, MPI_MAX, MPI_SUM, MPI_SUM};

static double calculateMean(int count, double sum) {
  double ret = 0.0;
  assert(count > 0);
  assert(sum >= 0.0);
  return sum/count;
}

static double calculateStdDev(int count, double sumsqr, double mean) {
  double ret = 0.0;
  assert(count > 0);
  assert(sumsqr >= 0.0);
  assert(mean >= 0.0);
  ret = (sumsqr/count) - (mean*mean);
  //  printf("%.16G %.16G\n", sumsqr, mean*mean);
  assert(ret >= 0.0);
  return sqrt(ret);
}

static void assignDerivedStats(double ****eventType, double ****gEventType,
			       int m, int i,
			       int globalNumThreads, int *numEventThreads) {
  (*eventType)[stat_mean_all][m][i] =
    calculateMean(globalNumThreads,(*gEventType)[step_sum][m][i]);
  (*eventType)[stat_mean_exist][m][i] =
    calculateMean(numEventThreads[i],(*gEventType)[step_sum][m][i]);
  (*eventType)[stat_stddev_all][m][i] =
    calculateStdDev(globalNumThreads,(*gEventType)[step_sumsqr][m][i],
		    (*eventType)[stat_mean_all][m][i]);
  (*eventType)[stat_stddev_exist][m][i] =
    calculateStdDev(numEventThreads[i],(*gEventType)[step_sumsqr][m][i],
		    (*eventType)[stat_mean_exist][m][i]);
}

static void assignDerivedStats(double ***eventType, double ***gEventType,
			       int i,
			       int globalNumThreads, int *numEventThreads) {
  (*eventType)[stat_mean_all][i] = 
    calculateMean(globalNumThreads,(*gEventType)[step_sum][i]);
  (*eventType)[stat_mean_exist][i] = 
    calculateMean(numEventThreads[i],(*gEventType)[step_sum][i]);
  (*eventType)[stat_stddev_all][i] = 
    calculateStdDev(globalNumThreads,
		    (*gEventType)[step_sumsqr][i],
		    (*eventType)[stat_mean_all][i]);
  (*eventType)[stat_stddev_exist][i] = 
    calculateStdDev(numEventThreads[i],
		    (*gEventType)[step_sumsqr][i],
		    (*eventType)[stat_mean_exist][i]);
}

/*********************************************************************
 * getStepValue returns the incremental thread-combined value, similar 
 * to how MPI_Reduce will combine values across ranks, we need to do this 
 * for threads.
 * Precondition: The initial values are assigned with the appropriate
 *               values for reduction purposes. This means -1 for 
 *               step_min and 0 for everything else.
 ********************************************************************/
static double getStepValue(collate_step step, double prevValue, 
			   double nextValue) {
  double ret = prevValue;
  switch (step) {
  case step_sum: {
    //    printf("next sum dbl: %.16G\n", nextValue);
    ret = prevValue + nextValue;
    assert(ret >= 0.0);
    break;
  }
  case step_sumsqr: {
    //    printf("next sumsqr dbl: %.16G\n", nextValue);
    ret = prevValue + (nextValue * nextValue);
    assert(ret >= 0.0);
    break;
  }
  case step_max: {
    ret = (nextValue > prevValue)?nextValue:prevValue;
    break;
  }
  case step_min: {
    if (nextValue < 0) {
      ret = prevValue;
    } else if (prevValue < 0) {
      ret = nextValue;
    } else {
      ret = (nextValue < prevValue)?nextValue:prevValue;
    }
    break;
  }
  }
  return ret;
}

/*********************************************************************
 * getStepValue returns the incremental thread-combined value, similar 
 * to how MPI_Reduce will combine values across ranks, we need to do this for threads.
 ********************************************************************/
static int getStepValue(collate_step step, int prevValue, int nextValue) {
  int ret = prevValue;
  switch (step) {
  case step_sum: {
    //    printf("next sum int: %d\n", nextValue);
    ret = prevValue + nextValue;
    assert(ret >= 0);
    break;
  }
  case step_sumsqr: {
    //    printf("next sumsqr int: %d\n", nextValue);
    ret = prevValue + (nextValue * nextValue);
    assert(ret >= 0);
    break;
  }
  case step_max: {
    ret = (nextValue > prevValue)?nextValue:prevValue;
    break;
  }
  case step_min: {
    if (nextValue < 0) {
      ret = prevValue;
    } else if (prevValue < 0) {
      ret = nextValue;
    } else {
      ret = (nextValue < prevValue)?nextValue:prevValue;
    }
    break;
  }
  }
  return ret;
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

void Tau_collate_allocateBuffers(double ***excl, double ***incl, int **numCalls, int **numSubr, int numItems) {
  *excl = (double **) TAU_UTIL_MALLOC(sizeof(double *) * Tau_Global_numCounters);
  *incl = (double **) TAU_UTIL_MALLOC(sizeof(double *) * Tau_Global_numCounters);
  // Please note the use of Calloc
  for (int m=0; m<Tau_Global_numCounters; m++) {
    (*excl)[m] = (double *) TAU_UTIL_CALLOC(sizeof(double) * numItems);
    (*incl)[m] = (double *) TAU_UTIL_CALLOC(sizeof(double) * numItems);
  }
  *numCalls = (int *) TAU_UTIL_CALLOC(sizeof(int) * numItems);
  *numSubr = (int *) TAU_UTIL_CALLOC(sizeof(int) * numItems);
}

void Tau_collate_allocateBuffers(double ***excl, double ***incl, 
				 double **numCalls, double **numSubr, 
				 int numItems) {
  *excl = (double **)TAU_UTIL_MALLOC(sizeof(double *)*Tau_Global_numCounters);
  *incl = (double **)TAU_UTIL_MALLOC(sizeof(double *)*Tau_Global_numCounters);
  // Please note the use of Calloc
  for (int m=0; m<Tau_Global_numCounters; m++) {
    (*excl)[m] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numItems);
    (*incl)[m] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numItems);
  }
  *numCalls = (double *)TAU_UTIL_CALLOC(sizeof(double)*numItems);
  *numSubr = (double *)TAU_UTIL_CALLOC(sizeof(double)*numItems);
}

void Tau_collate_freeBuffers(double ***excl, double ***incl, 
			     int **numCalls, int **numSubr) {
  free(*numCalls);
  free(*numSubr);
  for (int m=0; m<Tau_Global_numCounters; m++) {
    free((*excl)[m]);
    free((*incl)[m]);
  }
  free(*excl);
  free(*incl);
}

void Tau_collate_freeBuffers(double ***excl, double ***incl, 
			     double **numCalls, double **numSubr) {
  free(*numCalls);
  free(*numSubr);
  for (int m=0; m<Tau_Global_numCounters; m++) {
    free((*excl)[m]);
    free((*incl)[m]);
  }
  free(*excl);
  free(*incl);
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

/***
 *  2010-10-08 *CWL*
 *  Modularization of monitoring functionality for other non-TauMon
 *    purposes.
 */
void Tau_collate_compute_statistics(Tau_unify_object_t *functionUnifier,
				    int *globalEventMap, int numItems,
				    int globalNumThreads, int *numEventThreads,
				    double ****gExcl, double ****gIncl,
				    double ***gNumCalls, double ***gNumSubr,
				    double ****sExcl, double ****sIncl,
				    double ***sNumCalls, double ***sNumSubr) {
  int rank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // *CWL* - Minimum needs to be handled with out-of-band values for now.
  MPI_Op min_op;
  PMPI_Op_create (stat_min, 1, &min_op);
  collate_op[step_min] = min_op;

  // allocate memory for values to fill with performance data and sent to
  //   the root node
  double **excl, **incl;
  double *numCalls, *numSubr;
  Tau_collate_allocateBuffers(&excl, &incl, &numCalls, &numSubr, numItems);
  // Fill the data, once for each basic statistic
  for (int s=0; s<NUM_COLLATE_STEPS; s++) {
    // Initialize to -1 only for step_min to handle unrepresented values for
    //   minimum.
    double fillDbl = 0.0;
    if (s == step_min) {
      fillDbl = -1.0;
    }
    for (int i=0; i<numItems; i++) {
      for (int m=0; m<Tau_Global_numCounters; m++) {
	incl[m][i] = fillDbl;
	excl[m][i] = fillDbl;
      }
      numCalls[i] = fillDbl;
      numSubr[i] = fillDbl;
    }
    Tau_collate_allocateBuffers(&((*gExcl)[s]), &((*gIncl)[s]), 
				&((*gNumCalls)[s]), &((*gNumSubr)[s]), 
				numItems);

    for (int i=0; i<numItems; i++) { // for each event
      if (globalEventMap[i] != -1) { // if it occurred in our rank
	int local_index = functionUnifier->sortMap[globalEventMap[i]];
	FunctionInfo *fi = TheFunctionDB()[local_index];
	int numThreads = RtsLayer::getNumThreads();
	for (int tid = 0; tid<numThreads; tid++) { // for each thread
	  for (int m=0; m<Tau_Global_numCounters; m++) {
	    incl[m][i] = getStepValue((collate_step)s, incl[m][i],
				      fi->getDumpInclusiveValues(tid)[m]);
	    excl[m][i] = getStepValue((collate_step)s, excl[m][i],
				      fi->getDumpExclusiveValues(tid)[m]);
	  }
	  numCalls[i] = getStepValue((collate_step)s, numCalls[i],
				     (double)fi->GetCalls(tid));
	  numSubr[i] = getStepValue((collate_step)s, numSubr[i],
				    (double)fi->GetSubrs(tid));
	}
      }
    }
    
    // reduce data to rank 0
    for (int m=0; m<Tau_Global_numCounters; m++) {
      PMPI_Reduce(excl[m], (*gExcl)[s][m], numItems, MPI_DOUBLE, 
		  collate_op[s], 0, MPI_COMM_WORLD);
      PMPI_Reduce(incl[m], (*gIncl)[s][m], numItems, MPI_DOUBLE, 
		  collate_op[s], 0, MPI_COMM_WORLD);
    }
    PMPI_Reduce(numCalls, (*gNumCalls)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, MPI_COMM_WORLD);
    PMPI_Reduce(numSubr, (*gNumSubr)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, MPI_COMM_WORLD);
  }
  // Free allocated memory for basic info.
  Tau_collate_freeBuffers(&excl, &incl, &numCalls, &numSubr);

  // Now compute the actual statistics on rank 0 only. The assumption
  //   is that at least one thread would be active across the node-space
  //   and so negative values should never show up after reduction.
  if (rank == 0) {
    for (int s=0; s<NUM_STAT_TYPES; s++) {
      Tau_collate_allocateBuffers(&((*sExcl)[s]), &((*sIncl)[s]), 
				  &((*sNumCalls)[s]), &((*sNumSubr)[s]), 
				  numItems);
    }
    // *CWL* TODO - abstract the operations to avoid this nasty coding
    //     of individual operations.
    for (int i=0; i<numItems; i++) { // for each event
      for (int m=0; m<Tau_Global_numCounters; m++) {
	assignDerivedStats(sIncl, gIncl, m, i,
			   globalNumThreads, numEventThreads);
	assignDerivedStats(sExcl, gExcl, m, i,
			   globalNumThreads, numEventThreads);
      }
      assignDerivedStats(sNumCalls, gNumCalls, i,
			 globalNumThreads, numEventThreads);
      assignDerivedStats(sNumSubr, gNumSubr, i,
			 globalNumThreads, numEventThreads);
    }    
  }

  PMPI_Op_free(&min_op);
}

void Tau_collate_get_total_threads(int *globalNumThreads, 
				   int **numEventThreads,
				   int numEvents, int *globalEventMap) {
  int rank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int *numThreadsGlobal = (int *)TAU_UTIL_MALLOC(sizeof(int)*(numEvents+1));
  int *numThreadsLocal = (int *)TAU_UTIL_MALLOC(sizeof(int)*(numEvents+1));
  
  int numThreads = RtsLayer::getNumThreads();
  for (int i=0; i<numEvents; i++) {
    numThreadsLocal[i] = numThreads;
    if (globalEventMap[i] == -1) {
      numThreadsLocal[i] = 0;
    }
  }
  numThreadsLocal[numEvents] = numThreads;
  PMPI_Reduce(numThreadsLocal, numThreadsGlobal, numEvents+1, 
	      MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int i=0; i<numEvents; i++) {
      (*numEventThreads)[i] = numThreadsGlobal[i];
    }
    *globalNumThreads = numThreadsGlobal[numEvents];
  }
}

/* Only enable profile writing and dump operation if TAU MPI Monitoring
   is specified */
#ifdef TAU_EXP_COLLATE

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
    TAU_VERBOSE("TAU: Starting Mon MPI operations ...\n");
    start = TauMetrics_getTimeOfDay();
  }

  // create and assign specialized min and sum operators
  /*
  MPI_Op min_op, sum_op;
  PMPI_Op_create (stat_min, 1, &min_op);
  PMPI_Op_create (stat_sum, 1, &sum_op);
  collate_op[step_min] = min_op;
  collate_op[step_sum] = sum_op;
  collate_op[step_sumsqr] = sum_op;
  */

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

  // the global number of events
  int numItems = functionUnifier->globalNumItems;
  int globalNumThreads;
  int *numEventThreads = (int *)TAU_UTIL_MALLOC(sizeof(int)*numItems);

  // create a reverse mapping, not strictly necessary, but it makes things easier
  int *globalEventMap = (int*)TAU_UTIL_MALLOC(numItems*sizeof(int));
  // initialize all to -1
  for (int i=0; i<numItems; i++) {
    // -1 indicates that the event did not occur for this rank
    globalEventMap[i] = -1; 
  }
  for (int i=0; i<functionUnifier->localNumItems; i++) {
    globalEventMap[functionUnifier->mapping[i]] = i; // set reverse mapping
  }
  Tau_collate_get_total_threads(&globalNumThreads, &numEventThreads,
				numItems, globalEventMap);

  double ***gExcl, ***gIncl;
  double **gNumCalls, **gNumSubr;
  gExcl = (double ***) TAU_UTIL_MALLOC(sizeof(double **) * NUM_COLLATE_STEPS);
  gIncl = (double ***) TAU_UTIL_MALLOC(sizeof(double **) * NUM_COLLATE_STEPS);
  gNumCalls = (double **) TAU_UTIL_MALLOC(sizeof(double *) *NUM_COLLATE_STEPS);
  gNumSubr = (double **) TAU_UTIL_MALLOC(sizeof(double *) * NUM_COLLATE_STEPS);

  double ***sExcl, ***sIncl;
  double **sNumCalls, **sNumSubr;
  sExcl = (double ***)TAU_UTIL_MALLOC(sizeof(double **)*NUM_STAT_TYPES);
  sIncl = (double ***)TAU_UTIL_MALLOC(sizeof(double **)*NUM_STAT_TYPES);
  sNumCalls = (double **)TAU_UTIL_MALLOC(sizeof(double *)*NUM_STAT_TYPES);
  sNumSubr = (double **)TAU_UTIL_MALLOC(sizeof(double *)*NUM_STAT_TYPES);

  Tau_collate_compute_statistics(functionUnifier, globalEventMap, numItems, 
				 globalNumThreads, numEventThreads,
				 &gExcl, &gIncl, &gNumCalls, &gNumSubr,
				 &sExcl, &sIncl, &sNumCalls, &sNumSubr);
				 
  if (rank == 0) {
    end_aggregate = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: Mon MPI: Aggregation Complete, duration = %.4G seconds\n", ((double)((double)end_aggregate-start_aggregate))/1000000.0f);
  }

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
    if (globalEventMap[e] != -1) { // if it occurred in our rank

      int local_index = functionUnifier->sortMap[globalEventMap[e]];
      FunctionInfo *fi = TheFunctionDB()[local_index];
      
      double min, max;
      for (int tid = 0; tid<numThreads; tid++) { // for each thread
  	for (int m=0; m<Tau_Global_numCounters; m++) {
	  Tau_collate_incrementHistogram(&(histogram[(m*2)*numBins]), 
					 gExcl[step_min][m][e], gExcl[step_max][m][e], fi->getDumpExclusiveValues(tid)[m], numBins);
	  Tau_collate_incrementHistogram(&(histogram[(m*2+1)*numBins]), 
					 gIncl[step_min][m][e], gIncl[step_max][m][e], fi->getDumpInclusiveValues(tid)[m], numBins);
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
    TAU_VERBOSE("TAU: Mon MPI: Histogramming Complete, duration = %.4G seconds\n", ((double)((double)end_hist-start_hist))/1000000.0f);
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
    // *CWL* - templated_functions_MULTI_<metric name> should be the
    //         general output format. This should be added in subsequent
    //         revisions of the TauMon infrastructure.
    fprintf (profile, "%d templated_functions_MULTI_TIME\n", numItems);
    fprintf (profile, "# Name Calls Subrs Excl Incl ProfileCalls % <metadata><attribute><name>TAU Monitoring Transport</name><value>MPI</value></attribute>%s%s%s%s</metadata>\n", 
	     unifyMeta, aggregateMeta, histogramMeta, monitoringMeta);
    for (int i=0; i<numItems; i++) {
      /*
      printf("numthreads = %d\n", globalNumThreads);
      printf("Write: excl value = %.16G\n", gExcl[step_sum][0][i]);
      double exclusive = gExcl[step_sum][0][i] / globalNumThreads;
      double inclusive = gIncl[step_sum][0][i] / globalNumThreads;
      double numCalls = (double)gNumCalls[step_sum][i] / globalNumThreads;
      double numSubr = (double)gNumSubr[step_sum][i] / globalNumThreads;

      fprintf (profile, "\"%s\" %.16G %.16G %.16G %.16G 0 GROUP=\"TAU_OLD\"\n", functionUnifier->globalStrings[i], 
	       numCalls, numSubr, exclusive, inclusive);
      */
      /* The new */
      fprintf(profile, "\"%s\" %.16G %.16G %.16G %.16G 0 GROUP=\"TAU_DEFAULT\"\n",
	      functionUnifier->globalStrings[i], 
	      sNumCalls[stat_mean_all][i], 
	      sNumSubr[stat_mean_all][i], 
	      sExcl[stat_mean_all][0][i], 
	      sIncl[stat_mean_all][0][i]);
      /* Participants only - not used here.
      fprintf(profile, "\"%s\" %.16G %.16G %.16G %.16G 0 GROUP=\"TAU_FAKE\"\n",
	      functionUnifier->globalStrings[i], 
	      sNumCalls[stat_mean_exist][i], 
	      sNumSubr[stat_mean_exist][i], 
	      sExcl[stat_mean_exist][0][i], 
	      sIncl[stat_mean_exist][0][i]);
      */
    }
    fprintf (profile, "0 aggregates\n");
    fclose (profile);
    rename (profileNameTmp, profileName);
  }

  if (rank == 0) {
    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: Mon MPI: Operations complete, duration = %.4G seconds\n", ((double)((double)end-start))/1000000.0f);
  }

  /*
  PMPI_Op_free(&min_op);
  PMPI_Op_free(&sum_op);
  */

  return 0;
}

/*********************************************************************
 * For Dagstuhl demo 2010
 ********************************************************************/
extern "C" void Tau_collate_onlineDump() {
  // Not scalable output, even for verbose output. This is not a one-time
  //    operation.
  //  TAU_VERBOSE("collate online dump called\n");
  Tau_collate_writeProfile();
}

#endif /* TAU_EXP_COLLATE */

#endif /* TAU_UNIFY */
#endif /* TAU_MPI */
