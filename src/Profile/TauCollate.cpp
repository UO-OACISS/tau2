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

// The subsequent guards are for existing dependencies. These may go away as we
//   expand TAUmon MPI capabilities.
#ifdef TAU_UNIFY

#ifdef TAU_SHMEM
#include <shmem.h>
extern "C" void  __real_shmem_int_put(int * a1, const int * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_int_get(int * a1, const int * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_double_put(double * a1, const double * a2, size_t a3, int a4);
extern "C" void  __real_shmem_putmem(void * a1, const void * a2, size_t a3, int a4) ;
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
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */
#ifdef TAU_MPI
#include <mpi.h>
#else
// define some MPI things with dummy values, it makes it easier later.
#define MPI_Op int
#define MPI_MIN 0 
#define MPI_MAX 1
#define MPI_SUM 2
#endif /* TAU_MPI */
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
#include <strings.h>
#include <stdarg.h>

#define NDEBUG  // Disable to enable assertions
#include <assert.h>

using namespace std;
using namespace tau;

const int collate_num_op_items[NUM_COLLATE_OP_TYPES] =
  { NUM_COLLATE_STEPS, NUM_STAT_TYPES };
const char * collate_step_names[NUM_COLLATE_STEPS] =
  { "min", "max", "sum", "sum_of_squares" };
const char * stat_names[NUM_STAT_TYPES] =
  { "mean_all", "mean_no_null", "stddev_all", "stddev_no_null", "min_no_null", "max_no_null" };
const char** collate_op_names[NUM_COLLATE_OP_TYPES] = 
  { collate_step_names, stat_names };

//#define DEBUG
#ifdef DEBUG

void TAU_MPI_DEBUG0(const char *format, ...) {
  int rank = 0;
#ifdef TAU_MPI
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* TAU_MPI */
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
MPI_Op collate_op[NUM_COLLATE_STEPS] = { MPI_MIN, MPI_MAX, MPI_SUM, MPI_SUM };

static double calculateMean(int count, double sum) {
  if (count <= 0) return 0.0;
  assert(count >= 0);
  assert(sum >= 0.0);
  return sum/count;
}

static double calculateStdDev(int count, double sumsqr, double mean) {
  double ret = 0.0;
  //printf("Collate calculateStdDev count [%d] sumsqr [%.16G] meansqr [%.16G]\n", count, sumsqr, mean*mean);
  if (count <= 0) return 0.0;
  /*
  assert(count > 0);
  assert(sumsqr >= 0.0);
  assert(mean >= 0.0);
  */
  ret = (sumsqr/count) - (mean*mean);
  // assert(ret >= 0.0);
  return sqrt(fabs(ret));
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
  (*eventType)[stat_min_exist][m][i] =(*gEventType)[step_min][m][i];
  (*eventType)[stat_max_exist][m][i] =(*gEventType)[step_max][m][i];
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
  (*eventType)[stat_min_exist][i] =(*gEventType)[step_min][i];
  (*eventType)[stat_max_exist][i] =(*gEventType)[step_max][i];
}

/*********************************************************************
 * getStepValue returns the incremental thread-combined value, similar 
 * to how MPI_Reduce will combine values across ranks, we need to do this 
 * for threads.
 * Precondition: The initial values are assigned with the appropriate
 *               values for reduction purposes. This means -1 for 
 *               step_min and 0 for everything else.
 ********************************************************************/
static double getStepValue(collate_step step, double prevValue, double nextValue) {
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
    if (nextValue <= 0) {
      ret = prevValue;
    } else if (prevValue <= 0) {
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
/*
 // INTEGER CASE NEVER USED?
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
*/ 

/*********************************************************************
 * An MPI_Reduce operator similar to MPI_MIN, but it allows for -1 values
 * to represent "non-existent"
 ********************************************************************/
#ifdef TAU_MPI
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
#endif /* TAU_MPI */

void Tau_collate_allocateFunctionBuffers(double ****excl, double ****incl,
					 double ***numCalls, double ***numSubr,
					 int numEvents,
					 int numMetrics,
					 int collateOpType) {
  *excl = (double ***)TAU_UTIL_MALLOC(sizeof(double **)*
				      collate_num_op_items[collateOpType]);
  *incl = (double ***)TAU_UTIL_MALLOC(sizeof(double **)*
				      collate_num_op_items[collateOpType]);
  *numCalls = (double **)TAU_UTIL_MALLOC(sizeof(double *)*
					 collate_num_op_items[collateOpType]);
  *numSubr = (double **)TAU_UTIL_MALLOC(sizeof(double *)*
					collate_num_op_items[collateOpType]);
  for (int s=0; s<collate_num_op_items[collateOpType]; s++) {
    Tau_collate_allocateUnitFunctionBuffer(&((*excl)[s]), &((*incl)[s]),
					   &((*numCalls)[s]), &((*numSubr)[s]),
					   numEvents, numMetrics);
  }
}

void Tau_collate_allocateAtomicBuffers(double ***atomicMin, 
				       double ***atomicMax,
				       double ***atomicCalls, 
				       double ***atomicMean,
				       double ***atomicSumSqr,
				       int numEvents,
				       int collateOpType) {
  *atomicMin = 
    (double **)TAU_UTIL_MALLOC(sizeof(double *)*
			       collate_num_op_items[collateOpType]);
  *atomicMax = 
    (double **)TAU_UTIL_MALLOC(sizeof(double *)*
			       collate_num_op_items[collateOpType]);
  *atomicCalls = 
    (double **)TAU_UTIL_MALLOC(sizeof(double *)*
			       collate_num_op_items[collateOpType]);
  *atomicMean = 
    (double **)TAU_UTIL_MALLOC(sizeof(double *)*
			       collate_num_op_items[collateOpType]);
  *atomicSumSqr = 
    (double **)TAU_UTIL_MALLOC(sizeof(double *)*
			       collate_num_op_items[collateOpType]);
  
  for (int s=0; s<collate_num_op_items[collateOpType]; s++) {
    (*atomicMin)[s] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
    (*atomicMax)[s] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
    (*atomicCalls)[s] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
    (*atomicMean)[s] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
    (*atomicSumSqr)[s] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
  }
}

void Tau_collate_allocateUnitFunctionBuffer(double ***excl, double ***incl, 
					    double **numCalls, 
					    double **numSubr, 
					    int numEvents, int numMetrics) {
  *excl = (double **)TAU_UTIL_MALLOC(sizeof(double *)*numMetrics);
  *incl = (double **)TAU_UTIL_MALLOC(sizeof(double *)*numMetrics);
  // Please note the use of Calloc
  for (int m=0; m<numMetrics; m++) {
    (*excl)[m] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
    (*incl)[m] = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
  }
  *numCalls = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
  *numSubr = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
}

void Tau_collate_allocateUnitAtomicBuffer(double **atomicMin, 
					  double **atomicMax,
					  double **atomicCalls, 
					  double **atomicMean,
					  double **atomicSumSqr,
					  int numEvents) {
  *atomicMin = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
  *atomicMax = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
  *atomicCalls = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
  *atomicMean = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
  *atomicSumSqr = (double *)TAU_UTIL_CALLOC(sizeof(double)*numEvents);
}

void Tau_collate_freeFunctionBuffers(double ****excl, double ****incl,
				     double ***numCalls, double ***numSubr,
				     int numMetrics,
				     int collateOpType) {
  for (int s=0; s<collate_num_op_items[collateOpType]; s++) {
    Tau_collate_freeUnitFunctionBuffer(&((*excl)[s]), &((*incl)[s]),
				       &((*numCalls)[s]), &((*numSubr)[s]),
				       numMetrics);
  }
  free(*numCalls);
  free(*numSubr);
  free(*excl);
  free(*incl);
}

void Tau_collate_freeAtomicBuffers(double ***atomicMin, double ***atomicMax,
				   double ***atomicCalls, double ***atomicMean,
				   double ***atomicSumSqr,
				   int collateOpType) {
  for (int s=0; s<collate_num_op_items[collateOpType]; s++) {
    Tau_collate_freeUnitAtomicBuffer(&((*atomicMin)[s]), &((*atomicMax)[s]),
				     &((*atomicCalls)[s]), &((*atomicMean)[s]),
				     &((*atomicSumSqr)[s]));
  }
  free(*atomicMin);
  free(*atomicMax);
  free(*atomicCalls);
  free(*atomicMean);
  free(*atomicSumSqr);
}

void Tau_collate_freeUnitFunctionBuffer(double ***excl, double ***incl, 
					double **numCalls, double **numSubr,
					int numMetrics) {
  free(*numCalls);
  free(*numSubr);
  for (int m=0; m<numMetrics; m++) {
    free((*excl)[m]);
    free((*incl)[m]);
  }
  free(*excl);
  free(*incl);
}

void Tau_collate_freeUnitAtomicBuffer(double **atomicMin, double **atomicMax,
				      double **atomicCalls, 
				      double **atomicMean,
				      double **atomicSumSqr) {
  free(*atomicMin);
  free(*atomicMax);
  free(*atomicCalls);
  free(*atomicMean);
  free(*atomicSumSqr);
}


int Tau_collate_get_local_threads(int id, bool isAtomic){
    int numThreadsLocal=0;
    int numThreads = RtsLayer::getTotalThreads();
    if(isAtomic){
        TauUserEvent *userEvent = TheEventDB()[id];
		for (int t=0; t<numThreads; t++)
			{
				if (userEvent->GetNumEvents(t) > 0)
				{
					numThreadsLocal += 1;
				}
			}
    } else {/*It is a function*/
        FunctionInfo *fi = TheFunctionDB()[id];
		for (int t=0; t<numThreads; t++)
		{
			if (fi->GetCalls(t) > 0)
			{
				numThreadsLocal += 1;
			}
		}
    }
    return numThreadsLocal;
}


/* Parallel operation to acquire total number of threads for each event */
void Tau_collate_get_total_threads_MPI(Tau_unify_object_t *functionUnifier, int *globalNumThreads, 
				   int **numEventThreads,
				   int numEvents, int *globalEventMap,bool isAtomic) {
  int rank = 0;
#ifdef TAU_MPI
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* TAU_MPI */
  
  int *numThreadsLocal = (int *)TAU_UTIL_MALLOC(sizeof(int)*(numEvents+1));
#ifdef TAU_MPI
  int *numThreadsGlobal = (int *)TAU_UTIL_MALLOC(sizeof(int)*(numEvents+1));
#else
  // in the non-MPI case, just point to the same memory.
  int *numThreadsGlobal = numThreadsLocal;
#endif /* TAU_MPI */

  
  

  /* For each event, determine contributing threads */
  for (int i=0; i<numEvents; i++) {
		numThreadsLocal[i] = 0;
	}
	for (int i=0; i<numEvents; i++)
	{
	 int local_index = functionUnifier->sortMap[globalEventMap[i]];
/*   if (globalEventMap[i] != -1) { // if it occurred in our rank
	  FunctionInfo *fi = TheFunctionDB()[local_index];
			for (int t=0; t<numThreads; t++)
			{
				if (fi->GetCalls(t) > 0)
				{
					numThreadsLocal[i] += 1;
				}
			}
			numThreadsLocal[i], fi->GetName());
		}*/
		if(globalEventMap[i]!=-1){
		numThreadsLocal[i]=Tau_collate_get_local_threads(local_index,isAtomic);
		}
		else
		{	
			numThreadsLocal[i] = 0;
		}
	}
  /* Extra slot in array indicates number of threads on rank */
  numThreadsLocal[numEvents] = RtsLayer::getTotalThreads();
#ifdef TAU_MPI
  PMPI_Reduce(numThreadsLocal, numThreadsGlobal, numEvents+1, 
	      MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */

  /* Now rank 0 knows all about global thread counts */
  if (rank == 0) {
    for (int i=0; i<numEvents; i++) {
      (*numEventThreads)[i] = numThreadsGlobal[i];
    }
    *globalNumThreads = numThreadsGlobal[numEvents];
  }
}
void Tau_collate_get_total_threads_SHMEM(Tau_unify_object_t *functionUnifier, int *globalNumThreads, 
				   int **numEventThreads,
				   int numEvents, int *globalEventMap,bool isAtomic) {
  int rank = 0;
#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  rank = __real__my_pe();
#else
  rank = __real_shmem_my_pe();
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif
  
  int *numThreadsLocal = (int *)TAU_UTIL_MALLOC(sizeof(int)*(numEvents+1));
#ifdef TAU_SHMEM
  int *numThreadsGlobal = (int *)TAU_UTIL_MALLOC(sizeof(int)*(numEvents+1));
#else
  // in the non-MPI case, just point to the same memory.
  int *numThreadsGlobal = numThreadsLocal;
#endif /* TAU_MPI */

  
  

  /* For each event, determine contributing threads */
  for (int i=0; i<numEvents; i++) {
		numThreadsLocal[i] = 0;
	}
	for (int i=0; i<numEvents; i++)
	{
	 int local_index = functionUnifier->sortMap[globalEventMap[i]];
/*   if (globalEventMap[i] != -1) { // if it occurred in our rank
	  FunctionInfo *fi = TheFunctionDB()[local_index];
			for (int t=0; t<numThreads; t++)
			{
				if (fi->GetCalls(t) > 0)
				{
					numThreadsLocal[i] += 1;
				}
			}
			numThreadsLocal[i], fi->GetName());
		}*/
		if(globalEventMap[i]!=-1){
		numThreadsLocal[i]=Tau_collate_get_local_threads(local_index,isAtomic);
		}
		else
		{	
			numThreadsLocal[i] = 0;
		}
	}
  /* Extra slot in array indicates number of threads on rank */
  numThreadsLocal[numEvents] = RtsLayer::getTotalThreads();
#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  int size = __real__num_pes();
  int *numEventsMax = (int*)__real_shmalloc(sizeof(int));
  int *numEventsArr = (int*)__real_shmalloc(size*sizeof(int));
#else
  int size = __real_shmem_n_pes();
  int *numEventsMax = (int*)__real_shmem_malloc(sizeof(int));
  int *numEventsArr = (int*)__real_shmem_malloc(size*sizeof(int));
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  __real_shmem_barrier_all();
  // gather numItems from all pes
  __real_shmem_int_put(&numEventsArr[rank], &numEvents, 1, 0);
  // determine maximum value over all pes
  __real_shmem_barrier_all();
  if(rank == 0) {
    *numEventsMax = numEventsArr[0];
    for(int i=1; i<size; i++)
      if(*numEventsMax < numEventsArr[i]) *numEventsMax = numEventsArr[i];
    for(int i=1; i<size; i++) {
      __real_shmem_int_put(numEventsMax, numEventsMax, 1, i);
      __real_shmem_int_put(numEventsArr, numEventsArr, size, i);
    }
  }
  __real_shmem_barrier_all();
  
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  int *shlocal = (int*)__real_shmalloc((*numEventsMax+1)*sizeof(int));
#else
  int *shlocal = (int*)__real_shmem_malloc((*numEventsMax+1)*sizeof(int));
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  for(int i = 0; i < *numEventsMax+1; i++) {
    numThreadsGlobal[i] = 0;
    shlocal[i] = numThreadsLocal[i];
  }
  if(rank == 0) {
    for(int j=0; j<*numEventsMax+1; j++)
      numThreadsGlobal[j] = 0;
    int *tmp = (int*)malloc((*numEventsMax+1)*sizeof(int));
    for(int i=0; i<size; i++) {
      __real_shmem_int_get(tmp, shlocal, *numEventsMax+1, i);
      for(int j=0; j<*numEventsMax+1; j++)
        numThreadsGlobal[j] += tmp[j];
    }
    free(tmp);
  }
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  __real_shfree(shlocal);
  __real_shfree(numEventsMax);
  __real_shfree(numEventsArr);
#else
  __real_shmem_free(shlocal);
  __real_shmem_free(numEventsMax);
  __real_shmem_free(numEventsArr);
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */

  /* Now rank 0 knows all about global thread counts */
  if (rank == 0) {
    for (int i=0; i<numEvents; i++) {
      (*numEventThreads)[i] = numThreadsGlobal[i];
    }
    *globalNumThreads = numThreadsGlobal[numEvents];
  }
}

/***
 *  2011-10-10 *CWL*
 *  Computation of statistics for Atomic Events. These have to be handled differently
 *    because of the different internal representations for now.
 */
void Tau_collate_compute_atomicStatistics_MPI(Tau_unify_object_t *atomicUnifier,
					  int *globalEventMap, int numItems,
					  int globalNumThreads, 
					  int *numEventThreads,
					  double ***gAtomicMin, 
					  double ***gAtomicMax,
					  double ***gAtomicCalls, 
					  double ***gAtomicMean,
					  double ***gAtomicSumSqr,
					  double ***sAtomicMin, 
					  double ***sAtomicMax,
					  double ***sAtomicCalls, 
					  double ***sAtomicMean,
					  double ***sAtomicSumSqr) {
  int rank = 0;
#ifdef TAU_MPI
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* TAU_MPI */
  
  MPI_Op min_op = MPI_MIN;
#ifdef TAU_MPI
  PMPI_Op_create(stat_min, 1, &min_op);
#endif /* TAU_MPI */
  collate_op[step_min] = min_op;

  // allocate memory for values to fill with performance data and sent to
  //   the root node
  double *atomicMin, *atomicMax;
  double *atomicCalls, *atomicMean, *atomicSumSqr;

#ifdef TAU_MPI
  Tau_collate_allocateUnitAtomicBuffer(&atomicMin, &atomicMax, 
				       &atomicCalls, &atomicMean,
				       &atomicSumSqr,
				       numItems);
#endif /* TAU_MPI */

  for (int s=0; s<NUM_COLLATE_STEPS; s++) {
#ifndef TAU_MPI
    // in the non-MPI case, just point to the same memory.
    atomicMin = &((*gAtomicMin)[s][0]);
    atomicMax = &((*gAtomicMax)[s][0]);
    atomicCalls = &((*gAtomicCalls)[s][0]);
    atomicMean = &((*gAtomicMean)[s][0]);
    atomicSumSqr = &((*gAtomicSumSqr)[s][0]);
#endif /* !TAU_MPI */
    // Initialize to -1 only for step_min to handle unrepresented values for
    //   minimum.
    double fillDbl = 0.0;
    if (s == step_min) {
      fillDbl = -1.0;
    }
    for (int i=0; i<numItems; i++) {
      atomicMin[i] = fillDbl;
      atomicMax[i] = fillDbl;
      atomicCalls[i] = fillDbl;
      atomicMean[i] = fillDbl;
      atomicSumSqr[i] = fillDbl;
    }

    for (int i=0; i<numItems; i++) { // for each event
      if (globalEventMap[i] != -1) { // if it occurred in our rank
	int local_index = atomicUnifier->sortMap[globalEventMap[i]];
	TauUserEvent *event = TheEventDB()[local_index];
	//	int numThreads = RtsLayer::getNumThreads();
	int numThreads = RtsLayer::getTotalThreads();

	//synchronize
	RtsLayer::LockDB();

	for (int tid = 0; tid<numThreads; tid++) { // for each thread
	  atomicMin[i] = getStepValue((collate_step)s, atomicMin[i],
				      (double)event->GetMin(tid));
	  atomicMax[i] = getStepValue((collate_step)s, atomicMax[i],
				      (double)event->GetMax(tid));
	  atomicCalls[i] = getStepValue((collate_step)s, atomicCalls[i],
					(double)event->GetNumEvents(tid));
	  atomicMean[i] = getStepValue((collate_step)s, atomicMean[i],
				       (double)event->GetMean(tid));
	  atomicSumSqr[i] = getStepValue((collate_step)s, atomicSumSqr[i],
					 (double)event->GetSumSqr(tid));
	}
	
	//release lock
	RtsLayer::UnLockDB();
      }
    }

    // reduce data to rank 0
#ifdef TAU_MPI
    PMPI_Reduce(atomicMin, (*gAtomicMin)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, MPI_COMM_WORLD);
    PMPI_Reduce(atomicMax, (*gAtomicMax)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, MPI_COMM_WORLD);
    PMPI_Reduce(atomicCalls, (*gAtomicCalls)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, MPI_COMM_WORLD);
    PMPI_Reduce(atomicMean, (*gAtomicMean)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, MPI_COMM_WORLD);
    PMPI_Reduce(atomicSumSqr, (*gAtomicSumSqr)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */
  }
#ifndef TAU_MPI
  // free memory for basic information
  Tau_collate_freeUnitAtomicBuffer(&atomicMin, &atomicMax, 
				   &atomicCalls, &atomicMean, 
				   &atomicSumSqr);
#endif /* ! TAU_MPI */
  
  // Compute derived statistics on rank 0
  if (rank == 0) {
    for (int i=0; i<numItems; i++) { // for each event
      assignDerivedStats(sAtomicMin, gAtomicMin, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicMax, gAtomicMax, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicCalls, gAtomicCalls, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicMean, gAtomicMean, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicSumSqr, gAtomicSumSqr, i, globalNumThreads, numEventThreads);
    }    
  }
#ifdef TAU_MPI
  PMPI_Op_free(&min_op);
#endif /* TAU_MPI */
}

void Tau_collate_compute_atomicStatistics_MPI_with_minmaxloc(Tau_unify_object_t *atomicUnifier,
					  int *globalEventMap, int numItems,
					  int globalNumThreads, 
					  int *numEventThreads,
					  double ***gAtomicMin, 
					  double ***gAtomicMax,
					  double_int **gAtomicMin_min, 
					  double_int **gAtomicMax_max,
					  double ***gAtomicCalls, 
					  double ***gAtomicMean,
					  double ***gAtomicSumSqr,
					  double ***sAtomicMin, 
					  double ***sAtomicMax,
					  double ***sAtomicCalls, 
					  double ***sAtomicMean,
					  double ***sAtomicSumSqr
#ifdef TAU_MPI
  , MPI_Comm comm)
#else
  )
#endif

{
  int rank = 0;
#ifdef TAU_MPI
  PMPI_Comm_rank(comm, &rank);
#endif /* TAU_MPI */
  
  MPI_Op min_op = MPI_MIN;
#ifdef TAU_MPI
  PMPI_Op_create(stat_min, 1, &min_op);
#endif /* TAU_MPI */
  collate_op[step_min] = min_op;

  // allocate memory for values to fill with performance data and sent to
  //   the root node
  double *atomicMin, *atomicMax;
  double_int *atomicMin_, *atomicMax_;
  double *atomicCalls, *atomicMean, *atomicSumSqr;

#ifdef TAU_MPI
  Tau_collate_allocateUnitAtomicBuffer(&atomicMin, &atomicMax, 
				       &atomicCalls, &atomicMean,
				       &atomicSumSqr,
				       numItems);
   atomicMin_ = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numItems);
   atomicMax_ = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numItems);

#endif /* TAU_MPI */

  for (int s=0; s<NUM_COLLATE_STEPS; s++) {
#ifndef TAU_MPI
    // in the non-MPI case, just point to the same memory.
    atomicMin = &((*gAtomicMin)[s][0]);
    atomicMax = &((*gAtomicMax)[s][0]);
    atomicCalls = &((*gAtomicCalls)[s][0]);
    atomicMean = &((*gAtomicMean)[s][0]);
    atomicSumSqr = &((*gAtomicSumSqr)[s][0]);
#endif /* !TAU_MPI */
    // Initialize to -1 only for step_min to handle unrepresented values for
    //   minimum.
    double fillDbl = 0.0;
    if (s == step_min) {
      fillDbl = -1.0;
    }
    for (int i=0; i<numItems; i++) {
      atomicMin[i] = atomicMin_[i].value = fillDbl;
      atomicMax[i] = atomicMax_[i].value = fillDbl;
      atomicMin_[i].index = atomicMax_[i].index = rank;
      atomicCalls[i] = fillDbl;
      atomicMean[i] = fillDbl;
      atomicSumSqr[i] = fillDbl;
    }

    for (int i=0; i<numItems; i++) { // for each event
      if (globalEventMap[i] != -1) { // if it occurred in our rank
	int local_index = atomicUnifier->sortMap[globalEventMap[i]];
	TauUserEvent *event = TheEventDB()[local_index];
	//	int numThreads = RtsLayer::getNumThreads();
	int numThreads = RtsLayer::getTotalThreads();

	//synchronize
	RtsLayer::LockDB();

	for (int tid = 0; tid<numThreads; tid++) { // for each thread
	  atomicMin[i] = getStepValue((collate_step)s, atomicMin[i],
				      (double)event->GetMin(tid));
          atomicMin_[i].value = atomicMin[i];

	  atomicMax[i] = getStepValue((collate_step)s, atomicMax[i],
				      (double)event->GetMax(tid));
          atomicMax_[i].value = atomicMax[i];

	  atomicCalls[i] = getStepValue((collate_step)s, atomicCalls[i],
					(double)event->GetNumEvents(tid));
	  atomicMean[i] = getStepValue((collate_step)s, atomicMean[i],
				       (double)event->GetMean(tid));
	  atomicSumSqr[i] = getStepValue((collate_step)s, atomicSumSqr[i],
					 (double)event->GetSumSqr(tid));
	}
	
	//release lock
	RtsLayer::UnLockDB();
      }
    }

    // reduce data to rank 0
#ifdef TAU_MPI
    PMPI_Reduce(atomicMin, (*gAtomicMin)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, comm);
    PMPI_Reduce(atomicMax, (*gAtomicMax)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, comm);
    PMPI_Reduce(atomicCalls, (*gAtomicCalls)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, comm);
    PMPI_Reduce(atomicMean, (*gAtomicMean)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, comm);
    PMPI_Reduce(atomicSumSqr, (*gAtomicSumSqr)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, comm);

    if(s == step_min) {
      PMPI_Reduce(atomicMin_, *gAtomicMin_min, numItems, MPI_DOUBLE_INT,
                MPI_MINLOC, 0, comm);
    }
    if(s == step_max) {
      PMPI_Reduce(atomicMax_, *gAtomicMax_max, numItems, MPI_DOUBLE_INT,
                MPI_MAXLOC, 0, comm);
    }
#endif /* TAU_MPI */
  }
#ifndef TAU_MPI
  // free memory for basic information
  Tau_collate_freeUnitAtomicBuffer(&atomicMin, &atomicMax, 
				   &atomicCalls, &atomicMean, 
				   &atomicSumSqr);
#endif /* ! TAU_MPI */
  
  // Compute derived statistics on rank 0
  if (rank == 0) {
    for (int i=0; i<numItems; i++) { // for each event
      assignDerivedStats(sAtomicMin, gAtomicMin, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicMax, gAtomicMax, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicCalls, gAtomicCalls, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicMean, gAtomicMean, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicSumSqr, gAtomicSumSqr, i, globalNumThreads, numEventThreads);
    }    
  }
#ifdef TAU_MPI
  PMPI_Op_free(&min_op);
#endif /* TAU_MPI */
}

void Tau_collate_compute_atomicStatistics_SHMEM(Tau_unify_object_t *atomicUnifier,
					  int *globalEventMap, int numItems,
					  int globalNumThreads, 
					  int *numEventThreads,
					  double ***gAtomicMin, 
					  double ***gAtomicMax,
					  double ***gAtomicCalls, 
					  double ***gAtomicMean,
					  double ***gAtomicSumSqr,
					  double ***sAtomicMin, 
					  double ***sAtomicMax,
					  double ***sAtomicCalls, 
					  double ***sAtomicMean,
					  double ***sAtomicSumSqr) {
  int rank = 0;
#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  rank = __real__my_pe();
#else
  rank = __real_shmem_my_pe();
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */
  
  MPI_Op min_op = MPI_MIN;
  collate_op[step_min] = min_op;

  // allocate memory for values to fill with performance data and sent to
  //   the root node
  double *atomicMin, *atomicMax;
  double *atomicCalls, *atomicMean, *atomicSumSqr;

#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  int size = __real__num_pes();
  int *numItemsMax = (int*)__real_shmalloc(sizeof(int));
  int *numItemsArr = (int*)__real_shmalloc(size*sizeof(int));
#else
  int size = __real_shmem_n_pes();
  int *numItemsMax = (int*)__real_shmem_malloc(sizeof(int));
  int *numItemsArr = (int*)__real_shmem_malloc(size*sizeof(int));
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  Tau_collate_allocateUnitAtomicBuffer(&atomicMin, &atomicMax, 
				       &atomicCalls, &atomicMean,
				       &atomicSumSqr,
				       numItems);
  // gather numItems from all pes
  __real_shmem_int_put(&numItemsArr[rank], &numItems, 1, 0);
  // determine maximum value over all pes
  if(rank == 0) {
    *numItemsMax = numItemsArr[0];
    for(int i=1; i<size; i++)
      if(*numItemsMax < numItemsArr[i]) *numItemsMax = numItemsArr[i];
    for(int i=1; i<size; i++) {
      __real_shmem_int_put(numItemsMax, numItemsMax, 1, i);
      __real_shmem_int_put(numItemsArr, numItemsArr, size, i);
    }
  }
  __real_shmem_barrier_all();

#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  __real_shfree(numItemsMax);
  __real_shfree(numItemsArr);
#else
  __real_shmem_free(numItemsMax);
  __real_shmem_free(numItemsArr);
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  
#endif /* TAU_SHMEM */

  for (int s=0; s<NUM_COLLATE_STEPS; s++) {
#ifndef TAU_SHMEM
    // in the non-MPI case, just point to the same memory.
    atomicMin = &((*gAtomicMin)[s][0]);
    atomicMax = &((*gAtomicMax)[s][0]);
    atomicCalls = &((*gAtomicCalls)[s][0]);
    atomicMean = &((*gAtomicMean)[s][0]);
    atomicSumSqr = &((*gAtomicSumSqr)[s][0]);
#endif /* !TAU_SHMEM */
    // Initialize to -1 only for step_min to handle unrepresented values for
    //   minimum.
    double fillDbl = 0.0;
    if (s == step_min) {
      fillDbl = -1.0;
    }
    for (int i=0; i<numItems; i++) {
      atomicMin[i] = fillDbl;
      atomicMax[i] = fillDbl;
      atomicCalls[i] = fillDbl;
      atomicMean[i] = fillDbl;
      atomicSumSqr[i] = fillDbl;
    }

    for (int i=0; i<numItems; i++) { // for each event
      if (globalEventMap[i] != -1) { // if it occurred in our rank
	int local_index = atomicUnifier->sortMap[globalEventMap[i]];
	TauUserEvent *event = TheEventDB()[local_index];
	//	int numThreads = RtsLayer::getNumThreads();
	int numThreads = RtsLayer::getTotalThreads();

	//synchronize
	RtsLayer::LockDB();

	for (int tid = 0; tid<numThreads; tid++) { // for each thread
	  atomicMin[i] = getStepValue((collate_step)s, atomicMin[i],
				      (double)event->GetMin(tid));
	  atomicMax[i] = getStepValue((collate_step)s, atomicMax[i],
				      (double)event->GetMax(tid));
	  atomicCalls[i] = getStepValue((collate_step)s, atomicCalls[i],
					(double)event->GetNumEvents(tid));
	  atomicMean[i] = getStepValue((collate_step)s, atomicMean[i],
				       (double)event->GetMean(tid));
	  atomicSumSqr[i] = getStepValue((collate_step)s, atomicSumSqr[i],
					 (double)event->GetSumSqr(tid));
	}
	
	//release lock
	RtsLayer::UnLockDB();
      }
    }

    // reduce data to rank 0
  }
#ifndef TAU_SHMEM
  // free memory for basic information
  Tau_collate_freeUnitAtomicBuffer(&atomicMin, &atomicMax, 
				   &atomicCalls, &atomicMean, 
				   &atomicSumSqr);
#endif /* ! TAU_SHMEM */
  
  // Compute derived statistics on rank 0
  if (rank == 0) {
    for (int i=0; i<numItems; i++) { // for each event
      assignDerivedStats(sAtomicMin, gAtomicMin, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicMax, gAtomicMax, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicCalls, gAtomicCalls, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicMean, gAtomicMean, i, globalNumThreads, numEventThreads);
      assignDerivedStats(sAtomicSumSqr, gAtomicSumSqr, i, globalNumThreads, numEventThreads);
    }    
  }
}

/***
 *  2010-10-08 *CWL*
 *  Modularization of monitoring functionality for other non-TauMon
 *    purposes.
 */
void Tau_collate_compute_statistics_MPI(Tau_unify_object_t *functionUnifier,
				    int *globalEventMap, int numItems,
				    int globalNumThreads, int *numEventThreads,
				    double ****gExcl, double ****gIncl,
				    double ***gNumCalls, double ***gNumSubr,
				    double ****sExcl, double ****sIncl,
				    double ***sNumCalls, double ***sNumSubr) {
  int rank = 0;
#ifdef TAU_MPI
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* TAU_MPI */

  // *CWL* - Minimum needs to be handled with out-of-band values for now.
  MPI_Op min_op = MPI_MIN;
#ifdef TAU_MPI
  PMPI_Op_create (stat_min, 1, &min_op);
#endif /* TAU_MPI */
  collate_op[step_min] = min_op;

  // allocate memory for values to fill with performance data and sent to
  //   the root node
  double **excl;
  double **incl;
  double *numCalls, *numSubr;
#ifdef TAU_MPI
  Tau_collate_allocateUnitFunctionBuffer(&excl, &incl, 
					 &numCalls, &numSubr, 
					 numItems, Tau_Global_numCounters);
#endif /* TAU_MPI */

  // Fill the data, once for each basic statistic
  for (int s=0; s<NUM_COLLATE_STEPS; s++) {
#ifndef TAU_MPI
    // in the non-MPI case, just point to the same memory.
    excl = &((*gExcl)[s][0]);
    incl = &((*gIncl)[s][0]);
    numCalls = &((*gNumCalls)[s][0]);
    numSubr = &((*gNumSubr)[s][0]);
#endif /* !TAU_MPI */
    // Initialize to -1 only for step_min to handle unrepresented values for
    //   minimume
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
    for (int i=0; i<numItems; i++) { // for each event
      if (globalEventMap[i] != -1) { // if it occurred in our rank
	int local_index = functionUnifier->sortMap[globalEventMap[i]];
	FunctionInfo *fi = TheFunctionDB()[local_index];
	//	int numThreads = RtsLayer::getNumThreads();
	int numThreads = RtsLayer::getTotalThreads();
	//synchronize
	RtsLayer::LockDB();

	for (int tid = 0; tid<numThreads; tid++) { // for each thread
	  for (int m=0; m<Tau_Global_numCounters; m++) {
			//this make no sense but you need to use a different data-structure in
			//FunctionInfo if you are quering thread 0.
			if (tid == 0)
			{	
				incl[m][i] = getStepValue((collate_step)s, incl[m][i],
								fi->getDumpInclusiveValues(tid)[m]);
				excl[m][i] = getStepValue((collate_step)s, excl[m][i],
								fi->getDumpExclusiveValues(tid)[m]);
			}	
			else // thread != 0
			{
				incl[m][i] = getStepValue((collate_step)s, incl[m][i],
								fi->GetInclTimeForCounter(tid,m));
				excl[m][i] = getStepValue((collate_step)s, excl[m][i],
								fi->GetExclTimeForCounter(tid,m));
			}	
		
	  }
			numCalls[i] = getStepValue((collate_step)s, numCalls[i],
							 (double)fi->GetCalls(tid));
			numSubr[i] = getStepValue((collate_step)s, numSubr[i],
							(double)fi->GetSubrs(tid));
	}
	//release lock
	RtsLayer::UnLockDB();
      }
    }
    
#ifdef TAU_MPI
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
#endif /* TAU_MPI */
  }
#ifdef TAU_MPI
  // Free allocated memory for basic info.
  Tau_collate_freeUnitFunctionBuffer(&excl, &incl, &numCalls, &numSubr, Tau_Global_numCounters);
#endif

  // Now compute the actual statistics on rank 0 only. The assumption
  //   is that at least one thread would be active across the node-space
  //   and so negative values should never show up after reduction.
  if (rank == 0) {
    // *CWL* TODO - abstract the operations to avoid this nasty coding
    //     of individual operations.
    for (int i=0; i<numItems; i++) { // for each event
      for (int m=0; m<Tau_Global_numCounters; m++) {
	    assignDerivedStats(sIncl, gIncl, m, i, globalNumThreads, numEventThreads);
	    assignDerivedStats(sExcl, gExcl, m, i, globalNumThreads, numEventThreads);
	  }
	  assignDerivedStats(sNumCalls, gNumCalls, i, globalNumThreads, numEventThreads);
	  assignDerivedStats(sNumSubr, gNumSubr, i, globalNumThreads, numEventThreads);
	}    
  }
#ifdef TAU_MPI
  PMPI_Op_free(&min_op);
#endif /* TAU_MPI */
}

void Tau_collate_compute_statistics_MPI_with_minmaxloc(Tau_unify_object_t *functionUnifier,
				    int *globalEventMap, int numItems,
				    int globalNumThreads, int *numEventThreads,
				    double ****gExcl, double ****gIncl,
                                    double_int ***gExcl_min, double_int ***gIncl_min,
                                    double_int ***gExcl_max, double_int ***gIncl_max,
				    double ***gNumCalls, double ***gNumSubr,
				    double ****sExcl, double ****sIncl,
				    double ***sNumCalls, double ***sNumSubr

#ifdef TAU_MPI
 , MPI_Comm comm)
#else
  )
#endif
{

  int rank = 0;
#ifdef TAU_MPI
  PMPI_Comm_rank(comm, &rank);
#endif /* TAU_MPI */

  // *CWL* - Minimum needs to be handled with out-of-band values for now.
  MPI_Op min_op = MPI_MIN;
#ifdef TAU_MPI
  PMPI_Op_create (stat_min, 1, &min_op);
#endif /* TAU_MPI */
  collate_op[step_min] = min_op;

  // allocate memory for values to fill with performance data and sent to
  //   the root node
  double **excl;
  double **incl;
  double_int **excl_, **incl_;
  double *numCalls, *numSubr;
#ifdef TAU_MPI
  Tau_collate_allocateUnitFunctionBuffer(&excl, &incl, 
					 &numCalls, &numSubr, 
					 numItems, Tau_Global_numCounters);

  excl_ = (double_int **)TAU_UTIL_MALLOC(sizeof(double_int *)*Tau_Global_numCounters);
  incl_ = (double_int **)TAU_UTIL_MALLOC(sizeof(double_int *)*Tau_Global_numCounters);
  // Please note the use of Calloc
   for (int m=0; m<Tau_Global_numCounters; m++) {
     excl_[m] = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numItems);
     incl_[m] = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numItems);
   }

#endif /* TAU_MPI */

  // Fill the data, once for each basic statistic
  for (int s=0; s<NUM_COLLATE_STEPS; s++) {
#ifndef TAU_MPI
    // in the non-MPI case, just point to the same memory.
    excl = &((*gExcl)[s][0]);
    incl = &((*gIncl)[s][0]);
    numCalls = &((*gNumCalls)[s][0]);
    numSubr = &((*gNumSubr)[s][0]);
#endif /* !TAU_MPI */
    // Initialize to -1 only for step_min to handle unrepresented values for
    //   minimume
    double fillDbl = 0.0;
    if (s == step_min) {
      fillDbl = -1.0;
    }
    for (int i=0; i<numItems; i++) {
      for (int m=0; m<Tau_Global_numCounters; m++) {
	incl_[m][i].value = incl[m][i] = fillDbl;
	excl_[m][i].value = excl[m][i] = fillDbl;
        incl_[m][i].index = excl_[m][i].index = rank;
      }
      numCalls[i] = fillDbl;
      numSubr[i] = fillDbl;
    }
    for (int i=0; i<numItems; i++) { // for each event
      if (globalEventMap[i] != -1) { // if it occurred in our rank
	int local_index = functionUnifier->sortMap[globalEventMap[i]];
	FunctionInfo *fi = TheFunctionDB()[local_index];
	//	int numThreads = RtsLayer::getNumThreads();
	int numThreads = RtsLayer::getTotalThreads();
	//synchronize
	RtsLayer::LockDB();

	for (int tid = 0; tid<numThreads; tid++) { // for each thread
	  for (int m=0; m<Tau_Global_numCounters; m++) {
			//this make no sense but you need to use a different data-structure in
			//FunctionInfo if you are quering thread 0.
			if (tid == 0)
			{	
				incl[m][i] = getStepValue((collate_step)s, incl[m][i],
								fi->getDumpInclusiveValues(tid)[m]);
				excl[m][i] = getStepValue((collate_step)s, excl[m][i],
								fi->getDumpExclusiveValues(tid)[m]);
                                excl_[m][i].value = excl[m][i];
                                incl_[m][i].value = incl[m][i];
			}	
			else // thread != 0
			{
				incl[m][i] = getStepValue((collate_step)s, incl[m][i],
								fi->GetInclTimeForCounter(tid,m));
				excl[m][i] = getStepValue((collate_step)s, excl[m][i],fi->GetExclTimeForCounter(tid,m));

                                excl_[m][i].value = excl[m][i];
                                incl_[m][i].value = incl[m][i];
                                 
			}	
		
	  }
			numCalls[i] = getStepValue((collate_step)s, numCalls[i],
							 (double)fi->GetCalls(tid));
			numSubr[i] = getStepValue((collate_step)s, numSubr[i],
							(double)fi->GetSubrs(tid));
	}
	//release lock
	RtsLayer::UnLockDB();
      }
    }
    
#ifdef TAU_MPI
    // reduce data to rank 0
    for (int m=0; m<Tau_Global_numCounters; m++) {
      PMPI_Reduce(excl[m], (*gExcl)[s][m], numItems, MPI_DOUBLE, 
		  collate_op[s], 0, comm);
      PMPI_Reduce(incl[m], (*gIncl)[s][m], numItems, MPI_DOUBLE, 
		  collate_op[s], 0, comm);

      /* Awfully inefficient stuff!!*/
      if(s == step_min) {
        PMPI_Reduce(excl_[m], (*gExcl_min)[m], numItems, MPI_DOUBLE_INT,
                    MPI_MINLOC, 0, comm);
        PMPI_Reduce(incl_[m], (*gIncl_min)[m], numItems, MPI_DOUBLE_INT,
                    MPI_MINLOC, 0, comm);
      }

      if(s == step_max) {
        PMPI_Reduce(excl_[m], (*gExcl_max)[m], numItems, MPI_DOUBLE_INT,
                    MPI_MAXLOC, 0, comm);
        PMPI_Reduce(incl_[m], (*gIncl_max)[m], numItems, MPI_DOUBLE_INT,
                    MPI_MAXLOC, 0, comm);
      }
    }

    PMPI_Reduce(numCalls, (*gNumCalls)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, comm);
    PMPI_Reduce(numSubr, (*gNumSubr)[s], numItems, MPI_DOUBLE, 
		collate_op[s], 0, comm);
#endif /* TAU_MPI */
  }
#ifdef TAU_MPI
  // Free allocated memory for basic info.
  Tau_collate_freeUnitFunctionBuffer(&excl, &incl, &numCalls, &numSubr, Tau_Global_numCounters);
#endif

  // Now compute the actual statistics on rank 0 only. The assumption
  //   is that at least one thread would be active across the node-space
  //   and so negative values should never show up after reduction.
  if (rank == 0) {
    // *CWL* TODO - abstract the operations to avoid this nasty coding
    //     of individual operations.
    for (int i=0; i<numItems; i++) { // for each event
      for (int m=0; m<Tau_Global_numCounters; m++) {
	    assignDerivedStats(sIncl, gIncl, m, i, globalNumThreads, numEventThreads);
	    assignDerivedStats(sExcl, gExcl, m, i, globalNumThreads, numEventThreads);
	  }
	  assignDerivedStats(sNumCalls, gNumCalls, i, globalNumThreads, numEventThreads);
	  assignDerivedStats(sNumSubr, gNumSubr, i, globalNumThreads, numEventThreads);
	}    
  }
#ifdef TAU_MPI
  PMPI_Op_free(&min_op);
#endif /* TAU_MPI */
}

void Tau_collate_compute_statistics_SHMEM(Tau_unify_object_t *functionUnifier,
				    int *globalEventMap, int numItems,
				    int globalNumThreads, int *numEventThreads,
				    double ****gExcl, double ****gIncl,
				    double ***gNumCalls, double ***gNumSubr,
				    double ****sExcl, double ****sIncl,
				    double ***sNumCalls, double ***sNumSubr) {
  int rank = 0;
#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  rank = __real__my_pe();
#else
  rank = __real_shmem_my_pe();
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */

  // *CWL* - Minimum needs to be handled with out-of-band values for now.
  MPI_Op min_op = MPI_MIN;
  collate_op[step_min] = min_op;

  // allocate memory for values to fill with performance data and sent to
  //   the root node
  double **excl, **incl;
  double *numCalls, *numSubr;
#ifdef TAU_SHMEM
  Tau_collate_allocateUnitFunctionBuffer(&excl, &incl,
					 &numCalls, &numSubr,
					 numItems, Tau_Global_numCounters);


#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  int size = __real__num_pes();
  int *numItemsMax = (int*)__real_shmalloc(sizeof(int));
  int *numItemsArr = (int*)__real_shmalloc(size*sizeof(int));
#else
  int size = __real_shmem_n_pes();
  int *numItemsMax = (int*)__real_shmem_malloc(sizeof(int));
  int *numItemsArr = (int*)__real_shmem_malloc(size*sizeof(int));
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  // gather numItems from all pes
  __real_shmem_int_put(&numItemsArr[rank], &numItems, 1, 0);
  // determine maximum value over all pes
  if(rank == 0) {
    *numItemsMax = numItemsArr[0];
    for(int i=1; i<size; i++)
      if(*numItemsMax < numItemsArr[i]) *numItemsMax = numItemsArr[i];
    for(int i=1; i<size; i++) {
      __real_shmem_int_put(numItemsMax, numItemsMax, 1, i);
      __real_shmem_int_put(numItemsArr, numItemsArr, size, i);
    }
  }
  __real_shmem_barrier_all();
  // allocate g* shmem variables
  double ***shgExcl, ***shgIncl;
  double **shgNumCalls, **shgNumSubr;
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  shgExcl = (double***)__real_shmalloc(size*sizeof(double**));
  shgIncl = (double***)__real_shmalloc(size*sizeof(double**));
  shgNumCalls = (double**)__real_shmalloc(size*sizeof(double*));
  shgNumSubr = (double**)__real_shmalloc(size*sizeof(double*));
  for(int rnk=0; rnk<size; rnk++) {
    shgExcl[rnk] = (double**)__real_shmalloc(Tau_Global_numCounters*sizeof(double*));
    shgIncl[rnk] = (double**)__real_shmalloc(Tau_Global_numCounters*sizeof(double*));
    for(int m=0; m<Tau_Global_numCounters; m++) {
      shgExcl[rnk][m] = (double*)__real_shmalloc(*numItemsMax*sizeof(double));
      shgIncl[rnk][m] = (double*)__real_shmalloc(*numItemsMax*sizeof(double));
    }
    shgNumCalls[rnk] = (double*)__real_shmalloc(*numItemsMax*sizeof(double));
    shgNumSubr[rnk]  = (double*)__real_shmalloc(*numItemsMax*sizeof(double));
  }
#else
  shgExcl = (double***)__real_shmem_malloc(size*sizeof(double**));
  shgIncl = (double***)__real_shmem_malloc(size*sizeof(double**));
  shgNumCalls = (double**)__real_shmem_malloc(size*sizeof(double*));
  shgNumSubr = (double**)__real_shmem_malloc(size*sizeof(double*));
  for(int rnk=0; rnk<size; rnk++) {
    shgExcl[rnk] = (double**)__real_shmem_malloc(Tau_Global_numCounters*sizeof(double*));
    shgIncl[rnk] = (double**)__real_shmem_malloc(Tau_Global_numCounters*sizeof(double*));
    for(int m=0; m<Tau_Global_numCounters; m++) {
      shgExcl[rnk][m] = (double*)__real_shmem_malloc(*numItemsMax*sizeof(double));
      shgIncl[rnk][m] = (double*)__real_shmem_malloc(*numItemsMax*sizeof(double));
    }
    shgNumCalls[rnk] = (double*)__real_shmem_malloc(*numItemsMax*sizeof(double));
    shgNumSubr[rnk]  = (double*)__real_shmem_malloc(*numItemsMax*sizeof(double));
  }
#endif /* SHMEM_1_1 || SHMEM_1_2 */

#endif /* TAU_SHMEM */

  // Fill the data, once for each basic statistic
  for (int s=0; s<NUM_COLLATE_STEPS; s++) {
#ifndef TAU_SHMEM
    // in the non-MPI case, just point to the same memory.
    excl = &((*gExcl)[s][0]);
    incl = &((*gIncl)[s][0]);
    numCalls = &((*gNumCalls)[s][0]);
    numSubr = &((*gNumSubr)[s][0]);
#endif /* !TAU_SHMEM */
    // Initialize to -1 only for step_min to handle unrepresented values for
    //   minimume
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
    for (int i=0; i<numItems; i++) { // for each event
      if (globalEventMap[i] != -1) { // if it occurred in our rank
	int local_index = functionUnifier->sortMap[globalEventMap[i]];
	FunctionInfo *fi = TheFunctionDB()[local_index];
	//	int numThreads = RtsLayer::getNumThreads();
	int numThreads = RtsLayer::getTotalThreads();
	//synchronize
	RtsLayer::LockDB();

	for (int tid = 0; tid<numThreads; tid++) { // for each thread
	  for (int m=0; m<Tau_Global_numCounters; m++) {
			//this make no sense but you need to use a different data-structure in
			//FunctionInfo if you are quering thread 0.
			if (tid == 0)
			{	
				incl[m][i] = getStepValue((collate_step)s, incl[m][i],
								fi->getDumpInclusiveValues(tid)[m]);
				excl[m][i] = getStepValue((collate_step)s, excl[m][i],
								fi->getDumpExclusiveValues(tid)[m]);
			}	
			else // thread != 0
			{
				incl[m][i] = getStepValue((collate_step)s, incl[m][i],
								fi->GetInclTimeForCounter(tid,m));
				excl[m][i] = getStepValue((collate_step)s, excl[m][i],
								fi->GetExclTimeForCounter(tid,m));
			}	
		
	  }
			numCalls[i] = getStepValue((collate_step)s, numCalls[i],
							 (double)fi->GetCalls(tid));
			numSubr[i] = getStepValue((collate_step)s, numSubr[i],
							(double)fi->GetSubrs(tid));
	}
	//release lock
	RtsLayer::UnLockDB();
      }
    }
    
#ifdef TAU_SHMEM

    // Initialize the shg* arrays to 0

    for(int rnk=0; rnk<size; rnk++) {
      for(int i=0; i<*numItemsMax; i++) {
        for(int m=0; m<Tau_Global_numCounters; m++) {
          shgExcl[rnk][m][i] = -1.0;
          shgIncl[rnk][m][i] = -1.0;
        }
        shgNumCalls[rnk][i] = -1.0;
        shgNumSubr[rnk][i] = -1.0;
      }
    }
    __real_shmem_barrier_all();

    for(int m=0; m<Tau_Global_numCounters; m++) {
      __real_shmem_double_put(shgExcl[rank][m], excl[m], numItemsArr[rank], 0);
      __real_shmem_double_put(shgIncl[rank][m], incl[m], numItemsArr[rank], 0);
    }
    __real_shmem_double_put(shgNumCalls[rank], numCalls, numItemsArr[rank], 0);
    __real_shmem_double_put(shgNumSubr[rank], numSubr, numItemsArr[rank], 0);
    __real_shmem_barrier_all();

    if(rank == 0) {

      if(s == 0) // Min
      {
        for(int m=0; m<Tau_Global_numCounters; m++) {
          for(int i=0; i<*numItemsMax; i++) {
            (*gExcl)[s][m][i] = shgExcl[0][m][i];
            (*gIncl)[s][m][i] = shgIncl[0][m][i];
            for(int rnk=1; rnk<size; rnk++) {
              if(((*gExcl)[s][m][i] > shgExcl[rnk][m][i] && shgExcl[rnk][m][i] >= 0) || (*gExcl)[s][m][i] < 0) (*gExcl)[s][m][i] = shgExcl[rnk][m][i];
              if(((*gIncl)[s][m][i] > shgIncl[rnk][m][i] && shgIncl[rnk][m][i] >= 0) || (*gIncl)[s][m][i] < 0) (*gIncl)[s][m][i] = shgIncl[rnk][m][i];
            }
          }
        }
        for(int i=0; i<*numItemsMax; i++)
        {
          (*gNumCalls)[s][i] = shgNumCalls[0][i];
          (*gNumSubr)[s][i]  = shgNumSubr[0][i];
          for(int rnk=1; rnk<size; rnk++)
          {
            if(((*gNumCalls)[s][i] > shgNumCalls[rnk][i] && shgNumCalls[rnk][i] >= 0) || (*gNumCalls)[s][i] < 0) (*gNumCalls)[s][i] = shgNumCalls[rnk][i];
            if(((*gNumSubr)[s][i] > shgNumSubr[rnk][i] && shgNumSubr[rnk][i]) || (*gNumSubr)[s][i] < 0) (*gNumSubr)[s][i] = shgNumSubr[rnk][i];
          }
        }
      }
      else if(s == 1) // Max
      {
        for(int m=0; m<Tau_Global_numCounters; m++) {
          for(int i=0; i<*numItemsMax; i++) {
            (*gExcl)[s][m][i] = shgExcl[0][m][i];
            (*gIncl)[s][m][i] = shgIncl[0][m][i];
            for(int rnk=1; rnk<size; rnk++) {
              if((*gExcl)[s][m][i] < shgExcl[rnk][m][i]) (*gExcl)[s][m][i] = shgExcl[rnk][m][i];
              if((*gIncl)[s][m][i] < shgIncl[rnk][m][i]) (*gIncl)[s][m][i] = shgIncl[rnk][m][i];
            }
          }
        }
        for(int i=0; i<*numItemsMax; i++)
        {
          (*gNumCalls)[s][i] = shgNumCalls[0][i];
          (*gNumSubr)[s][i]  = shgNumSubr[0][i];
          for(int rnk=1; rnk<size; rnk++)
          {
            if((*gNumCalls)[s][i] < shgNumCalls[rnk][i]) (*gNumCalls)[s][i] = shgNumCalls[rnk][i];
            if((*gNumSubr)[s][i] < shgNumSubr[rnk][i]) (*gNumSubr)[s][i] = shgNumSubr[rnk][i];
          }
        }
      }
      else // Sum
      {
        for(int m=0; m<Tau_Global_numCounters; m++) {
          for(int i=0; i<*numItemsMax; i++) {
            (*gExcl)[s][m][i] = shgExcl[0][m][i];
            (*gIncl)[s][m][i] = shgIncl[0][m][i];
            for(int rnk=1; rnk<size; rnk++) {
              if(shgExcl[rnk][m][i] > 0) (*gExcl)[s][m][i] += shgExcl[rnk][m][i];
              if(shgIncl[rnk][m][i] > 0) (*gIncl)[s][m][i] += shgIncl[rnk][m][i];
            }
          }
        }
        for(int i=0; i<*numItemsMax; i++)
        {
          (*gNumCalls)[s][i] = shgNumCalls[0][i];
          (*gNumSubr)[s][i]  = shgNumSubr[0][i];
          for(int rnk=1; rnk<size; rnk++)
          {
            if(shgNumCalls[rnk][i] > 0) (*gNumCalls)[s][i] += shgNumCalls[rnk][i];
            if(shgNumSubr[rnk][i] > 0) (*gNumSubr)[s][i] += shgNumSubr[rnk][i];
          }
        }
      }
    }
#endif /* TAU_SHMEM */
  }
#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  for(int rnk=0; rnk<size; rnk++) {
    for(int m=0; m<Tau_Global_numCounters; m++) {
      __real_shfree(shgExcl[rnk][m]);
      __real_shfree(shgIncl[rnk][m]);
    }
    __real_shfree(shgExcl[rnk]);
    __real_shfree(shgIncl[rnk]);
    __real_shfree(shgNumCalls[rnk]);
    __real_shfree(shgNumSubr[rnk]);
  }
  __real_shfree(shgExcl);
  __real_shfree(shgIncl);
  __real_shfree(shgNumCalls);
  __real_shfree(shgNumSubr);
  __real_shfree(numItemsMax);
  __real_shfree(numItemsArr);
#else
  for(int rnk=0; rnk<size; rnk++) {
    for(int m=0; m<Tau_Global_numCounters; m++) {
      __real_shmem_free(shgExcl[rnk][m]);
      __real_shmem_free(shgIncl[rnk][m]);
    }
    __real_shmem_free(shgExcl[rnk]);
    __real_shmem_free(shgIncl[rnk]);
    __real_shmem_free(shgNumCalls[rnk]);
    __real_shmem_free(shgNumSubr[rnk]);
  }
  __real_shmem_free(shgExcl);
  __real_shmem_free(shgIncl);
  __real_shmem_free(shgNumCalls);
  __real_shmem_free(shgNumSubr);
  __real_shmem_free(numItemsMax);
  __real_shmem_free(numItemsArr);
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */

  // Now compute the actual statistics on rank 0 only. The assumption
  //   is that at least one thread would be active across the node-space
  //   and so negative values should never show up after reduction.
  if (rank == 0) {
    // *CWL* TODO - abstract the operations to avoid this nasty coding
    //     of individual operations.
    for (int i=0; i<numItems; i++) { // for each event
      for (int m=0; m<Tau_Global_numCounters; m++) {
	    assignDerivedStats(sIncl, gIncl, m, i, globalNumThreads, numEventThreads);
	    assignDerivedStats(sExcl, gExcl, m, i, globalNumThreads, numEventThreads);
	  }
	  assignDerivedStats(sNumCalls, gNumCalls, i, globalNumThreads, numEventThreads);
	  assignDerivedStats(sNumSubr, gNumSubr, i, globalNumThreads, numEventThreads);
	}    
  }
}

static void Tau_collate_incrementHistogram(int *histogram, double min, 
					   double max, double value, 
					   int numBins) {
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

void Tau_collate_compute_histograms(Tau_unify_object_t *functionUnifier,
				    int *globalEventMap, int numItems,
				    int numBins, int numHistograms,
				    int e, int **outHistogram,
				    double ***gExcl, double ***gIncl,
				    double **gNumCalls, double **gNumSubr) {
  // two for each metric (excl, incl) and numCalls/numSubr;
  int histogramBufSize = sizeof(int) * numBins * numHistograms;
#ifdef TAU_MPI
  int *histogram = (int *) TAU_UTIL_MALLOC(histogramBufSize);
#else /* !TAU_MPI */
  int *histogram = &((*outHistogram)[0]);
#endif /* TAU_MPI */

#ifndef TAU_WINDOWS
  bzero(histogram, histogramBufSize);
#endif /* TAU_WINDOWS */

#ifdef TAU_MPI
  int rank = 0;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* TAU_MPI */
  
  if (globalEventMap[e] != -1) { // if it occurred in our rank
    int local_index = functionUnifier->sortMap[globalEventMap[e]];
    FunctionInfo *fi = TheFunctionDB()[local_index];
    
    //    int numThreads = RtsLayer::getNumThreads();
    int numThreads = RtsLayer::getTotalThreads();
    for (int tid = 0; tid<numThreads; tid++) { // for each thread
      for (int m=0; m<Tau_Global_numCounters; m++) {
	Tau_collate_incrementHistogram(&(histogram[(m*2)*numBins]), 
				       gExcl[step_min][m][e], 
				       gExcl[step_max][m][e], 
				       fi->getDumpExclusiveValues(tid)[m], 
				       numBins);
	Tau_collate_incrementHistogram(&(histogram[(m*2+1)*numBins]), 
				       gIncl[step_min][m][e], 
				       gIncl[step_max][m][e], 
				       fi->getDumpInclusiveValues(tid)[m], 
				       numBins);
      }
      Tau_collate_incrementHistogram(&(histogram[(Tau_Global_numCounters*2)*
						   numBins]), 
				     gNumCalls[step_min][e], 
				     gNumCalls[step_max][e], 
				     fi->GetCalls(tid), numBins);
      Tau_collate_incrementHistogram(&(histogram[(Tau_Global_numCounters*2+1)*numBins]), 
				     gNumSubr[step_min][e], 
				     gNumSubr[step_max][e], 
				     fi->GetSubrs(tid), numBins);
    }
  }    
#ifdef TAU_MPI
  PMPI_Reduce (histogram, *outHistogram, 
	       numBins*numHistograms, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */
}

/* Only enable profile writing and dump operation if TAU MPI Monitoring
   is specified */
#ifdef TAU_MONITORING
#ifdef TAU_MON_MPI

extern "C" void Tau_mon_connect() {
  /* Nothing needs to happen for MPI-based monitoring */
}

extern "C" void Tau_mon_disconnect() {
  /* Nothing needs to happen for MPI-based monitoring */
}

/*********************************************************************
 * Write a profile with data from all nodes/threads
 ********************************************************************/
extern "C" int Tau_collate_writeProfile_MPI() {
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
  //  int numThreads = RtsLayer::getNumThreads();
  int numThreads = RtsLayer::getTotalThreads();
  for (int tid = 0; tid<numThreads; tid++) {
    TauProfiler_updateIntermediateStatistics(tid);
  }

  x_uint64 start_unify, end_unify;
  if (rank == 0) {
    start_unify = TauMetrics_getTimeOfDay();
  }

  // Unify events
  FunctionEventLister *functionEventLister = new FunctionEventLister();
  Tau_unify_object_t *functionUnifier = Tau_unify_unifyEvents_MPI(functionEventLister);
  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  Tau_unify_object_t *atomicUnifier = Tau_unify_unifyEvents_MPI(atomicEventLister);

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
  Tau_collate_get_total_threads_MPI(functionUnifier, &globalNumThreads, &numEventThreads,
				numItems, globalEventMap,false);

  double ***gExcl, ***gIncl;
  double **gNumCalls, **gNumSubr;
  double ***sExcl, ***sIncl;
  double **sNumCalls, **sNumSubr;
  Tau_collate_allocateFunctionBuffers(&gExcl, &gIncl,
				      &gNumCalls, &gNumSubr,
				      numItems,
				      Tau_Global_numCounters,
				      COLLATE_OP_BASIC);
  Tau_collate_allocateFunctionBuffers(&sExcl, &sIncl,
				      &sNumCalls, &sNumSubr,
				      numItems,
				      Tau_Global_numCounters,
				      COLLATE_OP_DERIVED);
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
  int numHistograms = (Tau_Global_numCounters * 2) + 2; 
  int histogramBufSize = sizeof(int) * numBins * numHistograms;
  int *outHistogram = (int *) TAU_UTIL_MALLOC(histogramBufSize);

  const char *profiledir = TauEnv_get_profiledir();

  FILE *histoFile;
  char histFileNameTmp[512];
  char histFileName[512];


  if (rank == 0) {
    snprintf (histFileName, sizeof(histFileName),  "%s/tau.histograms.%d", profiledir, 
	     invocationIndex);
    snprintf (histFileNameTmp, sizeof(histFileNameTmp),  "%s/.temp.tau.histograms.%d", profiledir,
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

  for (int e=0; e<numItems; e++) {
    // make parallel histogram call.
    bzero(outHistogram, histogramBufSize);
    Tau_collate_compute_histograms(functionUnifier,
				   globalEventMap, numItems,
				   numBins, numHistograms,
				   e, &outHistogram,
				   gExcl, gIncl,
				   gNumCalls, gNumSubr);
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
    end_hist = TauMetrics_getTimeOfDay();
  
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
    snprintf (profileNameTmp, sizeof(profileNameTmp),  "%s/.temp.mean.%d.0.0", profiledir,
	     invocationIndex);
    snprintf (profileName, sizeof(profileName),  "%s/mean.%d.0.0", profiledir, invocationIndex);
    snprintf(unifyMeta, sizeof(unifyMeta), "<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Unification Time", ((double)((double)end_unify-start_unify))/1000000.0f);
    snprintf(aggregateMeta, sizeof(aggregateMeta), "<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Mean Aggregation Time", ((double)((double)end_aggregate-start_aggregate))/1000000.0f);
    snprintf(histogramMeta, sizeof(histogramMeta), "<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Histogramming Time", ((double)((double)end_hist-start_hist))/1000000.0f);
    snprintf(monitoringMeta, sizeof(monitoringMeta), "<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
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
extern "C" int Tau_collate_writeProfile_SHMEM() {
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
  //  int numThreads = RtsLayer::getNumThreads();
  int numThreads = RtsLayer::getTotalThreads();
  for (int tid = 0; tid<numThreads; tid++) {
    TauProfiler_updateIntermediateStatistics(tid);
  }

  x_uint64 start_unify, end_unify;
  if (rank == 0) {
    start_unify = TauMetrics_getTimeOfDay();
  }

  // Unify events
  FunctionEventLister *functionEventLister = new FunctionEventLister();
  Tau_unify_object_t *functionUnifier = Tau_unify_unifyEvents_SHMEM(functionEventLister);
  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  Tau_unify_object_t *atomicUnifier = Tau_unify_unifyEvents_SHMEM(atomicEventLister);

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
  Tau_collate_get_total_threads_SHMEM(functionUnifier, &globalNumThreads, &numEventThreads,
				numItems, globalEventMap,false);

  double ***gExcl, ***gIncl;
  double **gNumCalls, **gNumSubr;
  double ***sExcl, ***sIncl;
  double **sNumCalls, **sNumSubr;
  Tau_collate_allocateFunctionBuffers(&gExcl, &gIncl,
				      &gNumCalls, &gNumSubr,
				      numItems,
				      Tau_Global_numCounters,
				      COLLATE_OP_BASIC);
  Tau_collate_allocateFunctionBuffers(&sExcl, &sIncl,
				      &sNumCalls, &sNumSubr,
				      numItems,
				      Tau_Global_numCounters,
				      COLLATE_OP_DERIVED);
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
  int numHistograms = (Tau_Global_numCounters * 2) + 2; 
  int histogramBufSize = sizeof(int) * numBins * numHistograms;
  int *outHistogram = (int *) TAU_UTIL_MALLOC(histogramBufSize);

  const char *profiledir = TauEnv_get_profiledir();

  FILE *histoFile;
  char histFileNameTmp[512];
  char histFileName[512];


  if (rank == 0) {
    snprintf (histFileName, sizeof(histFileName),  "%s/tau.histograms.%d", profiledir, 
	     invocationIndex);
    snprintf (histFileNameTmp, sizeof(histFileNameTmp),  "%s/.temp.tau.histograms.%d", profiledir,
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

  for (int e=0; e<numItems; e++) {
    // make parallel histogram call.
    bzero(outHistogram, histogramBufSize);
    Tau_collate_compute_histograms(functionUnifier,
				   globalEventMap, numItems,
				   numBins, numHistograms,
				   e, &outHistogram,
				   gExcl, gIncl,
				   gNumCalls, gNumSubr);
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
    end_hist = TauMetrics_getTimeOfDay();
  
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
    snprintf (profileNameTmp, sizeof(profileNameTmp),  "%s/.temp.mean.%d.0.0", profiledir,
	     invocationIndex);
    snprintf (profileName, sizeof(profileName),  "%s/mean.%d.0.0", profiledir, invocationIndex);
    snprintf(unifyMeta, sizeof(unifyMeta), "<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Unification Time", ((double)((double)end_unify-start_unify))/1000000.0f);
    snprintf(aggregateMeta, sizeof(aggregateMeta), "<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Mean Aggregation Time", ((double)((double)end_aggregate-start_aggregate))/1000000.0f);
    snprintf(histogramMeta, sizeof(histogramMeta), "<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	    "Histogramming Time", ((double)((double)end_hist-start_hist))/1000000.0f);
    snprintf(monitoringMeta, sizeof(monitoringMeta), "<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
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
extern "C" void Tau_mon_internal_onlineDump() {
  // Not scalable output, even for verbose output. This is not a one-time
  //    operation.
  //  TAU_VERBOSE("collate online dump called\n");
  Tau_collate_writeProfile();
}

#endif /* TAU_MON_MPI */
#endif /* TAU_MONITORING */

#endif /* TAU_UNIFY */
