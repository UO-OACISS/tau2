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

#include <assert.h>

#define DEBUG

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

/*********************************************************************
 * Write a profile with data from all nodes/threads
 ********************************************************************/
extern "C" int Tau_collate_writeProfile() {
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

  // Unify events
  FunctionEventLister *functionEventLister = new FunctionEventLister();
  Tau_unify_object_t *functionUnifier = Tau_unify_unifyEvents(functionEventLister);
  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  Tau_unify_object_t *atomicUnifier = Tau_unify_unifyEvents(atomicEventLister);

  TAU_MPI_DEBUG0 ("Found %d total regions\n", functionUnifier->globalNumItems);

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
  double **excl = (double **) TAU_UTIL_MALLOC(sizeof(double *) * Tau_Global_numCounters);
  double **incl = (double **) TAU_UTIL_MALLOC(sizeof(double *) * Tau_Global_numCounters);
  for (int m=0; m<Tau_Global_numCounters; m++) {
    excl[m] = (double *) TAU_UTIL_MALLOC(sizeof(double) * numItems);
    incl[m] = (double *) TAU_UTIL_MALLOC(sizeof(double) * numItems);
  }
  int *numCalls = (int *) TAU_UTIL_MALLOC(sizeof(int) * numItems);
  int *numSubr = (int *) TAU_UTIL_MALLOC(sizeof(int) * numItems);

  // we generate statistics in 4 steps
  for (int s=0; s < num_collate_steps; s++) {

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

    // create output buffers
    double **outExcl, **outIncl;
    int *outNumCalls, *outNumSubr;
    outExcl = (double **) TAU_UTIL_MALLOC(sizeof(double *) * Tau_Global_numCounters);
    outIncl = (double **) TAU_UTIL_MALLOC(sizeof(double *) * Tau_Global_numCounters);

    if (rank == 0) {
      for (int m=0; m<Tau_Global_numCounters; m++) {
	outExcl[m] = (double *) TAU_UTIL_CALLOC(sizeof(double) * numItems);
	outIncl[m] = (double *) TAU_UTIL_CALLOC(sizeof(double) * numItems);
      }
      outNumCalls = (int *) TAU_UTIL_CALLOC(sizeof(int) * numItems);
      outNumSubr = (int *) TAU_UTIL_CALLOC(sizeof(int) * numItems);
    }

    // reduce data to rank 0
    for (int m=0; m<Tau_Global_numCounters; m++) {
      PMPI_Reduce (excl[m], outExcl[m], numItems, MPI_DOUBLE, collate_op[s], 0, MPI_COMM_WORLD);
      PMPI_Reduce (incl[m], outIncl[m], numItems, MPI_DOUBLE, collate_op[s], 0, MPI_COMM_WORLD);
    }
    PMPI_Reduce (numCalls, outNumCalls, numItems, MPI_INT, collate_op[s], 0, MPI_COMM_WORLD);
    PMPI_Reduce (numSubr, outNumSubr, numItems, MPI_INT, collate_op[s], 0, MPI_COMM_WORLD);


    if (rank == 0) {
      printf ("\n----- data for statistic: %s\n", collate_step_name[s]);
      for (int i=0; i<numItems; i++) {
	printf ("[id=%2d] incl=%9.16G excl=%9.16G numcalls=%9d numsubr=%9d : %s\n", i, 
		outIncl[0][i], outExcl[0][i], outNumCalls[i], outNumSubr[i], functionUnifier->globalStrings[i]);

      }      
    }
  }


  if (rank == 0) {
    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: Collating Complete, duration = %.4G seconds\n", ((double)((double)end-start))/1000000.0f);
  }


  MPI_Op_free(&min_op);
  MPI_Op_free(&sum_op);


  // temp: write regular profiles too, for comparison
  int tid = 0;
  TauProfiler_DumpData(false, tid, "profile");


  return 0;
}

#endif /* TAU_EXP_UNIFY */
#endif /* TAU_MPI */
