/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File            : TauCollate.h                                     **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
**	Documentation	: See http://tau.uoregon.edu                       **
**                                                                         **
**      Description     : Profile merging                                  **
**                                                                         **
****************************************************************************/

#ifndef _TAU_COLLATE_H_
#define _TAU_COLLATE_H_

// To allow the use of Tau_unify_object_t in the interface.
#include "TauUnify.h"

#define NUM_COLLATE_OP_TYPES 2
#define COLLATE_OP_BASIC 0
#define COLLATE_OP_DERIVED 1

#define NUM_COLLATE_STEPS 4
#define NUM_STAT_TYPES 6

/* For Internal TAU C++ use only */
typedef enum {
  step_min,
  step_max,
  step_sum,
  step_sumsqr
} collate_step;

typedef enum {
  stat_mean_all,
  stat_mean_exist,
  stat_stddev_all,
  stat_stddev_exist,
  stat_min_exist,
  stat_max_exist  
} stat_derived_type;

extern "C" const int collate_num_op_items[NUM_COLLATE_OP_TYPES];
extern "C" const char *collate_step_names[NUM_COLLATE_STEPS];
extern "C" const char *stat_names[NUM_STAT_TYPES];
extern "C" const char **collate_op_names[NUM_COLLATE_OP_TYPES];

/* Modular Internal Operation headers */
void Tau_collate_allocateFunctionBuffers(double ****excl, double ****incl,
					 double ***numCalls, double ***numSubr,
					 int numEvents,
					 int numMetrics,
					 int collateOpType);
void Tau_collate_allocateAtomicBuffers(double ***atomicMin, double ***atomicMax,
				       double ***atomicSum, double ***atomicMean,
				       double ***atomicSumSqr,
				       int numEvents,
				       int collateOpType);
void Tau_collate_allocateUnitFunctionBuffer(double ***excl, double ***incl,
					    double **numCalls, double **numSubr,
					    int numEvents, 
					    int numMetrics);
void Tau_collate_allocateUnitAtomicBuffer(double **atomicMin, double **atomicMax,
					  double **atomicSum, double **atomicMean,
					  double **atomicSumSqr,
					  int numEvents);

void Tau_collate_freeFunctionBuffers(double ****excl, double ****incl,
				     double ***numCalls, double ***numSubr,
				     int numMetrics,
				     int collateOpType);
void Tau_collate_freeAtomicBuffers(double ***atomicMin, double ***atomicMax,
				   double ***atomicSum, double ***atomicMean,
				   double ***atomicSumSqr,
				   int collateOpType);
void Tau_collate_freeUnitFunctionBuffer(double ***excl, double ***incl,
					double **numCalls, double **numSubr,
					int numMetrics);
void Tau_collate_freeUnitAtomicBuffer(double **atomicMin, double **atomicMax,
				      double **atomicSum, double **atomicMean,
				      double **atomicSumSqr);

void Tau_collate_get_total_threads_MPI(Tau_unify_object_t *functionUnifier, int *globalNumThreads, 
				   int **numEventThreads,
				   int numItems, int *globalmap, bool isAtomic);
void Tau_collate_get_total_threads_SHMEM(Tau_unify_object_t *functionUnifier, int *globalNumThreads, 
				   int **numEventThreads,
				   int numItems, int *globalmap, bool isAtomic);

void Tau_collate_compute_atomicStatistics_MPI(Tau_unify_object_t *atomicUnifier,
					  int *globalEventMap, int numItems,
					  int globalNumThreads, int *numEventThreads,
					  double ***gAtomicMin, double ***gAtomicMax,
					  double ***gAtomicSum, double ***gAtomicMean,
					  double ***gAtomicSumSqr,
					  double ***sAtomicMin, double ***sAtomicMax,
					  double ***sAtomicSum, double ***sAtomicMean,
					  double ***sAtomicSumSqr);
void Tau_collate_compute_atomicStatistics_SHMEM(Tau_unify_object_t *atomicUnifier,
					  int *globalEventMap, int numItems,
					  int globalNumThreads, int *numEventThreads,
					  double ***gAtomicMin, double ***gAtomicMax,
					  double ***gAtomicSum, double ***gAtomicMean,
					  double ***gAtomicSumSqr,
					  double ***sAtomicMin, double ***sAtomicMax,
					  double ***sAtomicSum, double ***sAtomicMean,
					  double ***sAtomicSumSqr);
void Tau_collate_compute_statistics_MPI(Tau_unify_object_t *functionUnifier,
				    int *globalmap, int numItems, 
				    int globalNumThreads, 
				    int *numEventThreads,
				    double ****gExcl, double ****gIncl,
				    double ***gNumCalls, double ***gNumSubr,
				    double ****sExcl, double ****sIncl,
				    double ***sNumCalls, double ***sNumSubr);
void Tau_collate_compute_statistics_SHMEM(Tau_unify_object_t *functionUnifier,
				    int *globalmap, int numItems, 
				    int globalNumThreads, 
				    int *numEventThreads,
				    double ****gExcl, double ****gIncl,
				    double ***gNumCalls, double ***gNumSubr,
				    double ****sExcl, double ****sIncl,
				    double ***sNumCalls, double ***sNumSubr);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  /* API */
  int Tau_collate_writeProfile();


#ifdef __cplusplus
}
#endif /* __cplusplus */



#endif /* _TAU_COLLATE_H_ */
