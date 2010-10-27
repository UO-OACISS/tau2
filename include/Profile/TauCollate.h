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

#include <TauUnify.h>

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
  stat_stddev_exist
} stat_derived_type;

const int NUM_COLLATE_STEPS = 4;
const int NUM_STAT_TYPES = 4;

/* Modular Internal Operation headers */
void Tau_collate_allocateBuffers(double ***excl, double ***incl, 
				 int **numCalls, int **numSubr, 
				 int numItems);
void Tau_collate_allocateBuffers(double ***excl, double ***incl, 
				 double **numCalls, double **numSubr, 
				 int numItems);
void Tau_collate_freeBuffers(double ***excl, double ***incl,
			     int **numCalls, int **numSubr);
void Tau_collate_freeBuffers(double ***excl, double ***incl,
			     double **numCalls, double **numSubr);

void Tau_collate_get_total_threads(int *globalNumThreads, 
				   int **numEventThreads,
				   int numItems, int *globalmap);
void Tau_collate_compute_statistics(Tau_unify_object_t *functionUnifier,
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
