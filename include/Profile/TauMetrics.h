/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2009  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TauMetrics.cpp		        	   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Handles Metrics                                  **
**                                                                         **
****************************************************************************/




#ifndef _TAU_METRICS_H_
#define _TAU_METRICS_H_


/**
 * Sets the user definable clock
 */ 
void metric_write_userClock(int tid, double value);


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */



int TauMetrics_init();

void TauMetrics_getMetrics(int tid, double values[]);

const char *TauMetrics_getMetricName(int metric);
int TauMetrics_getMetricUsed(int metric);


/**
 * Trigger atomic events for each counter
 */
void TauMetrics_triggerAtomicEvents(unsigned long long timestamp, double *values, int tid);



/**
 * Returns a duplicated list of counter names, and writes the number of counters in numCounters
 */
void TauMetrics_getCounterList(const char ***counterNames, int *numCounters);


/**
 * Returns the index of the trace metric
 */
double TauMetrics_getTraceMetricIndex();

/**
 * Returns the index of the trace metric
 */
double TauMetrics_getTraceMetricValue(int tid);


x_uint64 TauMetrics_getInitialTimeStamp();

x_uint64 TauMetrics_getTimeOfDay();



#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_METRICS_H_ */
