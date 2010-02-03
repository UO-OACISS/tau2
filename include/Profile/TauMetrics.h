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

/**
 * Sets the user definable clock
 */ 
void metric_write_userClock(int tid, double value);
