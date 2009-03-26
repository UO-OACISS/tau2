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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <tau_internal.h>
#include <Profile/Profiler.h>

int TauMetrics_init();

static void metricv_add(const char* name);
static void reorder_metrics();
static void read_env_vars();
static void initialize_functionArray();

#define TAU_MAX_METRICS 25


// Global Variable holding the number of counters
int Tau_Global_numCounters = -1;





typedef void (*function)(int, int, double[]);



static const char* metricv[TAU_MAX_METRICS];
static int nmetrics = 0;

// nfunctions can be different from nmetrics because 
// a single call to PAPI can provide several metrics
static int nfunctions = 0; 

// traceMetric in the index used for the trace metric (might not be zero)
static int traceMetric = 0;

// array of function pointers used to get metric data
static function functionArray[TAU_MAX_METRICS];

void metric_read_nullClock(int tid, int idx, double values[]);
void metric_read_logicalClock(int tid, int idx, double values[]);
void metric_read_gettimeofday(int tid, int idx, double values[]);
void metric_read_bgtimers(int tid, int idx, double values[]);
void metric_read_craytimers(int tid, int idx, double values[]);
void metric_read_cputime(int tid, int idx, double values[]);
void metric_read_messagesize(int tid, int idx, double values[]);
void metric_read_papivirtual(int tid, int idx, double values[]);
void metric_read_papiwallclock(int tid, int idx, double values[]);
void metric_read_papi(int tid, int idx, double values[]);



static void metricv_add(const char* name) {
  if (nmetrics >= TAU_MAX_METRICS) {
    fprintf (stderr, "Number of counters exceeds TAU_MAX_METRICS\n");
  } else {
    metricv[nmetrics] = strdup(name);
    nmetrics++;
  }
}

/* This routine will reorder the metrics so that the PAPI ones all come last */
static void reorder_metrics() {
  const char* newmetricv[TAU_MAX_METRICS];
  int idx=0;

  for (int i=0; i<nmetrics; i++) {
    if (strncmp("PAPI", metricv[i], 4) != 0) {
      newmetricv[idx++] = metricv[i];
    }
  }

  for (int i=0; i<nmetrics; i++) {
    if (strncmp("PAPI", metricv[i], 4) == 0) {
      newmetricv[idx++] = metricv[i];
    }
  }

  for (int i=0; i<nmetrics; i++) {
    if (strcmp(newmetricv[i],metricv[0]) == 0) {
      traceMetric = i;
    }
  }

  for (int i=0; i<nmetrics; i++) {
    metricv[i] = newmetricv[i];
  }
}



static void read_env_vars() {
  const char *token;
  const char *taumetrics = getenv ("TAU_METRICS");

  if (taumetrics && strlen(taumetrics)==0) {
    taumetrics = NULL;
  }

  if (taumetrics) {
    char *metrics = strdup(taumetrics);
    token = strtok(metrics, ":");
    while (token) {
      metricv_add(token);
      token = strtok(NULL, ":");
    }
  } else {
    char counterName[256];
    for (int i=1; i<26; i++) {
      sprintf (counterName, "COUNTER%d", i);
      char *metric = getenv(counterName);
      if (metric && strlen(metric)==0) {
	metric = NULL;
      }
      if (metric) {
	metricv_add(metric);
      }
    }

    if (nmetrics == 0) {
      metricv_add("GET_TIME_OF_DAY");
    }
  }
}


static int compareMetricString(const char *one, const char *two) {
  if (strcmp(one, two) == 0) {
    return 1;
  } else {
    return 0;
  }
}


static void initialize_functionArray() {
  int pos = 0;
  for (int i=0; i<nmetrics; i++) {
    if (compareMetricString(metricv[i],"LOGICAL_CLOCK")) {
      functionArray[pos++] = metric_read_logicalClock;
    } else if (compareMetricString(metricv[i],"GET_TIME_OF_DAY")){
      functionArray[pos++] = metric_read_gettimeofday;
    } else if (compareMetricString(metricv[i],"CPU_TIME")){
      functionArray[pos++] = metric_read_cputime;
    } else if (compareMetricString(metricv[i],"BGL_TIMERS")){
      functionArray[pos++] = metric_read_bgtimers;
    } else if (compareMetricString(metricv[i],"BGP_TIMERS")){
      functionArray[pos++] = metric_read_bgtimers;
    } else if (compareMetricString(metricv[i],"CRAY_TIMERS")){
      functionArray[pos++] = metric_read_craytimers;
    } else if (compareMetricString(metricv[i],"TAU_MPI_MESSAGE_SIZE")){
      functionArray[pos++] = metric_read_messagesize;
    } else if (compareMetricString(metricv[i],"P_WALL_CLOCK_TIME")){
      functionArray[pos++] = metric_read_papiwallclock;
    } else if (compareMetricString(metricv[i],"P_VIRTUAL_TIME")){
      functionArray[pos++] = metric_read_papivirtual;
    } else {
      if (strncmp("PAPI", metricv[i], 4) != 0) { /* PAPI handled separately */
	fprintf (stderr, "TAU: Error: Unknown metric: %s\n", metricv[i]);
	functionArray[pos++] = metric_read_nullClock;
      }
    }
  }

  int usingPAPI=0;
  for (int i=0; i<nmetrics; i++) {
      if (strncmp("PAPI", metricv[i], 4) == 0) {
	usingPAPI = 1;
	if (strstr(metricv[i],"PAPI") != NULL) {
	  char *metricString = strdup(metricv[i]);

	  if (strstr(metricString,"NATIVE") != NULL) {
	    /* Fix the name for a native event */
	    int idx = 0;
	    while (metricString[12+idx]!='\0') {
	      metricString[idx] = metricString[12+idx];
	      idx++;
	    }
	    metricString[idx]='\0';
	  }
	  
	  int counterID = PapiLayer::addCounter(metricString);
	  free (metricString);
	}
      }
  }

  if (usingPAPI) {
    functionArray[pos++] = metric_read_papi;
  }

  nfunctions = pos;
}



const char *TauMetrics_getMetricName(int metric) {
  return metricv[metric];
}

int TauMetrics_getMetricUsed(int metric) {
  if (metric < nmetrics) {
    return 1;
  } else {
    return 0;
  }
}


void TauMetrics_getMetrics(int tid, double values[]) {
  for (int i=0; i<nfunctions; i++) {
    functionArray[i](tid, i, values);
  }
}


int TauMetrics_init() {


  read_env_vars();
  reorder_metrics();
  initialize_functionArray();


  Tau_Global_numCounters = nmetrics;
  return 0;
}
