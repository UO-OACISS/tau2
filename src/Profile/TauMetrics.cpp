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

int TauMetrics_init();

static void metricv_add(const char* name);
static void reorder_metrics();
static void read_env_vars();
static void initialize_functionArray();

#define TAU_MAX_METRICS 25


typedef void (*function)(int, int, double[]);



static const char* metricv[TAU_MAX_METRICS];
static int nmetrics = 0;
static int nfunctions = 0; // nfunctions can be different from nmetrics because only one call to PAPI can provide several metrics
static int traceMetric = 0;
static function functionArray[TAU_MAX_METRICS];

static void metric_read_logicalClock(int tid, int idx, double values[]);
static void metric_read_gettimeofday(int tid, int idx, double values[]);





static void metric_read_logicalClock(int tid, int idx, double values[]) {
  static long long value = 0;
  values[idx] = value++;
}

static void metric_read_gettimeofday(int tid, int idx, double values[]) {
  struct timeval tp;
  gettimeofday (&tp, 0);
  values[idx] = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
}



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
    if (strncmp("PAPI", metricv[i], 4)!=0) {
      newmetricv[idx++] = metricv[i];
    }
  }

  for (int i=0; i<nmetrics; i++) {
    if (strncmp("PAPI", metricv[i], 4)==0) {
      newmetricv[idx++] = metricv[i];
    }
  }

  for (int i=0; i<nmetrics; i++) {
    if (strcmp(newmetricv[i],metricv[0])==0) {
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
//       printf ("got token: %s\n", token);
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
  if (strcmp(one, two)==0) {
    return 1;
  } else {
    return 0;
  }
}


static void initialize_functionArray() {
  for (int i=0; i<nmetrics; i++) {
    if (compareMetricString(metricv[i],"LOGICAL_CLOCK")) {
      functionArray[i] = metric_read_logicalClock;
    } else if (compareMetricString(metricv[i],"MET_TIME_OF_DAY")){
      functionArray[i] = metric_read_gettimeofday;
    }
  }
}



const char *TauMetrics_getMetricName(int metric) {
  printf ("returning %d: %s\n", metric, metricv[metric]);
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


extern int Tau_Global_numCounters;


int TauMetrics_init() {

  printf ("TM: Initializing Metrics\n");

  read_env_vars();
  reorder_metrics();
  initialize_functionArray();

  for (int i=0; i<nmetrics; i++) {
    printf ("got: %s\n", metricv[i]);
    printf ("got: %s\n", TauMetrics_getMetricName(i));
  }
  printf ("trace metric is %d: %s\n", traceMetric, metricv[traceMetric]);

  nfunctions = nmetrics;
  Tau_Global_numCounters = nmetrics;

  printf ("TM: Done Initializing Metrics\n");
  printf ("nmetrics=%d\n", nmetrics);
  return 0;
}
