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

int TauMetrics_init();

static void metricv_add(char* name);
static void reorder_metrics();


#define TAU_MAX_METRICS 25





static const char* metricv[TAU_MAX_METRICS];
static int nmetrics = 0;

static void metricv_add(char* name) {
  if (nmetrics >= TAU_MAX_METRICS) {
    fprintf (stderr, "Number of counters exceeds TAU_MAX_METRICS\n");
  } else {
    metricv[nmetrics] = strdup(name);
    nmetrics++;
  }
}


static void reorder_metrics() {


}


int TauMetrics_init() {
  const char *token;

  printf ("TM: Initializing Metrics\n");
  const char *taumetrics = getenv ("TAU_METRICS");

  if (taumetrics && strlen(taumetrics)==0) {
    taumetrics = NULL;
  }

  if (taumetrics) {
    char *metrics = strdup(taumetrics);
    token = strtok(metrics, ":");
    while (token) {
      printf ("got token: %s\n", token);
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

  printf ("TM: Done Initializing Metrics\n");
  for (int i=0; i<nmetrics; i++) {
    printf ("got: %s\n", metricv[i]);
  }


  return 0;
}
