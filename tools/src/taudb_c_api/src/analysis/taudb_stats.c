#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

TAUDB_TIMER_CALLPATH* taudb_find_main_timer_callpath(TAUDB_TRIAL* trial, TAUDB_THREAD* thread, TAUDB_METRIC* metric) {
  // validate inputs
  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }
  if (thread == NULL) {
    fprintf(stderr, "Error: thread parameter null. Please provide a valid thread.\n");
    return NULL;
  }
  if (metric == NULL) {
    fprintf(stderr, "Error: metric parameter null. Please provide a valid metric.\n");
    return NULL;
  }

  double max_inclusive = 0.0;
  char* timestamp = NULL; // for now
  TAUDB_TIMER_CALLPATH *timer_callpath, *tmp, *found_callpath = NULL;

  // iterate over timer callpaths in the thread, and find the highest inclusive value
  HASH_ITER(hh2, trial->timer_callpaths_by_name, timer_callpath, tmp) {
    // find the measurement for this thread - it may not exist
    TAUDB_TIMER_CALL_DATA* timer_call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, thread, timestamp);
	if (timer_call_data != NULL) {
	  // find the measurement for this metric - it might not exist
      TAUDB_TIMER_VALUE *timer_value;
      //HASH_FIND(hh, timer_call_data->timer_values, metric->name, strlen(metric->name), timer_value);
      timer_value = taudb_get_timer_value(timer_call_data, metric);
      if (timer_value != NULL) {
        if (timer_value->inclusive > max_inclusive) {
          max_inclusive = timer_value->inclusive;
	      found_callpath = timer_callpath;
	    }
      }
	}
  }
  // there are cases where we find nothing. in which case, return the first one.
  if (found_callpath == NULL) {
    found_callpath = trial->timer_callpaths_by_name;
  }
  //printf("Main Function: %s, %f (%s)\n", found_callpath->name, max_inclusive, metric->name);
  return found_callpath;
}

void taudb_compute_percentages(TAUDB_TRIAL* trial, TAUDB_THREAD* thread, TAUDB_METRIC* metric, TAUDB_TIMER_CALLPATH* main_timer_callpath) {
  char* timestamp = NULL; // FOR NOW
  // given the main timer_callpath, find the inclusive value for this metric on this thread
  TAUDB_TIMER_CALL_DATA* timer_call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, main_timer_callpath, thread, timestamp);
  TAUDB_TIMER_VALUE *main_timer_value;
  //HASH_FIND(hh, timer_call_data->timer_values, metric->name, strlen(metric->name), main_timer_value);
  main_timer_value = taudb_get_timer_value(timer_call_data, metric);

  TAUDB_TIMER_CALLPATH *timer_callpath, *tmp;

  // iterate over the timer_callpaths
  HASH_ITER(hh2, trial->timer_callpaths_by_name, timer_callpath, tmp) {
    // find if this callpath was executed on this thread
    TAUDB_TIMER_CALL_DATA* timer_call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, thread, timestamp);
	// if so...
	if (timer_call_data != NULL) {
      TAUDB_TIMER_VALUE *timer_value;
	  // find the measurement for this timer callpath on this thread
      //HASH_FIND(hh, timer_call_data->timer_values, metric->name, strlen(metric->name), timer_value);
      timer_value = taudb_get_timer_value(timer_call_data, metric);
      if (timer_value != NULL) {
	    // ...and compute the inclusive and exclusive percentages.
	    timer_value->inclusive_percentage = (timer_value->inclusive / main_timer_value->inclusive) * 100.0;
	    timer_value->exclusive_percentage = (timer_value->exclusive / main_timer_value->inclusive) * 100.0;
	    //printf("%d, incl: %f, excl: %f, %s\n", thread->index, timer_value->inclusive, timer_value->exclusive, timer_callpath->name);
	    //printf("%d, incl: %f, excl: %f, %s\n", thread->index, timer_value->inclusive_percentage, timer_value->exclusive_percentage, timer_callpath->name);
      }
	}
  }
}

TAUDB_THREAD* taudb_create_stat_threads(TAUDB_TRIAL* trial) {
  TAUDB_THREAD* stat_threads = taudb_create_threads(7);
  stat_threads[0].node_rank = TAUDB_MEAN_WITHOUT_NULLS;
  stat_threads[0].context_rank = TAUDB_MEAN_WITHOUT_NULLS;
  stat_threads[0].thread_rank = TAUDB_MEAN_WITHOUT_NULLS;
  stat_threads[0].index = TAUDB_MEAN_WITHOUT_NULLS;
  stat_threads[0].trial = trial;
  stat_threads[1].node_rank = TAUDB_TOTAL;
  stat_threads[1].context_rank = TAUDB_TOTAL;
  stat_threads[1].thread_rank = TAUDB_TOTAL;
  stat_threads[1].index = TAUDB_TOTAL;
  stat_threads[1].trial = trial;
  stat_threads[2].node_rank = TAUDB_STDDEV_WITHOUT_NULLS;
  stat_threads[2].context_rank = TAUDB_STDDEV_WITHOUT_NULLS;
  stat_threads[2].thread_rank = TAUDB_STDDEV_WITHOUT_NULLS;
  stat_threads[2].index = TAUDB_STDDEV_WITHOUT_NULLS;
  stat_threads[2].trial = trial;
  stat_threads[3].node_rank = TAUDB_MIN;
  stat_threads[3].context_rank = TAUDB_MIN;
  stat_threads[3].thread_rank = TAUDB_MIN;
  stat_threads[3].index = TAUDB_MIN;
  stat_threads[3].trial = trial;
  stat_threads[4].node_rank = TAUDB_MAX;
  stat_threads[4].context_rank = TAUDB_MAX;
  stat_threads[4].thread_rank = TAUDB_MAX;
  stat_threads[4].index = TAUDB_MAX;
  stat_threads[4].trial = trial;
  stat_threads[5].node_rank = TAUDB_MEAN_WITH_NULLS;
  stat_threads[5].context_rank = TAUDB_MEAN_WITH_NULLS;
  stat_threads[5].thread_rank = TAUDB_MEAN_WITH_NULLS;
  stat_threads[5].index = TAUDB_MEAN_WITH_NULLS;
  stat_threads[5].trial = trial;
  stat_threads[6].node_rank = TAUDB_STDDEV_WITH_NULLS;
  stat_threads[6].context_rank = TAUDB_STDDEV_WITH_NULLS;
  stat_threads[6].thread_rank = TAUDB_STDDEV_WITH_NULLS;
  stat_threads[6].index = TAUDB_STDDEV_WITH_NULLS;
  stat_threads[6].trial = trial;

  TAUDB_METRIC *metric, *tmp_metric;
  TAUDB_TIMER_CALLPATH *timer_callpath, *tmp_timer_callpath;
  TAUDB_TIMER_CALL_DATA *timer_call_data;
  TAUDB_TIMER_VALUE *timer_value;
  int i;

  // iterate over our new threads...
  for (i = 0 ; i < 7 ; i++) {
    // ...iterate over all timer_callpaths...
    HASH_ITER(hh2, trial->timer_callpaths_by_name, timer_callpath, tmp_timer_callpath) {
      // and create timer_call_data objects for each callpath
      timer_call_data = taudb_create_timer_call_data(1);
	  timer_call_data->key.thread = &(stat_threads[i]);
	  timer_call_data->key.timer_callpath = timer_callpath;
	  timer_call_data->key.timestamp = NULL; // FOR NOW
	  timer_call_data->calls = 0;
	  timer_call_data->subroutines = 0;
	  taudb_add_timer_call_data_to_trial(trial, timer_call_data);
	  // for each metric, create a measurement, too...
      HASH_ITER(hh2, trial->metrics_by_name, metric, tmp_metric) {
	    timer_value = taudb_create_timer_values(1);
		timer_value->metric = metric;
		timer_value->inclusive = 0.0;
		timer_value->exclusive = 0.0;
		// ...and add it to our timer_call_data
	    taudb_add_timer_value_to_timer_call_data(timer_call_data, timer_value);
		timer_value = NULL;
      }
	  timer_call_data = NULL;
    }
  }

  return stat_threads;
}

void taudb_compute_statistics(TAUDB_TRIAL* trial) {
  TAUDB_METRIC *metric, *tmp_metric;
  TAUDB_THREAD *thread, *tmp_thread;
  TAUDB_TIMER_CALLPATH *timer_callpath, *tmp_timer_callpath;
  TAUDB_TIMER_CALL_DATA *timer_call_data, *mean1, *total, *stddev1, *min, *max, *mean2, *stddev2;
  TAUDB_TIMER_VALUE *timer_value, *total_timer_value, *min_timer_value, *max_timer_value, *mean_timer_value, *stddev_timer_value;
  TAUDB_THREAD* stat_threads = taudb_create_stat_threads(trial);
  int total_threads = HASH_CNT(hh,trial->threads);

  // iterate over all of the timer_callpaths...
  HASH_ITER(hh2, trial->timer_callpaths_by_name, timer_callpath, tmp_timer_callpath) {
    int threads_calling = 0; // for computing the different types of mean, stddev
	char* timestamp = NULL;

	// for this timer_callpath, create a timer_call_data for each statistic
    mean1 = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, &(stat_threads[0]), timestamp);
    total = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, &(stat_threads[1]), timestamp);
    stddev1 = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, &(stat_threads[2]), timestamp);
    min = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, &(stat_threads[3]), timestamp);
    max = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, &(stat_threads[4]), timestamp);
    mean2 = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, &(stat_threads[5]), timestamp);
    stddev2 = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, &(stat_threads[6]), timestamp);

    // ...iterate over the threads...
    HASH_ITER(hh, trial->threads, thread, tmp_thread) {
	  // get the timer_call_data for this timer_callpath on this thread
      timer_call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, thread, timestamp);
	  if (timer_call_data != NULL) {
	    // if this thread called this timer_callpath, then increment
	    if (timer_call_data->calls > 0) {
	      threads_calling = threads_calling + 1;
	    }
	    // update the total values
	    total->calls = total->calls + timer_call_data->calls;
	    total->subroutines = total->subroutines + timer_call_data->subroutines;
	    // update the min values - but only for non-zero values
	    if (min->calls == 0) {
	      min->calls = timer_call_data->calls;
	    } else {
	      if ((timer_call_data->calls > 0) && (min->calls > timer_call_data->calls)) {
            min->calls = timer_call_data->calls;
		  }
	    }
	    if (min->subroutines == 0) {
	      min->subroutines = timer_call_data->subroutines;
	    } else {
	      if ((timer_call_data->subroutines > 0) && (min->subroutines > timer_call_data->subroutines)) {
            min->subroutines = timer_call_data->subroutines;
		  }
	    }
	    // update the max values
	    if (max->calls < timer_call_data->calls) {
	      max->calls = timer_call_data->calls;
        }
	    if (max->subroutines < timer_call_data->subroutines) {
	      max->subroutines = timer_call_data->subroutines;
        }
	  
	    // now, iterate over the metrics so we can handle the measurements
        HASH_ITER(hh2, trial->metrics_by_name, metric, tmp_metric) {
	      timer_value = taudb_get_timer_value(timer_call_data, metric);
		  // update the total value
	      total_timer_value = taudb_get_timer_value(total, metric);
          total_timer_value->inclusive = total_timer_value->inclusive + timer_value->inclusive;
          total_timer_value->exclusive = total_timer_value->exclusive + timer_value->exclusive;
		  // update the min value - but only for non-zero values
	      min_timer_value = taudb_get_timer_value(min, metric);
   	      if (min_timer_value->inclusive == 0) {
	        min_timer_value->inclusive = timer_value->inclusive;
	        min_timer_value->exclusive = timer_value->exclusive;
	      } else {
	        if ((timer_value->inclusive > 0) && (min_timer_value->inclusive > timer_value->inclusive)) {
              min_timer_value->inclusive = timer_value->inclusive;
		    }
	        if ((timer_value->exclusive > 0) && (min_timer_value->exclusive > timer_value->exclusive)) {
              min_timer_value->exclusive = timer_value->exclusive;
		    }
	      }
		  // update the max value
	      max_timer_value = taudb_get_timer_value(max, metric);
	      if (max_timer_value->inclusive < timer_value->inclusive) {
	        max_timer_value->inclusive = timer_value->inclusive;
          }
	      if (max_timer_value->exclusive < timer_value->exclusive) {
	        max_timer_value->exclusive = timer_value->exclusive;
          }
        }
      }
    }

	// great! we computed total, min and max. now compute the means

	// first, the mean without nulls
	if (threads_calling == 0) {
	  mean1->calls = 0;
	  mean1->subroutines = 0;
	} else {
	  mean1->calls = total->calls / threads_calling;
	  mean1->subroutines = total->subroutines / threads_calling;
	}

	// second, the mean with nulls
	if (total_threads == 0) {
	  mean1->calls = 0;
	  mean1->subroutines = 0;
	} else {
	  mean2->calls = total->calls / total_threads;
	  mean2->subroutines = total->subroutines / total_threads;
	}

    // iterate over the metrics and handle the timer values
    HASH_ITER(hh2, trial->metrics_by_name, metric, tmp_metric) {
	  // update the total value
	  total_timer_value = taudb_get_timer_value(total, metric);
	  mean_timer_value = taudb_get_timer_value(mean1, metric);
	  if (threads_calling == 0) {
	    mean_timer_value->inclusive = 0;
	    mean_timer_value->exclusive = 0;
	  } else {
	    mean_timer_value->inclusive = total_timer_value->inclusive / threads_calling;
	    mean_timer_value->exclusive = total_timer_value->exclusive / threads_calling;
	  }
	  mean_timer_value = taudb_get_timer_value(mean2, metric);
	  if (total_threads == 0) {
	    mean_timer_value->inclusive = 0;
	    mean_timer_value->exclusive = 0;
	  } else {
	    mean_timer_value->inclusive = total_timer_value->inclusive / total_threads;
	    mean_timer_value->exclusive = total_timer_value->exclusive / total_threads;
	  }
	}

	// now, compute the stddevs
	double tmpVal = 0;
    // ...iterate over the threads...
    HASH_ITER(hh, trial->threads, thread, tmp_thread) {
	  // get the timer_call_data for this timer_callpath on this thread
      timer_call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, thread, timestamp);
	  if (timer_call_data != NULL) {
	    // if this thread called this timer_callpath, then increment
	    if (timer_call_data->calls > 0) {
	      threads_calling = threads_calling + 1;
	    }
	    // update the stddev (variance) values
		tmpVal = mean1->calls - timer_call_data->calls;
	    stddev1->calls = stddev1->calls + (tmpVal * tmpVal);
		tmpVal = mean2->calls - timer_call_data->calls;
	    stddev2->calls = stddev2->calls + (tmpVal * tmpVal);
		tmpVal = mean1->subroutines - timer_call_data->subroutines;
	    stddev1->subroutines = stddev1->subroutines + (tmpVal * tmpVal);
		tmpVal = mean2->subroutines - timer_call_data->subroutines;
	    stddev2->subroutines = stddev2->subroutines + (tmpVal * tmpVal);
        HASH_ITER(hh2, trial->metrics_by_name, metric, tmp_metric) {
	      // update the inclusive and exclusive values
	      timer_value = taudb_get_timer_value(timer_call_data, metric);
	      mean_timer_value = taudb_get_timer_value(mean1, metric);
		  stddev_timer_value = taudb_get_timer_value(stddev1, metric);
		  // inclusive
		  tmpVal = mean_timer_value->inclusive - timer_value->inclusive;
		  stddev_timer_value->inclusive = stddev_timer_value->inclusive + (tmpVal * tmpVal);
		  // exclusive
		  tmpVal = mean_timer_value->exclusive - timer_value->exclusive;
		  stddev_timer_value->exclusive = stddev_timer_value->exclusive + (tmpVal * tmpVal);

	      mean_timer_value = taudb_get_timer_value(mean2, metric);
		  stddev_timer_value = taudb_get_timer_value(stddev2, metric);
		  // inclusive
		  tmpVal = mean_timer_value->inclusive - timer_value->inclusive;
		  stddev_timer_value->inclusive = stddev_timer_value->inclusive + (tmpVal * tmpVal);
		  // exclusive
		  tmpVal = mean_timer_value->exclusive - timer_value->exclusive;
		  stddev_timer_value->exclusive = stddev_timer_value->exclusive + (tmpVal * tmpVal);
		}
	  }
	}

	// take the square roots of the variances
	stddev1->calls = sqrt(stddev1->calls);
	stddev2->calls = sqrt(stddev2->calls);
	stddev1->subroutines = sqrt(stddev1->subroutines);
	stddev2->subroutines = sqrt(stddev2->subroutines);
    HASH_ITER(hh2, trial->metrics_by_name, metric, tmp_metric) {
	  stddev_timer_value = taudb_get_timer_value(stddev1, metric);
	  stddev_timer_value->inclusive = sqrt(stddev_timer_value->inclusive);
	  stddev_timer_value->exclusive = sqrt(stddev_timer_value->exclusive);
	  stddev_timer_value = taudb_get_timer_value(stddev2, metric);
	  stddev_timer_value->inclusive = sqrt(stddev_timer_value->inclusive);
	  stddev_timer_value->exclusive = sqrt(stddev_timer_value->exclusive);
	}
  }

  int i;
  // great! now add our derived threads
  for (i = 0 ; i < 7 ; i++) {
    taudb_add_thread_to_trial(trial, &(stat_threads[i]));
  }

  // now compute the inclusive and exclusive percentages
  HASH_ITER(hh2, trial->metrics_by_name, metric, tmp_metric) {
    HASH_ITER(hh, trial->threads, thread, tmp_thread) {
	  TAUDB_TIMER_CALLPATH* timer_callpath = taudb_find_main_timer_callpath(trial, thread, metric);
      taudb_compute_percentages(trial, thread, metric, timer_callpath);
    }
  }
}




