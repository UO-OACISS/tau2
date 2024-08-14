#include "taudb_internal.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <dirent.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

extern void taudb_trim(char * s);
extern void taudb_parse_tau_profile_file(char* filename, TAUDB_TRIAL* trial);
extern void taudb_parse_tau_profile_function(char* line, TAUDB_TRIAL* trial, TAUDB_METRIC* metric, TAUDB_THREAD* thread);
extern void taudb_parse_tau_profile_counter(char* line, TAUDB_TRIAL* trial, TAUDB_THREAD* thread);
extern TAUDB_THREAD* taudb_parse_tau_profile_thread(char* filename, TAUDB_TRIAL* trial);
extern TAUDB_METRIC* taudb_parse_tau_profile_metric(char* line, TAUDB_TRIAL* trial);
extern TAUDB_TIMER* taudb_create_timer(TAUDB_TRIAL* trial, const char* timer_name);
extern TAUDB_COUNTER* taudb_create_counter(TAUDB_TRIAL* trial, const char* counter_name);
extern TAUDB_TIMER_CALLPATH* taudb_create_timer_callpath(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_TIMER_CALLPATH* parent);
extern TAUDB_TIMER_CALL_DATA* taudb_create_timer_call_datum(TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, int calls, int subroutines);
extern TAUDB_TIMER_VALUE* taudb_create_timer_value(TAUDB_TRIAL* trial, TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_METRIC* metric, int inclusive, int exclusive);
extern TAUDB_COUNTER_VALUE* taudb_create_counter_value(TAUDB_TRIAL* trial, TAUDB_COUNTER* counter, TAUDB_THREAD* thread, TAUDB_TIMER_CALLPATH* timer_callpath, int numevents, double max, double min, double mean, double sumsqr);
extern TAUDB_TIMER_CALLPATH* taudb_process_callpath_timer(TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath);
extern void taudb_process_timer_name(TAUDB_TIMER* timer);
extern void taudb_parse_tau_metadata(TAUDB_TRIAL* trial, TAUDB_THREAD* thread, const char* line);
extern boolean taudb_private_secondary_metadata_from_xml(TAUDB_TRIAL* trial, TAUDB_THREAD* thread, const char* xml);
extern xmlNodePtr taudb_private_find_xml_child_named(xmlNodePtr parent, const char * name);
extern void taudb_consolidate_metadata (TAUDB_TRIAL* trial);

extern void taudb_parse_timer_group_names(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, char* group_names);
extern char* taudb_getline(FILE* infile);

extern void count_profiles(const char* directory_name, int* counts);
extern void process_directory(const char* directory_name, TAUDB_TRIAL* trial);

TAUDB_TRIAL* taudb_parse_tau_profiles(const char* directory_name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_parse_tau_profiles(%s)\n", directory_name);
#endif

  printf("\nWarning: not fully functional. Missing support for:\n");
  printf(" - parsing timer parameters\n\n");

  // validate the config file name
  if (directory_name == NULL || strlen(directory_name) == 0) {
    fprintf(stderr, "ERROR: empty directory name.\n");
    return NULL;
  }

  TAUDB_TRIAL* trial = taudb_create_trials(1);

  int counts[3] = {0,0,0};
  count_profiles(directory_name, counts);
  printf("Profiles found: %d, %d, %d\n", counts[0], counts[1], counts[2]);
  trial->node_count = counts[0];
  trial->contexts_per_node = counts[1];
  trial->threads_per_context = counts[2];
  trial->total_threads = 0;

  process_directory(directory_name, trial);

  printf("Computing Stats...\n");
  taudb_compute_statistics(trial);

  taudb_consolidate_metadata (trial);
  return trial;
}

void process_directory(const char* directory_name, TAUDB_TRIAL* trial) {
  const char* profile_prefix = "profile.";
  const char* papi_prefix = "MULTI__";

  DIR *dp = NULL;
  struct dirent *ep = NULL;
  char profile_file[1024] = {0};
  dp = opendir (directory_name);
  if (dp != NULL) {
    ep = readdir(dp);
    while (ep != NULL) {
      // check for profile.x.x.x files
      if (strncmp(ep->d_name, profile_prefix, 8) == 0) {
        snprintf(profile_file, sizeof(profile_file),  "%s/%s", directory_name, ep->d_name);
#ifdef TAUDB_DEBUG
        printf("Parsing profile file %s...\n", profile_file);
#endif
        taudb_parse_tau_profile_file(profile_file, trial);
        trial->total_threads++;
      } else if (strncmp(ep->d_name, papi_prefix, 7) == 0) {
      // check for MULTI__* directories
        snprintf(profile_file, sizeof(profile_file),  "%s/%s", directory_name, ep->d_name);
        process_directory(profile_file, trial);
      }
      ep = readdir(dp);
    }
    closedir(dp);
  } else {
    fprintf(stderr, "No TAUdb config files found.\n");
  }
}

void count_profiles(const char* directory_name, int* counts) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling count_profiles(%s)\n", directory_name);
#endif

  const char* profile_prefix = "profile.";
  const char* papi_prefix = "MULTI__";
  int last_rank[3] = {-1,-1,-1};
  int rank[3] = {0,0,0};

  DIR *dp = NULL;
  struct dirent *ep = NULL;
  char subdir[1024] = {0};
  dp = opendir (directory_name);
  if (dp != NULL) {
    ep = readdir(dp);
    while (ep != NULL) {
      // check for profile.x.x.x files
      if (strncmp(ep->d_name, profile_prefix, 8) == 0) {
	    // get the node, context, thread values
        char* tmp = strtok(ep->d_name, ".");
	    tmp = strtok(NULL, ".");
        rank[0] = atoi(tmp);
        tmp = strtok(NULL, ".");
        rank[1] = atoi(tmp);
        tmp = strtok(NULL, ".");
	    rank[2] = atoi(tmp);
		// the process rank can start at 0, or can be a pid...
        if (rank[0] != last_rank[0]) {
		  counts[0] = counts[0] + 1;
		  last_rank[0] = rank[0];
		} 
		// thankfully, the context id is from 0
		if (rank[1]+1 > counts[1]) {
		  counts[1] = rank[1] + 1;
		} 
		// thankfully, the thread id is from 0
		if (rank[2]+1 > counts[2]) {
		  counts[2] = rank[2] + 1;
		}
      } else if (strncmp(ep->d_name, papi_prefix, 7) == 0) {
      // check for MULTI__* directories
        snprintf(subdir, sizeof(subdir),  "%s/%s", directory_name, ep->d_name);
	    // recurse
		int tmp_counts[3] = {0,0,0};
        count_profiles(subdir, tmp_counts);
		counts[0] = counts[0] > tmp_counts[0] ? counts[0] : tmp_counts[0];
		counts[1] = counts[1] > tmp_counts[1] ? counts[1] : tmp_counts[1];
		counts[2] = counts[2] > tmp_counts[2] ? counts[2] : tmp_counts[2];
      }
      ep = readdir(dp);
    }
    closedir(dp);
  } else {
    fprintf(stderr, "No TAUdb config files found.\n");
  }
}

void taudb_parse_tau_profile_file(char* filename, TAUDB_TRIAL* trial) {
  // open the file
  FILE* ifp = fopen (filename, "r");
  if (ifp == NULL) {
    fprintf(stderr, "ERROR: could not parse profile file %s\n", filename);
    return;
  }

  TAUDB_THREAD* thread = taudb_parse_tau_profile_thread(filename, trial);

  char* line = NULL;
  boolean functions = FALSE;
  boolean aggregates = FALSE;
  boolean userevents = FALSE;

  TAUDB_METRIC* metric = NULL;

  // parse the config file, one line at a time
  while (!feof(ifp)) {
    line = NULL;
    //fgets(line, MAX_RECORD_LENGTH, ifp);
    line = taudb_getline(ifp);
    taudb_trim(line);
    if (strlen(line) == 0) {
      // ignore empty lines
      continue;
    } else if (strstr(line, "templated_functions") != NULL) {
#ifdef TAUDB_DEBUG_DEBUG
      printf("%s\n", line);
#endif
      // parse the first line
      functions = FALSE;
      aggregates = FALSE;
      userevents = FALSE;
      metric = taudb_parse_tau_profile_metric(line, trial);
    } else if (strstr(line, "# Name Calls Subrs Excl Incl ProfileCalls") != NULL) {
      // parse the first line - metadata!
	  char* start = strstr(line, "<metadata>");
      taudb_parse_tau_metadata(trial, thread, start);
      functions = TRUE;
    } else if (strstr(line, "aggregates") != NULL) {
#ifdef TAUDB_DEBUG
      printf("%s\n", line);
#endif
      // parse the aggregates
      functions = FALSE;
      aggregates = TRUE;
      userevents = FALSE;
    } else if (strstr(line, "userevents") != NULL) {
#ifdef TAUDB_DEBUG
      printf("%s\n", line);
#endif
      // parse the userevents
      functions = FALSE;
      aggregates = FALSE;
      userevents = TRUE;
    } else if (strncmp(line, "#", 1) == 0) {
      // parse any other comment lines
      continue;
    } else {
      if (functions) {
        taudb_parse_tau_profile_function(line, trial, metric, thread);
      } else if (aggregates) {
      } else if (userevents) {
        taudb_parse_tau_profile_counter(line, trial, thread);
      }
    }
	free(line);
  }

  return;
}

TAUDB_THREAD* taudb_parse_tau_profile_thread(char* filename, TAUDB_TRIAL* trial) {
  // create a thread
  TAUDB_THREAD* thread = NULL;
  char* tmp = strtok(filename, ".");
  tmp = strtok(NULL, ".");
  int node = atoi(tmp);
  tmp = strtok(NULL, ".");
  int context = atoi(tmp);
  tmp = strtok(NULL, ".");
  int thr = atoi(tmp);
  int index = (node * trial->contexts_per_node * trial->threads_per_context) +
              (context * trial->threads_per_context) +
  			  (thr);

  thread = taudb_get_thread(trial->threads, index);
  if (thread == NULL) {
    thread = taudb_create_threads(1);
    thread->trial = trial;
    thread->node_rank = node;
    thread->context_rank = context;
    thread->thread_rank = thr;
    thread->index = index;
    //HASH_ADD(hh, trial->threads, index, sizeof(int), thread);
    taudb_add_thread_to_trial(trial, thread);
  }
  return thread;
}

TAUDB_METRIC* taudb_parse_tau_profile_metric(char* line, TAUDB_TRIAL* trial) {
  int timer_count = 0;
  const char* multi = "templated_functions_MULTI_";
  const char* single = "templated_functions_";
  const char* removeme;
  char* index = 0;
  if ((index = strstr(line, multi)) != NULL) {
    removeme = multi;
  }
  else if (strstr(line, single) != NULL) {
    removeme = single;
  }
  // get the number of functions
  char tmp[128] = {0};
  int count_length = strlen(line) - strlen(index);
  strncpy(tmp, line, count_length);
  timer_count = atoi(tmp);
  // get the metric name
  int metric_length = strlen(index) - strlen(removeme);
  index = index + (strlen(removeme));
  strncpy(tmp, index, metric_length);
  taudb_trim(tmp);
#ifdef TAUDB_DEBUG
  printf("%d, %s\n",timer_count, tmp);
#endif

  TAUDB_METRIC* metric = NULL;
  if (trial->metrics_by_name != NULL) {
    metric = taudb_get_metric_by_name(trial->metrics_by_name, tmp);
  }
  if (metric == NULL) {
    metric = taudb_create_metrics(1);
    metric->name = taudb_strdup(tmp);
    metric->derived = 0;
    //HASH_ADD_KEYPTR(hh2, trial->metrics_by_name, metric->name, strlen(metric->name), metric);
    taudb_add_metric_to_trial(trial, metric);
  }
  
  return metric;
}

void taudb_parse_tau_profile_function(char* line, TAUDB_TRIAL* trial, TAUDB_METRIC* metric, TAUDB_THREAD* thread) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("taudb_parse_tau_profile_function: %s\n", line);
#endif
  // it won't be this long, but it will be close.
  const int line_len = strlen(line);
  char* timer_name = calloc(line_len, sizeof(char));
  // likewise, the groups
  const int line_len = strlen(line);
  char* groups = calloc(line_len, sizeof(char));
  int calls = 0, subroutines = 0;
  double inclusive = 0.0, exclusive = 0.0, stddev = 0.0;
  // tokenize by quotes first, to get the function name
  char* tmp = strtok(line, "\"");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    strncpy(timer_name,  tmp, line_len); 
	// as much as I would love to trim the timer names, I can't. :(
	// it causes downstream problems.
	//taudb_trim(timer_name);
  }
  // tokenize by spaces now, to get the rest of the fields
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    calls = atoi(tmp);
  }
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    subroutines = atoi(tmp);
  }
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    exclusive = atof(tmp);
  }
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    inclusive = atof(tmp);
  }
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    stddev = atof(tmp);
  }
  tmp = strtok(NULL, "=");
  tmp = strtok(NULL, "\"");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    strncpy(groups,  tmp, line_len); 
  }
  
  const char* conjunction = " => ";
  TAUDB_TIMER* timer = NULL;
  TAUDB_TIMER_CALLPATH* timer_callpath = NULL;
  TAUDB_TIMER_CALL_DATA* timer_call_data = NULL;
  TAUDB_TIMER_VALUE* timer_value = NULL;
  if ((strstr(timer_name, conjunction)) != NULL) {
    // process callpath profile
    TAUDB_TIMER_CALLPATH tmp_timer_callpath;
	tmp_timer_callpath.id = 0;
	tmp_timer_callpath.name = timer_name;
    timer_callpath = taudb_process_callpath_timer(trial, &tmp_timer_callpath);
  } else {
#ifdef TAUDB_DEBUG_DEBUG
    printf("'%s' %d %d %f %f %f %s\n", timer_name, calls, subroutines, inclusive, exclusive, stddev, groups);
#endif
    timer = taudb_create_timer(trial, timer_name);
    if (strlen(groups) > 0) {
#ifdef TAUDB_DEBUG_DEBUG
	  printf("group: %s\n", groups);
#endif
      taudb_parse_timer_group_names(trial, timer, groups);
    }
    // get/create timer_callpath
    timer_callpath = taudb_create_timer_callpath(trial, timer, NULL);
  }
  // parse parameters, if exist

  // get/create timer_call_data
  timer_call_data = taudb_create_timer_call_datum(trial, timer_callpath, thread, calls, subroutines);
  // get/create timer_values
  timer_value = taudb_create_timer_value(trial, timer_call_data, metric, inclusive, exclusive);
}

TAUDB_TIMER* taudb_create_timer(TAUDB_TRIAL* trial, const char* timer_name) {
  TAUDB_TIMER* timer = NULL;
  if (trial->timers_by_name != NULL) {
#ifdef TAUDB_DEBUG_DEBUG
	printf("Searching for: '%s'\n", timer_name);
#endif
    timer = taudb_get_trial_timer_by_name(trial->timers_by_name, timer_name);
  }
  if (timer == NULL) {
    timer = taudb_create_timers(1);
    timer->name = taudb_strdup(timer_name);
	timer->trial = trial;
    // extract the short name
    // extract the file name
    // extract the line numbers
#ifdef TAUDB_DEBUG_DEBUG
	printf("Adding: '%s'\n", timer->name);
#endif
    taudb_add_timer_to_trial(trial, timer);
  }
  if (timer->short_name == NULL) {
    taudb_process_timer_name(timer);
  }
  return timer;
}

TAUDB_TIMER_CALLPATH* taudb_create_timer_callpath(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_TIMER_CALLPATH* parent) {
  TAUDB_TIMER_CALLPATH* timer_callpath = NULL;
  // get the timer callpath name, parent included
  char* tmp_name = NULL;
  if (parent == NULL) {
    tmp_name = timer->name;
  } else {
    const int size = (sizeof(char))*(strlen(timer_callpath->name) + strlen(parent->name) + 5);
    tmp_name = (char*)malloc(size);
	snprintf(tmp_name, size,  "%s => %s", parent->name, timer->name);
  }
  if (trial->timer_callpaths_by_name != NULL) {
    // does this timer_callpath exist?
    timer_callpath = taudb_get_timer_callpath_by_name(trial->timer_callpaths_by_name, tmp_name);
  }
  // if not, create one
  if (timer_callpath == NULL) {
    timer_callpath = taudb_create_timer_callpaths(1);
    timer_callpath->id = timer->id;
    timer_callpath->timer = timer;
    if (parent == NULL) {
      timer_callpath->name = taudb_strdup(timer->name);
    } else {
	  timer_callpath->name = tmp_name;
	}
	// as much as I would love to trim the timer names, I can't. :(
	// it causes downstream problems.
    // taudb_trim(timer_callpath->name);
    timer_callpath->parent = parent;
    //if (timer->id > 0) {
      //HASH_ADD(hh1, trial->timer_callpaths_by_id, id, sizeof(int), timer_callpath);
    //}
    //HASH_ADD_KEYPTR(hh2, trial->timer_callpaths_by_name, timer_callpath->name, strlen(timer_callpath->name), timer_callpath);
    taudb_add_timer_callpath_to_trial(trial, timer_callpath);
  }
  return timer_callpath;
}


char * taudb_getline(FILE * f)
{
    size_t size = 0;
    size_t len  = 0;
    size_t last = 0;
    char * buf  = NULL;

    do {
        //size += BUFSIZ; /* BUFSIZ is defined as "the optimal read size for this platform" */
        size += 8192; /* BUFSIZ is defined as "the optimal read size for this platform" */
		//printf("Getting %ld bytes...\n", size);
        buf = realloc(buf,size); /* realloc(NULL,n) is the same as malloc(n) */            
        /* Actually do the read. Note that fgets puts a terminal '\0' on the
           end of the string, so we make sure we overwrite this */
        fgets(buf+last,size,f);
        len = strlen(buf);
        last = len - 1;
    } while (!feof(f) && buf[last]!='\n');
    return buf;
}

TAUDB_TIMER_CALL_DATA* taudb_create_timer_call_datum(TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, int calls, int subroutines) {
  TAUDB_TIMER_CALL_DATA* timer_call_data = NULL;
  if (trial->timer_call_data_by_key != NULL) {
    timer_call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, thread, NULL);
  }
  if (timer_call_data == NULL) {
    timer_call_data = taudb_create_timer_call_data(1);
    timer_call_data->id = 0;
    timer_call_data->key.timer_callpath = timer_callpath;
    timer_call_data->key.thread = thread;
    timer_call_data->key.timestamp = NULL;
    timer_call_data->calls = calls;
    timer_call_data->subroutines = calls;
    //HASH_ADD(hh2, trial->timer_call_data_by_key, key, sizeof(TAUDB_TIMER_CALL_DATA_KEY), timer_call_data);
    taudb_add_timer_call_data_to_trial(trial, timer_call_data);
  //} else {
	//printf("TIMER_CALL_DATA: %s\n", timer_call_data->key.timer_callpath->name);
  }
  return timer_call_data;
}

extern TAUDB_TIMER_VALUE* taudb_create_timer_value(TAUDB_TRIAL* trial, TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_METRIC* metric, int inclusive, int exclusive) {
  TAUDB_TIMER_VALUE* timer_value = taudb_create_timer_values(1);
  timer_value->inclusive = inclusive;
  timer_value->exclusive = exclusive;
  timer_value->metric = metric;
  //HASH_ADD_KEYPTR(hh, timer_call_data->timer_values, metric->name, strlen(timer_value->metric->name), timer_value);
  taudb_add_timer_value_to_timer_call_data(timer_call_data, timer_value);
  return timer_value;
}

void taudb_process_timer_name(TAUDB_TIMER* timer) {
  const char* sampleunresolved = "[SAMPLE] UNRESOLVED";
  const char* sample = "[SAMPLE]";
  const char* intermediate = "[INTERMEDIATE]";
  const char* throttled = "[THROTTLED]";
  const char* openmp = "[OpenMP]";
  const char* openmplocation = "[OpenMP location:";
  char* working = taudb_strdup(timer->name);
  // printf("'%s'\n", working);
  // parse the components out of the timer name
  if (strstr(timer->name, sampleunresolved) != NULL) {
    // special case, handle it
	timer->short_name = taudb_strdup(timer->name);
  } else if (strstr(timer->name, throttled) != NULL) {
    // 'MPI_Irecv() [THROTTLED]'
    // special case, handle it
	int length = strlen(timer->name) - 11;
	timer->name[length] = 0; // new terminator
    taudb_process_timer_name(timer);
	strcpy(timer->name, working);
  } else if (strstr(timer->name, sample) != NULL) {
    // '[SAMPLE] uniform_space_dist_ [{/global/u2/k/khuck/src/XGC-1_CPU/./load.F95} {244}]'
    // special case, handle it
	// get the function signature
    char* tmp = strtok(working, "{");
	taudb_trim(tmp);
	timer->short_name = taudb_strdup(tmp);
	// trim the " ["
	timer->short_name[strlen(timer->short_name)-2] = 0;
    tmp = strtok(NULL, "}");
	timer->source_file = taudb_strdup(tmp);
    tmp = strtok(NULL, " {},-");
	timer->line_number = atoi(tmp);
	timer->line_number_end = atoi(tmp);
  } else if (strstr(timer->name, intermediate) != NULL) {
    // '[INTERMEDIATE] paralleldo [OpenMP location: file:/global/u2/k/khuck/src/XGC-1_CPU/pushe2.F95 <72, 150>]'
    // special case, handle it
	strcpy(timer->name, timer->name+15);
    taudb_process_timer_name(timer);
	strcpy(timer->name, working);
  } else if (strstr(timer->name, openmp) != NULL) {
    // 'barrier enter/exit [OpenMP]'
    // special case, handle it
	timer->short_name = taudb_strdup(timer->name);
  } else if (strstr(timer->name, openmplocation) != NULL) {
    // 'paralleldo [OpenMP location: file:/global/u2/k/khuck/src/XGC-1_CPU/pushe2.F95 <72, 150>]'
    // special case, handle it
    char* tmp = strtok(working, " ");
	taudb_trim(tmp);
	timer->short_name = taudb_strdup(tmp);
    tmp = strtok(NULL, ":");
    tmp = strtok(NULL, ":");
    tmp = strtok(NULL, " ");
	timer->source_file = taudb_strdup(tmp);
    tmp = strtok(NULL, " <,>]");
	timer->line_number = atoi(tmp);
    tmp = strtok(NULL, " <,>]");
	timer->line_number_end = atoi(tmp);
  } else if (strstr(timer->name, "[") != NULL) {
    // regular case
	// get the function signature
    char* tmp = strtok(working, "[");
	taudb_trim(tmp);
	timer->short_name = taudb_strdup(tmp);
	// get the filename
    tmp = strtok(NULL, "}");
	timer->source_file = taudb_strdup(tmp+1);
	// get the line and column numbers
    tmp = strtok(NULL, " {},-");
	timer->line_number = atoi(tmp);
    tmp = strtok(NULL, " {},-");
	timer->column_number = atoi(tmp);
    tmp = strtok(NULL, " {},-");
	timer->line_number_end = atoi(tmp);
    tmp = strtok(NULL, " {},-");
	timer->column_number_end = atoi(tmp);
  } else {
    // simple case.
	timer->short_name = taudb_strdup(timer->name);
  }
  free(working);
  return;
}

extern void taudb_parse_tau_metadata(TAUDB_TRIAL* trial, TAUDB_THREAD* thread, const char* line) {
  if (!taudb_private_secondary_metadata_from_xml(trial, thread, line)) {
    printf("Error Parsing metadata: %s\n", line);
  }
}

boolean taudb_private_secondary_metadata_from_xml(TAUDB_TRIAL * trial, TAUDB_THREAD* thread, const char * xml) {
	/* Initialize libxml and verify installed library is compatible with headers used */
	LIBXML_TEST_VERSION;

#ifdef TAUDB_DEBUG_DEBUG
    //printf("parsing xml: %s\n\n", xml);
#endif

	xmlDocPtr doc;
	doc = xmlReadMemory(xml, strlen(xml), "noname.xml", NULL, XML_PARSE_RECOVER | XML_PARSE_NONET);
	if(doc == NULL) {
		fprintf(stderr, "Unable to parse XML metadata\n");
        xmlCleanupParser();
		return FALSE;
	}
	
	xmlNodePtr metadata_tag = taudb_private_find_xml_child_named(xmlDocGetRootElement(doc), "metadata");
	if(metadata_tag == NULL) {
        xmlFreeDoc(doc);
        xmlCleanupParser();
		return FALSE;
	}
	/*
	xmlNodePtr common_profile_attributes_tag = taudb_private_find_xml_child_named(metadata_tag, "CommonProfileAttributes");
	if(common_profile_attributes_tag == NULL) {
        xmlFreeDoc(doc);
        xmlCleanupParser();
		return FALSE;
	}
	*/
	
	xmlNodePtr cur_node;
	for(cur_node = metadata_tag->children; cur_node != NULL; cur_node = cur_node -> next) {
		if(xmlStrcmp(cur_node->name,(unsigned char *)"attribute") == 0) {
			xmlNodePtr name_tag  = taudb_private_find_xml_child_named(cur_node, "name");
			xmlNodePtr value_tag = taudb_private_find_xml_child_named(cur_node, "value");
			if(name_tag != NULL && value_tag != NULL) {
				xmlChar * name_str  = xmlNodeListGetString(doc, name_tag->children,  1);
				xmlChar * value_str = xmlNodeListGetString(doc, value_tag->children, 1);
#ifdef TAUDB_DEBUG
				printf("Adding metadata %s : %s\n", name_str, value_str);
#endif
                TAUDB_SECONDARY_METADATA * secondary_metadata = taudb_create_secondary_metadata(1);
				secondary_metadata->key.timer_callpath = NULL;
				secondary_metadata->key.thread = thread;
				secondary_metadata->key.parent = NULL;
				secondary_metadata->key.time_range = NULL;
				secondary_metadata->key.name  = taudb_strdup((char *)name_str);
				secondary_metadata->num_values = 1;
				secondary_metadata->child_count = 0;
				secondary_metadata->children = NULL;
				secondary_metadata->value = (char**)malloc(sizeof(char*));
				secondary_metadata->value[0] = taudb_strdup((char *)value_str);
                //HASH_ADD_KEYPTR(hh2, trial->secondary_metadata_by_key, &(secondary_metadata->key), sizeof(secondary_metadata->key), secondary_metadata);
                taudb_add_secondary_metadata_to_trial(trial, secondary_metadata);
				// put it in the primary_metadata, if exists
                TAUDB_PRIMARY_METADATA* pm = NULL;
                if (thread->index == 0) {
                  pm = taudb_create_primary_metadata(1);
				  pm->name  = taudb_strdup((char *)name_str);
				  pm->value = taudb_strdup((char *)value_str);
                  //HASH_ADD_KEYPTR(hh, trial->primary_metadata, pm->name, strlen(pm->name), pm);
                  taudb_add_primary_metadata_to_trial(trial, pm);
                } else {
                  pm = taudb_get_primary_metadata_by_name(trial->primary_metadata, (const char*)name_str);
                  if (pm != NULL) {
                    if (strcmp(pm->value, (const char*)value_str) != 0) {
                      HASH_DELETE(hh, trial->primary_metadata, pm);
                    }
                  }
                }
                xmlFree(name_str);
                xmlFree(value_str);
			}
		}
	}

	xmlFreeDoc(doc);
	xmlCleanupParser();
	
	return TRUE;
}

void taudb_consolidate_metadata (TAUDB_TRIAL* trial) {
  // iterate over the threads
  int j;

  TAUDB_SECONDARY_METADATA *current, *tmp;
  HASH_ITER(hh2, trial->secondary_metadata_by_key, current, tmp) {
    TAUDB_PRIMARY_METADATA* pm = NULL;
	if (current->key.timer_callpath == NULL && current->key.time_range == NULL && current->key.parent == NULL ) {
	  pm = taudb_get_primary_metadata_by_name(trial->primary_metadata, current->key.name);
	  if (pm != NULL) {
        HASH_DELETE(hh2, trial->secondary_metadata_by_key, current);
        free(current->key.name);
        for (j = current->num_values-1 ; j >= 0 ; j--) {
          free(current->value[j]);
        }
        free(current->id);
        free(current);
	  }
	}
  }

  // iterate over the secondary metadata
}

void taudb_parse_tau_profile_counter(char* line, TAUDB_TRIAL* trial,  TAUDB_THREAD* thread) {
// eventname numevents max min mean sumsqr

#ifdef TAUDB_DEBUG_DEBUG
  printf("taudb_parse_tau_profile_counter: %s\n", line);
#endif
  // it won't be this long, but it will be close.
  const int line_len = strlen(line);
  char* counter_name = calloc(line_len, sizeof(char));
  int numevents = 0;
  double max = 0.0, min = 0.0, mean = 0.0, sumsqr = 0.0;
  // tokenize by quotes first, to get the function name
  char* tmp = strtok(line, "\"");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    strncpy(counter_name,  tmp, line_len); 
	// as much as I would love to trim the counter names, I can't. :(
	// it causes downstream problems.
	//taudb_trim(counter_name);
  }
  // tokenize by spaces now, to get the rest of the fields
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    numevents = atoi(tmp);
  } else { return; }
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    max = atof(tmp);
  } else { return; }
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    min = atof(tmp);
  } else { return; }
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    mean = atof(tmp);
  } else { return; }
  tmp = strtok(NULL, " ");
  if (tmp != NULL && (strlen(tmp) > 0)) {
    sumsqr = atof(tmp);
  } else { return; }
  const char* conjunction = " : ";
  TAUDB_COUNTER* counter = NULL;
  TAUDB_COUNTER_VALUE* counter_value = NULL;
  TAUDB_TIMER_CALLPATH* timer_callpath = NULL;
  if ((strstr(counter_name, conjunction)) != NULL) {
    // process callpath profile
    TAUDB_TIMER_CALLPATH tmp_timer_callpath;
	tmp_timer_callpath.id = 0;
	tmp_timer_callpath.name = strstr(counter_name, conjunction)+3;
#ifdef TAUDB_DEBUG_DEBUG
	printf("Searching for callpath: '%s'\n", tmp_timer_callpath.name);
#endif
    timer_callpath = taudb_process_callpath_timer(trial, &tmp_timer_callpath);
    char* tmp = strtok(counter_name, ":");
	char* tmp_counter_name = (char*)(calloc(strlen(tmp), sizeof(char)));
	strncpy(tmp_counter_name, tmp, strlen(tmp)-1);
    counter = taudb_create_counter(trial, tmp_counter_name);
  } else {
    counter = taudb_create_counter(trial, counter_name);
  }
#ifdef TAUDB_DEBUG_DEBUG
  printf("'%s' %d %f %f %f %f\n", counter_name, numevents, max, min, mean, sumsqr);
#endif
  // get/create timer_values
  counter_value = taudb_create_counter_value(trial, counter, thread, timer_callpath, numevents, max, min, mean, sumsqr);
}

TAUDB_COUNTER* taudb_create_counter(TAUDB_TRIAL* trial, const char* counter_name) {
  TAUDB_COUNTER* counter = NULL;
  if (trial->counters_by_name != NULL) {
#ifdef TAUDB_DEBUG_DEBUG
	printf("Searching for: '%s'\n", counter_name);
#endif
    counter = taudb_get_counter_by_name(trial->counters_by_name, counter_name);
  }
  if (counter == NULL) {
    counter = taudb_create_counters(1);
    counter->name = taudb_strdup(counter_name);
	counter->trial = trial;
#ifdef TAUDB_DEBUG_DEBUG
	printf("Adding: '%s'\n", counter->name);
#endif
    //HASH_ADD_KEYPTR(hh2, trial->counters_by_name, counter->name, strlen(counter->name), counter);
    taudb_add_counter_to_trial(trial, counter);
  }
  return counter;
}

TAUDB_COUNTER_VALUE* taudb_create_counter_value(TAUDB_TRIAL* trial, TAUDB_COUNTER* counter, TAUDB_THREAD* thread, TAUDB_TIMER_CALLPATH* timer_callpath, int numevents, double max, double min, double mean, double sumsqr) {
  TAUDB_COUNTER_VALUE* counter_value = taudb_create_counter_values(1);
  counter_value->key.counter = counter;
  counter_value->key.thread = thread;
  counter_value->key.context = timer_callpath;
  counter_value->key.timestamp = NULL;
  counter_value->sample_count = numevents;
  counter_value->maximum_value = max;
  counter_value->minimum_value = min;
  counter_value->mean_value = mean;
  counter_value->standard_deviation = sumsqr;
  //HASH_ADD(hh1, trial->counter_values, key, sizeof(counter_value->key), counter_value);
  taudb_add_counter_value_to_trial(trial, counter_value);
  return counter_value;
}



