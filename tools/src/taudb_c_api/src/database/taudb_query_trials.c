#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#ifndef fmax
#define fmax(a,b) ((a) < (b) ? (a) : (b))
#endif

xmlNodePtr taudb_private_find_xml_child_named(xmlNodePtr parent, const char * name) {
	xmlNodePtr cur_node = parent;
	if(xmlStrcmp(cur_node->name,(const unsigned char *)name) == 0) {
		return cur_node;
	}
	for(cur_node = parent->children; cur_node != NULL; cur_node = cur_node -> next) {
		if(xmlStrcmp(cur_node->name,(const unsigned char *)name) == 0) {
			break;
		}
	}
	if(cur_node == NULL) {
		fprintf(stderr, "XML did not contain %s tag\n", name);
		return NULL;
	}
	return cur_node;
}

boolean taudb_private_primary_metadata_from_xml(TAUDB_TRIAL * trial, char * xml) {
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
	xmlNodePtr common_profile_attributes_tag = taudb_private_find_xml_child_named(metadata_tag, "CommonProfileAttributes");
	if(common_profile_attributes_tag == NULL) {
        xmlFreeDoc(doc);
        xmlCleanupParser();
		return FALSE;
	}
	
	xmlNodePtr cur_node;
    size_t i = 0;
    trial->primary_metadata = NULL;
	for(cur_node = common_profile_attributes_tag->children; cur_node != NULL; cur_node = cur_node -> next) {
		if(xmlStrcmp(cur_node->name,(unsigned char *)"attribute") == 0) {
			xmlNodePtr name_tag  = taudb_private_find_xml_child_named(cur_node, "name");
			xmlNodePtr value_tag = taudb_private_find_xml_child_named(cur_node, "value");
			if(name_tag != NULL && value_tag != NULL) {
				xmlChar * name_str  = xmlNodeListGetString(doc, name_tag->children,  1);
				xmlChar * value_str = xmlNodeListGetString(doc, value_tag->children, 1);
#ifdef TAUDB_DEBUG
				printf("Adding metadata %s : %s\n", name_str, value_str);
#endif
                TAUDB_PRIMARY_METADATA * primary_metadata = taudb_create_primary_metadata(1);
				primary_metadata->name  = taudb_strdup((char *)name_str);
				primary_metadata->value = taudb_strdup((char *)value_str);
				taudb_add_primary_metadata_to_trial(trial, primary_metadata);
                xmlFree(name_str);
                xmlFree(value_str);
                i++;
			}
		}
	}

	xmlFreeDoc(doc);
	xmlCleanupParser();
	
	return TRUE;
}

TAUDB_TRIAL* taudb_private_query_trials(TAUDB_CONNECTION* connection, boolean full, char* my_query, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_trials(%d, %s)\n", full, my_query);
#endif
  int nFields;
  int i, j;

  taudb_begin_transaction(connection);
  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  TAUDB_TRIAL* trials = taudb_create_trials(nRows);

  nFields = taudb_get_num_columns(connection);
#ifdef TAUDB_DEBUG
  printf("Found %d rows, %d columns\n", nRows, nFields);
#endif

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    //int metaIndex = 0;
    //trials[i].primary_metadata = taudb_create_primary_metadata(nFields-6);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
        trials[i].id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        trials[i].name = taudb_strdup(taudb_get_value(connection,i,j));
      //} else if (strcmp(taudb_get_column_name(connection, j), "date") == 0) {
        //trials[i].collection_date = taudb_strdup(taudb_get_value(connection,i,j));
      } else if (strcmp(taudb_get_column_name(connection, j), "node_count") == 0) {
        trials[i].node_count = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "contexts_per_node") == 0) {
        trials[i].contexts_per_node = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "threads_per_context") == 0) {
        trials[i].threads_per_context = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "total_threads") == 0) {
        trials[i].total_threads = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "data_source") == 0) {
        int data_source = atoi(taudb_get_value(connection, i, j));
        trials[i].data_source = taudb_get_data_source_by_id(connection->data_sources_by_id, data_source);
      } else if (strcmp(taudb_get_column_name(connection, j), "xml_metadata") == 0) {
        // TODO we need to handle this!
        continue;
      } else if (strcmp(taudb_get_column_name(connection, j), "xml_metadata_gz") == 0) {
        char* value = taudb_get_binary_value(connection, i, j);
        taudb_private_primary_metadata_from_xml(&(trials[i]), value);
        continue;
      }
    } 
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  for (i = 0 ; i < nRows ; i++) {
    if (taudb_version != TAUDB_2005_SCHEMA) {
      trials[i].primary_metadata = taudb_query_primary_metadata(connection, &(trials[i]), taudb_numItems);
	}
    if (full) {
      if (taudb_version == TAUDB_2012_SCHEMA) {
	    printf("Threads\n");
        taudb_query_threads(connection, &(trials[i]),taudb_numItems);
	    printf("Metrics\n");
        taudb_query_metrics(connection, &(trials[i]), taudb_numItems);
        //taudb_query_time_range(connection, &(trials[i]));
	    printf("Timers\n");
        taudb_query_timers(connection, &(trials[i]), taudb_numItems);
	    printf("Timer_groups\n");
        taudb_query_timer_groups(connection, &(trials[i]), taudb_numItems);
	    printf("Timer call paths\n");
        taudb_query_timer_callpaths(connection, &(trials[i]), NULL,taudb_numItems);
	    printf("Timer call data\n");
        taudb_query_timer_call_data(connection, &(trials[i]), NULL, NULL, taudb_numItems);
	    printf("Timer values\n");
        taudb_query_timer_values(connection, &(trials[i]), NULL, NULL, NULL, taudb_numItems);
	    printf("Counters \n");
        taudb_query_counters(connection, &(trials[i]), taudb_numItems);
	    printf("Counter values\n");
        taudb_query_counter_values(connection, &(trials[i]),taudb_numItems);
	    printf("Secondary metadata\n");
        trials[i].secondary_metadata = taudb_query_secondary_metadata(connection, &(trials[i]), taudb_numItems);
      } else {
        taudb_query_threads(connection, &(trials[i]),taudb_numItems);
        taudb_query_timers(connection, &(trials[i]), taudb_numItems);
        taudb_query_all_timer_callpaths(connection, &(trials[i]), taudb_numItems);
        //taudb_query_all_timer_callpath_stats(connection, &(trials[i]));
        taudb_query_metrics(connection, &(trials[i]), taudb_numItems);
        //taudb_query_all_timer_values(connection, &(trials[i]));
        //taudb_query_counters(&(trials[i]));
        //taudb_query_counter_values(&(trials[i]));
	  }
    }
  }
  *taudb_numItems = nRows;

  return trials;
}

TAUDB_TRIAL* taudb_query_trials(TAUDB_CONNECTION* connection, boolean full, TAUDB_TRIAL* trial, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_trials(%d, %p)\n", full, trial);
#endif
  char my_query[4096]; // hopefully, this is long enough!
  if (trial->id > 0) { // the user wants a specific trial, so get it
    snprintf(my_query, sizeof(my_query), "select * from trial where id = %d", trial->id);
  } else if (trial->name != NULL) {
    snprintf(my_query, sizeof(my_query), "select * from trial where name = '%s'", trial->name);
  } else { 
    snprintf(my_query, sizeof(my_query), "select * from trial ");
    char *where1 = "where id in (select pm0.trial from primary_metadata pm0 ";
    char *join = "inner join primary_metadata pm%d on pm%d.trial = pm%d.trial";
    char conjunction[128];
	strncpy(conjunction,  where1, sizeof(conjunction)); 
	int index = 1;
	// are there metadata fields?
    TAUDB_PRIMARY_METADATA * current;
    for (current = trial->primary_metadata; current != NULL;
         current = taudb_next_primary_metadata_by_name_from_trial(current)) {
      snprintf(my_query, sizeof(my_query),  "%s %s ", my_query, conjunction);
      snprintf(conjunction, sizeof(conjunction),  join, index, index, index-1);
	  index = index + 1;
    }
	index = 0;
    char *where2 = "where";
    char *and = "and";
	char *equals = "=";
	char *like = "like";
	char *comparison = equals;
    char *conjunction2 = where2;
    for (current = trial->primary_metadata; current != NULL;
         current = taudb_next_primary_metadata_by_name_from_trial(current)) {
      if (strstr(current->value, "%") == NULL) {
	    comparison = equals;
      } else {
	    comparison = like;
      }
      snprintf(my_query, sizeof(my_query),  "%s %s pm%d.name = '%s' and pm%d.value %s '%s' ", my_query, conjunction2, index, current->name, index, comparison, current->value);
      conjunction2 = and;
	  index = index + 1;
    }
	if (conjunction2 == and) {
      snprintf(my_query, sizeof(my_query),  "%s)", my_query);
	}
  }
  printf("%s\n", my_query);
  return taudb_private_query_trials(connection, full, my_query, taudb_numItems);
}

TAUDB_TRIAL* perfdmf_query_trials(TAUDB_CONNECTION* connection, PERFDMF_EXPERIMENT* experiment, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_trials(%p)\n", experiment);
#endif
  char my_query[256];
  snprintf(my_query, sizeof(my_query), "select * from trial where experiment = %d", experiment->id);

  return taudb_private_query_trials(connection, FALSE, my_query, taudb_numItems);
}

void taudb_save_trial(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update, boolean cascade) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_save_trial()\n");
	printf("Trial data_source id is %d.\n", trial->data_source->id);
#endif
  const char* my_query;
  const char* statement_name;
  int nParams = 7;

  struct timeval begin_wall, end_wall, full_wall;  // to measure wall clock time
  clock_t begin, end, full; // to measure CPU time
  double time_spent, time_spent_wall;

  gettimeofday(&begin_wall, NULL);
  gettimeofday(&full_wall, NULL);
  begin = clock();
  full = begin;

  // Are we updating, or inserting?
  if (update && trial->id > 0) {
    nParams = 7;
	  statement_name = "TAUDB_UPDATE_TRIAL";
    my_query = "update trial set name=$1, data_source=$2, node_count=$3, contexts_per_node=$4, threads_per_context=$5, total_threads=$6 where id = $7;";
  } else {
    nParams = 6;
	statement_name = "TAUDB_INSERT_TRIAL";
    my_query = "insert into trial (name, data_source, node_count, contexts_per_node, threads_per_context, total_threads) values ($1, $2, $3, $4, $5, $6);";
  }

  // begin the transaction.
  taudb_begin_transaction(connection);
  // make the prepared statement.
  taudb_prepare_statement(connection, statement_name, my_query, nParams);
  
  // make array of nParms character pointers - if we have 1 too many, that's ok
  // static declaration is preferable to doing a malloc.
  const char* paramValues[7] = {0};
  // populate the array of string values
  paramValues[0] = trial->name;
  char data_source[32] = {0};
  if (trial->data_source == NULL) {
    paramValues[1] = NULL;
  } else {
    snprintf(data_source, sizeof(data_source),  "%d", trial->data_source->id);
    paramValues[1] = data_source;
  }
  char nodes[32] = {0};
  snprintf(nodes, sizeof(nodes),  "%d", trial->node_count);
  paramValues[2] = nodes;
  char contexts[32] = {0};
  snprintf(contexts, sizeof(contexts),  "%d", trial->contexts_per_node);
  paramValues[3] = contexts;
  char threads[32] = {0};
  snprintf(threads, sizeof(threads),  "%d", trial->threads_per_context);
  paramValues[4] = threads;
  char total[32] = {0};
  snprintf(total, sizeof(total),  "%d", trial->total_threads);
  paramValues[5] = total;

  // if we are updating, add the ID to the query
  char id[32] = {0};
  if (update && trial->id > 0) {
    snprintf(id, sizeof(id),  "%d", trial->id);
    paramValues[6] = id;
  }
	
#ifdef TAUDB_DEBUG_DEBUG
	printf("Before execute, trial data_source id is %d.\n", trial->data_source->id);
	printf("Before execute, paramValues[1] is %s\n", paramValues[1]);
#endif
	
  // execute the statement
  taudb_execute_statement(connection, statement_name, nParams, paramValues);

  // if we did an insert, get the new trial ID
  if (!(update && trial->id > 0)) {
    taudb_execute_query(connection, "select currval('trial_id_seq');");

    int nRows = taudb_get_num_rows(connection);
    if (nRows == 1) {
      trial->id = atoi(taudb_get_value(connection, 0, 0));
      printf("New Trial: %d\n", trial->id);
    } else {
      printf("Failed.\n");
    }
    taudb_close_query(connection);
  }

  // should we save the entire trial?
  if (cascade) {
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved trial in:              %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_metrics(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved metrics in:            %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_threads(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved threads in:            %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_timers(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved timers in:             %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_time_ranges(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved time_ranges in:        %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_timer_groups(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved timer_groups in:       %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_timer_parameters(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved timer_parameters in:   %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_timer_callpaths(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved timer_callpaths in:    %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_timer_call_data(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved timer_call_data in:    %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_timer_values(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved timer_values in:       %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_counters(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved counters in:           %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_counter_values(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved counter_values in:     %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_primary_metadata(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved primary_metadata in:   %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
    begin = clock();
    gettimeofday(&begin_wall, NULL);
    taudb_save_secondary_metadata(connection, trial, update);
    end = clock();
    gettimeofday(&end_wall, NULL);
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	time_spent_wall = (double) (end_wall.tv_usec - begin_wall.tv_usec)/1000000 +
	                  (double) (end_wall.tv_sec - begin_wall.tv_sec);
	printf("Saved secondary_metadata in: %.2f seconds (%.2fs CPU, %.2fs database latency)...\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
  }
  end = clock();
  gettimeofday(&end_wall, NULL);
  time_spent = (double)(end - full) / CLOCKS_PER_SEC;
  time_spent_wall = (double) (end_wall.tv_usec - full_wall.tv_usec)/1000000 +
                    (double) (end_wall.tv_sec - full_wall.tv_sec);
  printf("\nDone saving trial in: %.2f seconds (%.2fs CPU, %.2fs database latency).\n", time_spent_wall, time_spent, fmax(0.0, (time_spent_wall - time_spent)));
  taudb_close_transaction(connection);
  taudb_clear_result(connection);
}

