#include "taudb_internal.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

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
				primary_metadata->name  = taudb_create_and_copy_string((char *)name_str);
				primary_metadata->value = taudb_create_and_copy_string((char *)value_str);
                HASH_ADD_KEYPTR(hh, trial->primary_metadata, primary_metadata->name, strlen((char *)name_str), primary_metadata);
                xmlFree(name_str);
                xmlFree(value_str);
                i++;
			}
		}
	}

    trial->primary_metadata_count = i;
	
	xmlFreeDoc(doc);
	xmlCleanupParser();
	
	return TRUE;
}

TAUDB_TRIAL* taudb_private_query_trials(TAUDB_CONNECTION* connection, boolean full, char* my_query) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_trials(%d, %s)\n", full, my_query);
#endif
  void *res;
  int nFields;
  int i, j;

  taudb_begin_transaction(connection);
  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  TAUDB_TRIAL* trials = taudb_create_trials(nRows);

  nFields = taudb_get_num_columns(res);
#ifdef TAUDB_DEBUG
  printf("Found %d rows, %d columns\n", nRows, nFields);
#endif

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    //int metaIndex = 0;
    //trials[i].primary_metadata = taudb_create_primary_metadata(nFields-6);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
        trials[i].id = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
        trials[i].name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
      //} else if (strcmp(taudb_get_column_name(res, j), "date") == 0) {
        //trials[i].collection_date = taudb_create_and_copy_string(taudb_get_value(res,i,j));
      } else if (strcmp(taudb_get_column_name(res, j), "node_count") == 0) {
        trials[i].node_count = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "contexts_per_node") == 0) {
        trials[i].contexts_per_node = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "threads_per_context") == 0) {
        trials[i].threads_per_context = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "total_threads") == 0) {
        trials[i].total_threads = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "data_source") == 0) {
        int data_source = atoi(taudb_get_value(res, i, j));
        trials[i].data_source = taudb_get_data_source_by_id(connection->data_sources_by_id, data_source);
      } else if (strcmp(taudb_get_column_name(res, j), "xml_metadata") == 0) {
        // TODO we need to handle this!
        continue;
      } else if (strcmp(taudb_get_column_name(res, j), "xml_metadata_gz") == 0) {
        char* value = taudb_get_binary_value(res, i, j);
        taudb_private_primary_metadata_from_xml(&(trials[i]), value);
        continue;
      }
    } 
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  for (i = 0 ; i < nRows ; i++) {
    if (taudb_version != TAUDB_2005_SCHEMA) {
      trials[i].primary_metadata = taudb_query_primary_metadata(connection, &(trials[i]));
      trials[i].primary_metadata_count = taudb_numItems;
	}
    if (full) {
      trials[i].threads = taudb_query_threads(connection, &(trials[i]));
      trials[i].thread_count = taudb_numItems;
      trials[i].timers_by_id = taudb_query_timers(connection, &(trials[i]));
      trials[i].timer_count = taudb_numItems;
      trials[i].timer_callpaths_by_id = taudb_query_all_timer_callpaths(connection, &(trials[i]));
      trials[i].timer_callpath_count = taudb_numItems;
      //trials[i].timer_callpath_stats = taudb_query_all_timer_callpath_stats(connection, &(trials[i]));
      //trials[i].callpath_stat_count = taudb_numItems;
      trials[i].metrics_by_id = taudb_query_metrics(connection, &(trials[i]));
      trials[i].metric_count = taudb_numItems;
      //trials[i].timer_values = taudb_query_all_timer_values(connection, &(trials[i]));
      //trials[i].value_count = taudb_numItems;
      //trials[i].counters = taudb_query_counters(&(trials[i]));
      //trials[i].counter_count = taudb_numItems;
      //taudb_query_counter_values(&(trials[i]));
      if (taudb_version == TAUDB_2012_SCHEMA) {
        trials[i].secondary_metadata = taudb_query_secondary_metadata(connection, &(trials[i]));
        trials[i].secondary_metadata_count = taudb_numItems;
      }
    }
  }
  taudb_numItems = nRows;

  return trials;
}

TAUDB_TRIAL* taudb_query_trials(TAUDB_CONNECTION* connection, boolean full, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_trials(%d, %p)\n", full, trial);
#endif
  char my_query[1024];
  if (trial->id > 0) { // the user wants a specific trial, so get it
    sprintf(my_query,"select * from trial where id = %d", trial->id);
  } else {
    sprintf(my_query,"select * from trial where");
    if (trial->name != NULL) {
      sprintf(my_query,"%s name = '%s'", my_query, trial->name);
    } 
  }
  return taudb_private_query_trials(connection, full, my_query);
}

TAUDB_TRIAL* perfdmf_query_trials(TAUDB_CONNECTION* connection, PERFDMF_EXPERIMENT* experiment) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_trials(%p)\n", experiment);
#endif
  char my_query[256];
  sprintf(my_query,"select * from trial where experiment = %d", experiment->id);

  return taudb_private_query_trials(connection, FALSE, my_query);
}

