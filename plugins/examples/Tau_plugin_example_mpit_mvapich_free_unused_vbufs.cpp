#ifdef TAU_MPI
#ifdef TAU_MPI_T

#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauEnv.h>
#include <Profile/TauPlugin.h>

/*User defined macros*/
#define TAU_NAME_LENGTH 1024
#define CVAR_ENABLING_POOL_CONTROL "MPIR_CVAR_VBUF_POOL_CONTROL"
#define CVAR_SPECIFYING_REDUCED_POOL_SIZE "MPIR_CVAR_VBUF_POOL_REDUCED_VALUE"
#define PVAR_VBUF_ALLOCATED "mv2_vbuf_allocated_array"
#define PVAR_MAX_VBUF_USAGE "mv2_vbuf_max_use_array"
#define PVAR_VBUF_WASTED_THRESHOLD 10

/*Externs*/
extern int Tau_mpi_t_parse_and_write_cvars(const char * metric, const char * value);
extern int Tau_mpi_t_initialize();
extern int Tau_mpi_t_get_pvar_count(int pvarindex);
extern MPI_T_pvar_session * Tau_mpi_t_get_pvar_session();
extern MPI_T_pvar_handle * Tau_mpi_t_get_pvar_handles();

static int pvar_max_vbuf_usage_index, pvar_vbuf_allocated_index = -1;
static MPI_T_pvar_session * tau_pvar_session = NULL;
static MPI_T_pvar_handle  * tau_pvar_handles = NULL;
unsigned long long int * reduced_value_array = NULL;
unsigned long long int * pvar_max_vbuf_usage = NULL;
unsigned long long int * pvar_vbuf_allocated = NULL;

static int num_vbuf_pools = 0;

int Tau_plugin_example_mpit_recommend_sharp_usage(Tau_plugin_event_interrupt_trigger_data_t* data) {
  char metric_string[TAU_NAME_LENGTH], value_string[TAU_NAME_LENGTH];
  unsigned long long int reduced_value_array[5] = {0};
  static char reduced_value_cvar_string[TAU_NAME_LENGTH] = "";
  static char reduced_value_cvar_value_string[TAU_NAME_LENGTH] = "";
  int i;


  int has_threshold_been_breached_in_any_pool = 0;

  if((pvar_max_vbuf_usage_index == -1) || (pvar_vbuf_allocated_index == -1)) {
    TAU_VERBOSE("TAU PLUGIN: Unable to find the indexes of PVARs required for tuning. Returning without doing any tuning\n");
    return 0;
  }

  if(tau_pvar_session == NULL || tau_pvar_handles == NULL || tau_pvar_handles[pvar_max_vbuf_usage_index] == NULL || tau_pvar_handles[pvar_vbuf_allocated_index] == NULL) return 0; 

  for(i = 0; i < num_vbuf_pools; i++) reduced_value_array[i] = 0;

  strncpy(reduced_value_cvar_string,  "", sizeof(reduced_value_cvar_string)); 
  strncpy(reduced_value_cvar_value_string,  "", sizeof(reduced_value_cvar_value_string)); 

  MPI_T_pvar_read(*tau_pvar_session, tau_pvar_handles[pvar_max_vbuf_usage_index], (void*)pvar_max_vbuf_usage);
  MPI_T_pvar_read(*tau_pvar_session, tau_pvar_handles[pvar_vbuf_allocated_index], (void*)pvar_vbuf_allocated);

  
  /*Tuning logic: If the difference between allocated vbufs and max use vbufs in a given
 *   * vbuf pool is higher than a set threshhold, then we will free from that pool.*/
  for(i = 0 ; i < num_vbuf_pools; i++) {

    if(pvar_max_vbuf_usage[i] > 1000) pvar_max_vbuf_usage[i] = 0; /*HACK - we are getting garbage values for pool2. Doesn't seem to be an issue in TAU*/

    if((pvar_vbuf_allocated[i] - pvar_max_vbuf_usage[i]) > PVAR_VBUF_WASTED_THRESHOLD) {
      has_threshold_been_breached_in_any_pool = 1;
      reduced_value_array[i] = pvar_max_vbuf_usage[i];
      TAU_VERBOSE("Threshold breached: Max usage for %d pool is %llu but vbufs allocated are %llu\n", i, pvar_max_vbuf_usage[i], pvar_vbuf_allocated[i]);
    } else {
      reduced_value_array[i] = pvar_vbuf_allocated[i] + 10; //Some value higher than current allocated
    }

    if(i == num_vbuf_pools) {
      snprintf(metric_string, sizeof(metric_string), "%s[%d]", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      snprintf(value_string, sizeof(value_string), "%llu", reduced_value_array[i]);
    } else {
      snprintf(metric_string, sizeof(metric_string), "%s[%d],", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      snprintf(value_string, sizeof(value_string), "%llu,", reduced_value_array[i]);
    }

    strcat(reduced_value_cvar_string, metric_string);
    strcat(reduced_value_cvar_value_string, value_string);

  }

  if(has_threshold_been_breached_in_any_pool) {
    snprintf(metric_string, sizeof(metric_string), "%s,%s", CVAR_ENABLING_POOL_CONTROL, reduced_value_cvar_string);
    snprintf(value_string, sizeof(value_string), "%d,%s", 1, reduced_value_cvar_value_string);
    TAU_VERBOSE("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  } else {
    snprintf(metric_string, sizeof(metric_string), "%s", CVAR_ENABLING_POOL_CONTROL);
    snprintf(value_string, sizeof(value_string), "%d", 0);
    TAU_VERBOSE("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  }
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to.
 * In addition, this init function stores the index of the PVARs it is interested in for future use inside the tuning routine*/

int Tau_plugin_init_func(int argc, char **argv, int id) {

  int return_val = Tau_mpi_t_initialize();

  if(return_val != MPI_SUCCESS) {
    printf("TAU PLUGIN: Failed to initalize the MPI_T interface from the plugin. Bailing.\n");
    return -1;
  }

  Tau_plugin_callbacks_t * cb = (Tau_plugin_callbacks_t*)malloc(sizeof(Tau_plugin_callbacks_t));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
  cb->InterruptTrigger = Tau_plugin_example_mpit_recommend_sharp_usage;
  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  int i, namelen, verb, varclass, bind, threadsup;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;
  
  int num_pvars;
  MPI_T_pvar_get_num(&num_pvars);

  /*Find the index of the PVARs that are need for tuning. If they are not found, do not perform any tuning.*/
  for(i = 0; i < num_pvars; i++) {
      namelen = desc_len = TAU_NAME_LENGTH;
      return_val = MPI_T_pvar_get_info(i/*IN*/,
      event_name /*OUT*/,
      &namelen /*INOUT*/,
      &verb /*OUT*/,
      &varclass /*OUT*/,
      &datatype /*OUT*/,
      &enumtype /*OUT*/,
      description /*description: OUT*/,
      &desc_len /*desc_len: INOUT*/,
      &bind /*OUT*/,
      &readonly /*OUT*/,
      &continuous /*OUT*/,
      &atomic/*OUT*/);

      if(strcmp(event_name, PVAR_MAX_VBUF_USAGE) == 0) {
        pvar_max_vbuf_usage_index = i;
      } else if (strcmp(event_name, PVAR_VBUF_ALLOCATED) == 0) {
        pvar_vbuf_allocated_index = i;
      }
  }

  if((pvar_max_vbuf_usage_index == -1) || (pvar_vbuf_allocated_index == -1)) {
    printf("TAU PLUGIN: Unable to find the indexes of PVARs required for tuning. Not doing any tuning.\n");
    return -1;
  }

  tau_pvar_session = Tau_mpi_t_get_pvar_session();
  tau_pvar_handles = Tau_mpi_t_get_pvar_handles();
  num_vbuf_pools = Tau_mpi_t_get_pvar_count(pvar_max_vbuf_usage_index);
  reduced_value_array = (unsigned long long int *)malloc(sizeof(unsigned long long int)*num_vbuf_pools);
  pvar_max_vbuf_usage = (unsigned long long int *)malloc(sizeof(unsigned long long int)*num_vbuf_pools);
  pvar_vbuf_allocated = (unsigned long long int *)malloc(sizeof(unsigned long long int)*num_vbuf_pools);

  return 0;
}

#endif /* TAU_MPI_T */
#endif /* TAU_MPI */
