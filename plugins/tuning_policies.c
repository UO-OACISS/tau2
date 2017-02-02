
#include <mpi.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauMpiTTypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

void init_control_policies()
{

 fprintf(stdout, "Tuning policies DSO init.....\n");

}

/*Implement user based CVAR tuning policy based on a policy file (?)
 * TODO: This tuning logic should be in a separate module/file. Currently implementing hard-coded policies for MVAPICH meant only for experimentation purposes*/
//void Tau_enable_user_cvar_tuning_policy(const int num_pvars, int *tau_pvar_count, unsigned long long int **pvar_value_buffer) {
void plugin_tuning_policy(int argc, void **args) {

  int return_val, i, namelen, verb, varclass, bind, threadsup;
  int index;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  char metric_string[TAU_NAME_LENGTH], value_string[TAU_NAME_LENGTH];
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;
  static int firsttime = 1;
  static unsigned long long int *reduced_value_array = NULL;
  static char *reduced_value_cvar_string = NULL;
  static char *reduced_value_cvar_value_string = NULL;
  
  fprintf(stdout, "plugin tuning policy ...\n");

  assert(argc=3);

  const int num_pvars 				= (const int)(args[0]);
  int *tau_pvar_count 				= (int *)(args[1]);
  unsigned long long int **pvar_value_buffer 	= (unsigned long long int **)(args[2]);

  /*MVAPICH specific thresholds and names*/
  char PVAR_MAX_VBUF_USAGE[TAU_NAME_LENGTH] = "mv2_vbuf_max_use_array";
  char PVAR_VBUF_ALLOCATED[TAU_NAME_LENGTH] = "mv2_vbuf_allocated_array";
  int PVAR_VBUF_WASTED_THRESHOLD = 10; //This is the threshold above which we will be free from the pool

  char CVAR_ENABLING_POOL_CONTROL[TAU_NAME_LENGTH] = "MPIR_CVAR_VBUF_POOL_CONTROL";
  char CVAR_SPECIFYING_REDUCED_POOL_SIZE[TAU_NAME_LENGTH] = "MPIR_CVAR_VBUF_POOL_REDUCED_VALUE";

  int pvar_max_vbuf_usage_index, pvar_vbuf_allocated_index, has_threshold_been_breached_in_any_pool;
  pvar_max_vbuf_usage_index = -1;
  pvar_vbuf_allocated_index = -1;
  has_threshold_been_breached_in_any_pool = 0;

 if(firsttime) {
  firsttime = 0;
  for(i = 0; i < num_pvars; i++){
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
      reduced_value_array = (unsigned long long int *)calloc(sizeof(unsigned long long int), tau_pvar_count[pvar_max_vbuf_usage_index]);
      reduced_value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH);
      strcpy(reduced_value_cvar_string, "");
      reduced_value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH);
      strcpy(reduced_value_cvar_value_string, "");
  }

  if((pvar_max_vbuf_usage_index == -1) || (pvar_vbuf_allocated_index == -1)) {
    printf("Unable to find the indexes of PVARs required for tuning\n");
    return;
  } else {
    dprintf("Index of %s is %d and index of %s is %d\n", PVAR_MAX_VBUF_USAGE, pvar_max_vbuf_usage_index, PVAR_VBUF_ALLOCATED, pvar_vbuf_allocated_index);
  }
 }
  /*Tuning logic: If the difference between allocated vbufs and max use vbufs in a given
 *   * vbuf pool is higher than a set threshhold, then we will free from that pool.*/
  for(i = 0 ; i < tau_pvar_count[pvar_max_vbuf_usage_index]; i++) {
    if(pvar_value_buffer[pvar_max_vbuf_usage_index][i] > 1000) pvar_value_buffer[pvar_max_vbuf_usage_index][i] = 0; /*HACK - we are getting garbage values for pool2. Doesn't seem to be an issue in TAU*/

    if((pvar_value_buffer[pvar_vbuf_allocated_index][i] - pvar_value_buffer[pvar_max_vbuf_usage_index][i]) > PVAR_VBUF_WASTED_THRESHOLD) {
      has_threshold_been_breached_in_any_pool = 1;
      reduced_value_array[i] = pvar_value_buffer[pvar_max_vbuf_usage_index][i];
      dprintf("Threshold breached: Max usage for %d pool is %llu but vbufs allocated are %llu\n", i, pvar_value_buffer[pvar_max_vbuf_usage_index][i], pvar_value_buffer[pvar_vbuf_allocated_index][i]);
    } else {
      reduced_value_array[i] = pvar_value_buffer[pvar_vbuf_allocated_index][i] + 10; //Some value higher than current allocated
    }

    if(i == (tau_pvar_count[pvar_max_vbuf_usage_index])) {
      sprintf(metric_string,"%s[%d]", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      sprintf(value_string,"%llu", reduced_value_array[i]);
    } else {
      sprintf(metric_string,"%s[%d],", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      sprintf(value_string,"%llu,", reduced_value_array[i]);
    }
    
    strcat(reduced_value_cvar_string, metric_string);
    strcat(reduced_value_cvar_value_string, value_string);

  }

  if(has_threshold_been_breached_in_any_pool) {
    sprintf(metric_string,"%s,%s", CVAR_ENABLING_POOL_CONTROL, reduced_value_cvar_string);
    sprintf(value_string,"%d,%s", 1, reduced_value_cvar_value_string);
    dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  } else {
    sprintf(metric_string,"%s", CVAR_ENABLING_POOL_CONTROL);
    sprintf(value_string,"%d", 0);
    dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  }
}
