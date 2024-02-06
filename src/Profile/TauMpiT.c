/****************************************************************************
 **                     TAU Portable Profiling Package                     **
 **                     http://www.cs.uoregon.edu/research/tau             **
 *****************************************************************************
 **    Copyright 1997-2009                                                 **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **     File            : TauMpiT.c                                       **
 **     Description     : TAU Profiling Package                           **
 **     Contact         : tau-bugs@cs.uoregon.edu                         **
 **     Documentation   : See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////
#include <mpi.h>
#include <Profile/Profiler.h> 
#include <Profile/TauEnv.h> 
#include <Profile/TauMpiTTypes.h>
#include <Profile/TauPluginInternals.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h> // pow

//////////////////////////////////////////////////////////////////////
// Global variables
//////////////////////////////////////////////////////////////////////
static MPI_T_pvar_handle  *tau_pvar_handles; 
static MPI_T_pvar_session tau_pvar_session;
static MPI_T_cvar_handle *tau_cvar_handles;
static unsigned long long int **pvar_value_buffer;
static void *read_value_buffer; // values are read into this buffer.
static MPI_Datatype *tau_mpi_datatype;
static int *tau_pvar_count;
static int *tau_cvar_num_vals;
static int tau_initial_pvar_count = 0;
static int tau_mpi_t_is_initialized = 0;
static MPI_Comm default_comm = MPI_COMM_WORLD;
int num_cvars = 0; //For now, we don't support case where number of CVARS changes dynamically at runtime

//////////////////////////////////////////////////////////////////////
// externs
//////////////////////////////////////////////////////////////////////
extern void Tau_track_pvar_event(int current_pvar_index, int current_pvar_subindex, const int *tau_pvar_count, int num_pvars, double data);extern void Tau_disable_tracking_mpi_t(void);
extern void Tau_allocate_pvar_event(int num_pvars, const int *tau_pvar_count);
extern void *Tau_MemMgr_malloc(int tid, size_t size);
extern void Tau_MemMgr_free(int tid, void *addr, size_t size);
extern char * Tau_get_pvar_name(int i, int j);
int Tau_msg_ini();

int Tau_mpi_t_initialize();

#define dprintf TAU_VERBOSE

/*Helper functions that ensure that we initialize the MPI_T interface only once from TAU
 * As of the time being, we do not want multiple performance tracking sessions in flight.*/
int Tau_mpi_t_is_initialized() {
  return tau_mpi_t_is_initialized;
}

void Tau_mpi_t_set_initialized() {
  tau_mpi_t_is_initialized = 1;
}

/*Returns the count associated with a PVAR specified by an index.
 * MPI_T doesn't yet support such operations directly, so we have no option but to
 * store and return this information from within TAU */
int Tau_mpi_t_get_pvar_count(int pvarindex) {
  if(!Tau_mpi_t_is_initialized()) Tau_mpi_t_initialize();
  if(pvarindex > tau_initial_pvar_count || pvarindex < 0) return -1;
  return tau_pvar_count[pvarindex];
}

/*Returns a pointer to the global PVAR session*/
MPI_T_pvar_session * Tau_mpi_t_get_pvar_session() {
   if(!Tau_mpi_t_is_initialized()) Tau_mpi_t_initialize();
   return &tau_pvar_session;
}

/*Returns a pointer to the global PVAR handles*/
MPI_T_pvar_handle * Tau_mpi_t_get_pvar_handles() {
  if(!Tau_mpi_t_is_initialized()) Tau_mpi_t_initialize();
  return tau_pvar_handles;
}

//////////////////////////////////////////////////////////////////////
int Tau_mpi_t_initialize(void) {
  int return_val, thread_provided, num_pvars, i, namelen, verb, varclass, bind, threadsup;
  int index;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;

  /* if TAU_TRACK_MPI_T_PVARS is not set to true, return with a success but do nothing 
   * to initialize MPI_T PVAR interface*/
  dprintf("TAU: Before MPI_T initialization\n");
  if (TauEnv_get_track_mpi_t_pvars() == 0) {
    return MPI_SUCCESS; 
  } 
  
  /*Return without doing anything if MPI_T is already initialized*/
   if(Tau_mpi_t_is_initialized()) {
    return MPI_SUCCESS;
  }

  /* Initialize MPI_T */
  return_val = MPI_T_init_thread(MPI_THREAD_SINGLE, &thread_provided);

  if (return_val != MPI_SUCCESS) 
  {
    perror("MPI_T_init_thread ERROR:");
    return return_val;
  }
  else {
    /* track a performance pvar session */
    return_val = MPI_T_pvar_session_create(&tau_pvar_session);
    if (return_val != MPI_SUCCESS) {
      perror("MPI_T_pvar_session_create ERROR:");
      return return_val;
    }  
  }

  dprintf("MPI_T initialized successfully!\n");
  /* get the number of pvars exported by the implmentation */
  return_val = MPI_T_pvar_get_num(&num_pvars);
  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_pvar_get_num ERROR:");
    return return_val;
  }
  dprintf("TAU MPI_T STARTED session: pvars exposed = %d\n", num_pvars);

  pvar_value_buffer = (unsigned long long int**)malloc(sizeof(unsigned long long int*) * (num_pvars + 1));
  tau_mpi_datatype = (MPI_Datatype *) malloc(sizeof(MPI_Datatype *) * (num_pvars+1));
  tau_pvar_handles = (MPI_T_pvar_handle*)malloc(sizeof(MPI_T_pvar_handle) * (num_pvars + 1));
  tau_pvar_count = (int*)malloc(sizeof(int) * (num_pvars + 1));
  memset(tau_pvar_count, 0, sizeof(int) * (num_pvars + 1));
  read_value_buffer = (void*)malloc(sizeof(unsigned long long int) * (TAU_NAME_LENGTH + 1));

  /* Initialize variables. Get the names of performance variables */
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
    tau_mpi_datatype[i] = datatype;

  /* allocate a pvar handle that will be used later */
  if(bind == MPI_T_BIND_MPI_COMM) {
    return_val = MPI_T_pvar_handle_alloc(tau_pvar_session, i, &default_comm, &tau_pvar_handles[i], &tau_pvar_count[i]);
  } else {
    return_val = MPI_T_pvar_handle_alloc(tau_pvar_session, i, NULL, &tau_pvar_handles[i], &tau_pvar_count[i]);
  }
  
  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_pvar_handle_alloc ERROR:");
    return return_val;
  }

  /*Non-continuous variables need to be started before being read. If this is not done
   *TODO:Currently, the MVAPICH and MPICH implementations error out if non-continuous PVARs are not started before being read.
   *Check if this is expected behaviour from an MPI implementation. No mention of the need to do this from a clients perspective in the 3.1 standard.*/
  if(continuous == 0) {
    return_val = MPI_T_pvar_start(tau_pvar_session, tau_pvar_handles[i]);
    if (return_val != MPI_SUCCESS) {
      perror("MPI_T_pvar_start ERROR:");
      return return_val;
    }
  }

  /* and a buffer to store the results in */
  pvar_value_buffer[i] = (unsigned long long int*)malloc(sizeof(unsigned long long int) * (tau_pvar_count[i] + 1));
  dprintf("Name: %s (%s), i = %d, varclass = %d, bind = %d, readonly = %d, continuous = %d, atomic = %d\n", event_name, description, i, varclass, bind, readonly, continuous, atomic);

  }
  
  Tau_allocate_pvar_event(num_pvars, tau_pvar_count);

  tau_initial_pvar_count = num_pvars;
  Tau_mpi_t_set_initialized();

  return return_val;
}

/*Cleanup by freeing handles and session*/
int Tau_mpi_t_cleanup(void) {
  int return_val, num_pvars, i;

  for(i=0;i < num_cvars;i++) {
    return_val = MPI_T_cvar_handle_free(&(tau_cvar_handles[i]));
    if (return_val != MPI_SUCCESS) {
      printf("TAU: Can't free cvar handle for index %d in this MPI implementation\n", i);
      return return_val;
    }
  }

  /*Don't free any handles and handles if nobody is tracking anything!*/
  if (TauEnv_get_track_mpi_t_pvars() == 0) {
    return MPI_SUCCESS;
  }

  /* get number of pvars from MPI_T */
  /*return_val = MPI_T_pvar_get_num(&num_pvars);
  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_pvar_get_num ERROR:");
    return return_val;
  }*/
  
  for(i=0; i < tau_initial_pvar_count; i++) {
    return_val = MPI_T_pvar_handle_free(tau_pvar_session, &(tau_pvar_handles[i]));
    if (return_val != MPI_SUCCESS) {
      perror("MPI_T_pvar_handle_free ERROR:");
      return return_val;
    }
  }
  
  return_val = MPI_T_pvar_session_free(&tau_pvar_session);
  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_pvar_session_free ERROR:");
    return return_val;
  }
  
  Tau_disable_tracking_mpi_t();
  return return_val;
}

/*Iterates through all the cvars provided by the implementation and prints the name and description of each cvar*/
int Tau_mpi_t_print_all_cvar_desc(int num_cvars) {
  int i, name_len, desc_len, verbosity, binding, scope, rank, return_val;
  char name[TAU_NAME_LENGTH]= "";
  char desc[TAU_NAME_LENGTH]= "";
  char tau_metric_name[TAU_NAME_LENGTH]= "";

  rank = Tau_get_node();
  MPI_Datatype datatype;
  MPI_T_enum enumtype;

  /*Iterate through the entire list of cvars exposed and print the name and description*/
  for (i=0; i < num_cvars; i++) {
    name_len = desc_len = TAU_NAME_LENGTH;
    return_val = MPI_T_cvar_get_info(i, name, &name_len, &verbosity, &datatype, &enumtype, desc, &desc_len, &binding, &scope);
    if (return_val != MPI_SUCCESS) {
      printf("TAU: Rank %d: Can't get cvar info i=%d, num_cvars=%d\n", rank, i, num_cvars);
      return return_val;
    }
    // This code gets called before MPI_Init has taken place. 
    // rank is -1 when TAU is linked in (as opposed to tau_exec). 
    //if (rank == 0) {
      dprintf("CVAR[%d] = %s \t \t desc = %s\n", i, name, desc);
      snprintf(tau_metric_name, sizeof(tau_metric_name),  "MPI_T CVAR: %s", name);
      TAU_METADATA(tau_metric_name, desc);
    //} 
  }

  return MPI_SUCCESS;
}

/*Parses cvar string using strtok, and separates the tokens using "," as the delimiter*/
void Tau_mpi_t_parse_cvar_string(int num_cvars, const char *cvar_string, char **cvar_array, int *number_of_elements) {

  /*A copy of the cvar string needs to be created because strtok doesn't accept const */
  char *cvar_copy = Tau_MemMgr_malloc(Tau_get_thread(), (strlen(cvar_string)+1)*sizeof(char));
  char *token;
  int current_index = 0;
  int iter = 0;
  char *save_ptr;

  strcpy(cvar_copy, cvar_string);
  token = strtok_r(cvar_copy, ",", &save_ptr);

  /*Initialize the string pointers to NULL*/
  for(iter=0; iter<num_cvars; iter++) {
    cvar_array[iter] = NULL;
  }

  /*Handling case where only one cvar metric or value is provided*/
  if(token == NULL) {
    cvar_array[current_index] = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char)*TAU_NAME_LENGTH);
    strcpy(cvar_array[current_index], cvar_string);
    *number_of_elements = 1;
  }

  /*Handling case where there are more than one cvar metric or values are provided. We choose to ignore
   *trailing "," */
  while(token != NULL && token != "") {
    cvar_array[current_index] = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char)*TAU_NAME_LENGTH);
    strcpy(cvar_array[current_index], token);
    current_index = current_index + 1;
    *number_of_elements = current_index;

    token = strtok_r(NULL, ",", &save_ptr);
  }

  //Tau_MemMgr_free(Tau_get_thread(), cvar_copy);
}

/*Iterates through the current list of vectors to see if the current vector element being added is new or already exists
 * If new, creates a new vector element. If vector element already exists, it appends a key value pair to the list of the corresponding vector element with the same name*/
void Tau_mpi_t_add_vector_element(VectorControlVariable *vectors, char *vector_name, char *key, int index_into_value_array, int *current_vector, char **cvar_values) {
  int iter, counter;
  ListStringPair *current_string_pair;

  for(iter = 0; iter < *current_vector; iter++) {
    //Found a vector with the same name that already exists
    if(strcmp(vectors[iter].name, vector_name) == 0) {
      counter = 0;
      current_string_pair = vectors[iter].list;
      while(counter < vectors[iter].number_of_elements - 1) {
        current_string_pair = current_string_pair->link;
        counter = counter + 1;
      }
      current_string_pair->link = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(ListStringPair));
      current_string_pair = current_string_pair->link;
      strcpy(current_string_pair->pair.first, key);
      strcpy(current_string_pair->pair.second, cvar_values[index_into_value_array]);
      vectors[iter].number_of_elements = vectors[iter].number_of_elements + 1;
      return;
    }
  }
  //Vector with this name doesn't exist. Add it!
  strcpy(vectors[*current_vector].name, vector_name);
  vectors[*current_vector].number_of_elements = 1;
  vectors[*current_vector].list = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(ListStringPair));
  strcpy(vectors[*current_vector].list->pair.first, key);
  strcpy(vectors[*current_vector].list->pair.second, cvar_values[index_into_value_array]);
  vectors[*current_vector].list->link = NULL;
  current_string_pair = NULL;
  *current_vector = *current_vector + 1;

}
 
/*Counts number of scalar and vector metrics*/
void Tau_mpi_t_count_scalar_and_vector_metrics(const char *cvars, char **cvar_metrics, int num_cvar_metrics, int *num_scalars, int *num_vectors) {

  int iter, current_scalar, current_vector;
  char *vector_metric_names;
  char *temporary_string;
  char *temporary_string2;
  int position_of_open_bracket, position_of_close_bracket;

  *num_scalars = 0;
  *num_vectors = 0;
  current_scalar = current_vector = 0;

  temporary_string = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char)*TAU_NAME_LENGTH);
  temporary_string2 = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char)*TAU_NAME_LENGTH);
  vector_metric_names = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char)*strlen(cvars));

  /*Iterates through all cvar metrics to find out how many scalar and vector elements are present. Vector elements in the cvar metrics 
   * string are considered unique only if they have different names*/
  for(iter=0; iter < num_cvar_metrics; iter++) {
    if(strchr(cvar_metrics[iter],'[') != NULL) {
        position_of_open_bracket = strcspn(cvar_metrics[iter], "[");
        strncpy(temporary_string, cvar_metrics[iter], position_of_open_bracket);
        temporary_string[position_of_open_bracket] = ',';
        temporary_string[position_of_open_bracket+1] = '\0';
        if(strstr(vector_metric_names,temporary_string) == NULL) {
          *num_vectors = *num_vectors + 1;
        }
        strcat(vector_metric_names, temporary_string);
    } else {
      *num_scalars = *num_scalars + 1;
    }
  }

  ///Tau_MemMgr_free(Tau_get_thread(), temporary_string); Tau_MemMgr_free(Tau_get_thread(), temporary_string2); Tau_MemMgr_free(Tau_get_thread(), vector_metric_names);
}

/*Separates out scalar and vector elements into two different lists for ease of use*/
void Tau_mpi_t_map_cvar_metrics_to_values(char **cvar_metrics, char **cvar_values, int num_cvar_metrics, VectorControlVariable *vectors,ScalarControlVariable *scalars, int num_scalars, int num_vectors) {
  int iter, current_scalar, current_vector;
  char *vector_metric_names;
  char *temporary_string;
  char *temporary_string2;
  int position_of_open_bracket, position_of_close_bracket;

  current_scalar = current_vector = 0;

  temporary_string = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char)*TAU_NAME_LENGTH);
  temporary_string2 = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char)*TAU_NAME_LENGTH);

  /*Logic for separating the scalars and vectors into two separate lists*/
  for(iter=0; iter < num_cvar_metrics; iter++) {
    if(strchr(cvar_metrics[iter],'[') == NULL) {
      strcpy(scalars[current_scalar].name, cvar_metrics[iter]);
      strcpy(scalars[current_scalar].value, cvar_values[iter]);
      current_scalar = current_scalar + 1;
    } else {
        position_of_open_bracket = strcspn(cvar_metrics[iter], "[");
        position_of_close_bracket = strcspn(cvar_metrics[iter], "]");
        strncpy(temporary_string, cvar_metrics[iter], position_of_open_bracket); //Copy in the name of the vector
        strncpy(temporary_string2, &cvar_metrics[iter][position_of_open_bracket+1], position_of_close_bracket - position_of_open_bracket - 1); //Copy in the number contained with the [].
        temporary_string[position_of_open_bracket] = '\0';
        temporary_string2[position_of_close_bracket - position_of_open_bracket - 1] = '\0';
        Tau_mpi_t_add_vector_element(vectors, temporary_string, temporary_string2, iter, &current_vector, cvar_values);
    }
  }

  //Tau_MemMgr_free(Tau_get_thread(), temporary_string); Tau_MemMgr_free(Tau_get_thread(), temporary_string2);
}

/*Type the read, write and reset_value buffers depending on the MPI_Datatype of the cvar*/
void Tau_mpi_t_allocate_read_write_buffers(void **val, void **oldval, void **reset_value, MPI_Datatype datatype, int num_vals) {

  if(datatype == MPI_UNSIGNED || datatype == MPI_UNSIGNED_LONG || datatype == MPI_UNSIGNED_LONG_LONG || datatype == MPI_COUNT) {
    *val = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(unsigned long long));
    *oldval = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(unsigned long long));
    *reset_value = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(unsigned long long));
  }

  if(datatype == MPI_INT) {
    *val = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(signed int));
    *oldval = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(signed int));
    *reset_value = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(signed int));
  }

  if(datatype == MPI_CHAR) {
    *val = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(char));
    *oldval = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(char));
    *reset_value = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(char));
  }

  if(datatype == MPI_DOUBLE) {
    *val = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(double));
    *oldval = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(double));
    *reset_value = Tau_MemMgr_malloc(Tau_get_thread(), num_vals*sizeof(double));
  }
}

/*This function verifies that the scalar cvar metric has been written correctly by reading and checking against the reset_value. Logs metadata*/
void Tau_mpi_t_verify_write_and_log_scalar_metadata(int rank, void *val, void *reset_value, void *oldval, MPI_Datatype datatype, int num_vals, char *metric_name, char desc[TAU_NAME_LENGTH]) {
  char metastring[TAU_NAME_LENGTH] = "";
  char tau_metric_name[TAU_NAME_LENGTH] = "";

  if(datatype == MPI_UNSIGNED || datatype == MPI_UNSIGNED_LONG || datatype == MPI_UNSIGNED_LONG_LONG || datatype == MPI_COUNT) {
    if ((rank == 0) && (*(unsigned long long*)reset_value == *(unsigned long long*)val)) {
      dprintf("ResetValue %lld matches what we set for cvars=%s\n", *(unsigned long long*)reset_value, metric_name);
      snprintf(metastring, sizeof(metastring), "%lld (old) -> %lld (new), %s", *(unsigned long long*)oldval, *(unsigned long long*)reset_value, desc);
    }
  } else if(datatype == MPI_INT) {
    if ((rank == 0) && (*(signed int*)reset_value == *(signed int*)val)) {
      dprintf("ResetValue %d matches what we set for cvars=%s\n", *(signed int*)reset_value, metric_name);
      snprintf(metastring, sizeof(metastring), "%d (old) -> %d (new), %s", *(signed int*)oldval, *(signed int*)reset_value, desc);
    }
  } else if(datatype == MPI_DOUBLE) {
    if ((rank == 0) && (*(double*)reset_value == *(double*)val)) {
      dprintf("ResetValue %lf matches what we set for cvars=%s\n", *(double*)reset_value, metric_name);
      snprintf(metastring, sizeof(metastring), "%lf (old) -> %lf (new), %s", *(double*)oldval, *(double*)reset_value, desc);
    }
  } else if(datatype == MPI_CHAR) {
    if ((rank == 0) && (strcmp((char*)reset_value, (char*)val) == 0)) {
      dprintf("ResetValue %s matches what we set for cvars=%s\n", (char *)reset_value, metric_name);
      snprintf(metastring, sizeof(metastring), "%s (old) -> %s (new), %s", (char *)oldval, (char *)reset_value, desc);
    }
  }
 
  if (rank == 0) {
    snprintf(tau_metric_name, sizeof(tau_metric_name),  "MPI_T CVAR: %s", metric_name);
    TAU_METADATA(tau_metric_name, metastring);
  }
}

/*This function verifies that the vector cvar metric has been written correctly by reading and checking against the reset_value. Logs metadata*/
void Tau_mpi_t_verify_write_and_log_vector_metadata(int rank, void *val, void *reset_value, void *oldval, MPI_Datatype datatype, int num_vals, char *metric_name) {

  int is_reset_value_equal_to_written_value = 1;
  int iter;
  char old_value_string[TAU_NAME_LENGTH] = "";
  char reset_value_string[TAU_NAME_LENGTH] = "";
  char metastring[TAU_NAME_LENGTH] = "";
  char metric_name_with_array_index[TAU_NAME_LENGTH] = "";

/*First check the datatype of the values, and then accordingly log metadata*/
  if(datatype == MPI_UNSIGNED || datatype == MPI_UNSIGNED_LONG || datatype == MPI_UNSIGNED_LONG_LONG || datatype == MPI_COUNT) {
    for(iter=0; iter < num_vals; iter++) {
      if(((unsigned long long*)reset_value)[iter] != ((unsigned long long*)val)[iter]) {
        is_reset_value_equal_to_written_value = 0;
        break;
      }
    }
    if((rank == 0) && is_reset_value_equal_to_written_value) {
      dprintf("ResetValue matches what we set for cvars=%s\n", metric_name);
      for(iter=0; iter < num_vals; iter++) {
        if(((unsigned long long*)oldval)[iter] != ((unsigned long long*)reset_value)[iter]) {
          snprintf(old_value_string, sizeof(old_value_string), "%lld,",((unsigned long long*)oldval)[iter]);
          snprintf(reset_value_string, sizeof(reset_value_string), "%lld,",((unsigned long long*)reset_value)[iter]);
          snprintf(metastring, sizeof(metastring), "%s (old) -> %s (new)", old_value_string, reset_value_string);
          snprintf(metric_name_with_array_index, sizeof(metric_name_with_array_index), "MPI_T CVAR: %s[%d]", metric_name, iter);
          TAU_METADATA(metric_name_with_array_index, metastring);
          strcpy(old_value_string, ""); strcpy(reset_value_string, ""); strcpy(metastring, ""); strcpy(metric_name_with_array_index, "");
        }
      }
    }
  } else if(datatype == MPI_INT) {
    for(iter=0; iter < num_vals; iter++) {
      if(((signed int*)reset_value)[iter] != ((signed int*)val)[iter]) {
        is_reset_value_equal_to_written_value = 0;
        break;
      }
    }
    if((rank == 0) && is_reset_value_equal_to_written_value) {
      dprintf("ResetValue matches what we set for cvars=%s\n", metric_name);
      for(iter=0; iter < num_vals; iter++) {
        if(((signed int*)oldval)[iter] != ((signed int*)reset_value)[iter]) {
          snprintf(old_value_string, sizeof(old_value_string), "%lld,",((signed int*)oldval)[iter]);
          snprintf(reset_value_string, sizeof(reset_value_string), "%lld,",((signed int*)reset_value)[iter]);
          snprintf(metastring, sizeof(metastring), "%s (old) -> %s (new)", old_value_string, reset_value_string);
          snprintf(metric_name_with_array_index, sizeof(metric_name_with_array_index), "MPI_T CVAR: %s[%d]", metric_name, iter);
          TAU_METADATA(metric_name_with_array_index, metastring);
          strcpy(old_value_string, ""); strcpy(reset_value_string, ""); strcpy(metastring, ""); strcpy(metric_name_with_array_index, "");
        }
      }
    }
  } else if(datatype == MPI_DOUBLE) {
    for(iter=0; iter < num_vals; iter++) {
      if(((double*)reset_value)[iter] != ((double*)val)[iter]) {
        is_reset_value_equal_to_written_value = 0;
        break;
      }
    }
    if((rank == 0) && is_reset_value_equal_to_written_value) {
      dprintf("ResetValue matches what we set for cvars=%s\n", metric_name);
      for(iter=0; iter < num_vals; iter++) {
        if(((double*)oldval)[iter] != ((double*)reset_value)[iter]) {
          snprintf(old_value_string, sizeof(old_value_string), "%lld,",((double*)oldval)[iter]);
          snprintf(reset_value_string, sizeof(reset_value_string), "%lld,",((double*)reset_value)[iter]);
          snprintf(metastring, sizeof(metastring), "%s (old) -> %s (new)", old_value_string, reset_value_string);
          snprintf(metric_name_with_array_index, sizeof(metric_name_with_array_index), "MPI_T CVAR: %s[%d]", metric_name, iter);
          TAU_METADATA(metric_name_with_array_index, metastring);
          strcpy(old_value_string, ""); strcpy(reset_value_string, ""); strcpy(metastring, ""); strcpy(metric_name_with_array_index, "");
        }
      }
    }
  }
}
/*Convert a string to a typed value, depending on the MPI_Datatype.*/
void Tau_mpi_t_convert_string_to_typed_value(char *string_value, MPI_Datatype datatype, void *val, int index) {
  if(datatype == MPI_UNSIGNED || datatype == MPI_UNSIGNED_LONG || datatype == MPI_UNSIGNED_LONG_LONG || datatype == MPI_COUNT) {
    sscanf(string_value, "%lld", &((unsigned long long*)val)[index]);
  } else if(datatype == MPI_INT) {
    sscanf(string_value, "%d", &((signed int*)val)[index]);
  } else if(datatype == MPI_CHAR) {
    sscanf(string_value, "%s", &((char*)val)[index]);
  } else if(datatype == MPI_DOUBLE) {
    sscanf(string_value, "%lf", &((double*)val)[index]);
  }
}

int Tau_mpi_t_parse_and_write_cvars(const char *cvars, const char *values) {

      int cindex, num_cvar_metrics, num_cvar_values, num_scalars, num_vectors, array_index, thread_provided;
      int return_val, iter, rank;
       
      char name[TAU_NAME_LENGTH]= ""; 
      char desc[TAU_NAME_LENGTH]= ""; 
      int verbosity, binding, scope, i, j, k;
      int name_len;
      int desc_len;
      MPI_Datatype datatype; 
      MPI_T_enum enumtype; 
      char **cvar_metrics_array, **cvar_values_array;
      VectorControlVariable *vectors;
      ScalarControlVariable *scalars;
      void *val, *oldval, *reset_value;
      ListStringPair *current_string_pair;

      rank = Tau_get_node(); /* MPI_Init may not been invoked yet */

      /*Initialize the MPI_T interface in case we have not done so already. No harm in re-initializing the interface.*/
      return_val = MPI_T_init_thread(MPI_THREAD_SINGLE, &thread_provided);

      if (return_val != MPI_SUCCESS) {
        perror("MPI_T_init_thread ERROR:");
        return return_val;
      }

      cvar_metrics_array = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char*)*num_cvars);
      cvar_values_array = Tau_MemMgr_malloc(Tau_get_thread(), sizeof(char*)*num_cvars);

      /*Parse the metrics and values using strtok. At this stage, we aren't concerned with differenting between scalar and vector metrics.*/
      Tau_mpi_t_parse_cvar_string(num_cvars, cvars, cvar_metrics_array, &num_cvar_metrics);
      Tau_mpi_t_parse_cvar_string(num_cvars, values, cvar_values_array, &num_cvar_values);
   

      /*Checking if number of cvar metrics and cvar values provided are equal*/
      if(num_cvar_metrics != num_cvar_values) { 
        printf("TAU: Rank %d: Number of CVAR metrics don't match number of CVAR values provided. Please check environment variables TAU_MPI_T_CVAR_METRICS and TAU_MPI_T_CVAR_VALUES", rank);
	return return_val;
      }
 
      /*Counts the number of scalar and vector metrics*/ 
      Tau_mpi_t_count_scalar_and_vector_metrics(cvars, cvar_metrics_array, num_cvar_metrics, &num_scalars, &num_vectors);


      scalars = Tau_MemMgr_malloc(Tau_get_thread(), (num_scalars)*sizeof(ScalarControlVariable));
      vectors = Tau_MemMgr_malloc(Tau_get_thread(), (num_vectors)*sizeof(VectorControlVariable));

      /*Maps cvar metrics to values, handling both scalars and vectors appropriately*/
      Tau_mpi_t_map_cvar_metrics_to_values(cvar_metrics_array, cvar_values_array, num_cvar_metrics, vectors, scalars, num_scalars, num_vectors);
      
      for (i=0; i < num_cvars; i++) {
        name_len = desc_len = TAU_NAME_LENGTH;
        return_val = MPI_T_cvar_get_info(i, name, &name_len, &verbosity, &datatype, &enumtype, desc, &desc_len, &binding, &scope);
        if (return_val != MPI_SUCCESS) {
	  printf("TAU: Rank %d: Can't get cvar info i=%d, num_cvars=%d\n", rank, i, num_cvars);
	  return return_val; 
        }

        /*Iterate through vector metrics*/
        for(iter=0;iter < num_vectors; iter++) {
          if (strcmp(name, vectors[iter].name)==0) {
            if (rank == 0) {
              dprintf("Rank: %d FOUND CVAR match: %s, cvars = %s, desc = %s\n", rank, name, vectors[iter].name, desc);
            }
            cindex = i;

            Tau_mpi_t_allocate_read_write_buffers(&val, &oldval, &reset_value, datatype, tau_cvar_num_vals[cindex]);

            return_val = MPI_T_cvar_read(tau_cvar_handles[cindex], oldval);
            
            return_val = MPI_T_cvar_read(tau_cvar_handles[cindex], val);

            if (return_val != MPI_SUCCESS) {
               printf("TAU: Rank %d: Can't read cvar %s in this MPI implementation\n", rank, vectors[iter].name);
               return return_val;
            } else {
               //TODO: Print old values based on type
            }
           
            current_string_pair = vectors[iter].list;
            k = 0;
            while(k < vectors[iter].number_of_elements) {
              sscanf(current_string_pair->pair.first, "%d", &array_index);
              Tau_mpi_t_convert_string_to_typed_value(current_string_pair->pair.second, datatype, val, array_index);
              current_string_pair = current_string_pair->link;
              k = k + 1;
            }

            return_val = MPI_T_cvar_write(tau_cvar_handles[cindex], val);
            if (return_val != MPI_SUCCESS) {
               printf("TAU: Rank %d: Can't write cvar %s in this MPI implementation\n", rank, vectors[iter].name);
               return return_val;
            }
            return_val = MPI_T_cvar_read(tau_cvar_handles[cindex], reset_value);

            if (return_val != MPI_SUCCESS) {
              printf("TAU: Rank %d: Can't read cvar %s in this MPI implementation\n", rank, vectors[iter].name);
              return return_val;
            } else {
              Tau_mpi_t_verify_write_and_log_vector_metadata(rank, val, reset_value, oldval, datatype, tau_cvar_num_vals[cindex], vectors[iter].name);
            }
            //Tau_MemMgr_free(Tau_get_thread(), val); Tau_MemMgr_free(Tau_get_thread(), oldval); Tau_MemMgr_free(Tau_get_thread(), reset_value);

          }
        }
        /*Iterate through all the scalar metrics*/
        for(iter=0;iter < num_scalars; iter++) {
          if (strcmp(name, scalars[iter].name)==0) {
            if (rank == 0) {
              dprintf("Rank: %d FOUND CVAR match: %s, cvars = %s, desc = %s\n", rank, name, scalars[iter].name, desc);
            }
            cindex = i;
            /*This check is in place to handle the case where user inputs scalar values for a cvar that is a vector, except for a variable of type MPI_CHAR*/
            if(tau_cvar_num_vals[cindex] != 1 && datatype != MPI_CHAR) { 
              printf("TAU: Rank %d: CVAR %s assumed scalar by user, but is actually a vector\n", rank, scalars[iter].name);
              return return_val;
            }
 
            Tau_mpi_t_allocate_read_write_buffers(&val, &oldval, &reset_value, datatype, tau_cvar_num_vals[cindex]);

            return_val = MPI_T_cvar_read(tau_cvar_handles[cindex], oldval);
            if (return_val != MPI_SUCCESS) {
               printf("TAU: Rank %d: Can't read cvar %s in this MPI implementation\n", rank, scalars[iter].name);
               return return_val;
            } else {
              if (rank == 0) {
                //TODO: Print old values based on type dprintf("Oldval = %d, newval=%lld, cvars=%s\n", oldval, val, cvar_metrics_array[iter]);
              }
            }
           
            array_index = 0; //for scalars, array index into the val array is 0 as there is only one element
            Tau_mpi_t_convert_string_to_typed_value(scalars[iter].value, datatype, val, array_index);

            return_val = MPI_T_cvar_write(tau_cvar_handles[cindex], val);
            if (return_val != MPI_SUCCESS) {
               printf("TAU: Rank %d: Can't write cvar %s in this MPI implementation\n", rank, scalars[iter].name);
               return return_val;
            }
            return_val = MPI_T_cvar_read(tau_cvar_handles[cindex], reset_value);
            if (return_val != MPI_SUCCESS) {
              printf("TAU: Rank %d: Can't read cvar %s in this MPI implementation\n", rank, scalars[iter].name);
              return return_val;
            } else {
              Tau_mpi_t_verify_write_and_log_scalar_metadata(rank, val, reset_value, oldval, datatype, tau_cvar_num_vals[cindex], scalars[iter].name, desc);
            }
            //Tau_MemMgr_free(Tau_get_thread(), val); Tau_MemMgr_free(Tau_get_thread(), oldval); Tau_MemMgr_free(Tau_get_thread(), reset_value);
         
          }
        }

      /* NOT implemented: return_val = MPI_T_cvar_get_index(cvars, &cindex);
      if (return_val != MPI_SUCCESS) { 
        printf("TAU: Rank %d: Can't access MPI_T variable %s in this MPI implementation\n", Tau_get_node(), cvars);
        return return_val;
      }
      */
      }  

      //Tau_MemMgr_free(Tau_get_thread(), cvar_metrics_array); Tau_MemMgr_free(Tau_get_thread(), cvar_values_array); Tau_MemMgr_free(Tau_get_thread(), vectors); Tau_MemMgr_free(Tau_get_thread(), scalars);

      /*Finalize the MPI_T since we called MPI_T_init_thread above. This ensure the tools interface is back to the state
       *state it was in before this function call.*/
      /*return_val = MPI_T_finalize();
      if (return_val != MPI_SUCCESS) {
        printf("TAU: Rank %d: Call to MPI_T_finalize failed\n", rank);
        return return_val;
      }*/

      return return_val;
}

/*Read CVARs to be set and their values from the environment and write them to the MPI_T interface */
int Tau_mpi_t_cvar_initialize(void) {
  int return_val, iter, thread_provided, rank, i;
  const char *cvars = TauEnv_get_cvar_metrics();
  const char *values = TauEnv_get_cvar_values();
  char metastring[TAU_NAME_LENGTH] = "";

  /* MPI_Comm_rank(MPI_COMM_WORLD, &rank); */
  rank = Tau_get_node(); /* MPI_Init has not been invoked yet */

  /*Initialize the MPI_T interface in case we have not done so already. This can happen when performance variables are not being tracked.*/
  return_val = MPI_T_init_thread(MPI_THREAD_SINGLE, &thread_provided);

  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_init_thread ERROR:");
    return return_val;
  }

  /* get number of cvars from MPI_T */
  return_val = MPI_T_cvar_get_num(&num_cvars);
  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_cvar_get_num ERROR:");
    return return_val;
  }

  tau_cvar_handles = Tau_MemMgr_malloc(Tau_get_thread(), num_cvars*sizeof(MPI_T_cvar_handle));
  tau_cvar_num_vals = Tau_MemMgr_malloc(Tau_get_thread(), num_cvars*sizeof(int));

  for(i=0;i < num_cvars;i++) {
    return_val = MPI_T_cvar_handle_alloc(i, NULL, &(tau_cvar_handles[i]), &(tau_cvar_num_vals[i]));
    if (return_val != MPI_SUCCESS) {
      printf("TAU: Can't allocate cvar handle for index %d in this MPI implementation\n", i);
      return return_val;
    }
  }

#ifdef TAU_MPI_T_TRACK_GPU_MSGS
  Tau_msg_init(); 
#endif /* TAU_MPI_T_TRACK_GPU_MSGS */

  return_val = Tau_mpi_t_print_all_cvar_desc(num_cvars);
  if(return_val != MPI_SUCCESS) {
    printf("TAU: Rank %d: Can't read the MPI_T control variables in this MPI implementation\n", rank);
    return return_val;
  }

  if (strcmp(cvars, "") == 0) {
    dprintf("TAU: No CVARS specified using TAU_MPI_T_CVAR_METRICS and TAU_MPI_T_CVAR_VALUES\n");

  } else {
    dprintf("CVAR_METRICS=%s\n", cvars);
    if (strcmp(values, "") == 0) {
      printf("TAU: WARNING: Environment variable TAU_MPI_T_CVAR_METRICS is not specified for TAU_MPI_T_CVAR_METRICS=%s\n", 
	cvars);
    } else { // both cvars and values are specified*/
      /*Parse the cvars and values and write to the MPI_T interface */
      return_val = Tau_mpi_t_parse_and_write_cvars(cvars, values);

      TAU_METADATA("TAU_MPI_T_CVAR_METRICS", cvars);
      TAU_METADATA("TAU_MPI_T_CVAR_VALUES", values);
      snprintf(metastring, sizeof(metastring),  "%d", TauEnv_get_track_mpi_t_pvars());
      TAU_METADATA("TAU_TRACK_MPI_T_PVARS", metastring);
    }
  }

  return return_val; 
}

//////////////////////////////////////////////////////////////////////
int Tau_track_mpi_t_here(void) {
  int return_val, num_pvars, i;
  int index;
  MPI_Datatype datatype;
  MPI_T_enum enumtype;
  int mpi_t_enable_user_tuning_policy = 0;

  /* if TAU_TRACK_MPI_T_PVARS is not set to true, return with a success but do nothing 
   * to process MPI_T events */
  if (TauEnv_get_track_mpi_t_pvars() == 0) {
    return MPI_SUCCESS; 
  } 

  /*Double check to ensure that the MPI_T interface is initialized*/
  return_val = Tau_mpi_t_initialize();
  if(return_val != MPI_SUCCESS) 
    return return_val;

  /* get number of pvars from MPI_T */
  return_val = MPI_T_pvar_get_num(&num_pvars);
  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_pvar_get_num ERROR:");
    return return_val;
  }

  /* The MPI library can dynamically increase the number of PVARs exposed during runtime
   * We don't currently support this feature because we need to re-allocate PVAR handles 
   * and other data-structures inside a signal-handler, and calling malloc inside a 
   * signal handler is unsafe. 
   * Until we find an alternate mechanism, we will just read the number of pvars that
   * were exposed during  Tau_mpi_t_initialize() */
  if (num_pvars > tau_initial_pvar_count) {
    dprintf("TAU: Number of PVARS increased from %d to %d. Only reading %d PVARS\n", tau_initial_pvar_count, num_pvars, tau_initial_pvar_count);

  }

  int rank = Tau_get_node(); 
  int size, j; 

  
  for(i = 0; i < tau_initial_pvar_count; i++) {
    // get data from event
    MPI_T_pvar_read(tau_pvar_session, tau_pvar_handles[i], read_value_buffer);

    /*This statement causes a deadlock with MVAPICH2 when using large process counts.
      Issue is within MVAPICH2 - they are using some SIGNAL unsafe routines which cause a hang
      when we are sampling*/
    //MPI_Type_size(tau_mpi_datatype[i], &size); 

    for(j = 0; j < tau_pvar_count[i]; j++) {
      pvar_value_buffer[i][j] = 0;
    }
     
    //Please do not memcpy inside this routine! It is NOT signal safe and will
    //cause you unending pain through deadlocks...
    for(j = 0; j < tau_pvar_count[i]; j++)
     pvar_value_buffer[i][j] = ((unsigned long long int*)read_value_buffer)[j];

    for(j = 0; j < tau_pvar_count[i]; j++){
      // unsigned long long int to double conversion can result in an error. 
      // We first convert it to a long long int.
      long long int mydata = (long long int) pvar_value_buffer[i][j]; 
      int is_double = 0; 
      if (tau_mpi_datatype[i] == MPI_DOUBLE) is_double=1; 
      if (is_double) {
        // long long int mydata_d = (long long int) (pvar_value_buffer[i][j]); 
        // First convert it from unsigned long long to long long and then to a double
        //double double_data = ((double)(pvar_value_buffer[i][j]));
        double double_data = ((double)(mydata));

        dprintf("RANK:%d: pvar_value_buffer[%d][%d]=%lld, double_data=%g, is_double=%d\n",rank,i,j,mydata, double_data, is_double);
	if (double_data > 1e-14 )  {
      
        // Double values are really large for timers. Please check 1E18?? 
          //Tau_track_pvar_event(i, num_pvars, double_data);
          dprintf("Not tracking anything for now \n");
        } 
      } else {
        dprintf("RANK:%d: pvar_value_buffer[%d][%d]=%lld, is_double=%d\n",rank,i,j,mydata, is_double);
        // Trigger the TAU event if it is non-zero
	if (mydata > 0L) {
          Tau_track_pvar_event(i, j, tau_pvar_count, tau_initial_pvar_count, mydata);
     
          //TauInternalFunctionGuard protects_this_function; 
          dprintf("MPI_T: pvar initial count=%d, pvar count=%d, pvar index=%d, pvar sub index=%d, pvar value=%llu\n", tau_initial_pvar_count, tau_pvar_count[i], i, j, mydata);
          /*Invoke plugins only if both plugin path and plugins are specified*/
          /*if(TauEnv_get_plugins_enabled()) {
            Tau_plugin_event_mpit_data_t plugin_data;
 
            char * pvar_name_char = Tau_get_pvar_name(i,j);
            plugin_data.pvar_name = Tau_get_pvar_name(i,j);
            plugin_data.pvar_index = j;
            plugin_data.pvar_value = mydata;
            dprintf("MPI-T invoke callback\n"); 
            Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_MPIT, plugin_data.pvar_name, &plugin_data);
          }*/
 
        }
        /*Invoke plugins only if both plugin path and plugins are specified*/
        if(TauEnv_get_plugins_enabled()) {
          Tau_plugin_event_mpit_data_t plugin_data;

          char * pvar_name_char = Tau_get_pvar_name(i,j);
          plugin_data.pvar_name = Tau_get_pvar_name(i,j);
          plugin_data.pvar_index = j;
          plugin_data.pvar_value = mydata;
          dprintf("MPI-T invoke callback\n");
          Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_MPIT, plugin_data.pvar_name, &plugin_data);
        }

      }
    }

  }

  dprintf("Finished!!\n");
}

////////////////////////////////////////////////////////////////////////////////////
int Tau_set_mpi_t_cvar_value_for_communicator(char* cname, int value, void* comm, const char *comm_name){
  MPI_T_cvar_handle handle;
  int err = MPI_SUCCESS;
  int read_value, count, cvar_index = 0;
 
#if (defined MVAPICH2_NUMVERSION && MVAPICH2_NUMVERSION > 20200300) || (!defined MVAPICH2_NUMVERSION)
  printf("TAU: rank [%d] Tau_set_mpi_t_cvar_value_for_communicator: CVAR name=%s, value = %d, comm_name=%s, comm = %p\n", Tau_get_node(), cname, value, comm_name, comm);
  err = MPI_T_cvar_get_index( cname , &cvar_index);
  if(err!=MPI_SUCCESS) {
    fprintf(stderr,"\nTAU: Error: cvar_get_index err=%d\n",err);
  }
  else {
    dprintf("Comm_name = %s, cvar = %s, value = %d, cvar index = %d, comm = %p\n", 
      comm_name, cname, value, cvar_index, comm); 
  } 
  err = MPI_T_cvar_handle_alloc(cvar_index, comm, &handle, &count);
  if(err!=MPI_SUCCESS) {
    fprintf(stderr, "\nTAU: Error: cvar_handle_alloc err=%d\n", err);
  }
  
  err = MPI_T_cvar_read(handle, &read_value);
  if(err!=MPI_SUCCESS) {
    fprintf(stderr, "\nTAU: Error cvar_read error err=%d\n", err);
  }

  err = MPI_T_cvar_write(handle, &value);
  if(err!=MPI_SUCCESS) {
    fprintf(stderr,"\nTAU: Error: cvar_write err=%d\n",err);
  }

  /* check if the cvar can be read properly - it is the same as what we wrote above. */
  err = MPI_T_cvar_read(handle, &read_value);
  if(err!=MPI_SUCCESS) {
    fprintf(stderr, "\nTAU: Error: cvar_read err=%d\n", err);
  }


  /* free the handle. */
  MPI_T_cvar_handle_free(&handle);
#else
  fprintf(stderr, "\nTAU: Warning: Communicator-based CVAR support not available for MVAPICH2.2 and older. Please use a newer MVAPICH2 version.\n");
#endif
  return 0;
}

void Tau_show_mpi_t_usage() {
  if (Tau_get_node() == 0) {
    printf("TAU: Error: Please set TAU_MPI_T_COMM_METRIC_VALUES environment variable to <comm_name,cvar_name,cvar_value>,<...>\n");
    return;  
  }
}
//////////////////////////////////////////////////////////////////////
void Tau_mpi_t_check_communicator(void *comm, const char *comm_name) {
  const char *communicator;
  const char *cvar_metric;
  const char *cvar_value;
  const char *comm_metric_values = TauEnv_get_mpi_t_comm_metric_values(); 
  
  dprintf("TAU_MPI_T_Check_communicator: comm: %p, name = %s\n", comm, comm_name);
  if (TauEnv_get_track_mpi_t_comm_metric_values) {
    //comm_metric_values = TauEnv_get_mpi_t_comm_metric_values();
    char *cvar_copy = Tau_MemMgr_malloc(Tau_get_thread(), (strlen(comm_metric_values)+1)*sizeof(char));
    char *save_ptr, *token, *start;
    char *comm_extracted, *metric;
    char *value;

    strcpy(cvar_copy, comm_metric_values);
    dprintf("TAU: cvar_copy = %s\n comm=%s\n", cvar_copy, comm_name); 
    
    while(cvar_copy) {
      dprintf("TAU:Tau_mpi_t_check_communicator: In loop: rank=%d: cvar_copy=%s\n", Tau_get_node(), cvar_copy);
      start = strstr(cvar_copy, comm_name);
      save_ptr = NULL;
      if(start) {
        dprintf("cvar_copy = %s, start=%s\n", cvar_copy, start); 
        comm_extracted = strtok_r(start, ",", &save_ptr);
        if(comm_extracted) { 
          dprintf("CVAR Communicator Name = %s\n", comm_extracted);
        } else {
          Tau_show_mpi_t_usage();  
          return;
        } 
        metric = strtok_r(NULL, ",", &save_ptr);
        if(metric) {
          dprintf("CVAR METRIC = %s\n", metric);
        } else {
          Tau_show_mpi_t_usage();  
          return;
        } 
        value = strtok_r(NULL, ",", &save_ptr);
        if(value) {
          dprintf("CVAR VALUE = %s\n", value);
          TAU_METADATA(metric, value);
        } else {
          Tau_show_mpi_t_usage();
          return;
        }
        int intvalue;
        sscanf(value, "%d", &intvalue); 
        // We now have the metric, integer value, and the communicator address:
        Tau_set_mpi_t_cvar_value_for_communicator(metric, intvalue, comm, comm_name);
        //printf("save_ptr = %s\n", save_ptr);
      }
      cvar_copy = save_ptr; 
    }
  }
}

/* Here we define all the profile path based tracking calls */
typedef struct {
    /* pvar metadata variables */
    char        *name;
    char        *desc;
    int         name_len;
    int         desc_len;
    int         verbosity;
    int         var_class;
    int         bind;
    int         readonly;
    int         continuous;
    int         atomic;
    int         count;
    MPI_Datatype    datatype;
    MPI_T_enum      enumtype;
} pvar_data_t;

static MPI_T_pvar_handle tau_handle_send;
static MPI_T_pvar_handle tau_handle_recv;
static MPI_T_pvar_session tau_session; 
static unsigned long long * tau_send_value;
static unsigned long long * tau_recv_value;
static int tau_send_count = 0;
static int tau_recv_count = 0;


//////////////////////////////////////////////////////////////////////
int Tau_msg_init(void){
  int err, i=0; 
  char *send_name = malloc(sizeof(char) * TAU_NAME_LENGTH);
  char *send_description = malloc(sizeof(char) * TAU_NAME_LENGTH);
  char *recv_name = malloc(sizeof(char) * TAU_NAME_LENGTH);
  char *recv_description = malloc(sizeof(char) * TAU_NAME_LENGTH);
  int pvar_send_index = 0;
  int pvar_recv_index = 0;
  pvar_data_t *pvar_send = malloc(sizeof(pvar_data_t));
  pvar_data_t *pvar_recv = malloc(sizeof(pvar_data_t));
  pvar_send->name_len = TAU_NAME_LENGTH;
  pvar_send->desc_len = TAU_NAME_LENGTH;
  pvar_recv->name_len = TAU_NAME_LENGTH;
  pvar_recv->desc_len = TAU_NAME_LENGTH;

  MPI_T_pvar_session_create(&tau_session);
  /* For MVAPICH2: check if the link utilization for send is there */
  PMPI_T_pvar_get_index("mv2_hardware_link_utilization_send", MPI_T_PVAR_CLASS_COUNTER, &pvar_send_index);
  PMPI_T_pvar_get_info(pvar_send_index,
            send_name,
            &pvar_send->name_len,
            &pvar_send->verbosity,
            &pvar_send->var_class,
            &pvar_send->datatype,
            &pvar_send->enumtype,
            send_description,
            &pvar_send->desc_len,
            &pvar_send->bind,
            &pvar_send->readonly,
            &pvar_send->continuous,
            &pvar_send->atomic);
  pvar_send->name = malloc(pvar_send->name_len);
  pvar_send->desc = malloc(pvar_send->desc_len);
  strcpy(pvar_send->name, send_name);
  strcpy(pvar_send->desc, send_description);

  /* For MVAPICH2: check if the link utilization for recv is there */
  PMPI_T_pvar_get_index("mv2_hardware_link_utilization_recv", MPI_T_PVAR_CLASS_COUNTER, &pvar_recv_index);
  PMPI_T_pvar_get_info(pvar_recv_index,
            recv_name,
            &pvar_recv->name_len,
            &pvar_recv->verbosity,
            &pvar_recv->var_class,
            &pvar_recv->datatype,
            &pvar_recv->enumtype,
            recv_description,
            &pvar_recv->desc_len,
            &pvar_recv->bind,
            &pvar_recv->readonly,
            &pvar_recv->continuous,
            &pvar_recv->atomic);
  pvar_recv->name = malloc(pvar_recv->name_len);
  pvar_recv->desc = malloc(pvar_recv->desc_len);
  strcpy(pvar_recv->name, recv_name);
  strcpy(pvar_recv->desc, recv_description);

  /* Next, we allocate handles: */
  MPI_Comm comm = MPI_COMM_WORLD; 
  err = MPI_T_pvar_handle_alloc(tau_session,pvar_send_index, &comm, &tau_handle_send, &tau_send_count);
  if (err != MPI_SUCCESS) {
    fprintf(stderr, "\nTAU: ERROR pvar_handle_alloc failed for NO_OBJ (%d)\n", err);
  }

  err = MPI_T_pvar_handle_alloc(tau_session,pvar_recv_index, &comm, &tau_handle_recv, &tau_recv_count);
  if (err != MPI_SUCCESS) {
    fprintf(stderr, "\nTAU: ERROR pvar_handle_alloc failed for NO_OBJ (%d)\n", err);
  }
    
  /* Parse the description of PVARs for all paths between CPUs and GPUs */
  char **send_paths = malloc(tau_send_count * sizeof(char*));
  TAU_VERBOSE("send_description=%s, tau_send_count=%d\n", send_description, tau_send_count);
  TAU_VERBOSE("recv_description=%s, tau_recv_count=%d\n", recv_description, tau_recv_count);
  int nl_found; // does it find a new line in the string?
  for(i = 0; i < tau_send_count; i++)
  {
    send_paths[i] = malloc(sizeof(char) * TAU_NAME_LENGTH);
    nl_found = 0; 
    char *nextLine = strchr(send_description, '\n');
    if (nextLine) {
      nl_found = 1; // new line is found! 
      *nextLine = '\0'; // null terminate it
    } 
    strcpy(send_paths[i], send_description); // chop off and store the path
    if (nl_found) {
      *nextLine = '\n';  
      send_description = nextLine + 1; // if new line is found
    }
    TAU_METADATA_ITERATION("TAU_SEND_PATH_ID", i, send_paths[i]);
  }

  char **recv_paths = malloc(tau_recv_count * sizeof(char*));
  for(i = 0; i < tau_recv_count; i++)
  {
    recv_paths[i] = malloc(sizeof(char) * TAU_NAME_LENGTH);
    nl_found = 0; 
    char *nextLine = strchr(recv_description, '\n');
    if (nextLine) {
      nl_found = 1; // new line is found! 
      *nextLine = '\0'; // null terminate it
    }
    strcpy(recv_paths[i], recv_description);
    if (nl_found) {
      *nextLine = '\n';  
      recv_description = nextLine + 1; // if new line is found
    }
    TAU_METADATA_ITERATION("TAU_RECV_PATH_ID", i, recv_paths[i]);
  }
    
  /* Initialize value arrays to get link utilization for within the node and inter-node links */
  tau_send_value = malloc(sizeof(unsigned long long) * tau_send_count);
  tau_recv_value = malloc(sizeof(unsigned long long) * tau_recv_count);

}

//////////////////////////////////////////////////////////////////////
int Tau_msg_send_prolog(void){
  TAU_VERBOSE("Inside message send prolog\n");
  int err; 
  err = PMPI_T_pvar_start(tau_session, tau_handle_send);
  if (err != MPI_SUCCESS)
    fprintf(stderr,"TAU: ERROR in MPI_T pvar start in send prolog (%d)", err);
  TAU_VERBOSE("Exiting from message send prolog\n");
  fflush(stdout);
  return 0;
} 

//////////////////////////////////////////////////////////////////////
int Tau_msg_recv_prolog(void){
  TAU_VERBOSE("Inside message recv prolog\n");
  int err; 
  err = PMPI_T_pvar_start(tau_session, tau_handle_recv);
  if (err != MPI_SUCCESS)
    fprintf(stderr,"TAU: ERROR in MPI_T pvar start in recv prolog (%d)", err);
  return 0;
} 

//////////////////////////////////////////////////////////////////////
long Tau_get_message_send_path(void) {
  // Query MPI_T and return the path associated with the pvars. 
  int err;
  int i;  
  long id = 0; 
  int found = 0;
  
  err = PMPI_T_pvar_stop(tau_session, tau_handle_send);
  if (err != MPI_SUCCESS)
    fprintf(stderr,"TAU: ERROR in MPI_T pvar stop in send path (%d)", err);

  /* Read send value from pvar */
  err = PMPI_T_pvar_read(tau_session, tau_handle_send, tau_send_value);
  if (err != MPI_SUCCESS)
    fprintf(stderr,"TAU: ERROR in MPI_T pvar read in send path (%d)", err);

  for(i = 0; i < tau_send_count; i++)
  { 
    /* If delta is > 0, send_paths[i] was used */
    if (tau_send_value[i] > 0)
    {
      id = id+(pow(100,found)*i);
      found++;
    }
  }
  return id; /* default path id */
}

//////////////////////////////////////////////////////////////////////
long Tau_get_message_recv_path(void) {
  // Query MPI_T and return the path associated with the pvars. 
  int err; 
  int i, found;
  long id = 0;

  err = PMPI_T_pvar_stop(tau_session, tau_handle_recv);
  if (err != MPI_SUCCESS)
    fprintf(stderr,"TAU: ERROR in MPI_T pvar stop in recv path (%d)", err);


   
  /* Read recv value from pvar */
  err = PMPI_T_pvar_read(tau_session, tau_handle_recv, tau_recv_value);
  if (err != MPI_SUCCESS)
    fprintf(stderr,"TAU: ERROR in MPI_T pvar read in recv path (%d)", err);

  found = 0;
  for(i = 0; i < tau_recv_count; i++)
  { 
    /* If delta is > 0, recv_paths[i] was used */
    if (tau_recv_value[i] > 0) {
      id = id+(pow(100,found)*i);
      found++;
    }
  }
  return id; /* default path id */

}
//////////////////////////////////////////////////////////////////////
// EOF : TauMpiT.c
//////////////////////////////////////////////////////////////////////


