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
#include <stdio.h>

//////////////////////////////////////////////////////////////////////
// Global variables
//////////////////////////////////////////////////////////////////////
static MPI_T_pvar_handle  *tau_pvar_handles; 
static MPI_T_pvar_session tau_pvar_session;

//////////////////////////////////////////////////////////////////////
// externs
//////////////////////////////////////////////////////////////////////
extern void Tau_track_pvar_event(int index, int total_events, double data);

#define dprintf TAU_VERBOSE

//////////////////////////////////////////////////////////////////////
int Tau_mpi_t_initialize(void) {
  int return_val, thread_provided, num_pvars;   

  /* if TAU_TRACK_MPI_T_PVARS is not set to true, return with a success but do nothing 
   * to initialize MPI_T */
  if (TauEnv_get_track_mpi_t_pvars() == false) {
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

  /* get the number of pvars exported by the implmentation */
  return_val = MPI_T_pvar_get_num(&num_pvars);
  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_pvar_get_num ERROR:");
    return return_val;
  }
  dprintf("TAU STARTED session: pvars exposed = %d\n", num_pvars);
  
  return return_val; 
}

//////////////////////////////////////////////////////////////////////
void Tau_track_mpi_t(void) {

}

#define TAU_NAME_LENGTH 1024

static unsigned long long int **pvar_value_buffer;
static void *read_value_buffer; // values are read into this buffer.
static MPI_Datatype *tau_mpi_datatype; 

//////////////////////////////////////////////////////////////////////
int Tau_track_mpi_t_here(void) {
  static int first_time = 1; 
  int return_val, num_pvars, i, namelen, verb, varclass, bind, threadsup;
  int index;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;
  static int *tau_pvar_count; 
  int returnVal;

  
  /* if TAU_TRACK_MPI_T_PVARS is not set to true, return with a success but do nothing 
   * to process MPI_T events */
  if (TauEnv_get_track_mpi_t_pvars() == 0) {
    return MPI_SUCCESS; 
  } 

  /* get number of pvars from MPI_T */
  return_val = MPI_T_pvar_get_num(&num_pvars);
  if (return_val != MPI_SUCCESS) {
    perror("MPI_T_pvar_get_num ERROR:");
    return return_val;
  }

  /* The first time this function is entered, allocate memory for the pvar data structures */
  if (first_time == 1) {
    first_time = 0;
    pvar_value_buffer = (unsigned long long int**)malloc(sizeof(unsigned long long int*) * (num_pvars + 1));
    tau_mpi_datatype = (MPI_Datatype *) malloc(sizeof(MPI_Datatype *) * (num_pvars+1)); 
    tau_pvar_handles = (MPI_T_pvar_handle*)malloc(sizeof(MPI_T_pvar_handle) * (num_pvars + 1));
    tau_pvar_count = (int*)malloc(sizeof(int) * (num_pvars + 1));
    memset(tau_pvar_count, 0, sizeof(int) * (num_pvars + 1));

    read_value_buffer = (void*)malloc(sizeof(unsigned long long int) * (TAU_NAME_LENGTH + 1));


    /* Initialize variables. Get the names of performance variables */
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
     tau_mpi_datatype[i] = datatype;

     /* allocate a pvar handle that will be used later */
     returnVal = MPI_T_pvar_handle_alloc(tau_pvar_session, i, NULL, &tau_pvar_handles[i], &tau_pvar_count[i]);
     if (return_val != MPI_SUCCESS) {
       perror("MPI_T_pvar_handle_alloc ERROR:");
       return return_val;
     }

     /* and a buffer to store the results in */
     pvar_value_buffer[i] = (unsigned long long int*)malloc(sizeof(unsigned long long int) * (tau_pvar_count[i] + 1));

     dprintf("Name: %s (%s), i = %d\n", event_name, description, i); 
    }
  }
  int rank; 
  int size, j; 

  /* We need the rank */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for(i = 0; i < num_pvars; i++){
    // get data from event 
    MPI_T_pvar_read(tau_pvar_session, tau_pvar_handles[i], read_value_buffer);
    MPI_Type_size(tau_mpi_datatype[i], &size); 
    for(j = 0; j < tau_pvar_count[j]; j++){
      pvar_value_buffer[i][j] = 0;
      memcpy(&(pvar_value_buffer[i][j]), read_value_buffer, size);
      long long int mydata = (long long int) (pvar_value_buffer[i][j]); 
      /* unsigned long long int to double conversion can result in an error. 
       * We first convert it to a long long int. */
      int is_double = 0; 
      if (tau_mpi_datatype[i] == MPI_DOUBLE) is_double=1; 
      if (is_double) {
        double double_data = *((double*)(pvar_value_buffer[i][j]));
        dprintf("RANK:%d: pvar_value_buffer[%d][%d]=%g, size = %d, is_double=%d\n",rank,i,j,double_data, size, is_double);

        Tau_track_pvar_event(i, num_pvars, double_data);
      } else {
        dprintf("RANK:%d: pvar_value_buffer[%d][%d]=%lld, size = %d, is_double=%d\n",rank,i,j,mydata, size, is_double);
        /* Trigger the TAU event if it is non-zero */
	if (mydata != 0L) 
          Tau_track_pvar_event(i, num_pvars, mydata);
      }
    }

  }
  dprintf("Finished!!\n");
}

//////////////////////////////////////////////////////////////////////
void Tau_enable_tracking_mpi_t(void) {
  TauEnv_set_track_mpi_t_pvars(1); 
}

//////////////////////////////////////////////////////////////////////
void Tau_disable_tracking_mpi_t(void) {
  TauEnv_set_track_mpi_t_pvars(0); 
}


//////////////////////////////////////////////////////////////////////
// EOF : TauMpiT.c
//////////////////////////////////////////////////////////////////////


