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
  dprintf("TAU: Before MPI_T initialization\n");
  if (TauEnv_get_track_mpi_t_pvars() == 0) {
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
  return return_val;
} 

int Tau_mpi_t_cvar_initialize(void) {
  int return_val;
  const char *cvars = TauEnv_get_cvar_metrics();
  const char *values = TauEnv_get_cvar_values();
  if (cvars == (const char *) NULL) {
    dprintf("TAU: No CVARS specified using TAU_MPI_T_CVAR_METRICS and TAU_MPI_T_CVAR_VALUES\n");
  } else {
    dprintf("CVAR_METRICS=%s\n", cvars);
    if (values == (char *) NULL) {
      printf("TAU: WARNING: Environment variable TAU_MPI_T_CVAR_METRICS is not specified for TAU_MPI_T_CVAR_METRICS=%s\n", 
	cvars);
    } else { // both cvars and values are specified
    // Use strtok and parse the names of all CVARS using , as a delimiter. For now assume only one is specifed. 
      long long val; 
      sscanf(values, "%lld", &val);
      dprintf("TAU: cvars=%s, values=%s, val = %lld\n", cvars, values, val); 
      MPI_T_cvar_handle chandle; 
      int cindex, num_vals, num_cvars;
      
      char name[TAU_NAME_LENGTH]= ""; 
      char desc[TAU_NAME_LENGTH]= ""; 
      int verbosity, binding, scope, i;
      int name_len;
      int desc_len;
      MPI_Datatype datatype; 
      MPI_T_enum enumtype; 
      char metastring[TAU_NAME_LENGTH]; 
      
      int rank ;
      /* MPI_Comm_rank(MPI_COMM_WORLD, &rank); */
      rank = Tau_get_node(); /* MPI_Init has not been invoked yet */

      return_val = MPI_T_cvar_get_num(&num_cvars); 
      if (return_val != MPI_SUCCESS) { 
	printf("TAU: Rank %d: Can't read the number of MPI_T control variables in this MPI implementation\n", rank);
        return return_val;
      }
      for (i=0; i < num_cvars; i++) {
        name_len = desc_len = TAU_NAME_LENGTH;
        return_val = MPI_T_cvar_get_info(i, name, &name_len, &verbosity, &datatype, &enumtype, desc, &desc_len, &binding, &scope);
        if (return_val != MPI_SUCCESS) {
	  printf("TAU: Rank %d: Can't get cvar info i=%d, num_cvars=%d\n", rank, i, num_cvars);
	  return return_val; 
        }
        if (rank == 0) {
	  dprintf("CVAR[%d] = %s \t \t desc = %s\n", i, name, desc);
	  TAU_METADATA(name, desc);
        }
	if (strcmp(name,cvars)==0) {
	  if (rank == 0) {
            dprintf("Rank: %d FOUND CVAR match: %s, cvars = %s, desc = %s\n", rank, name, cvars, desc);
          }
          cindex = i; 
          return_val = MPI_T_cvar_handle_alloc(cindex, NULL, &chandle, &num_vals);
          if (return_val != MPI_SUCCESS) {
	    printf("TAU: Rank %d: Can't allocate cvar handle in this MPI implementation\n", rank);
            return return_val;
          }
          int oldval=0; 
          return_val = MPI_T_cvar_read(chandle, &oldval); 
          if (return_val != MPI_SUCCESS) {
	     printf("TAU: Rank %d: Can't read cvar %s = %d in this MPI implementation\n", rank, cvars, oldval);
             return return_val;
          } else {
            if (rank == 0) {
              dprintf("Oldval = %d, newval=%lld, cvars=%s\n", oldval, val, cvars);
            }
          }
          return_val = MPI_T_cvar_write(chandle, &val); 
          if (return_val != MPI_SUCCESS) {
	     printf("TAU: Rank %d: Can't write cvar %s = %lld in this MPI implementation\n", rank, cvars, val);
             return return_val;
          }
          int reset_value; 
          return_val = MPI_T_cvar_read(chandle, &reset_value); 
          if (return_val != MPI_SUCCESS) {
	     printf("TAU: Rank %d: Can't read cvar %s = %d in this MPI implementation\n", rank, cvars, reset_value);
             return return_val;
          } else {
            if ((rank == 0) && (reset_value == (int) val)) {
              dprintf("ResetValue=%d matches what we set for cvars=%s\n", reset_value, cvars);
	      sprintf(metastring,"%d (old) -> %d (new), %s", oldval, reset_value, desc);
	      TAU_METADATA(name, metastring);
	      TAU_METADATA("TAU_MPI_T_CVAR_METRICS", cvars);
	      TAU_METADATA("TAU_MPI_T_CVAR_VALUES", values);
	      sprintf(metastring, "%d", TauEnv_get_track_mpi_t_pvars());
	      TAU_METADATA("TAU_TRACK_MPI_T_PVARS", metastring);
            }
          }

          MPI_T_cvar_handle_free(&chandle);
        }
      }
      /* NOT implemented: return_val = MPI_T_cvar_get_index(cvars, &cindex);
      if (return_val != MPI_SUCCESS) { 
	printf("TAU: Rank %d: Can't access MPI_T variable %s in this MPI implementation\n", Tau_get_node(), cvars);
        return return_val;
      }
      */
    
    }
  }
  return return_val; 
}

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
     //printf("MPI_T_PVAR_CLASS_TIMER: %d,  varclass = %d, i = %d, event_name = %s\n", MPI_T_PVAR_CLASS_TIMER, varclass, i, event_name);
    }
  }
  int rank = Tau_get_node(); 
  int size, j; 

  for(i = 0; i < num_pvars; i++){
    // get data from event 
    MPI_T_pvar_read(tau_pvar_session, tau_pvar_handles[i], read_value_buffer);
    MPI_Type_size(tau_mpi_datatype[i], &size); 
    for(j = 0; j < tau_pvar_count[j]; j++){
      pvar_value_buffer[i][j] = 0;
      memcpy(&(pvar_value_buffer[i][j]), read_value_buffer, size);
      /* unsigned long long int to double conversion can result in an error. 
       * We first convert it to a long long int. */
      long long int mydata = (long long int) pvar_value_buffer[i][j]; 
      int is_double = 0; 
      if (tau_mpi_datatype[i] == MPI_DOUBLE) is_double=1; 
      if (is_double) {
        // long long int mydata_d = (long long int) (pvar_value_buffer[i][j]); 
        // First convert it from unsigned long long to long long and then to a double
        //double double_data = ((double)(pvar_value_buffer[i][j]));
        double double_data = ((double)(mydata));

        dprintf("RANK:%d: pvar_value_buffer[%d][%d]=%lld, double_data=%g, size = %d, is_double=%d\n",rank,i,j,mydata, double_data, size, is_double);
	if (double_data > 1e-14 )  {
      
        // Double values are really large for timers. Please check 1E18?? 
          //Tau_track_pvar_event(i, num_pvars, double_data);
        } 
      } else {
        dprintf("RANK:%d: pvar_value_buffer[%d][%d]=%lld, size = %d, is_double=%d\n",rank,i,j,mydata, size, is_double);
        /* Trigger the TAU event if it is non-zero */
	if (mydata > 0L) {
          Tau_track_pvar_event(i, num_pvars, mydata);
        }
      }
    }

  }
  dprintf("Finished!!\n");
}


//////////////////////////////////////////////////////////////////////
// EOF : TauMpiT.c
//////////////////////////////////////////////////////////////////////


