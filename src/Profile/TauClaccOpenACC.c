#ifndef TAU_PGI_OPENACC
#include <stdio.h>
#include <TAU.h>
#include <stdlib.h>
#include <acc_prof.h>

#define TAU_ACC_NAME_LEN 4096
#define VERSION 0.1

/* Init */

void dev_init_start( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_device_init" );
  TAU_START( sourceinfo );
  //  printf( "dev init start\n " );
}

void dev_init_stop( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_device_init" );
  //  printf( "dev init stop\n " );
  TAU_STOP( sourceinfo );
}

/* Enter data */

void enter_data_start( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_enter_data" );
  TAU_START( sourceinfo );
  //  printf( "enter data start\n " );
}

void enter_data_end( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_enter_data" );
  //  printf( "enter data end\n " );
  TAU_STOP( sourceinfo );
}

/* Exit data */

void exit_data_start( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_exit_data" );
  TAU_START( sourceinfo );
  //  printf( "exit data start\n " );
}

void exit_data_end( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_exit_data" );
  //  printf( "exit data end\n " );
  TAU_STOP( sourceinfo );
}

/* Compute construct */

void compute_construct_start( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_compute_construct" );
  TAU_START( sourceinfo );
  //  printf( "compute construct start\n " );
}

void compute_construct_end( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_compute_construct" );
  //  printf( "compute construct end\n " );
  TAU_STOP( sourceinfo );
}

/* Device shutdown */

void device_shutdown_start( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_device_shutdown" );
  TAU_START( sourceinfo );
  //  printf( "device shutdown start\n " );
}

void device_shutdown_end( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_device_shutdown" );
  //  printf( "device shutdown end\n " );
  TAU_STOP( sourceinfo );
}

/* Enqueue launch */

void ev_enqueue_launch_start( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_enqueue_launch" );
  TAU_START( sourceinfo );
  //  printf( "ev enqueue launch start\n " );
}

void ev_enqueue_launch_end( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_enqueue_launch" );
  //  printf( "ev enqueue launch end\n " );
  TAU_STOP( sourceinfo );
}

/* Enqueue upload */

void ev_enqueue_upload_start( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_enqueue_upload" );
  TAU_START( sourceinfo );
  //  printf( "ev enqueue upload start\n " );
}

void ev_enqueue_upload_end( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_enqueue_upload" );
  //  printf( "ev enqueue upload end\n " );
  TAU_STOP( sourceinfo );
}

/* Enqueue download */

void ev_enqueue_download_start( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_enqueue_download" );
  TAU_START( sourceinfo );
  //  printf( "ev enqueue download start\n " );
}

void ev_enqueue_download_end( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_enqueue_download" );
  //  printf( "ev enqueue download end\n " );
  TAU_STOP( sourceinfo );
}

/* Runtime shutdown */

void runtime_shutdown( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_runtime_shutdown" );
  TAU_START( sourceinfo );
  //  printf( "runtime shutdown\n " );
  TAU_STOP( sourceinfo );
}

/* Events */

void ev_create( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_create" );
  TAU_START( sourceinfo );
  //  printf( "ev create\n " );
  TAU_STOP( sourceinfo );
}

void ev_delete( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_delete" );
  TAU_START( sourceinfo );
  //  printf( "ev delete\n " );
  TAU_STOP( sourceinfo );
}

void ev_alloc( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_alloc" );
  TAU_START( sourceinfo );
  //  printf( "ev alloc\n " );
  TAU_STOP( sourceinfo );
}

void ev_free( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info ){
  char sourceinfo[TAU_ACC_NAME_LEN];
  sprintf( sourceinfo, "openacc_ev_free" );
  TAU_START( sourceinfo );
  //  printf( "ev free\n " );
  TAU_STOP( sourceinfo );
}

/* Register the actions */

void acc_register_library(acc_prof_reg reg, acc_prof_reg unreg,
                          acc_prof_lookup lookup) {

  reg( acc_ev_device_init_start, &dev_init_start, acc_reg );
  reg( acc_ev_device_init_end, &dev_init_stop, acc_reg );
  reg( acc_ev_device_shutdown_start, &device_shutdown_start, acc_reg );
  reg( acc_ev_device_shutdown_end, &device_shutdown_end, acc_reg ); 
  reg( acc_ev_runtime_shutdown, &runtime_shutdown, acc_reg );
  reg( acc_ev_create, &ev_create, acc_reg );
  reg( acc_ev_delete, &ev_delete, acc_reg );
  reg( acc_ev_alloc, &ev_alloc, acc_reg ); 
  reg( acc_ev_free, &ev_free, acc_reg );
  reg( acc_ev_enter_data_start, &enter_data_start, acc_reg );
  reg( acc_ev_enter_data_end, &enter_data_end, acc_reg );
  reg( acc_ev_exit_data_start, &exit_data_start, acc_reg );
  reg( acc_ev_exit_data_end, &exit_data_end, acc_reg ); 
  reg( acc_ev_compute_construct_start, &compute_construct_start, acc_reg );
  reg( acc_ev_compute_construct_end, &compute_construct_end, acc_reg );
  reg( acc_ev_enqueue_launch_start, &ev_enqueue_launch_start, acc_reg );
  reg( acc_ev_enqueue_launch_end, &ev_enqueue_launch_end, acc_reg );
  reg( acc_ev_enqueue_upload_start, &ev_enqueue_upload_start, acc_reg );  
  reg( acc_ev_enqueue_upload_end, &ev_enqueue_upload_end, acc_reg ); 
  reg( acc_ev_enqueue_download_start, &ev_enqueue_download_start, acc_reg );  
  reg( acc_ev_enqueue_download_end, &ev_enqueue_download_end, acc_reg ); 

}


  /* Unimplemented:
     
     acc_ev_wait_start
     acc_ev_wait_end
     acc_ev_update_start
     acc_ev_update_end
  */  
#endif // ndef TAU_PGI_OPENACC
