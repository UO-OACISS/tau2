/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997-2015 	          			   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **	File 		: TauOpenACC.cpp				  **
 **	Description 	: TAU Profiling Package				  **
 **	Contact		: tau-bugs@cs.uoregon.edu 		 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////



#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <openacc.h>
#ifdef TAU_PGI_OPENACC
#include "pgi_acc_prof.h"
#endif /* TAU_PGI_OPENACC */
#include <Profile/Profiler.h>


#define TAU_SET_EVENT_NAME(event_name, str) strcpy(event_name, str); break 
////////////////////////////////////////////////////////////////////////////
extern "C" static void
Tau_openacc_callback( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info )
{
  char event_name[256]; 
/* 
  acc_data_event_info*   data_event_info = NULL;
  acc_launch_event_info* launch_event_info = NULL;
  acc_other_event_info*  other_event_info = NULL;
*/

  
  switch (prof_info->eventtype) {
    case acc_ev_init_start 	            : Tau_create_top_level_timer_if_necessary(); TAU_SET_EVENT_NAME(event_name, ">openacc_init"); 
    case acc_ev_init_end   	            : TAU_SET_EVENT_NAME(event_name, "<openacc_init");
    case acc_ev_shutdown_start              : TAU_SET_EVENT_NAME(event_name, ">openacc_shutdown");
    case acc_ev_shutdown_end                : TAU_SET_EVENT_NAME(event_name, "<openacc_shutdown");
    case acc_ev_done                        : TAU_SET_EVENT_NAME(event_name, "openacc_done");
    case acc_ev_enqueue_launch_start        : TAU_SET_EVENT_NAME(event_name, ">openacc_enqueue_launch");
    case acc_ev_enqueue_launch_end          : TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_launch");
    case acc_ev_enqueue_upload_start        : TAU_SET_EVENT_NAME(event_name, ">openacc_enqueue_upload");
    case acc_ev_enqueue_upload_end          : TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_upload");
    case acc_ev_enqueue_download_start      : TAU_SET_EVENT_NAME(event_name, ">openacc_enqueue_download");
    case acc_ev_enqueue_download_end        : TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_download");
    case acc_ev_wait_start                  : TAU_SET_EVENT_NAME(event_name, ">openacc_wait");
    case acc_ev_wait_end                    : TAU_SET_EVENT_NAME(event_name, "<openacc_wait");
    case acc_ev_implicit_wait_start         : TAU_SET_EVENT_NAME(event_name, ">openacc_implicit_wait");
    case acc_ev_implicit_wait_end           : TAU_SET_EVENT_NAME(event_name, "<openacc_implicit_wait");
    case acc_ev_compute_construct_start     : TAU_SET_EVENT_NAME(event_name, ">openacc_compute_construct");
    case acc_ev_compute_construct_end       : TAU_SET_EVENT_NAME(event_name, "<openacc_compute_construct");
    case acc_ev_data_construct_enter_start  : TAU_SET_EVENT_NAME(event_name, ">openacc_data_construct_enter");
    case acc_ev_data_construct_enter_end    : TAU_SET_EVENT_NAME(event_name, "<opeancc_data_construct_enter");
    case acc_ev_data_construct_exit_start   : TAU_SET_EVENT_NAME(event_name, ">openacc_data_construct_exit");
    case acc_ev_data_construct_exit_end     : TAU_SET_EVENT_NAME(event_name, "<openacc_data_construct_exit");
    case acc_ev_update_construct_start      : TAU_SET_EVENT_NAME(event_name, ">openacc_update_construct");
    case acc_ev_update_construct_end        : TAU_SET_EVENT_NAME(event_name, "<openacc_update_construct");
    case acc_ev_enter_data_start            : TAU_SET_EVENT_NAME(event_name, ">openacc_enter_data");
    case acc_ev_enter_data_end              : TAU_SET_EVENT_NAME(event_name, "<openacc_enter_data");
    case acc_ev_exit_data_start             : TAU_SET_EVENT_NAME(event_name, ">openacc_exit_data");
    case acc_ev_exit_data_end               : TAU_SET_EVENT_NAME(event_name, "<openacc_exit_data");
    case acc_ev_create                      : TAU_SET_EVENT_NAME(event_name, "openacc_create");
    case acc_ev_delete                      : TAU_SET_EVENT_NAME(event_name, "openacc_delete");
    case acc_ev_alloc                       : TAU_SET_EVENT_NAME(event_name, "openacc_alloc");
    case acc_ev_free                        : TAU_SET_EVENT_NAME(event_name, "openacc_free");
    default                                 : TAU_SET_EVENT_NAME(event_name, "default");
  }
  char srcinfo[1024]; 
  char lineinfo[256]; 

  if (prof_info) {
    TAU_VERBOSE("Device=%d ", prof_info->devnum);
    TAU_VERBOSE("Thread=%d ", prof_info->threadid);
    sprintf(srcinfo, " %s [{%s}", prof_info->funcname, prof_info->srcfile);
    if (prof_info->lineno) { 
      sprintf(lineinfo, " {%d,0}", prof_info->lineno); 
      strcat(srcinfo,lineinfo);
      if ((prof_info->endlineno) && (prof_info->endlineno > prof_info->lineno)) {
        sprintf(lineinfo, "-{%d,0}", prof_info->endlineno); 
        strcat(srcinfo,lineinfo);
      }
    }
    strcat(srcinfo,"]");
    strcat(event_name, srcinfo); 
  }
  if (event_name[0] == '>') {
    TAU_VERBOSE("START>>%s\n", &event_name[1]);
    TAU_START(&event_name[1]);
  }  else if (event_name[0] == '<') {
    TAU_VERBOSE("STOP <<%s\n", &event_name[1]);
    TAU_STOP(&event_name[1]);
     } else {
        TAU_VERBOSE("event_name = %s\n", event_name);
     }
  

}

////////////////////////////////////////////////////////////////////////////
typedef void (*Tau_openacc_registration_routine_t)( acc_event_t, acc_prof_callback_t, int );

////////////////////////////////////////////////////////////////////////////
extern "C" void
acc_register_library(Tau_openacc_registration_routine_t reg, Tau_openacc_registration_routine_t unreg )
{
    TAU_VERBOSE("Inside acc_register_library\n");

    reg( acc_ev_init_start, Tau_openacc_callback, 0 );
    reg( acc_ev_init_end, Tau_openacc_callback, 0 );
    reg( acc_ev_shutdown_start, Tau_openacc_callback, 0 );
    reg( acc_ev_shutdown_end, Tau_openacc_callback, 0 );
    reg( acc_ev_done, Tau_openacc_callback, 0 );
    reg( acc_ev_enqueue_launch_start, Tau_openacc_callback, 0 );
    reg( acc_ev_enqueue_launch_end, Tau_openacc_callback, 0 );
    reg( acc_ev_enqueue_upload_start, Tau_openacc_callback, 0 );
    reg( acc_ev_enqueue_upload_end, Tau_openacc_callback, 0 );
    reg( acc_ev_enqueue_download_start, Tau_openacc_callback, 0 );
    reg( acc_ev_enqueue_download_end, Tau_openacc_callback, 0 );
    reg( acc_ev_wait_start, Tau_openacc_callback, 0 );
    reg( acc_ev_wait_end, Tau_openacc_callback, 0 );
    reg( acc_ev_implicit_wait_start, Tau_openacc_callback, 0 );
    reg( acc_ev_implicit_wait_end, Tau_openacc_callback, 0 );
/*
    reg( acc_ev_compute_construct_start, Tau_openacc_callback, 0 );
    reg( acc_ev_compute_construct_end, Tau_openacc_callback, 0 );
    reg( acc_ev_data_construct_enter_start, Tau_openacc_callback, 0 );
    reg( acc_ev_data_construct_enter_end, Tau_openacc_callback, 0 );
    reg( acc_ev_data_construct_exit_start, Tau_openacc_callback, 0 );
    reg( acc_ev_data_construct_exit_end, Tau_openacc_callback, 0 );
*/
    reg( acc_ev_update_construct_start, Tau_openacc_callback, 0 );
    reg( acc_ev_update_construct_end, Tau_openacc_callback, 0 );
    reg( acc_ev_enter_data_start, Tau_openacc_callback, 0 );
    reg( acc_ev_enter_data_end, Tau_openacc_callback, 0 );
    reg( acc_ev_exit_data_start, Tau_openacc_callback, 0 );
    reg( acc_ev_exit_data_end, Tau_openacc_callback, 0 );
    reg( acc_ev_create, Tau_openacc_callback, 0 );
    reg( acc_ev_delete, Tau_openacc_callback, 0 );
    reg( acc_ev_alloc, Tau_openacc_callback, 0 );
    reg( acc_ev_free, Tau_openacc_callback, 0 );

} // acc_register_library 

////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////
