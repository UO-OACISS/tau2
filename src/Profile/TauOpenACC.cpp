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
//#include <cupti_openacc.h>
#ifdef CUPTI
#include <cupti.h>
#include <cuda.h>
#endif

//#include <Profile/TauOpenACC.h>
#include <Profile/TauGpuAdapterOpenACC.h>
//#include <Profile/CuptiActivity.h> // solely for get_taskid_from_context_id

#define TAU_SET_EVENT_NAME(event_name, str) strcpy(event_name, str); break 
////////////////////////////////////////////////////////////////////////////
extern "C" static void
Tau_openacc_callback( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info )
{
  char event_name[256], user_event_name[256];

  //acc_event_t *event_type_info = NULL; 
  acc_data_event_info*   data_event_info = NULL;
  acc_launch_event_info* launch_event_info = NULL;
  //acc_other_event_info*  other_event_info = NULL;

  
#ifdef TAU_PGI_OPENACC_OLD
  switch (prof_info->eventtype) {
    case acc_ev_init_start 	            : Tau_create_top_level_timer_if_necessary(); TAU_SET_EVENT_NAME(event_name, ">openacc_init"); 
    case acc_ev_init_end   	            : TAU_SET_EVENT_NAME(event_name, "<openacc_init");
    case acc_ev_shutdown_start              : TAU_SET_EVENT_NAME(event_name, ">openacc_shutdown");
    case acc_ev_shutdown_end                : TAU_SET_EVENT_NAME(event_name, "<openacc_shutdown");
    case acc_ev_done                        : TAU_SET_EVENT_NAME(event_name, "openacc_done");
    case acc_ev_data_construct_enter_start  : TAU_SET_EVENT_NAME(event_name, ">openacc_data_construct_enter");
    case acc_ev_data_construct_enter_end    : TAU_SET_EVENT_NAME(event_name, "<openacc_data_construct_enter");
    case acc_ev_data_construct_exit_start   : TAU_SET_EVENT_NAME(event_name, ">openacc_data_construct_exit");
    case acc_ev_data_construct_exit_end     : TAU_SET_EVENT_NAME(event_name, "<openacc_data_construct_exit");
    case acc_ev_update_construct_start      : TAU_SET_EVENT_NAME(event_name, ">openacc_update_construct");
    case acc_ev_update_construct_end        : TAU_SET_EVENT_NAME(event_name, "<openacc_update_construct");
#else /* TAU_PGI_OPENACC_OLD */
  switch (prof_info->event_type) {
    case acc_ev_device_init_start 	            : Tau_create_top_level_timer_if_necessary(); TAU_SET_EVENT_NAME(event_name, ">openacc_init"); 
    case acc_ev_device_init_end   	            : TAU_SET_EVENT_NAME(event_name, "<openacc_init");
    case acc_ev_device_shutdown_start              : TAU_SET_EVENT_NAME(event_name, ">openacc_shutdown");
    case acc_ev_device_shutdown_end                : TAU_SET_EVENT_NAME(event_name, "<openacc_shutdown");
    // case acc_ev_done                        : TAU_SET_EVENT_NAME(event_name, "openacc_done");
    case acc_ev_enter_data_start: 
       TAU_SET_EVENT_NAME(event_name, ">openacc_enter_data");
    case acc_ev_enter_data_end: TAU_SET_EVENT_NAME(event_name, "<openacc_enter_data");
    case acc_ev_exit_data_start: TAU_SET_EVENT_NAME(event_name, ">openacc_exit_data");
    case acc_ev_exit_data_end: TAU_SET_EVENT_NAME(event_name, "<openacc_exit_data");
    case acc_ev_update_start                : TAU_SET_EVENT_NAME(event_name, ">openacc_update");
    case acc_ev_update_end                  : TAU_SET_EVENT_NAME(event_name, "<openacc_update");
#endif /* TAU_PGI_OPENACC_OLD */

    case acc_ev_enqueue_launch_start        : 
      if (event_info) {
        launch_event_info = &(event_info->launch_event); 
        sprintf(event_name, ">openacc_enqueue_launch kernel=%s <num_gangs=%d, num_workers=%d, vector_length=%d>", 
		launch_event_info->kernel_name, 
		launch_event_info->num_gangs, launch_event_info->num_workers, launch_event_info->vector_length);
      }
      break;
    case acc_ev_enqueue_launch_end          : 
      if (event_info) {
        launch_event_info = &(event_info->launch_event); 
        sprintf(event_name, "<openacc_enqueue_launch kernel=%s <num_gangs=%d, num_workers=%d, vector_length=%d>", 
		launch_event_info->kernel_name, 
		launch_event_info->num_gangs, launch_event_info->num_workers, launch_event_info->vector_length);
      }
      break;
    case acc_ev_enqueue_upload_start        : 
      if (event_info) {
        data_event_info = &(event_info->data_event); 
        TAU_VERBOSE("UPLOAD start: Var_name = %s, bytes=%d \n", data_event_info->var_name, 
		event_info->data_event.bytes);
        if (data_event_info->var_name) {
          sprintf(user_event_name, "Data transfer from host to device <variable=%s>", data_event_info->var_name);
        } else {
          sprintf(user_event_name, "Data transfer from host to device <other>");
        }
        TAU_TRIGGER_EVENT(user_event_name, event_info->data_event.bytes);
      }
      TAU_SET_EVENT_NAME(event_name, ">openacc_enqueue_upload");
    case acc_ev_enqueue_upload_end          : 
      TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_upload");
    case acc_ev_enqueue_download_start      : 
      if (event_info) {
        data_event_info = &(event_info->data_event); 
        TAU_VERBOSE("DOWNLOAD start: Var_name = %s, bytes=%d \n", data_event_info->var_name, 
		event_info->data_event.bytes);
        if (data_event_info->var_name) {
          sprintf(user_event_name, "Data transfer from device to host <variable=%s>", data_event_info->var_name);
        } else {
          sprintf(user_event_name, "Data transfer from device to host <other>");
        }
        TAU_TRIGGER_EVENT(user_event_name, event_info->data_event.bytes);
      }
      TAU_SET_EVENT_NAME(event_name, ">openacc_enqueue_download");
    case acc_ev_enqueue_download_end        : TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_download");
    case acc_ev_wait_start                  : TAU_SET_EVENT_NAME(event_name, ">openacc_wait");
    case acc_ev_wait_end                    : TAU_SET_EVENT_NAME(event_name, "<openacc_wait");
#ifdef TAU_PGI_OPENACC_15
    case acc_ev_implicit_wait_start         : TAU_SET_EVENT_NAME(event_name, ">openacc_implicit_wait");
    case acc_ev_implicit_wait_end           : TAU_SET_EVENT_NAME(event_name, "<openacc_implicit_wait");
#endif /* TAU_PGI_OPENACC_15 */
    case acc_ev_compute_construct_start     : TAU_SET_EVENT_NAME(event_name, ">openacc_compute_construct");
    case acc_ev_compute_construct_end       : TAU_SET_EVENT_NAME(event_name, "<openacc_compute_construct");
    case acc_ev_create                      : TAU_SET_EVENT_NAME(event_name, "openacc_create");
    case acc_ev_delete                      : TAU_SET_EVENT_NAME(event_name, "openacc_delete");
    case acc_ev_alloc                       : TAU_SET_EVENT_NAME(event_name, "openacc_alloc");
    case acc_ev_free                        : TAU_SET_EVENT_NAME(event_name, "openacc_free");
    default                                 : TAU_SET_EVENT_NAME(event_name, "default");
  }
  char srcinfo[1024]; 
  char lineinfo[256]; 

  if (prof_info) {
#ifdef TAU_PGI_OPENACC_OLD
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
#else /* TAU_PGI_OPENACC_OLD */
    TAU_VERBOSE("Device=%d ", prof_info->device_number);
    TAU_VERBOSE("Thread=%d ", prof_info->thread_id);
    sprintf(srcinfo, " %s [{%s}", prof_info->func_name, prof_info->src_file);
    if ((event_name[15] == 'd' && event_name[16] == 'a' && prof_info->line_no) || (event_name[17] == 'c' && event_name[18] == 'o' && prof_info->line_no)) {
	TAU_VERBOSE("Do not extract line number info for %s\n", event_name); 
	// PGI has messed up line numbers for entry and exit for construct 
	// and data events 
    /* do nothing */ 
    } else {
      sprintf(lineinfo, " {%d,0}", prof_info->line_no); 
      strcat(srcinfo,lineinfo);
      if ((prof_info->end_line_no) && (prof_info->end_line_no > prof_info->line_no)) {
        sprintf(lineinfo, "-{%d,0}", prof_info->end_line_no); 
        strcat(srcinfo,lineinfo);
      }
    }
#endif /* TAU_PGI_OPENACC_OLD */
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

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      /*if(_status == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED)          \
          exit(0);                                                      \
      else*/                                                              \
          exit(-1);                                                     \
    }                                                                   \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))


//array enumerating CUpti_OpenAccEventKind strings
char* openacc_event_names[] = {
		"CUPTI_OPENACC_EVENT_KIND_INVALD",
		"CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT",
		"CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN",
		"CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN",
		"CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH",
		"CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD",
		"CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD",
		"CUPTI_OPENACC_EVENT_KIND_WAIT",
		"CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT",
		"CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT",
		"CUPTI_OPENACC_EVENT_KIND_UPDATE",
		"CUPTI_OPENACC_EVENT_KIND_ENTER_DATA",
		"CUPTI_OPENACC_EVENT_KIND_EXIT_DATA",
		"CUPTI_OPENACC_EVENT_KIND_CREATE",
		"CUPTI_OPENACC_EVENT_KIND_DELETE",
		"CUPTI_OPENACC_EVENT_KIND_ALLOC",
		"CUPTI_OPENACC_EVENT_KIND_FREE"
	};

static size_t openacc_records = 0;
#ifdef CUPTI
static void
printActivity(CUpti_Activity *record)
{                                                                                  
  switch (record->kind) {
	//TODO: tau_cupti_register_gpu_event for each of these; could probably piggyback on existing code, or:
	// write a tau_openacc_register_thingy for each type of event (more likely)
	// make an event mappy thing a la CuptiActivity
	// write a GpuEvent as seen in TauGpuAdapterCupti for each of these
	// probably only events that matter are 11, 12, and 9
	// find out what events are actually under LAUNCH
	// https://docs.nvidia.com/cuda/cupti/structCUpti__ActivityOpenAcc.html#structCUpti__ActivityOpenAcc
	// https://docs.nvidia.com/cuda/cupti/structCUpti__ActivityOpenAccData.html#structCUpti__ActivityOpenAccData
	// https://docs.nvidia.com/cuda/cupti/group__CUPTI__ACTIVITY__API.html#group__CUPTI__ACTIVITY__API_1g0e638b0b6a210164345ab159bcba6717
        case CUPTI_ACTIVITY_KIND_OPENACC_DATA:                                        
        case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH:
        case CUPTI_ACTIVITY_KIND_OPENACC_OTHER:
        {                                                                                    
					CUpti_ActivityOpenAcc *oacc = (CUpti_ActivityOpenAcc *)record;
					if (oacc->deviceType != acc_device_nvidia) { 
						printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n", oacc->deviceType, acc_device_nvidia);
						exit(-1);
					}
					
					uint32_t context = oacc->cuContextId;
					uint32_t device = oacc->cuDeviceId;
					uint32_t stream = oacc->cuStreamId;
					uint32_t corr_id = oacc->externalId; // pretty sure this is right
					uint64_t start = oacc->start;
					uint64_t end = oacc->end;

					int task = oacc->cuThreadId;

					//TODO: empty for now
					GpuEventAttributes* map;
					int map_size = 0;

					//TODO: could do better but this'll do for now
					const char* name = openacc_event_names[oacc->eventKind];
					
					Tau_openacc_register_gpu_event(name, device, stream, context, task, corr_id, map, map_size, start/1e3, end/1e3);

          openacc_records++;
        }
        break;

		default:
      ;
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}
#endif
void finalize()
{
#ifdef CUPTI
  cuptiActivityFlushAll(0);
#endif
  printf("Found %llu OpenACC records\n", (long long unsigned) openacc_records);
}

////////////////////////////////////////////////////////////////////////////
typedef void (*Tau_openacc_registration_routine_t)( acc_event_t, acc_prof_callback_t, int );
#ifdef TAU_PGI_OPENACC_OLD 
extern "C" void
acc_register_library(Tau_openacc_registration_routine_t reg, Tau_openacc_registration_routine_t unreg )
{
    TAU_VERBOSE("Inside acc_register_library\n");
#else /* TAU_PGI_OPENACC_OLD */
////////////////////////////////////////////////////////////////////////////
//typedef void (*Tau_openacc_prof_fn_t)();

////////////////////////////////////////////////////////////////////////////
//typedef Tau_acc_prof_fn_t (*Tau_acc_prof_lookup) (const char *name);

////////////////////////////////////////////////////////////////////////////
extern "C" void
acc_register_library(acc_prof_reg reg, acc_prof_reg unreg, acc_prof_lookup lookup)
{
    TAU_VERBOSE("Inside acc_register_library\n");
#endif /* TAU_PGI_OPENACC_OLD */

#ifdef TAU_PGI_OPENACC_OLD 
    reg( acc_ev_init_start, Tau_openacc_callback, 0 );
    reg( acc_ev_init_end, Tau_openacc_callback, 0 );
    reg( acc_ev_shutdown_start, Tau_openacc_callback, 0 );
    reg( acc_ev_shutdown_end, Tau_openacc_callback, 0 );
    reg( acc_ev_done, Tau_openacc_callback, 0 );
    reg( acc_ev_update_construct_start, Tau_openacc_callback, 0 );
    reg( acc_ev_update_construct_end, Tau_openacc_callback, 0 );
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
    reg( acc_ev_enter_data_start, Tau_openacc_callback, 0 );
    reg( acc_ev_enter_data_end, Tau_openacc_callback, 0 );
    reg( acc_ev_exit_data_start, Tau_openacc_callback, 0 );
    reg( acc_ev_exit_data_end, Tau_openacc_callback, 0 );
    reg( acc_ev_create, Tau_openacc_callback, 0 );
    reg( acc_ev_delete, Tau_openacc_callback, 0 );
    reg( acc_ev_alloc, Tau_openacc_callback, 0 );
    reg( acc_ev_free, Tau_openacc_callback, 0 );
#else /* TAU_PGI_OPENACC_OLD */
    reg( acc_ev_device_init_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_device_init_end, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_device_shutdown_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_device_shutdown_end, Tau_openacc_callback, (acc_register_t) 0 );
    // reg( acc_ev_done, Tau_openacc_callback, 0 );
    reg( acc_ev_update_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_update_end, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_enqueue_launch_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_enqueue_launch_end, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_enqueue_upload_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_enqueue_upload_end, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_enqueue_download_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_enqueue_download_end, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_wait_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_wait_end, Tau_openacc_callback, (acc_register_t) 0 );
#ifdef TAU_PGI_OPENACC_15
    reg( acc_ev_implicit_wait_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_implicit_wait_end, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_exit_data_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_exit_data_end, Tau_openacc_callback, (acc_register_t) 0 );
#endif /* TAU_PGI_OPENACC_15 */
    reg( acc_ev_create, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_delete, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_alloc, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_free, Tau_openacc_callback, (acc_register_t) 0 );
// Added these four to the new version: (line numbers are omitted for PGI)
    reg( acc_ev_compute_construct_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_compute_construct_end, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_enter_data_start, Tau_openacc_callback, (acc_register_t) 0 );
    reg( acc_ev_enter_data_end, Tau_openacc_callback, (acc_register_t) 0 );
#endif /* TAU_PGI_OPENACC_OLD */
/*
    reg( acc_ev_compute_construct_start, Tau_openacc_callback, 0 );
    reg( acc_ev_compute_construct_end, Tau_openacc_callback, 0 );
    reg( acc_ev_data_construct_enter_start, Tau_openacc_callback, 0 );
    reg( acc_ev_data_construct_enter_end, Tau_openacc_callback, 0 );
    reg( acc_ev_data_construct_exit_start, Tau_openacc_callback, 0 );
    reg( acc_ev_data_construct_exit_end, Tau_openacc_callback, 0 );
*/
#ifdef CUPTI
    if (cuptiOpenACCInitialize(reg, unreg, lookup) != CUPTI_SUCCESS) {
        printf("ERROR: failed to initialize CUPTI OpenACC support\n");
    }

    printf("Initialized CUPTI for OpenACC\n");

    CUptiResult cupti_err = CUPTI_SUCCESS;

    cupti_err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_DATA);
    cupti_err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH);
    cupti_err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_OTHER);

    if (cupti_err != CUPTI_SUCCESS) {
        printf("ERROR: unable to enable some OpenACC CUPTI measurements\n");
    }

    cupti_err = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);

    if (cupti_err != CUPTI_SUCCESS) {
        printf("ERROR: unable to register buffers with CUPTI\n");
    }
#endif
    atexit(finalize);

} // acc_register_library 

////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////
