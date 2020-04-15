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
#include "pgi_acc_prof.h"

#ifdef CUPTI
#include <cupti.h>
#include <cuda.h>
#endif

#include <Profile/Profiler.h>
#include <Profile/TauOpenACC.h>
#include <Profile/TauGpuAdapterOpenACC.h>
#include <Profile/TauGpu.h>

#define TAU_SET_EVENT_NAME(event_name, str) strcpy(event_name, str)
////////////////////////////////////////////////////////////////////////////

// strings for OpenACC parent constructs; based on enum acc_construct_t
char* acc_constructs[] = {
	"parallel",
	"kernels",
	"loop",
	"data",
	"enter data",
	"exit data",
	"host data",
	"atomic",
	"declare",
	"init",
	"shutdown",
	"set",
	"update",
	"routine",
	"wait",
	"runtime api",
	"serial",
};

extern "C" static void
Tau_openacc_launch_callback(acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info)
{
	acc_launch_event_info* launch_event = &(event_info->launch_event);
	char file_name[256];
	char event_name[256];
	char event_data[256];
	int start = 0; // 0 = stop, 1 = start, -1 = something weird happened

	switch(prof_info->event_type) {
		// note: these don't correspond to when kernels are actually run, just when they're put in the 
		// execution queue on the device
		case acc_ev_enqueue_launch_start:
			start = 1;
			sprintf(event_name, "OpenACC enqueue launch");
			break;
		case acc_ev_enqueue_launch_end:
			start = 0;
			sprintf(event_name, "OpenACC enqueue launch");
			break;
		default:
			start = -1;
			sprintf(event_name, "UNKNOWN OPENACC LAUNCH EVENT");
			fprintf(stderr, "ERROR: Non-launch event passed to OpenACC launch event callback.");
	}

	sprintf(file_name, "%s:%s-%s", 
			prof_info->src_file, 
			(prof_info->line_no > 0) ? std::to_string(prof_info->line_no).c_str() : "?",
			(prof_info->end_line_no > 0) ? std::to_string(prof_info->end_line_no).c_str() : "?");

	
	sprintf(event_data, " kernel name = %s %s; parent construct = %s; gangs=%zu, workers=%zu, vector lanes=%zu (%s)", 
	//                            name ^  ^ (implicit?)                                   file and line no. ^
			(launch_event->kernel_name) ? launch_event->kernel_name : "unknown kernel",
			(launch_event->implicit) ? "(implicit)" : "",
			acc_constructs[launch_event->parent_construct],
			launch_event->num_gangs,
			launch_event->num_workers,
			launch_event->vector_length,
			file_name);

	//printf("parent construct %d\n", launch_event->parent_construct);

	strcat(event_name, event_data);

	if (start == 1) {
		TAU_START(&event_name[0]); //has to be &str[index] to get around const-ness problems
	}
	else if (start == 0) {
		TAU_STOP(&event_name[0]);
	}
	else {
		TAU_TRIGGER_EVENT(&event_name[0], 0);
	}
}

//TODO: split this into multiple functions to get rid of the nasty switch case
extern "C" static void
Tau_openacc_callback( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info )
{
  char event_name[256], user_event_name[256];

  //acc_event_t *event_type_info = NULL; 
  acc_data_event_info*   data_event_info = NULL;
  acc_launch_event_info* launch_event_info = NULL;
  //acc_other_event_info*  other_event_info = NULL;

  switch (prof_info->event_type) {
    case acc_ev_device_init_start: 
			Tau_create_top_level_timer_if_necessary(); 
			TAU_SET_EVENT_NAME(event_name, ">openacc_init"); 
			break;
    case acc_ev_device_init_end: 
			TAU_SET_EVENT_NAME(event_name, "<openacc_init");
			break;
    case acc_ev_device_shutdown_start: 
			TAU_SET_EVENT_NAME(event_name, ">openacc_shutdown");
			break;
    case acc_ev_device_shutdown_end: 
			TAU_SET_EVENT_NAME(event_name, "<openacc_shutdown");
			break;
    case acc_ev_enter_data_start: 
      TAU_SET_EVENT_NAME(event_name, ">openacc_enter_data");
			break;
    case acc_ev_enter_data_end: 
			TAU_SET_EVENT_NAME(event_name, "<openacc_enter_data");
			break;
    case acc_ev_exit_data_start: 
			TAU_SET_EVENT_NAME(event_name, ">openacc_exit_data");
			break;
    case acc_ev_exit_data_end: 
			TAU_SET_EVENT_NAME(event_name, "<openacc_exit_data");
			break;
    case acc_ev_update_start: 
			TAU_SET_EVENT_NAME(event_name, ">openacc_update");
			break;
    case acc_ev_update_end: 
			TAU_SET_EVENT_NAME(event_name, "<openacc_update");
			break;
    case acc_ev_enqueue_upload_start: 
      if (event_info) {
        data_event_info = &(event_info->data_event); 
        TAU_VERBOSE("UPLOAD start: Var_name = %s, bytes=%d \n", 
					data_event_info->var_name, event_info->data_event.bytes);
        if (data_event_info->var_name) {
          sprintf(user_event_name, "Data transfer from host to device <variable=%s>", data_event_info->var_name);
        } else {
          sprintf(user_event_name, "Data transfer from host to device <other>");
        }
        TAU_TRIGGER_EVENT(user_event_name, event_info->data_event.bytes);
      }
      TAU_SET_EVENT_NAME(event_name, ">openacc_enqueue_upload");
			break;
    case acc_ev_enqueue_upload_end: 
      TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_upload");
			break;
    case acc_ev_enqueue_download_start: 
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
			break;
    case acc_ev_enqueue_download_end: 
			TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_download");
			break;
    case acc_ev_wait_start: 
			TAU_SET_EVENT_NAME(event_name, ">openacc_wait");
			break;
    case acc_ev_wait_end: 
			TAU_SET_EVENT_NAME(event_name, "<openacc_wait");
			break;
    case acc_ev_compute_construct_start: 
			TAU_SET_EVENT_NAME(event_name, ">openacc_compute_construct");
			break;
    case acc_ev_compute_construct_end: 
			TAU_SET_EVENT_NAME(event_name, "<openacc_compute_construct");
			break;
    case acc_ev_create: 
			TAU_SET_EVENT_NAME(event_name, "openacc_create");
			break;
    case acc_ev_delete: 
			TAU_SET_EVENT_NAME(event_name, "openacc_delete");
			break;
    case acc_ev_alloc: 
			TAU_SET_EVENT_NAME(event_name, "openacc_alloc");
			break;
    case acc_ev_free: 
			TAU_SET_EVENT_NAME(event_name, "openacc_free");
			break;
    default: 
			TAU_SET_EVENT_NAME(event_name, "unknown OpenACC event");
			break;
  }
  char srcinfo[1024]; 
  char lineinfo[256]; 

//TODO: fix this, ew
  if (prof_info) {
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
	GpuEventAttributes* map;
	int map_size;                                              
  switch (record->kind) {
	//TODO:
	// make an event mappy thing a la CuptiActivity
        case CUPTI_ACTIVITY_KIND_OPENACC_DATA:
				{
					CUpti_ActivityOpenAccData *oacc_data = (CUpti_ActivityOpenAccData*) record;
					if (oacc_data->deviceType != acc_device_nvidia) {
            printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n", oacc_data->deviceType, acc_device_nvidia);
            exit(-1);
          }

					map_size = 2;
					map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);

					static TauContextUserEvent* bytes;
					Tau_get_context_userevent((void**) &bytes, "Bytes transfered");
					map[0].userEvent = bytes;
					map[0].data = oacc_data->bytes;
					break;
				}                                        
        case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH:
				{
					CUpti_ActivityOpenAccLaunch *oacc_launch = (CUpti_ActivityOpenAccLaunch*) record;
					if (oacc_launch->deviceType != acc_device_nvidia) {
            printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n", oacc_launch->deviceType, acc_device_nvidia);
            exit(-1);
          }

					map_size = 4;
					map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);

					static TauContextUserEvent* gangs;
					Tau_get_context_userevent((void**) &gangs, "Num gangs");
					map[0].userEvent = gangs;
					map[0].data = oacc_launch->numGangs;

					static TauContextUserEvent* workers;
					Tau_get_context_userevent((void**) &workers, "Num workers");
					map[1].userEvent = workers;
					map[1].data = oacc_launch->numWorkers;

					static TauContextUserEvent* vector;
					Tau_get_context_userevent((void**) &vector, "Vector lanes");
					map[2].userEvent = vector;
					map[2].data = oacc_launch->vectorLength;

					break;
				}
        case CUPTI_ACTIVITY_KIND_OPENACC_OTHER:
        {                                                                                    
					CUpti_ActivityOpenAccData *oacc_other = (CUpti_ActivityOpenAccData*) record;
					if (oacc_other->deviceType != acc_device_nvidia) {
            printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n", oacc_other->deviceType, acc_device_nvidia);
            exit(-1);
          }

					map_size = 1;
					map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
        break;
        }

		default:
      ;
  }

	CUpti_ActivityOpenAcc* oacc = (CUpti_ActivityOpenAcc*) record;
	// TODO: are we guaranteed to only get openacc events? I don't know. Guess we'll find out.
	// always add duration at the end
	uint32_t context = oacc->cuContextId;
	uint32_t device = oacc->cuDeviceId;
	uint32_t stream = oacc->cuStreamId;
	uint32_t corr_id = oacc->externalId; // pretty sure this is right
	uint64_t start = oacc->start;
	uint64_t end = oacc->end;

	int task = oacc->cuThreadId;

	static TauContextUserEvent* duration;
	Tau_get_context_userevent((void**) &duration, "Duration");
	map[map_size-1].userEvent = duration;
	map[map_size-1].data = (double)(end - start) / (double)1e6;

	//TODO: could do better but this'll do for now
	const char* name = openacc_event_names[oacc->eventKind];
	
	Tau_openacc_register_gpu_event(name, device, stream, context, task, corr_id, map, map_size, start/1e3, end/1e3);

	openacc_records++;
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
extern "C" void
acc_register_library(acc_prof_reg reg, acc_prof_reg unreg, acc_prof_lookup lookup)
{
    TAU_VERBOSE("Inside acc_register_library\n");

		// Launch events
    reg( acc_ev_enqueue_launch_start,      Tau_openacc_launch_callback, acc_reg );
    reg( acc_ev_enqueue_launch_end,        Tau_openacc_launch_callback, acc_reg );
    
		// Other events
		reg( acc_ev_device_init_start,         Tau_openacc_callback, acc_reg );
    reg( acc_ev_device_init_end,           Tau_openacc_callback, acc_reg );
    reg( acc_ev_device_shutdown_start,     Tau_openacc_callback, acc_reg );
    reg( acc_ev_device_shutdown_end,       Tau_openacc_callback, acc_reg );
    reg( acc_ev_update_start,              Tau_openacc_callback, acc_reg );
    reg( acc_ev_update_end,                Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_upload_start,      Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_upload_end,        Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_download_start,    Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_download_end,      Tau_openacc_callback, acc_reg );
    reg( acc_ev_wait_start,                Tau_openacc_callback, acc_reg );
    reg( acc_ev_wait_end,                  Tau_openacc_callback, acc_reg );
    reg( acc_ev_create,                    Tau_openacc_callback, acc_reg );
    reg( acc_ev_delete,                    Tau_openacc_callback, acc_reg );
    reg( acc_ev_alloc,                     Tau_openacc_callback, acc_reg );
    reg( acc_ev_free,                      Tau_openacc_callback, acc_reg );
    reg( acc_ev_compute_construct_start,   Tau_openacc_callback, acc_reg );
    reg( acc_ev_compute_construct_end,     Tau_openacc_callback, acc_reg );
    reg( acc_ev_enter_data_start,          Tau_openacc_callback, acc_reg );
    reg( acc_ev_enter_data_end,            Tau_openacc_callback, acc_reg );


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

