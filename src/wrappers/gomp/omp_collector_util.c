/*
 Collector Support for Open64's OpenMP runtime library

 Copyright (C) 2009 University of Houston.

 This program is free software; you can redistribute it and/or modify it
 under the terms of version 2 of the GNU General Public License as
 published by the Free Software Foundation.

 This program is distributed in the hope that it would be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

 Further, this software is distributed without any warranty that it is
 free of the rightful claim of any third person regarding infringement
 or the like.  Any license provided herein, whether implied or
 otherwise, applies only to this software file.  Patent licenses, if
 any, provided herein do not apply to combinations of this program with
 other software, or any other product whatsoever.

 You should have received a copy of the GNU General Public License along
 with this program; if not, write the Free Software Foundation, Inc., 59
 Temple Place - Suite 330, Boston MA 02111-1307, USA.

 Contact information:
 http://www.cs.uh.edu/~hpctools
*/

#include "omp_collector_api.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <Profile/Profiler.h>
#include "omp.h"

typedef void (*callback) (OMP_COLLECTORAPI_EVENT e);

char GOMP_OMP_EVENT_NAME[35][50]= {
  "OMP_EVENT_FORK",
  "OMP_EVENT_JOIN",
  "OMP_EVENT_THR_BEGIN_IDLE",
  "OMP_EVENT_THR_END_IDLE",
  "OMP_EVENT_THR_BEGIN_IBAR",
  "OMP_EVENT_THR_END_IBAR",
  "OMP_EVENT_THR_BEGIN_EBAR",
  "OMP_EVENT_THR_END_EBAR",
  "OMP_EVENT_THR_BEGIN_LKWT",
  "OMP_EVENT_THR_END_LKWT",
  "OMP_EVENT_THR_BEGIN_CTWT",
  "OMP_EVENT_THR_END_CTWT",
  "OMP_EVENT_THR_BEGIN_ODWT",
  "OMP_EVENT_THR_END_ODWT",
  "OMP_EVENT_THR_BEGIN_MASTER",
  "OMP_EVENT_THR_END_MASTER",
  "OMP_EVENT_THR_BEGIN_SINGLE",
  "OMP_EVENT_THR_END_SINGLE",
  "OMP_EVENT_THR_BEGIN_ORDERED",
  "OMP_EVENT_THR_END_ORDERED",
  "OMP_EVENT_THR_BEGIN_ATWT",
  "OMP_EVENT_THR_END_ATWT",
  "OMP_EVENT_THR_BEGIN_CREATE_TASK",
  "OMP_EVENT_THR_END_CREATE_TASK_IMM",
  "OMP_EVENT_THR_END_CREATE_TASK_DEL",
  "OMP_EVENT_THR_BEGIN_SCHD_TASK",
  "OMP_EVENT_THR_END_SCHD_TASK",
  "OMP_EVENT_THR_BEGIN_SUSPEND_TASK",
  "OMP_EVENT_THR_END_SUSPEND_TASK",
  "OMP_EVENT_THR_BEGIN_STEAL_TASK",
  "OMP_EVENT_THR_END_STEAL_TASK",
  "OMP_EVENT_THR_FETCHED_TASK",
  "OMP_EVENT_THR_BEGIN_EXEC_TASK",
  "OMP_EVENT_THR_BEGIN_FINISH_TASK",
  "OMP_EVENT_THR_END_FINISH_TASK"
 };


char OMP_STATE_NAME[16][50]= {
  "THR_OVHD_STATE",          /* Overhead */
  "THR_WORK_STATE",          /* Useful work, excluding reduction, master, single, critical */
  "THR_IBAR_STATE",          /* In an implicit barrier */
  "THR_EBAR_STATE",          /* In an explicit barrier */
  "THR_IDLE_STATE",          /* Slave waiting for work */
  "THR_SERIAL_STATE",        /* thread not in any OMP parallel region (initial thread only) */
  "THR_REDUC_STATE",         /* Reduction */
  "THR_LKWT_STATE",          /* Waiting for lock */
  "THR_CTWT_STATE",          /* Waiting to enter critical region */
  "THR_ODWT_STATE",          /* Waiting to execute an ordered region */
  "THR_ATWT_STATE",          /* Waiting to enter an atomic region */
  "THR_TASK_CREATE_STATE",        /* Creating new explicit task */
  "THR_TASK_SCHEDULE_STATE",      /* Find explicit task from queue */
  "THR_TASK_SUSPEND_STATE",       /* Suspending current explicit task */
  "THR_TASK_STEAL_STATE", /* Stealing explicit task */
  "THR_TASK_FINISH_STATE"         /* Completing explicit task */
};

static callback callbacks[OMP_EVENT_THR_END_FINISH_TASK+1];

__thread OMP_COLLECTOR_API_THR_STATE gomp_state = THR_IDLE_STATE;

static unsigned long current_region_id;

int __omp_collector_api(void *arg);

static omp_lock_t init_lock;
int collector_initialized=0;
static omp_lock_t paused_lock;
int collector_paused=0;
static omp_lock_t event_lock;
int process_top_request(omp_collector_message * req);
int register_event(omp_collector_message * req);
int unregister_event(omp_collector_message *req);
int return_state(omp_collector_message *req);
int return_current_prid(omp_collector_message *req);
int return_parent_prid(omp_collector_message *req);

int __omp_collector_api(void *arg)
{
  if(arg!=NULL) {
    char *traverse = (char *) arg;

    while((int)(*traverse)!=0) {
      omp_collector_message req;
      req.sz = (int)(*traverse); // todo: add check for consistency    
      traverse+=sizeof(int);
      req.r = (OMP_COLLECTORAPI_REQUEST)(*traverse);  // todo: add check for a valid request
      traverse+=sizeof(int);      
      req.ec= (OMP_COLLECTORAPI_EC *) traverse;  // copy address for response of error code
      traverse+=sizeof(int);    
      req.rsz = (int *)(traverse);
      traverse+=sizeof(int);
      req.mem = traverse;
      traverse+=req.sz-(4*sizeof(int));
      process_top_request(&req);  
    } 

    return 0;
  }
  return -1;
}

void __ompc_req_start(omp_collector_message *req)
{
  int i;
  
  if(!collector_initialized) {
    for (i=0; i< OMP_EVENT_THR_END_FINISH_TASK+1; i++) {
      omp_set_lock(&event_lock);
      callbacks[i]= NULL;
      omp_unset_lock(&event_lock);
    } // note check callback boundaries.
    omp_set_lock(&init_lock);
    collector_initialized = 1;
    gomp_state = THR_SERIAL_STATE; // everyone is initialized to IDLE except thread 0
    omp_unset_lock(&init_lock);
    *(req->ec) = OMP_ERRCODE_OK;
  } else {
    *(req->ec) = OMP_ERRCODE_SEQUENCE_ERR;
  }
   
  *(req->rsz) =0;
}


void __ompc_req_stop(omp_collector_message *req)
{
  int i;

  if(collector_initialized) {
    omp_set_lock(&init_lock);
    collector_initialized = 0;
    omp_unset_lock(&init_lock);
    omp_set_lock(&event_lock);
    for (i=0; i< OMP_EVENT_THR_END_FINISH_TASK+1; i++) {
      callbacks[i]= NULL;
    } 
    omp_unset_lock(&event_lock);
    // note check callback boundaries.
  
    *(req->ec) = OMP_ERRCODE_OK;
  } else {
    *(req->ec) = OMP_ERRCODE_SEQUENCE_ERR;
  } 
  *(req->rsz) =0;
}

int __ompc_req_pause(omp_collector_message *req)
{
  if(collector_initialized) {
                 
    omp_set_lock(&paused_lock); 
    collector_paused = 1;
    omp_unset_lock(&paused_lock);     
    *(req->ec) = OMP_ERRCODE_OK;
  } else {
    *(req->ec) = OMP_ERRCODE_SEQUENCE_ERR;
  } 
  *(req->rsz) = 0;
  return 1;
}


int __ompc_req_resume(omp_collector_message *req)
{
  if(collector_initialized) { 
    omp_set_lock(&paused_lock); 
    collector_paused = 0;
    omp_unset_lock(&paused_lock);
    *(req->ec) = OMP_ERRCODE_OK;
  } else {
    *(req->ec) = OMP_ERRCODE_SEQUENCE_ERR;
  } 
  *(req->rsz) = 0;
  return 1;
}

int process_top_request(omp_collector_message *req)
{
  switch(req->r) {
  case OMP_REQ_START:
    __ompc_req_start(req);   
    break;

  case OMP_REQ_REGISTER:
    register_event(req);
    break;

  case OMP_REQ_UNREGISTER:
    unregister_event(req);
    break; 
          
  case OMP_REQ_STATE:
    return_state(req);
    break; 

  case OMP_REQ_CURRENT_PRID:
    return_current_prid(req);
    break;

  case OMP_REQ_PARENT_PRID:
    return_parent_prid(req);
    break;

  case OMP_REQ_STOP:
    __ompc_req_stop(req);
    break;

  case OMP_REQ_PAUSE:
    __ompc_req_pause(req); 
    break;

  case OMP_REQ_RESUME:
    __ompc_req_resume(req);
    break;

  default:
    *(req->ec) = OMP_ERRCODE_UNKNOWN;
    *(req->rsz) = 0;   
    break;
  }
  return 1;
   
}

int event_is_valid(OMP_COLLECTORAPI_EVENT e)
{
  /* this needs to be improved with something more portable when we extend the events in the runtime */
  if (e>=OMP_EVENT_FORK && e<=OMP_EVENT_THR_END_FINISH_TASK)
    return 1; 
  else
    return 0;
}

int event_is_supported(OMP_COLLECTORAPI_EVENT e)
{
  int event_supported=1;
  switch (e) {
  case OMP_EVENT_THR_BEGIN_ATWT:
  case OMP_EVENT_THR_END_FINISH_TASK:
    event_supported=0;
    break;

  default:
    break;
  }

  return event_supported;
}
int register_event(omp_collector_message *req)
{    
  if(collector_initialized) {
    OMP_COLLECTORAPI_EVENT  *event = (OMP_COLLECTORAPI_EVENT *)req->mem;
    unsigned long *temp_mem = (unsigned long *)(req->mem + sizeof(OMP_COLLECTORAPI_EVENT));
    if(event_is_valid(*event)) {
      omp_set_lock(&event_lock);
      callbacks[*event] = (void (*)(OMP_COLLECTORAPI_EVENT)) (*temp_mem);
      omp_unset_lock(&event_lock); 
      if(event_is_supported(*event)) *(req->ec)=OMP_ERRCODE_OK;
      else *(req->ec)=OMP_ERRCODE_UNSUPPORTED;
    } else {
      *(req->ec)=OMP_ERRCODE_UNKNOWN;
    }
  } else {
    *(req->ec)=OMP_ERRCODE_SEQUENCE_ERR;
  }
  
  *(req->rsz) = 0;

  return 1;
}

int unregister_event(omp_collector_message *req)
{

  if(collector_initialized) {
    OMP_COLLECTORAPI_EVENT  *event = (OMP_COLLECTORAPI_EVENT *)req->mem;
    if(event_is_valid(*event)) {
      omp_set_lock(&event_lock);
      callbacks[*event] = NULL;
      omp_unset_lock(&event_lock);

      if(event_is_supported(*event))
        *(req->ec)=OMP_ERRCODE_OK;
      else
        *(req->ec)=OMP_ERRCODE_UNSUPPORTED;
    } else {
      *(req->ec)=OMP_ERRCODE_UNKNOWN;
    }
  } else {
    *(req->ec)=OMP_ERRCODE_SEQUENCE_ERR;
  }
  return 1;
}

/* needs to be thread safe */
int return_state_id(omp_collector_message *req,long id)
{
  int possible_mem_prob = (req->sz - 4*sizeof(int)) < (sizeof(OMP_COLLECTOR_API_THR_STATE)+sizeof(unsigned long)); 
  if(!possible_mem_prob){ 
    *((unsigned long *)(req->mem+sizeof(OMP_COLLECTOR_API_THR_STATE)))=id; 
    *(req->rsz) = sizeof(OMP_COLLECTOR_API_THR_STATE)+sizeof(unsigned long);
    *(req->ec) = OMP_ERRCODE_OK; 
  } else {
    *(req->ec) = OMP_ERRCODE_MEM_TOO_SMALL;
    *(req->rsz)=0;
    return 0;
  }
  return 1;
}
int return_state(omp_collector_message *req)
{
  if(!collector_initialized) {
    *(req->rsz)=0;
    *(req->ec)=OMP_ERRCODE_SEQUENCE_ERR;
    return 0;
  } 

  if((req->sz - 4*sizeof(int)) < sizeof(OMP_COLLECTOR_API_THR_STATE)) {
    *(req->ec) = OMP_ERRCODE_MEM_TOO_SMALL;
    return 0;
  } else {
    *((unsigned long *)(req->mem))=gomp_state; 
    *(req->rsz) = sizeof(OMP_COLLECTOR_API_THR_STATE)+sizeof(unsigned long);
    *(req->ec) = OMP_ERRCODE_OK; 
    return 1;
/*
    switch(gomp_state) {
  
    case THR_IBAR_STATE:
      return return_state_id(req,gomp_state);
    case THR_EBAR_STATE:    	      
      return return_state_id(req,gomp_state);    
    case THR_LKWT_STATE:
      return return_state_id(req,gomp_state);     
    case THR_CTWT_STATE:
      return return_state_id(req,gomp_state);     
    case THR_ODWT_STATE:
      return return_state_id(req,gomp_state);     
    case THR_ATWT_STATE:
      return return_state_id(req,gomp_state);     
    default:
      *(req->rsz)=sizeof(OMP_COLLECTOR_API_THR_STATE);
      *(req->ec) = OMP_ERRCODE_OK;
      return 1;
      break; 
 
    }
*/
  }

  return 1;
}

int return_current_prid(omp_collector_message *req)
{ 
  if(!collector_initialized) {
    *(req->rsz)=0;
    *(req->ec)=OMP_ERRCODE_SEQUENCE_ERR;
    return 0;
  } 

  if((req->sz - 4*sizeof(int)) < sizeof(unsigned long)) {
    *(req->ec) = OMP_ERRCODE_MEM_TOO_SMALL;
    *(req->rsz)=0;
    return 0;
  } else {
    if(gomp_state!=THR_SERIAL_STATE) {
      *((unsigned long *)req->mem) = current_region_id; }
    else *((unsigned long *)req->mem) = 0;
    *(req->rsz)=sizeof(unsigned long);
  }
  return 1;
}       

int return_parent_prid(omp_collector_message *req)
{
  if(!collector_initialized) {
    *(req->rsz)=0;
    *(req->ec)=OMP_ERRCODE_SEQUENCE_ERR;
    return 0;
  } 


  if((req->sz - 4*sizeof(int)) < sizeof(unsigned long)) {
    *(req->rsz)=0;
    *(req->ec) = OMP_ERRCODE_MEM_TOO_SMALL;
    return 0;
  } else {
    if(gomp_state!=THR_SERIAL_STATE) {
      *((unsigned long *)req->mem) = current_region_id; }
    else *((unsigned long *)req->mem) = 0;   
    *(req->rsz)=sizeof(unsigned long);
  }
  return 1;
}
       
void __omp_collector_init() {
  omp_init_lock(&init_lock);
  omp_init_lock(&paused_lock);
  omp_init_lock(&event_lock);
  current_region_id = 0;
}

void incr_current_region_id() {
  omp_set_lock(&event_lock);
  current_region_id = current_region_id + 1;
  omp_unset_lock(&event_lock);
}

OMP_COLLECTOR_API_THR_STATE __ompc_set_state(OMP_COLLECTOR_API_THR_STATE state)
{
  OMP_COLLECTOR_API_THR_STATE previous = gomp_state;
  gomp_state = state;
  return previous;
}
void __ompc_event_callback(OMP_COLLECTORAPI_EVENT event)
{
  if( callbacks[event] && collector_initialized && (!collector_paused))
  callbacks[event](event);
}



