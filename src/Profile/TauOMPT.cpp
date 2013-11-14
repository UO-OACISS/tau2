#ifdef TAU_IBM_OMPT
#include <lomp/omp.h>
#endif /* TAU_IBM_OMPT */

#include "omp.h"
#include "ompt.h"
#include <stdio.h>
#include <pthread.h>

#include <Profile/Profiler.h>

// set to enable debugging print statements
#ifdef DEBUG_PROF 
#define DEBUG 1
#endif /* DEBUG_PROF */

// set to enable numerous callbacks
#define TAU_OMPT_CALLBACK 1


#ifdef DEBUG
#define PRINTF(_args...) {printf(_args);}
#else
#define PRINTF(_args...) 
#endif /* DEBUG */

////////////////////////////////////////////////////////////////////////////////
// my tool structures

class TauOMPT {
public:
  TauOMPT();
  ~TauOMPT();
  int GetNewThreadId();
  int GetNewTaskId();
  int GetNewParallelId();
private:
  pthread_mutex_t mutex;
  int threadId;
  int taskId;
  int parallelId;
};

////////////////////////////////////////////////////////////////////////////////
// data

TauOMPT tau_ompt;

////////////////////////////////////////////////////////////////////////////////
// mandatory callbacks

extern "C" void Tau_ompt_parallel_create(
  ompt_data_t  *parent_data,  /* data of parent task               */
  ompt_frame_t *parent_frame, /* frame of parent task 		   */
  ompt_parallel_id_t parallel_id    /* id of parallel region       */
  )								   
{
  // init data
  #if DEBUG
    PRINTF(
      "##m parallel create: id %Ld, parent data %Ld, parent frame (0x%Lx, 0x%Lx), outlined fct 0x%L, TAU Tid=%d\n",
      (long long) parallel_id, parent_data->value, 
      parent_frame->exit_runtime_frame, parent_frame->reenter_runtime_frame,
      0, Tau_get_tid());
  #endif
  TAU_START("OMPT parallel");

}

extern "C" void Tau_ompt_parallel_exit(
  ompt_data_t  *parent_data,  /* data of parent task               */
  ompt_frame_t *parent_frame, /* frame of parent task 		   */
  ompt_parallel_id_t parallel_id    /* id of parallel region       */
  )								   
{
  #if DEBUG
    PRINTF(
      "##m parallel exit: id %Ld, parent data %Ld, parent frame (0x%Lx, 0x%Lx), outlined fct 0x%Lx, TAU Tid=%d\n",
      parallel_id, parent_data->value, 
      parent_frame->exit_runtime_frame, parent_frame->reenter_runtime_frame,
      0, Tau_get_tid());
  #endif
  TAU_STOP("OMPT parallel");
}

extern "C" void Tau_ompt_task_create(
  ompt_data_t  *parent_data,  /* data of parent task               */
  ompt_frame_t *parent_frame, /* frame of parent task 		   */
  ompt_data_t  *data          /* data of new task begin greated    */
  )								   
{
  // init data
  data->value = 1;
  #if DEBUG
    data->value = tau_ompt.GetNewTaskId();
    PRINTF(
      "##m task create: data %Ld, parent data %Ld, parent frame (0x%Lx, 0x%Lx), outlined fct 0x%Lx\n",
      data->value, parent_data->value, 
      parent_frame->exit_runtime_frame, parent_frame->reenter_runtime_frame,
      0);
  #endif
  TAU_START("OMPT task");
}

extern "C" void Tau_ompt_task_exit(ompt_data_t *data)
{
  #if DEBUG
    PRINTF("##m task exit: data %Ld\n", data->value);
  #endif
  TAU_STOP("OMPT task");
}

extern "C" void Tau_ompt_thread_create(ompt_data_t *data)								   
{
  data->value = 1;
  #if DEBUG
    data->value = tau_ompt.GetNewThreadId();
    PRINTF("##m thread create: data %Ld, TAU Tid=%d\n", data->value, Tau_get_tid());
  #endif
  Tau_create_top_level_timer_if_necessary();
  TAU_START("OMPT thread");
}

extern "C" void Tau_ompt_thread_exit(ompt_data_t *data)								   
{
  #if DEBUG
    PRINTF("##m thread exit: data %Ld, TAU Tid = %d\n", data->value, Tau_get_tid());
  #endif
  TAU_STOP("OMPT thread");
  Tau_stop_top_level_timer_if_necessary();
}

////////////////////////////////////////////////////////////////////////////////
// blame shifting callbacks

extern "C" void Tau_ompt_idle_begin(ompt_data_t *data)
{
  #if DEBUG
    PRINTF("##b idle begin: data %Ld, TAU Tid=%d\n", data->value, Tau_get_tid() );
  #endif
  Tau_create_top_level_timer_if_necessary();
  TAU_START("OMPT idle");
}

extern "C" void Tau_ompt_idle_end(ompt_data_t *data)
{
  #if DEBUG
    PRINTF("##b idle end: data %Ld, TAU Tid=%d\n", data->value, Tau_get_tid());
  #endif
  TAU_STOP("OMPT idle");
  Tau_stop_top_level_timer_if_necessary();
}

extern "C" void Tau_ompt_wait_barrier_begin(ompt_data_t *data, ompt_parallel_id_t parallel_id)
{
  #if DEBUG
    PRINTF("##b wait barrier begin: id %Ld, data %Ld, TAU Tid=%d\n", parallel_id, data->value, Tau_get_tid());
  #endif
  //TAU_START("OMPT barrier");
}

extern "C" void Tau_ompt_wait_barrier_end(ompt_data_t *data, ompt_parallel_id_t parallel_id)
{
  #if DEBUG
    PRINTF("##b wait barrier end: id %Ld, data %Ld, TAU Tid=%d\n", parallel_id, data->value, Tau_get_tid());
  #endif
  //TAU_STOP("OMPT barrier");
}

extern "C" void Tau_ompt_wait_taskwait_begin(ompt_data_t *data, ompt_parallel_id_t parallel_id)
{
  #if DEBUG
    PRINTF("##b wait taskwait begin: id %Ld, data %Ld\n", parallel_id, data->value);
  #endif
  TAU_START("OMPT taskwait");
}

extern "C" void Tau_ompt_wait_taskwait_end(ompt_data_t *data, ompt_parallel_id_t parallel_id)
{
  #if DEBUG
    PRINTF("##b wait taskwait end: id %Ld, data %Ld\n", parallel_id, data->value);
  #endif
  TAU_STOP("OMPT taskwait");
}

extern "C" void Tau_ompt_release_lock(ompt_wait_id_t waitId)
{
  #if DEBUG_OFF
    PRINTF("##b release lock: wait id %Ld\n", waitId);
  #endif
}

extern "C" void Tau_ompt_release_nest_lock_last(ompt_wait_id_t waitId)
{
  #if DEBUG_OFF
    PRINTF("##b release nest lock last: wait id 0x%Lx\n", waitId);
  #endif
}

extern "C" void Tau_ompt_release_critical(ompt_wait_id_t waitId)
{
  #if DEBUG
    PRINTF("##b release critical: wait id 0x%Lx\n", waitId);
  #endif
}

extern "C" void Tau_ompt_release_atomic(ompt_wait_id_t waitId)
{
  #if DEBUG
    PRINTF("##b release atomic: wait id 0x%Lx\n", waitId);
  #endif
}

extern "C" void Tau_ompt_release_ordered(ompt_wait_id_t waitId)
{
  #if DEBUG
    PRINTF("##b release ordered: wait id 0x%Lx\n", waitId);
  #endif
}

////////////////////////////////////////////////////////////////////////////////
// implicit

extern "C" void Tau_ompt_implicit_task_create(ompt_data_t *data, ompt_parallel_id_t parallel_id)								   
{
  data->value = 1;
  #if DEBUG
    data->value = tau_ompt.GetNewTaskId();
    PRINTF("##m implicit task create: id %Ld, data %Ld\n", parallel_id, data->value);
  #endif
  TAU_START("OMPT implicit task");
}

extern "C" void Tau_ompt_implicit_task_exit(ompt_data_t *data, ompt_parallel_id_t parallel_id)								   
{
  #if DEBUG
  PRINTF("##m implicit task exit: id %Ld, data %Ld\n", parallel_id, data->value);
  #endif
  TAU_STOP("OMPT implicit task");
}

////////////////////////////////////////////////////////////////////////////////
// init

extern "C" void Tau_ompt_register(ompt_event_t e, ompt_callback_t c)
{
  int rc = ompt_set_callback(e, c);
  if (!rc) printf("failed to register event %d\n", (int) e);
}

extern "C" void Tau_ompt_initialize(void)
{
    PRINTF("Inside ompt_initialize()\n");
  #if DEBUG
    printf("enter my tool initialize \n");
  #endif
  #if TAU_OMPT_CALLBACK
    PRINTF("tool defined and used\n");
    Tau_ompt_register(ompt_event_parallel_create       , (ompt_callback_t) Tau_ompt_parallel_create); 
    //ompt_set_callback(ompt_event_parallel_create       , (ompt_callback_t) Tau_ompt_parallel_create); 
    Tau_ompt_register(ompt_event_parallel_exit         , (ompt_callback_t) Tau_ompt_parallel_exit); 				    		      
    Tau_ompt_register(ompt_event_task_create           , (ompt_callback_t) Tau_ompt_task_create); 
    Tau_ompt_register(ompt_event_task_exit             , (ompt_callback_t) Tau_ompt_task_exit); 
    Tau_ompt_register(ompt_event_thread_create         , (ompt_callback_t) Tau_ompt_thread_create); 
    Tau_ompt_register(ompt_event_thread_exit           , (ompt_callback_t) Tau_ompt_thread_exit); 
    ///Tau_ompt_register(ompt_event_control              , (ompt_callback_t) ); 
    Tau_ompt_register(ompt_event_idle_begin	     , (ompt_callback_t) Tau_ompt_idle_begin);  
    Tau_ompt_register(ompt_event_idle_end	  	     , (ompt_callback_t) Tau_ompt_idle_end);  
    Tau_ompt_register(ompt_event_wait_barrier_begin    , (ompt_callback_t) Tau_ompt_wait_barrier_begin); 
    Tau_ompt_register(ompt_event_wait_barrier_end      , (ompt_callback_t) Tau_ompt_wait_barrier_end); 
    Tau_ompt_register(ompt_event_wait_taskwait_begin   , (ompt_callback_t) Tau_ompt_wait_taskwait_begin); 
    Tau_ompt_register(ompt_event_wait_taskwait_end     , (ompt_callback_t) Tau_ompt_wait_taskwait_end); 
    //Tau_ompt_register(ompt_event_wait_taskgroup_begin  , (ompt_callback_t) ); 
    //Tau_ompt_register(ompt_event_wait_taskgroup_end    , (ompt_callback_t) ); 
    Tau_ompt_register(ompt_event_release_lock          , (ompt_callback_t) Tau_ompt_release_lock); 
    Tau_ompt_register(ompt_event_release_nest_lock_last, (ompt_callback_t) Tau_ompt_release_nest_lock_last); 
    Tau_ompt_register(ompt_event_release_critical      , (ompt_callback_t) Tau_ompt_release_critical); 
    Tau_ompt_register(ompt_event_release_atomic        , (ompt_callback_t) Tau_ompt_release_atomic); 
    Tau_ompt_register(ompt_event_release_ordered       , (ompt_callback_t) Tau_ompt_release_ordered); 
    //Tau_ompt_register(ompt_event_implicit_task_create  , (ompt_callback_t) Tau_ompt_implicit_task_create); 
    //Tau_ompt_register(ompt_event_implicit_task_end     , (ompt_callback_t) Tau_ompt_implicit_task_exit); 
  #else
    printf("tool defined but not used\n");
  #endif
  //return 1;
}

    
////////////////////////////////////////////////////////////////////////////////
// my tool methods
//

extern "C" void TauInitOMPT(void) 
{
#ifdef DEBUG 
  printf("TauInitOMPT\n");
#endif /* DEBUG */
  //ompt_register_tool(ompt_initialize);
}

TauOMPT::TauOMPT()
{
  #if DEBUG 
    printf("my tool register initialization entry\n");
  #endif
  pthread_mutex_init(&mutex, NULL);
  taskId = 1;
  parallelId = 100;
  threadId = 10000;

  PRINTF("before calling ompt_initialize()\n");
  //ompt_register_tool(ompt_initialize);
  PRINTF("after ompt_initialize()\n");
}
TauOMPT::~TauOMPT()
{
  #if DEBUG && 0
    printf("my tool done\n");
  #endif
  pthread_mutex_destroy(&mutex);
}

int TauOMPT::GetNewThreadId()
{
  pthread_mutex_lock(&mutex);
  int id = threadId++;
  pthread_mutex_unlock(&mutex);
  return id;
}

int TauOMPT::GetNewTaskId()
{
  pthread_mutex_lock(&mutex);
  int id = taskId++;
  pthread_mutex_unlock(&mutex);
  return id;
}

int TauOMPT::GetNewParallelId()
{
  pthread_mutex_lock(&mutex);
  int id = parallelId++;
  pthread_mutex_unlock(&mutex);
  return id;
}

