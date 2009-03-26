/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009					   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.h					  **
**	Description 	: TAU Profiling Package API			  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _TAU_API_H_
#define _TAU_API_H_

#ifdef TAU_ENABLED

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */





/* for consistency, we provide the long form */
#define TAU_STATIC_TIMER_START TAU_START
#define TAU_STATIC_TIMER_STOP TAU_STOP




#define TAU_PROFILE_TIMER(var,name, type, group) static void *var=NULL; Tau_profile_c_timer(&var, name, type, group, #group);
#define TAU_PROFILE_TIMER_DYNAMIC(var,name, type, group) void *var=NULL; Tau_profile_c_timer(&var, name, type, group, #group);

#define TAU_PROFILE_DECLARE_TIMER(var) static void *var=NULL;
#define TAU_PROFILE_CREATE_TIMER(var,name,type,group) Tau_profile_c_timer(&var, name, type, group, #group);


#define TAU_PROFILE_START(var) Tau_start_timer(var, 0, Tau_get_tid());
#define TAU_PROFILE_STOP(var) Tau_stop_timer(var, Tau_get_tid());
#define TAU_PROFILE_STMT(stmt) stmt;


/* TAU_PROFILE_CREATE_DYNAMIC_AUTO embeds the counter in the name. isPhase = 0 implies a timer */
#define TAU_PROFILE_CREATE_DYNAMIC_AUTO(var, name, type, group)  \
        void *var = NULL;\
        { \
          static int tau_dy_timer_counter = 1; \
          Tau_profile_dynamic_auto(tau_dy_timer_counter++, &var, name, type, group, #group, 0); \
        }

#define TAU_PHASE_CREATE_DYNAMIC_AUTO(var, name, type, group)  \
        void *var##finfo = NULL;\
        { \
          static int tau_dy_timer_counter = 1; \
          Tau_profile_dynamic_auto(tau_dy_timer_counter++, &var##finfo, name, type, group, #group, 0); \
        }



#define TAU_DYNAMIC_TIMER_START(name) Tau_dynamic_start(name, 0);
#define TAU_DYNAMIC_TIMER_STOP(name) Tau_dynamic_stop(name, 0);
#define TAU_DYNAMIC_PHASE_START(name) Tau_dynamic_start(name, 1);
#define TAU_DYNAMIC_PHASE_STOP(name) Tau_dynamic_stop(name, 1);


#define TAU_STATIC_PHASE_START(name) Tau_static_phase_start(name)
#define TAU_STATIC_PHASE_STOP(name)  Tau_static_phase_stop(name)



#define TAU_PHASE_CREATE_STATIC(var, name, type, group) static void *var##finfo = NULL; Tau_profile_c_timer(&var##finfo, name, type, group, Tau_phase_enable_once(#group, &var##finfo))


#define TAU_PHASE_CREATE_DYNAMIC(var, name, type, group) void *var##finfo = NULL; Tau_profile_c_timer(&var##finfo, name, type, group, Tau_phase_enable_once(#group, &var##finfo))


#define TAU_PHASE_START(var) Tau_start_timer(var##finfo, 1)
#define TAU_PHASE_STOP(var) Tau_stop_timer(var##finfo, Tau_get_tid())



#define TAU_ENABLE_GROUP(group)			Tau_enable_group(group);
#define TAU_DISABLE_GROUP(group)		Tau_disable_group(group);
#define TAU_ENABLE_GROUP_NAME(group)            Tau_enable_group_name(group) 
#define TAU_DISABLE_GROUP_NAME(group)           Tau_disable_group_name(group)
#define TAU_ENABLE_ALL_GROUPS()            	Tau_enable_all_groups() 
#define TAU_DISABLE_ALL_GROUPS()            	Tau_disable_all_groups() 
#define TAU_DISABLE_GROUP_NAME(group)           Tau_disable_group_name(group)
#define TAU_GET_PROFILE_GROUP(group)            Tau_get_profile_group(group)







#define TAU_PROFILE_INIT(argc, argv)		Tau_init(argc, argv);
#define TAU_INIT(argc, argv)			Tau_init_ref(argc, argv);
#define TAU_PROFILE_STMT(stmt) stmt;
#define TAU_PROFILE_EXIT(msg)  			Tau_exit(msg);

#define TAU_PROFILE_SET_NODE(node) 		Tau_set_node(node);
#define TAU_PROFILE_SET_CONTEXT(context)	Tau_set_context(context);
#define TAU_PROFILE_SET_THREAD(thread)          Tau_set_thread(thread);


#define TAU_REGISTER_THREAD()			Tau_register_thread();	
#define TAU_REGISTER_FORK(nodeid, op) 		Tau_register_fork(nodeid, op);
#define TAU_ENABLE_INSTRUMENTATION()		Tau_enable_instrumentation();
#define TAU_DISABLE_INSTRUMENTATION()		Tau_disable_instrumentation();

/* DB Access */
#define TAU_DB_DUMP()                           Tau_dump();
#define TAU_DB_DUMP_PREFIX(prefix)              Tau_dump_prefix(prefix);
#define TAU_DB_DUMP_INCR()                      Tau_dump_incr();
#define TAU_DB_PURGE()                          Tau_purge();
#define TAU_GET_FUNC_NAMES(functionList, num)   Tau_the_function_list(&functionList, &num);
#define TAU_DUMP_FUNC_NAMES()                   Tau_dump_function_names();
#define TAU_GET_COUNTER_NAMES(counterList, num) Tau_get_counter_names(&counterList, &num);
#define TAU_GET_FUNC_VALS(v1,v2,v3,v4,v5,v6,v7,v8) Tau_get_function_values(v1,v2,&v3,&v4,&v5,&v6,&v7,&v8);
#define TAU_DUMP_FUNC_VALS(functionList, num)   Tau_dump_function_values(functionList, num);
#define TAU_DUMP_FUNC_VALS_INCR(functionList, num)  Tau_dump_function_values_incr(functionList, num);
#define TAU_GET_EVENT_NAMES(eventList, num)     Tau_get_event_names(&eventList, &num);
#define TAU_GET_EVENT_VALS(v1,v2,v3,v4,v5,v6,v7)   Tau_get_event_vals(v1,v2,&v3,&v4,&v5,&v6,&v7);





/* Atomic Events */
#define TAU_REGISTER_EVENT(event, name)	static void *event = 0; \
                                 if (event == 0) event = Tau_get_userevent(name);
				
#define TAU_REGISTER_CONTEXT_EVENT(event, name)	static void *event = 0; \
                                 if (event == 0) Tau_get_context_userevent(&event, name);  
                                   #define TAU_EVENT(event, data)			Tau_userevent(event, data);
#define TAU_CONTEXT_EVENT(event, data)		Tau_context_userevent(event, data);
#define TAU_EVENT_SET_NAME(event, name)	Tau_set_event_name(event, name); 	
#define TAU_REPORT_STATISTICS()		Tau_report_statistics();
#define TAU_REPORT_THREAD_STATISTICS()  Tau_report_thread_statistics();
#define TAU_EVENT_DISABLE_MIN(event) 	Tau_event_disable_min(event);
#define TAU_EVENT_DISABLE_MAX(event)	Tau_event_disable_max(event);
#define TAU_EVENT_DISABLE_MEAN(event)	Tau_event_disable_mean(event);
#define TAU_EVENT_DISABLE_STDDEV(event) Tau_event_disable_stddev(event);



#define TAU_DISABLE_CONTEXT_EVENT(event) Tau_disable_context_event(event);
#define TAU_ENABLE_CONTEXT_EVENT(event) Tau_enable_context_event(event);



#define TAU_ENABLE_TRACKING_MEMORY()	        Tau_enable_tracking_memory()
#define TAU_DISABLE_TRACKING_MEMORY()	        Tau_disable_tracking_memory()
#define TAU_TRACK_MEMORY()		        Tau_track_memory()
#define TAU_TRACK_MEMORY_HERE()	        	Tau_track_memory_here()
#define TAU_TRACK_MEMORY_HEADROOM()        	Tau_track_memory_headroom()
#define TAU_TRACK_MEMORY_HEADROOM_HERE()	Tau_track_memory_headroom_here()
#define TAU_ENABLE_TRACKING_MEMORY_HEADROOM()	Tau_enable_tracking_memory_headroom()
#define TAU_DISABLE_TRACKING_MEMORY_HEADROOM()	Tau_disable_tracking_memory_headroom()


#define TAU_SET_INTERRUPT_INTERVAL(value)	Tau_set_interrupt_interval(value)





#define TAU_GLOBAL_TIMER(timer, name, type, group) void *TauGlobal##timer(void) \
{ void *ptr = NULL; \
  Tau_profile_c_timer(&ptr, name, type, group, #group); \
  return ptr; \
} 
#define TAU_GLOBAL_TIMER_START(timer) { void *ptr = TauGlobal##timer(); \
	Tau_start_timer(ptr, 0); }
#define TAU_GLOBAL_TIMER_STOP()  Tau_global_stop()
#define TAU_GLOBAL_TIMER_EXTERNAL(timer)  extern void* TauGlobal##timer(void);


#define TAU_GLOBAL_PHASE(timer, name, type, group) void * TauGlobalPhase##timer(void) \
{ static void *ptr = NULL; \
  Tau_profile_c_timer(&ptr, name, type, group, #group); \
  return ptr; \
} 

#define TAU_GLOBAL_PHASE_START(timer) { void *ptr = TauGlobalPhase##timer(); \
	Tau_start_timer(ptr, 1); } 

#define TAU_GLOBAL_PHASE_STOP(timer)  { void *ptr = TauGlobalPhase##timer(); \
	Tau_stop_timer(ptr, Tau_get_tid()); }

#define TAU_GLOBAL_PHASE_EXTERNAL(timer)  extern void * TauGlobalPhase##timer(void)


#define TAU_PROFILE_SNAPSHOT(name)              Tau_profile_snapshot(name);
#define TAU_PROFILE_SNAPSHOT_1L(name, expr)     Tau_profile_snapshot_1l(name, expr);
#define TAU_METADATA(name, value)               Tau_metadata(name, value);
#define TAU_CONTEXT_METADATA(name, value)       Tau_context_metadata(name, value);
#define TAU_PHASE_METADATA(name, value)         Tau_phase_metadata(name, value);







#ifdef TAU_PROFILEPARAM
#define TAU_PROFILE_PARAM1L(b,c)  	Tau_profile_param1l(b,c)
#else  /* TAU_PROFILEPARAM */
#define TAU_PROFILE_PARAM1L(b,c)  	
#endif /* TAU_PROFILEPARAM */





#define TAU_TRACE_SENDMSG(type, destination, length) \
        Tau_trace_sendmsg(type, destination, length);
#define TAU_TRACE_RECVMSG(type, source, length) \
        Tau_trace_recvmsg(type, source, length);





#define TAU_PROFILER_CREATE(handle, name, type, group)  handle=Tau_get_profiler(name, type, group, #group);
#define TAU_PROFILER_START(handle) Tau_start_timer(handle, 0, Tau_get_tid());
#define TAU_PROFILER_STOP(handle) Tau_stop_timer(handle, Tau_get_tid());
#define TAU_PROFILER_GET_INCLUSIVE_VALUES(handle, data) Tau_get_inclusive_values(handle, (double *) data, Tau_get_tid());
#define TAU_PROFILER_GET_EXCLUSIVE_VALUES(handle, data) Tau_get_exclusive_values(handle, (double *) data, Tau_get_tid());
#define TAU_PROFILER_GET_CALLS(handle, number) Tau_get_calls(handle, number, Tau_get_tid())
#define TAU_PROFILER_GET_CHILD_CALLS(handle, number) Tau_get_child_calls(handle, number, Tau_get_tid());
#define TAU_PROFILER_GET_COUNTER_INFO(counters, numcounters) Tau_get_counter_info((const char ***)counters, numcounters);


#define TAU_CREATE_TASK(taskid) taskid = Tau_create_task()
#define TAU_PROFILER_START_TASK(handle, taskid) Tau_start_timer(handle, 0, taskid);
#define TAU_PROFILER_STOP_TASK(handle, taskid) Tau_stop_timer(handle, taskid);
#define TAU_PROFILER_GET_INCLUSIVE_VALUES_TASK(handle, data, taskid) Tau_get_inclusive_values(handle, (double *) data, taskid)
#define TAU_PROFILER_GET_EXCLUSIVE_VALUES_TASK(handle, data, taskid) Tau_get_exclusive_values(handle, (double *) data, taskid)
#define TAU_PROFILER_GET_CALLS_TASK(handle, number, taskid) Tau_get_calls(handle, number, taskid)
#define TAU_PROFILER_GET_CHILD_CALLS_TASK(handle, number, taskid) Tau_get_child_calls(handle, number, taskid)
#define TAU_PROFILER_GET_COUNTER_INFO_TASK(counters, numcounters, taskid) Tau_get_counter_info((const char ***)counters, numcounters);


#define TAU_BCAST_DATA(data)  	                Tau_bcast_data(data)
#define TAU_REDUCE_DATA(data)  	                Tau_reduce_data(data)
#define TAU_ALLTOALL_DATA(data)                 Tau_alltoall_data(data) 
#define TAU_SCATTER_DATA(data)                  Tau_scatter_data(data) 
#define TAU_GATHER_DATA(data)  	                Tau_gather_data(data)
#define TAU_ALLREDUCE_DATA(data)  	        Tau_allreduce_data(data)
#define TAU_ALLGATHER_DATA(data)  	        Tau_allgather_data(data)
#define TAU_REDUCESCATTER_DATA(data)  	        Tau_reducescatter_data(data)
#define TAU_SCAN_DATA(data)  		        Tau_scan_data(data)



/* dead macros */
#define TAU_PROFILE_CALLSTACK()
#define PROFILED_BLOCK(name, type)
#define TAU_ENABLE_TRACKING_MUSE_EVENTS()
#define TAU_DISABLE_TRACKING_MUSE_EVENTS()
#define TAU_TRACK_MUSE_EVENTS()




/****************************
* Forward declarations
*****************************/

TauGroup_t Tau_enable_all_groups(void);
TauGroup_t Tau_disable_all_groups(void);
void Tau_disable_group(TauGroup_t group);
void Tau_enable_group(TauGroup_t group);

void Tau_start(const char *name);
void Tau_stop(const char *name);
int Tau_stop_current_timer();
char * Tau_phase_enable(const char *group);

void Tau_profile_snapshot(char *name);
void Tau_profile_snapshot_1l(char *name, int number);
void Tau_dynamic_start(char *name, int isPhase); 
void Tau_dynamic_stop(char *name, int isPhase); 
void Tau_static_phase_start(char *name);
void Tau_static_phase_stop(char *name);
void* Tau_get_profiler(const char *name, const char *type, TauGroup_t group, const char *gr_name);
void Tau_get_calls(void *handle, long* values, int tid);
void Tau_get_child_calls(void *handle, long* values, int tid);
void Tau_get_inclusive_values(void *handle, double* values, int tid);
void Tau_get_exclusive_values(void *handle, double* values, int tid);
void Tau_get_counter_info(const char ***counterlist, int *numcounters);
int  Tau_get_tid(void);
int  Tau_create_task(void);
void Tau_destructor_trigger();

void Tau_profile_set_name(void *ptr, const char *name);
void Tau_profile_set_type(void *ptr, const char *type);
void Tau_profile_set_group(void *ptr, TauGroup_t group);
void Tau_profile_set_group_name(void *ptr, const char *groupname);

const char *Tau_profile_get_group_name(void *ptr);
const char *Tau_profile_get_name(void *ptr);
const char *Tau_profile_get_type(void *ptr);
TauGroup_t Tau_profile_get_group(void *ptr);



void Tau_disable_context_event(void *event);
void Tau_enable_context_event(void *event);

void Tau_the_function_list(const char ***functionList, int *num);
int Tau_dump_prefix(const char *prefix);

void Tau_get_event_names(const char ***eventList, int *num);
void Tau_get_event_vals(const char **inUserEvents, int numUserEvents,
				  int **numEvents, double **max, double **min,
				   double **mean, double **sumSqr);


void Tau_profile_dynamic_auto(int iteration, void **ptr, char *fname, char *type, TauGroup_t group, char *group_name, int isPhase);
void Tau_exit(const char *msg);

void Tau_start(const char *name);
void Tau_stop(const char *name);
void Tau_specify_mapping_data1(long data, const char *name);

void *Tau_get_profiler(const char *name, const char *type, TauGroup_t group, const char *gr_name);


void TAUDECL Tau_profile_c_timer(void **ptr, const char *fname, const char *type, TauGroup_t group, const char *group_name);

void TAUDECL Tau_bcast_data(int data);
void TAUDECL Tau_reduce_data(int data);
void TAUDECL Tau_alltoall_data(int data);
void TAUDECL Tau_scatter_data(int data);
void TAUDECL Tau_gather_data(int data);
void TAUDECL Tau_allreduce_data(int data);
void TAUDECL Tau_allgather_data(int data);
void TAUDECL Tau_reducescatter_data(int data);
void TAUDECL Tau_scan_data(int data);
void TAUDECL Tau_set_node(int node);
void TAUDECL Tau_start_timer(void *profiler, int phase, int tid);
int TAUDECL Tau_stop_timer(void *profiler, int tid); 
void TAUDECL Tau_trace_sendmsg(int type, int destination, int length);
void TAUDECL Tau_trace_recvmsg(int type, int source, int length);
void TAUDECL Tau_create_top_level_timer_if_necessary(void);
void TAUDECL Tau_stop_top_level_timer_if_necessary(void);
void TAUDECL Tau_metadata(char *name, char *value);
void TAUDECL Tau_phase_metadata(char *name, char *value);
void TAUDECL Tau_context_metadata(char *name, char *value);

void Tau_init(int argc, char **argv);
void Tau_init_ref(int* argc, char ***argv);
void Tau_set_context(int context);
void Tau_set_thread(int thread);
void Tau_callstack(void);
int Tau_dump(void);
int Tau_dump_incr(void);
void Tau_purge(void);
void Tau_theFunctionList(const char ***functionList, int *num);
void Tau_dump_function_names();
void Tau_get_counter_names(const char ***counterList, int *num);
void Tau_get_function_values(const char **inFuncs, int numOfFuncs,
				    double ***counterExclusiveValues,
				    double ***counterInclusiveValues,
				    int **numOfCalls, int **numOfSubRoutines,
				    const char ***counterNames, int *numOfCounters);
void Tau_dump_function_values(const char **functionList, int num);
void Tau_dump_function_values_incr(const char **functionList, int num);
void Tau_register_thread();
void Tau_register_fork(int nodeid, enum TauFork_t opcode);
void * Tau_get_userevent(char *name);
void Tau_get_context_userevent(void **ptr, char *name);
void Tau_userevent(void *event, double data);
void Tau_context_userevent(void *event, double data);
void Tau_set_event_name(void *event, char * name);
void Tau_report_statistics(void);
void Tau_report_thread_statistics(void);
void Tau_event_disable_min(void *event);
void Tau_event_disable_max(void *event);
void Tau_event_disable_mean(void *event);
void Tau_event_disable_stddev(void *event);
TauGroup_t Tau_enable_group_name(char *group);
TauGroup_t Tau_disable_group_name(char *group);
TauGroup_t Tau_get_profile_group(char *group);
void Tau_track_memory(void);
void Tau_enable_tracking_memory(void);
void Tau_disable_tracking_memory(void);
void Tau_set_interrupt_interval(int value);
void Tau_enable_instrumentation(void);
void Tau_disable_instrumentation(void);
void Tau_global_stop(void);
char * Tau_phase_enable_once(const char *group, void **ptr);

void Tau_profile_snapshot(char *name);
void Tau_profile_snapshot_1l(char *name, int number);

void Tau_dynamic_start(char *name, int isPhase); 
void Tau_dynamic_stop(char *name, int isPhase); 
void Tau_static_phase_start(char *name);
void Tau_static_phase_stop(char *name);
void Tau_profile_dynamic_auto(int iteration, void **ptr, char *fname, char *type, TauGroup_t group, char *group_name, int isPhase);

void Tau_get_calls(void *handle, long* values, int tid);
void Tau_get_child_calls(void *handle, long* values, int tid);
void Tau_get_inclusive_values(void *handle, double* values, int tid);
void Tau_get_exclusive_values(void *handle, double* values, int tid);
void Tau_get_counter_info(const char ***counterlist, int *numcounters);
int Tau_get_tid(void);


void Tau_enable_tracking_memory_headroom();
void Tau_disable_tracking_memory_headroom();
void Tau_track_memory_here(void);
void Tau_track_memory_headroom(void);
void Tau_track_memory_headroom_here(void);
void Tau_profile_param1l(long data, const char *dataname);


#ifdef __cplusplus
/* Include the C++ API header */
#include <Profile/TauCppAPI.h>
#else /* __cplusplus */
/* These can't be used in C, only C++, so they are dummy macros here */
#define TAU_PROFILE(name, type, group) 
#define TAU_DYNAMIC_PROFILE(name, type, group) 
#define TYPE_STRING(profileString, str)
#define CT(obj)
#endif /* __cplusplus */ 


#ifdef __cplusplus
} /* for extern "C" */
#endif /* __cplusplus */






#define TAU_PROFILE_SET_GROUP_NAME(newname) Tau_profile_set_group_name(tauFI,newname);
#define TAU_PROFILE_TIMER_SET_NAME(t, newname)	Tau_profile_set_name(t,newname);
#define TAU_PROFILE_TIMER_SET_TYPE(t, newname)  Tau_profile_set_type(t,newname);
#define TAU_PROFILE_TIMER_SET_GROUP(t, id) Tau_profile_set_group(t,id); 
#define TAU_PROFILE_TIMER_SET_GROUP_NAME(t, newname) Tau_profile_set_group_name(t,newname);

#define TAU_PROFILE_TIMER_GET_NAME(timer) Tau_profile_get_name(timer)
#define TAU_PROFILE_TIMER_GET_TYPE(timer) Tau_profile_get_type(timer)
#define TAU_PROFILE_TIMER_GET_GROUP(timer) Tau_profile_get_group(timer)
#define TAU_PROFILE_TIMER_GET_GROUP_NAME(timer) Tau_profile_get_group_name(timer)


/**************************************************************************/









#endif /* TAU_ENABLED */


#endif /* _TAU_API_H_ */
/***************************************************************************
 * $RCSfile: TauAPI.h,v $   $Author: sameer $
 * $Revision: 1.94 $   $Date: 2009/03/26 19:12:39 $
 * POOMA_VERSION_ID: $Id: TauAPI.h,v 1.94 2009/03/26 19:12:39 sameer Exp $ 
 ***************************************************************************/
