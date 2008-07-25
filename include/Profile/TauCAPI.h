
/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauCAPI.h					  **
**	Description 	: TAU Profiling Package API for C		  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**			  -DSGI_TIMERS  for SGI fast nanosecs timer	  **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _TAU_CAPI_H_
#define _TAU_CAPI_H_

#if ((! defined( __cplusplus)) || defined (TAU_USE_C_API))
#ifdef TAU_USE_C_API
extern "C" {
#endif /* TAU_USE_C_API */
/* For C */
#include <stdio.h>
#include <Profile/ProfileGroups.h>
/* C API Definitions follow */
#if (defined(PROFILING_ON) || defined(TRACING_ON) )


/* These can't be used in C, only C++, so they are dummy macros here */
#define TAU_PROFILE(name, type, group) 
#define TAU_DYNAMIC_PROFILE(name, type, group) 
/* C doesn't support runtime type information */
#define CT(obj)


#define TAU_PROFILE_TIMER(var,name, type, group) static void *var=NULL; Tau_profile_c_timer(&var, name, type, group, #group);

#define TAU_PROFILE_DECLARE_TIMER(var) static void *var=NULL;
#define TAU_PROFILE_CREATE_TIMER(var,name,type,group) Tau_profile_c_timer(&var, name, type, group, #group);


#define TAU_PROFILE_START(var) 			Tau_start_timer(var, 0);
#define TAU_PROFILE_STOP(var) 			Tau_stop_timer(var);
#define TAU_PROFILE_STMT(stmt) 			stmt;
#define TAU_PROFILE_EXIT(msg)  			Tau_exit(msg);
#define TAU_PROFILE_INIT(argc, argv)		Tau_init(argc, argv);
#define TAU_INIT(argc, argv)			Tau_init_ref(argc, argv);
#define TAU_PROFILE_SET_NODE(node) 		Tau_set_node(node);
#define TAU_PROFILE_SET_CONTEXT(context)	Tau_set_context(context);
#define TAU_PROFILE_CALLSTACK()			Tau_profile_callstack();
#define TAU_DB_DUMP()                           Tau_dump();
#define TAU_DB_DUMP_PREFIX(prefix)              Tau_dump_prefix(prefix);
#define TAU_DB_DUMP_INCR()                      Tau_dump_incr();
#define TAU_DB_PURGE()                          Tau_purge();
#define TAU_GET_FUNC_NAMES(functionList, num)   Tau_the_function_list(&functionList, &num);
#define TAU_DUMP_FUNC_NAMES()                   Tau_dump_function_names();
#define TAU_GET_COUNTER_NAMES(counterList, num) Tau_the_counter_names(counterList, num);
#define TAU_GET_FUNC_VALS(v1,v2,v3,v4,v5,v6,v7,v8) Tau_get_function_values(v1,v2,&v3,&v4,&v5,&v6,&v7,&v8);
#define TAU_DUMP_FUNC_VALS(functionList, num)   Tau_dump_function_values(functionList, num);
#define TAU_DUMP_FUNC_VALS_incr(functionList, num)  Tau_dump_function_values_incr(functionList, num);


#define TAU_GET_EVENT_NAMES(eventList, num)     Tau_get_event_names(&eventList, &num);
#define TAU_GET_EVENT_VALS(v1,v2,v3,v4,v5,v6)   Tau_get_event_vals(v1,v2,&v3,&v4,&v5,&v6);

#define TAU_REGISTER_EVENT(event, name)	static int taufirst##event = 1;\
                                 static void *event; \
                                 if (taufirst##event == 1) { \
                                   event = Tau_get_userevent(name); \
                                   taufirst##event = 0; }
				
#define TAU_REGISTER_CONTEXT_EVENT(event, name)	static int taufirst##event = 1;\
                                 static void *event = 0; \
                                 if (taufirst##event == 1) { \
                                   Tau_get_context_userevent(&event, name); \
                                   taufirst##event = 0; } 
                                   
#define TAU_EVENT(event, data)			Tau_userevent(event, data);
#define TAU_CONTEXT_EVENT(event, data)		Tau_context_userevent(event, data);
#define TAU_EVENT_SET_NAME(event, name)	Tau_set_event_name(event, name); 	
#define TAU_REPORT_STATISTICS()		Tau_report_statistics();
#define TAU_REPORT_THREAD_STATISTICS()  Tau_report_thread_statistics();
#define TAU_EVENT_DISABLE_MIN(event) 	Tau_event_disable_min(event);
#define TAU_EVENT_DISABLE_MAX(event)	Tau_event_disable_max(event);
#define TAU_EVENT_DISABLE_MEAN(event)	Tau_event_disable_mean(event);
#define TAU_EVENT_DISABLE_STDDEV(event) Tau_event_disable_stddev(event);
#define TAU_STORE_ALL_EVENTS
#define TYPE_STRING(profileString, str)
#define PROFILED_BLOCK(name, type)

#define TAU_REGISTER_THREAD()			Tau_register_thread();	
#define TAU_REGISTER_FORK(nodeid, op) 		Tau_register_fork(nodeid, op);
#define TAU_ENABLE_INSTRUMENTATION()		Tau_enable_instrumentation();
#define TAU_DISABLE_INSTRUMENTATION()		Tau_disable_instrumentation();
#define TAU_ENABLE_GROUP(group)			Tau_enable_group(group);
#define TAU_DISABLE_GROUP(group)		Tau_disable_group(group);
#define TAU_ENABLE_GROUP_NAME(group)            Tau_enable_group_name(group) 
#define TAU_DISABLE_GROUP_NAME(group)           Tau_disable_group_name(group)
#define TAU_ENABLE_ALL_GROUPS()            	Tau_enable_all_groups() 
#define TAU_DISABLE_ALL_GROUPS()            	Tau_disable_all_groups() 
#define TAU_DISABLE_GROUP_NAME(group)           Tau_disable_group_name(group)
#define TAU_GET_PROFILE_GROUP(group)            Tau_get_profile_group(group)
#define TAU_BCAST_DATA(data)  	                Tau_bcast_data(data)
#define TAU_REDUCE_DATA(data)  	                Tau_reduce_data(data)
#define TAU_ALLTOALL_DATA(data)                 Tau_alltoall_data(data) 
#define TAU_SCATTER_DATA(data)                  Tau_scatter_data(data) 
#define TAU_GATHER_DATA(data)  	                Tau_gather_data(data)
#define TAU_ALLREDUCE_DATA(data)  	        Tau_allreduce_data(data)
#define TAU_ALLGATHER_DATA(data)  	        Tau_allgather_data(data)
#define TAU_REDUCESCATTER_DATA(data)  	        Tau_reducescatter_data(data)
#define TAU_SCAN_DATA(data)  		        Tau_scan_data(data)
#define TAU_ENABLE_TRACKING_MEMORY()	        Tau_enable_tracking_memory()
#define TAU_DISABLE_TRACKING_MEMORY()	        Tau_disable_tracking_memory()
#define TAU_TRACK_MEMORY()		        Tau_track_memory()
#define TAU_TRACK_MEMORY_HERE()	        	Tau_track_memory_here()
#define TAU_TRACK_MEMORY_HEADROOM()        	Tau_track_memory_headroom()
#define TAU_TRACK_MEMORY_HEADROOM_HERE()	Tau_track_memory_headroom_here()
#define TAU_ENABLE_TRACKING_MEMORY_HEADROOM()	Tau_enable_tracking_memory_headroom()
#define TAU_DISABLE_TRACKING_MEMORY_HEADROOM()	Tau_disable_tracking_memory_headroom()
#define TAU_ENABLE_TRACKING_MUSE_EVENTS()	Tau_enable_tracking_muse_events()	
#define TAU_DISABLE_TRACKING_MUSE_EVENTS()	Tau_disable_tracking_muse_events()
#define TAU_TRACK_MUSE_EVENTS()			Tau_track_muse_events()		
#define TAU_SET_INTERRUPT_INTERVAL(value)	Tau_set_interrupt_interval(value)
#define TAU_PHASE_CREATE_STATIC(var, name, type, group) static void *var##finfo = NULL; Tau_profile_c_timer(&var##finfo, name, type, group, Tau_phase_enable_once(#group, &var##finfo))
#define TAU_PHASE_CREATE_DYNAMIC(var, name, type, group) void *var##finfo = NULL; Tau_profile_c_timer(&var##finfo, name, type, group, Tau_phase_enable_once(#group, &var##finfo))
#define TAU_PROFILE_TIMER_DYNAMIC(var, name, type, group) void *var = NULL; Tau_profile_c_timer(&var, name, type, group, #group)

/* TAU_PHASE_CREATE_DYNAMIC_AUTO embeds the counter in the name. isPhase = 1 */
#define TAU_PHASE_CREATE_DYNAMIC_AUTO(var, name, type, group)  \
        void *var##finfo = NULL;\
    { static int tau_dy_phase_counter = 1; \
  Tau_profile_dynamic_auto(tau_dy_phase_counter++, &var##finfo, name, type, group, #group, 1); \
    }  

/* TAU_PROFILE_CREATE_DYNAMIC_AUTO embeds the counter in the name. isPhase = 0 implies a timer */
#define TAU_PROFILE_CREATE_DYNAMIC_AUTO(var, name, type, group)  \
        void *var = NULL;\
    { static int tau_dy_timer_counter = 1; \
  Tau_profile_dynamic_auto(tau_dy_timer_counter++, &var, name, type, group, #group, 0); \
    }

#define TAU_PHASE_START(var) Tau_start_timer(var##finfo, 1)
#define TAU_PHASE_STOP(var) Tau_stop_timer(var##finfo)
#define TAU_STATIC_PHASE_START(name) Tau_static_phase_start(name)
#define TAU_STATIC_PHASE_STOP(name)  Tau_static_phase_stop(name)
#define TAU_DYNAMIC_PHASE_START(name) \
{ static void *tau_counter=NULL; \
  Tau_dynamic_start(name, &tau_counter, 1); \
}

#define TAU_DYNAMIC_PHASE_STOP(name) \
{ static void *tau_counter=NULL; \
  Tau_dynamic_stop(name, &tau_counter, 1); \
}

#define TAU_DYNAMIC_TIMER_START(name) \
{ static void *tau_counter=NULL; \
  Tau_dynamic_start(name, &tau_counter, 0); \
}

#define TAU_DYNAMIC_TIMER_STOP(name) \
{ static void *tau_counter=NULL; \
  Tau_dynamic_stop(name, &tau_counter, 0); \
}

#define TAU_GLOBAL_PHASE(timer, name, type, group) void * TauGlobalPhase##timer(void) \
{ static void *ptr = NULL; \
  Tau_profile_c_timer(&ptr, name, type, group, #group); \
  return ptr; \
} 

#define TAU_GLOBAL_PHASE_START(timer) { void *ptr = TauGlobalPhase##timer(); \
	Tau_start_timer(ptr, 1); } 

#define TAU_GLOBAL_PHASE_STOP(timer)  { void *ptr = TauGlobalPhase##timer(); \
	Tau_stop_timer(ptr); }

#define TAU_GLOBAL_PHASE_EXTERNAL(timer)  extern void * TauGlobalPhase##timer(void)

#define TAU_GLOBAL_TIMER(timer, name, type, group) void * TauGlobal##timer(void) \
{ void *ptr = NULL; \
  Tau_profile_c_timer(&ptr, name, type, group, #group); \
  return ptr; \
} 


#define TAU_GLOBAL_TIMER_START(timer) { void *ptr = TauGlobal##timer(); \
	Tau_start_timer(ptr, 0); }

#define TAU_GLOBAL_TIMER_STOP()  Tau_global_stop()

#define TAU_GLOBAL_TIMER_EXTERNAL(timer)  extern void* TauGlobal##timer(void);

#ifdef TAU_PROFILEPARAM
#define TAU_PROFILE_PARAM1L(b,c)  	Tau_profile_param1l(b,c)
#else  /* TAU_PROFILEPARAM */
#define TAU_PROFILE_PARAM1L(b,c)  	
#endif /* TAU_PROFILEPARAM */

#define TAU_PROFILE_SNAPSHOT(name)              Tau_profile_snapshot(name);
#define TAU_PROFILE_SNAPSHOT_1L(name, expr)     Tau_profile_snapshot(name, expr);
#define TAU_METADATA(name, value)               Tau_metadata(name, value);

/* for profiler objects created by name */

#define TAU_PROFILER_CREATE(handle, name, type, group)  handle=Tau_get_profiler(name, type, group, #group);
#define TAU_PROFILER_START(handle) Tau_start_timer(handle, 0);
#define TAU_PROFILER_STOP(handle) Tau_stop_timer(handle);
#define TAU_PROFILER_GET_INCLUSIVE_VALUES(handle, data) Tau_get_inclusive_values(handle, (double *) data, Tau_get_tid());
#define TAU_PROFILER_GET_EXCLUSIVE_VALUES(handle, data) Tau_get_exclusive_values(handle, (double *) data, Tau_get_tid());
#define TAU_PROFILER_GET_CALLS(handle, number) Tau_get_calls(handle, number, Tau_get_tid())
#define TAU_PROFILER_GET_CHILD_CALLS(handle, number) Tau_get_child_calls(handle, number, Tau_get_tid());
#define TAU_PROFILER_GET_COUNTER_INFO(counters, numcounters) Tau_get_counter_info((const char ***)counters, numcounters);




extern void Tau_start(char *name);
extern void Tau_stop(char *name);
extern void Tau_specify_mapping_data1(long data, const char *name);
extern void TAUDECL Tau_bcast_data(int data);
extern void TAUDECL Tau_reduce_data(int data);
extern void TAUDECL Tau_alltoall_data(int data);
extern void TAUDECL Tau_scatter_data(int data);
extern void TAUDECL Tau_gather_data(int data);
extern void TAUDECL Tau_allreduce_data(int data);
extern void TAUDECL Tau_allgather_data(int data);
extern void TAUDECL Tau_reducescatter_data(int data);
extern void TAUDECL Tau_scan_data(int data);

extern void * Tau_get_profiler(char *fname, char *type, TauGroup_t  group, char *gr_name);
extern void TAUDECL Tau_start_timer(void *profiler, int phase);
extern void TAUDECL Tau_stop_timer(void *profiler);
extern void Tau_exit(char *msg);
extern void Tau_init(int argc, char **argv);
extern void Tau_init_ref(int* argc, char ***argv);
extern void TAUDECL Tau_set_node(int node);
extern void Tau_set_context(int context);
extern void Tau_callstack(void);
extern int Tau_dump(void);
extern int Tau_dump_incr(void);
extern void Tau_purge(void);
extern void Tau_theFunctionList(const char ***functionList, int *num);
extern void Tau_dump_function_names();
extern void Tau_the_counter_names(const char **counterList, int num);
extern void Tau_get_function_values(const char **inFuncs, int numOfFuncs,
				    double ***counterExclusiveValues,
				    double ***counterInclusiveValues,
				    int **numOfCalls, int **numOfSubRoutines,
				    const char ***counterNames, int *numOfCounters);
extern void Tau_dump_function_values(const char **functionList, int num);
extern void Tau_dump_function_values_incr(const char **functionList, int num);
extern void Tau_register_thread();
extern void Tau_register_fork(int nodeid, enum TauFork_t opcode);
extern void * Tau_get_userevent(char *name);
extern void Tau_get_context_userevent(void **ptr, char *name);
extern void Tau_userevent(void *event, double data);
extern void Tau_context_userevent(void *event, double data);
extern void Tau_set_event_name(void *event, char * name);
extern void Tau_report_statistics(void);
extern void Tau_report_thread_statistics(void);
extern void Tau_event_disable_min(void *event);
extern void Tau_event_disable_max(void *event);
extern void Tau_event_disable_mean(void *event);
extern void Tau_event_disable_stddev(void *event);
extern void TAUDECL Tau_trace_sendmsg(int type, int destination, int length);
extern void TAUDECL Tau_trace_recvmsg(int type, int source, int length);
extern TauGroup_t Tau_enable_group_name(char *group);
extern TauGroup_t Tau_disable_group_name(char *group);
extern TauGroup_t Tau_get_profile_group(char *group);
extern void TAUDECL Tau_profile_c_timer(void **ptr, char *fname, char *type, TauGroup_t group, char *group_name);
extern void TAUDECL Tau_create_top_level_timer_if_necessary(void);
extern void TAUDECL Tau_stop_top_level_timer_if_necessary(void);
extern void Tau_track_memory(void);
extern void Tau_track_muse_events(void);
extern void Tau_enable_tracking_memory(void);
extern void Tau_disable_tracking_memory(void);
extern void Tau_enable_tracking_muse_events(void);
extern void Tau_disable_tracking_muse_events(void);
extern void Tau_set_interrupt_interval(int value);
extern void Tau_enable_instrumentation(void);
extern void Tau_disable_instrumentation(void);
extern void Tau_global_stop(void);
extern char * Tau_phase_enable_once(const char *group, void **ptr);

extern void Tau_profile_snapshot(char *name);
extern void Tau_profile_snapshot_1l(char *name, int number);
extern void TAUDECL Tau_metadata(char *name, char *value);

extern void Tau_dynamic_start(char *name, void *tau_counter, int isPhase); 
extern void Tau_dynamic_stop(char *name, void *tau_counter, int isPhase); 
extern void Tau_static_phase_start(char *name);
extern void Tau_static_phase_stop(char *name);
extern void Tau_profile_dynamic_auto(int iteration, void **ptr, char *fname, char *type, TauGroup_t group, char *group_name, int isPhase);

extern void Tau_get_calls(void *handle, long* values, int tid);
extern void Tau_get_child_calls(void *handle, long* values, int tid);
extern void Tau_get_inclusive_values(void *handle, double* values, int tid);
extern void Tau_get_exclusive_values(void *handle, double* values, int tid);
extern void Tau_get_counter_info(const char ***counterlist, int *numcounters);
extern int Tau_get_tid(void);

#endif /* PROFILING_ON */

#ifdef TRACING_ON
#define TAU_TRACE_SENDMSG(type, destination, length) \
        Tau_trace_sendmsg(type, destination, length);
#define TAU_TRACE_RECVMSG(type, source, length) \
        Tau_trace_recvmsg(type, source, length);

#else /* TRACING_ON */
#define TAU_TRACE_SENDMSG(type, destination, length) \
        Tau_trace_sendmsg(type, destination, length);
#define TAU_TRACE_RECVMSG(type, source, length) \
        Tau_trace_recvmsg(type, source, length);
#endif /* TRACING_ON */


#ifdef TAU_USE_C_API
}
#endif /* TAU_USE_C_API */

/* for consistency, we provide the long form */
#define TAU_STATIC_TIMER_START TAU_START
#define TAU_STATIC_TIMER_STOP TAU_STOP

#endif /* ! __cplusplus || TAU_C_API */
#endif /* _TAU_CAPI_H_ */

/***************************************************************************
 * $RCSfile: TauCAPI.h,v $   $Author: sameer $
 * $Revision: 1.56 $   $Date: 2008/07/25 21:32:50 $
 * POOMA_VERSION_ID: $Id: TauCAPI.h,v 1.56 2008/07/25 21:32:50 sameer Exp $
 ***************************************************************************/

