/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
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
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

#ifndef _TAU_CAPI_H_
#define _TAU_CAPI_H_

#ifndef __cplusplus
/* For C */
#include <stdio.h>
#include <Profile/ProfileGroups.h>
/* C API Definitions follow */
#if (defined(PROFILING_ON) || defined(TRACING_ON) )


#define TAU_PROFILE(name, type, group) 
/* OLD CODE. Not thread-safe  */
/*
#define TAU_PROFILE_TIMER(var,name, type, group) static int taufirst##var = 1;\
                                 static void *var; \
                                 if (taufirst##var == 1) { \
                                   var = Tau_get_profiler(name, type, group, #group); \
                                   taufirst##var = 0; }
*/

#define TAU_PROFILE_TIMER(var,name, type, group) static void *var=NULL; Tau_profile_c_timer(&var, name, type, group, #group);

#define TAU_PROFILE_DECLARE_TIMER(var) static void *var=NULL;
#define TAU_PROFILE_CREATE_TIMER(var,name,type,group) Tau_profile_c_timer(&var, name, type, group, #group);


#define TAU_PROFILE_START(var) 			Tau_start_timer(var);
#define TAU_PROFILE_STOP(var) 			Tau_stop_timer(var);
#define TAU_PROFILE_STMT(stmt) 			stmt;
#define TAU_PROFILE_EXIT(msg)  			Tau_exit(msg);
#define TAU_PROFILE_INIT(argc, argv)		Tau_init(argc, argv);
#define TAU_INIT(argc, argv)			Tau_init_ref(argc, argv);
#define TAU_PROFILE_SET_NODE(node) 		Tau_set_node(node);
#define TAU_PROFILE_SET_CONTEXT(context)	Tau_set_context(context);
#define TAU_PROFILE_CALLSTACK()			Tau_callstack();
#define TAU_DB_DUMP()                           Tau_dump();
#define TAU_DB_DUMP_INCR()                      Tau_dump_incr();
#define TAU_DB_PURGE()                          Tau_purge();
#define TAU_GET_FUNC_NAMES(functionList, num)   Tau_the_function_list(functionList, num);
#define TAU_DUMP_FUNC_NAMES()                   Tau_dump_function_names();
#define TAU_GET_COUNTER_NAMES(counterList, num) Tau_the_counter_names(counterList, num);
#define TAU_GET_FUNC_VALS(v1,v2,v3,v4,v5,v6,v7,v8) Tau_get_function_values(v1,v2,v3,v4,v5,v6,v7,v8);
#define TAU_DUMP_FUNC_VALS(functionList, num)   Tau_dump_function_values(functionList, num);
#define TAU_DUMP_FUNC_VALS_incr(functionList, num)  Tau_dump_function_values_incr(functionList, num);

/*
#define TAU_REGISTER_EVENT(event, name)	static int tauuser##event = 1;\
				        static void *event; \ 
					if (tauuser##event == 1) { \ 
					  event = Tau_get_userevent(name); \
					  tauuser##event = 0; }
*/
#define TAU_REGISTER_EVENT(event, name)	static int taufirst##event = 1;\
                                 static void *event; \
                                 if (taufirst##event == 1) { \
                                   event = Tau_get_userevent(name); \
                                   taufirst##event = 0; }
				
#define TAU_EVENT(event, data)			Tau_userevent(event, data);
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
/* C doesn't support runtime type information */
#define CT(obj)

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


extern void * Tau_get_profiler(char *fname, char *type, TauGroup_t  group, char *gr_name);
extern void Tau_start_timer(void *profiler);
extern void Tau_stop_timer(void *profiler);
extern void Tau_exit(char *msg);
extern void Tau_init(int argc, char **argv);
extern void Tau_init_ref(int* argc, char ***argv);
extern void Tau_set_node(int node);
extern void Tau_set_context(int context);
extern void Tau_callstack(void);
extern int Tau_dump(void);
extern int Tau_dump_incr(void);
extern void Tau_purge(void);
extern void Tau_theFunctionList(const char **functionList, int num);
extern void Tau_dump_function_names();
extern void Tau_the_counter_names(const char **counterList, int num);
extern void Tau_get_function_values(const char **inFuncs, int numOfFuncs,
				    double **counterExclusiveValues,
				    double **counterInclusiveValues,
				    int *numOfCalls, int *numOfSubRoutines,
				    const char **counterNames, int numOfCounters);
extern void Tau_dump_function_values(const char **functionList, int num);
extern void Tau_dump_function_values_incr(const char **functionList, int num);
extern void Tau_register_thread();
extern void Tau_register_fork(int nodeid, enum TauFork_t opcode);
extern void * Tau_get_userevent(char *name);
extern void Tau_userevent(void *event, double data);
extern void Tau_set_event_name(void *event, char * name);
extern void Tau_report_statistics(void);
extern void Tau_report_thread_statistics(void);
extern void Tau_event_disable_min(void *event);
extern void Tau_event_disable_max(void *event);
extern void Tau_event_disable_mean(void *event);
extern void Tau_event_disable_stddev(void *event);
extern void Tau_trace_sendmsg(int type, int destination, int length);
extern void Tau_trace_recvmsg(int type, int source, int length);
extern TauGroup_t Tau_enable_group_name(char *group);
extern TauGroup_t Tau_disable_group_name(char *group);
extern TauGroup_t Tau_get_profile_group(char *group);
extern void Tau_profile_c_timer(void **ptr, char *fname, char *type, TauGroup_t group, char *group_name);




#else /* PROFILING_ON */
/* In the absence of profiling, define the functions as null */
#define TYPE_STRING(profileString, str)
#define PROFILED_BLOCK(name, type)

#define TAU_TYPE_STRING(profileString, str)
#define TAU_PROFILE(name, type, group)
#define TAU_PROFILE_TIMER(var, name, type, group)
#define TAU_PROFILE_START(var)
#define TAU_PROFILE_STOP(var)
#define TAU_PROFILE_STMT(stmt)
#define TAU_PROFILE_EXIT(msg)
#define TAU_PROFILE_INIT(argc, argv)
#define TAU_PROFILE_SET_NODE(node)
#define TAU_PROFILE_SET_CONTEXT(context)
#define TAU_PROFILE_CALLSTACK()
#define TAU_DB_DUMP()
#define TAU_DB_DUMP_INCR()
#define TAU_DB_PURGE()
#define TAU_GET_FUNC_NAMES(functionList, num)
#define TAU_DUMP_FUNC_NAMES()
#define TAU_GET_COUNTER_NAMES(counterList, num)
#define TAU_GET_FUNC_VALS(v1,v2,v3,v4,v5,v6,v7,v8)
#define TAU_DUMP_FUNC_VALS(functionList, num)
#define TAU_DUMP_FUNC_VALS_incr(functionList, num)

#define TAU_REGISTER_EVENT(event, name)
#define TAU_EVENT(event, data)
#define TAU_EVENT_SET_NAME(event, name)
#define TAU_REPORT_STATISTICS()
#define TAU_REPORT_THREAD_STATISTICS()
#define TAU_EVENT_DISABLE_MIN(event)
#define TAU_EVENT_DISABLE_MAX(event)
#define TAU_EVENT_DISABLE_MEAN(event)
#define TAU_EVENT_DISABLE_STDDEV(event)
#define TAU_STORE_ALL_EVENTS
#define TAU_REGISTER_THREAD()
#define TAU_REGISTER_FORK(nodeid, op) 		
#define TAU_ENABLE_INSTRUMENTATION()	
#define TAU_DISABLE_INSTRUMENTATION()
#define TAU_ENABLE_GROUP(group)	
#define TAU_DISABLE_GROUP(group)
#define TAU_ENABLE_GROUP_NAME(group)            
#define TAU_DISABLE_GROUP_NAME(group)          
#define TAU_GET_PROFILE_GROUP(group) 

#define CT(obj)

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


#endif /* __cplusplus */
#endif /* _TAU_CAPI_H_ */

/***************************************************************************
 * $RCSfile: TauCAPI.h,v $   $Author: sameer $
 * $Revision: 1.21 $   $Date: 2002/11/08 02:25:06 $
 * POOMA_VERSION_ID: $Id: TauCAPI.h,v 1.21 2002/11/08 02:25:06 sameer Exp $
 ***************************************************************************/

