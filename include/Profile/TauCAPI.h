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
#include <Profile/ProfileGroups.h>
/* C API Definitions follow */
#if (defined(PROFILING_ON) || defined(TRACING_ON) )


#define TAU_PROFILE(name, type, group) 
/* OLD CODE. Not thread-safe  */
/*
#define TAU_PROFILE_TIMER(var,name, type, group) static int taufirst##var = 1;\
                                 static void *var; \
                                 if (taufirst##var == 1) { \
                                   var = tau_get_profiler(name, type, group); \
                                   taufirst##var = 0; }
*/

#define TAU_PROFILE_TIMER(var,name, type, group) static void *var=NULL; tau_profile_c_timer(&var, name, type, group);


#define TAU_PROFILE_START(var) 			tau_start_timer(var);
#define TAU_PROFILE_STOP(var) 			tau_stop_timer(var);
#define TAU_PROFILE_STMT(stmt) 			stmt;
#define TAU_PROFILE_EXIT(msg)  			tau_exit(msg);
#define TAU_PROFILE_INIT(argc, argv)		tau_init(argc, argv);
#define TAU_PROFILE_SET_NODE(node) 		tau_set_node(node);
#define TAU_PROFILE_SET_CONTEXT(context)	tau_set_context(context);
#define TAU_PROFILE_CALLSTACK()			tau_callstack();
/*
#define TAU_REGISTER_EVENT(event, name)	static int tauuser##event = 1;\
				        static void *event; \ 
					if (tauuser##event == 1) { \ 
					  event = tau_get_userevent(name); \
					  tauuser##event = 0; }
*/
#define TAU_REGISTER_EVENT(event, name)	static int taufirst##event = 1;\
                                 static void *event; \
                                 if (taufirst##event == 1) { \
                                   event = tau_get_userevent(name); \
                                   taufirst##event = 0; }
				
#define TAU_EVENT(event, data)			tau_userevent(event, data);
#define TAU_REPORT_STATISTICS()		tau_report_statistics();
#define TAU_REPORT_THREAD_STATISTICS()  tau_report_thread_statistics();
#define TAU_EVENT_DISABLE_MIN(event) 	tau_event_disable_min(event);
#define TAU_EVENT_DISABLE_MAX(event)	tau_event_disable_max(event);
#define TAU_EVENT_DISABLE_MEAN(event)	tau_event_disable_mean(event);
#define TAU_EVENT_DISABLE_STDDEV(event) tau_event_disable_stddev(event);
#define TAU_STORE_ALL_EVENTS
#define TYPE_STRING(profileString, str)
#define PROFILED_BLOCK(name, type)
/* C doesn't support runtime type information */
#define CT(obj)

#define TAU_REGISTER_THREAD()			tau_register_thread();	
#define TAU_REGISTER_FORK(nodeid, op) 		tau_register_fork(nodeid, op);
#define TAU_ENABLE_INSTRUMENTATION()		tau_enable_instrumentation();
#define TAU_DISABLE_INSTRUMENTATION()		tau_disable_instrumentation();
#define TAU_ENABLE_GROUP(group)			tau_enable_group(group);
#define TAU_DISABLE_GROUP(group)		tau_disable_group(group);

extern void * tau_get_profiler(char *fname, char *type, TauGroup_t  group);
extern void tau_start_timer(void *profiler);
extern void tau_stop_timer(void *profiler);
extern void tau_exit(char *msg);
extern void tau_init(int argc, char **argv);
extern void tau_set_node(int node);
extern void tau_set_context(int context);
extern void tau_callstack(void);
extern void tau_register_thread();
extern void tau_register_fork(int nodeid, enum TauFork_t opcode);
extern void * tau_get_userevent(char *name);
extern void tau_userevent(void *event, double data);
extern void tau_report_statistics(void);
extern void tau_report_thread_statistics(void);
extern void tau_event_disable_min(void *event);
extern void tau_event_disable_max(void *event);
extern void tau_event_disable_mean(void *event);
extern void tau_event_disable_stddev(void *event);
extern void tau_trace_sendmsg(int type, int destination, int length);
extern void tau_trace_recvmsg(int type, int source, int length);


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

#define TAU_REGISTER_EVENT(event, name)
#define TAU_EVENT(event, data)
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

#define CT(obj)

#endif /* PROFILING_ON */

#ifdef TRACING_ON
#define TAU_TRACE_SENDMSG(type, destination, length) \
        tau_trace_sendmsg(type, destination, length);
#define TAU_TRACE_RECVMSG(type, source, length) \
        tau_trace_recvmsg(type, source, length);

#else /* TRACING_ON */
#define TAU_TRACE_SENDMSG(type, destination, length)
#define TAU_TRACE_RECVMSG(type, source, length)
#endif /* TRACING_ON */


#endif /* __cplusplus */
#endif /* _TAU_CAPI_H_ */

/***************************************************************************
 * $RCSfile: TauCAPI.h,v $   $Author: sameer $
 * $Revision: 1.9 $   $Date: 2001/01/05 22:30:49 $
 * POOMA_VERSION_ID: $Id: TauCAPI.h,v 1.9 2001/01/05 22:30:49 sameer Exp $
 ***************************************************************************/

