/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.h					  **
**	Description 	: TAU Profiling Package API			  **
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

#ifndef _TAU_API_H_
#define _TAU_API_H_

#if (defined(PROFILING_ON) || defined(TRACING_ON) )

//////////////////////////////////////////////////////////////////////
// TAU PROFILING API MACROS. 
// To ensure that Profiling does not add any runtime overhead when it 
// is turned off, these macros expand to null.
//////////////////////////////////////////////////////////////////////
#define TAU_TYPE_STRING(profileString, str) static string profileString(str);
#define TAU_PROFILE(name, type, group) \
	static TauGroup_t tau_gr = group; \
	static FunctionInfo tauFI(name, type, tau_gr, #group); \
	Profiler tauFP(&tauFI, tau_gr); 
#define TAU_PROFILE_TIMER(var, name, type, group) \
	static TauGroup_t var##tau_gr = group; \
	static FunctionInfo var##fi(name, type, var##tau_gr, #group); \
	Profiler var(&var##fi, var##tau_gr, true); 

// Construct a Profiler obj and a FunctionInfo obj with an extended name
// e.g., FunctionInfo loop1fi(); Profiler loop1(); 
#define TAU_PROFILE_START(var) var.Start();
#define TAU_PROFILE_STOP(var)  var.Stop();
#define TAU_PROFILE_STMT(stmt) stmt;
#define TAU_PROFILE_EXIT(msg)  Profiler::ProfileExit(msg); 
#define TAU_PROFILE_INIT(argc, argv) RtsLayer::ProfileInit(argc, argv);
#define TAU_INIT(argc, argv) RtsLayer::ProfileInit(*argc, *argv);
#define TAU_PROFILE_SET_NODE(node) RtsLayer::setMyNode(node);
#define TAU_PROFILE_SET_CONTEXT(context) RtsLayer::setMyContext(context);
#define TAU_PROFILE_SET_GROUP_NAME(newname) tauFI.SetPrimaryGroupName(newname);
#define TAU_PROFILE_TIMER_SET_GROUP_NAME(t, newname) t##fi.SetPrimaryGroupName(newname);

#define TAU_GLOBAL_TIMER(timer, name, type, group) static FunctionInfo timer##fi(name, type, group, #group);
#define TAU_GLOBAL_TIMER_START(timer) FunctionInfo *timer##fptr = &timer##fi; \
        int tau_tid = RtsLayer::myThread(); \
        Profiler *timer = new Profiler(timer##fptr, timer##fptr != (FunctionInfo *) 0 ? timer##fptr->GetProfileGroup() : TAU_DEFAULT, true, tau_tid); \
        timer->Start(tau_tid);

#define TAU_GLOBAL_TIMER_STOP()  {int tau_threadid = RtsLayer::myThread(); \
                Profiler::CurrentProfiler[tau_threadid]->Stop(tau_threadid);}

/* The above macros are for use with global timers in a multi-threaded application */

#ifdef PROFILE_CALLSTACK
#define TAU_PROFILE_CALLSTACK()    Profiler::CallStackTrace();
#else
#define TAU_PROFILE_CALLSTACK() 
#endif /* PROFILE_CALLSTACK */

#define TAU_DB_DUMP() Profiler::DumpData();
#define TAU_DB_PURGE() Profiler::PurgeData();

// UserEvents
#define TAU_REGISTER_EVENT(event, name)  	TauUserEvent event(name);
#define TAU_EVENT(event, data) 		 	(event).TriggerEvent(data);
#define TAU_EVENT_DISABLE_MIN(event) 		(event).SetDisableMin(true);
#define TAU_EVENT_DISABLE_MAX(event) 		(event).SetDisableMax(true);
#define TAU_EVENT_DISABLE_MEAN(event) 		(event).SetDisableMean(true);
#define TAU_EVENT_DISABLE_STDDEV(event) 	(event).SetDisableStdDev(true);
#define TAU_REPORT_STATISTICS()			TauUserEvent::ReportStatistics();
#define TAU_REPORT_THREAD_STATISTICS()		TauUserEvent::ReportStatistics(true);

#define TAU_REGISTER_THREAD()			RtsLayer::RegisterThread();
#define TAU_REGISTER_FORK(id, op) 		RtsLayer::RegisterFork(id, op);
#define TAU_ENABLE_INSTRUMENTATION() 		RtsLayer::TheEnableInstrumentation() = true;
#define TAU_DISABLE_INSTRUMENTATION() 		RtsLayer::TheEnableInstrumentation() = false;
#define TAU_ENABLE_GROUP(group)			RtsLayer::enableProfileGroup(group)
#define TAU_DISABLE_GROUP(group)		RtsLayer::disableProfileGroup(group)
#define TAU_ENABLE_GROUP_NAME(group)		RtsLayer::enableProfileGroupName(group)
#define TAU_DISABLE_GROUP_NAME(group)		RtsLayer::disableProfileGroupName(group)
#define TAU_ENABLE_ALL_GROUPS()			RtsLayer::enableAllGroups()
#define TAU_DISABLE_ALL_GROUPS()		RtsLayer::disableAllGroups()
#define TAU_GET_PROFILE_GROUP(group)		RtsLayer::getProfileGroup(group)

#ifdef NO_RTTI
/* #define CT(obj) string(#obj) */
#define CT(obj) " "
#else // RTTI is present
#define CT(obj) string(RtsLayer::CheckNotNull(typeid(obj).name())) 
#endif //NO_RTTI

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
#define TAU_PROFILE_SET_GROUP_NAME(newname)
#define TAU_PROFILE_TIMER_SET_GROUP_NAME(t, newname)
#define TAU_PROFILE_CALLSTACK()    
#define TAU_DB_DUMP()
#define TAU_DB_PURGE()

#define TAU_REGISTER_EVENT(event, name)
#define TAU_EVENT(event, data)
#define TAU_EVENT_DISABLE_MIN(event)
#define TAU_EVENT_DISABLE_MAX(event)
#define TAU_EVENT_DISABLE_MEAN(event)
#define TAU_EVENT_DISABLE_STDDEV(event)
#define TAU_REPORT_STATISTICS()
#define TAU_REPORT_THREAD_STATISTICS()
#define TAU_REGISTER_THREAD()
#define TAU_REGISTER_FORK(id, op) 
#define TAU_ENABLE_INSTRUMENTATION() 		
#define TAU_DISABLE_INSTRUMENTATION() 	
#define TAU_ENABLE_GROUP(group)
#define TAU_DISABLE_GROUP(group)
#define TAU_ENABLE_GROUP_NAME(group)
#define TAU_DISABLE_GROUP_NAME(group)
#define TAU_ENABLE_ALL_GROUPS()			
#define TAU_DISABLE_ALL_GROUPS()	

#define CT(obj)

#endif /* PROFILING_ON */

#ifdef TRACING_ON
#define TAU_TRACE_SENDMSG(type, destination, length) \
	RtsLayer::TraceSendMsg(type, destination, length); 
#define TAU_TRACE_RECVMSG(type, source, length) \
	RtsLayer::TraceRecvMsg(type, source, length); 

#else /* TRACING_ON */
#define TAU_TRACE_SENDMSG(type, destination, length) 
#define TAU_TRACE_RECVMSG(type, source, length)
#endif /* TRACING_ON */


#ifdef DEBUG_PROF
#define DEBUGPROFMSG(msg) { cout<< msg; }
#else
#define DEBUGPROFMSG(msg) 
#endif // DEBUG_PROF

#endif /* _TAU_API_H_ */
/***************************************************************************
 * $RCSfile: TauAPI.h,v $   $Author: sameer $
 * $Revision: 1.16 $   $Date: 2002/01/15 21:30:12 $
 * POOMA_VERSION_ID: $Id: TauAPI.h,v 1.16 2002/01/15 21:30:12 sameer Exp $ 
 ***************************************************************************/
