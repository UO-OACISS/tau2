/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.cpp					  **
**	Description 	: TAU Mappings for relating profile data from one **
**			  layer to another				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/
/* TAU Mappings */
#ifndef _TAU_MAPPING_H_
#define _TAU_MAPPING_H_

#if (PROFILING_ON || TRACING_ON)
Profiler *& TheTauMapProf();
// For Mapping, global variables used between layers
#define TAU_MAPPING(stmt)   \
  { static char TauStmtUsed[1] = {'Y'}; \
    static Profiler *TauThisStatement; \
    static FunctionInfo taufimap(#stmt, " ", TAU_USER, "TAU_USER"); \
    if (TauStmtUsed[0] != (char) NULL) \
    { \
	TheTauMapProf() = new Profiler(&taufimap, TAU_USER, true);\
	TauThisStatement = TheTauMapProf(); \
    } \
    else { \
	TauStmtUsed[0] = (char) NULL; \
	TheTauMapProf() = TauThisStatement; \
    } \
  } \
  TheTauMapProf()->Start(); \
  stmt; \
  cout <<#stmt <<endl; \
  TheTauMapProf()->Stop();

#else
#define TAU_MAPPING(stmt) stmt

#endif /* PROFILING_ON or TRACING_ON  */
#endif /* _TAU_MAPPING_H_ */
