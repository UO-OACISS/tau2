/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 2009  						   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/****************************************************************************
 **	File 		: TauBacktrace.h					   **
 **	Description 	: TAU Profiling Package API for C++		   **
 **	Contact		: tau-team@cs.uoregon.edu               	   **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
 ****************************************************************************/

#ifndef _TAU_BACKTRACE_H_
#define _TAU_BACKTRACE_H_

#ifdef __cplusplus
extern "C" {
#endif

int Tau_backtrace_record_backtrace(int trim);
void Tau_print_simple_backtrace(int tid);
void Tau_backtrace_exit_with_backtrace(int trim, char const * fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* _TAU_BACKTRACE_H_ */
