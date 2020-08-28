/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauWrapSyscalls.c				  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**			  for wrapping syscalls like exit                 **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


// Include Files
//////////////////////////////////////////////////////////////////////

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <TAU.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <Profile/TauEnv.h>

extern int Tau_init_check_initialized();

#define dprintf TAU_VERBOSE

#if (defined (TAU_BGP) || defined(TAU_XLC))
#define TAU_DISABLE_SYSCALL_WRAPPER
#endif /* TAU_BGP || TAU_XLC */

int Tau_wrap_syscalls_checkPassThrough() {
	//Do not wrap system calls that occur outside of TAU
	if (Tau_init_check_initialized() == 0) {
		return 1;
	}
	else {
		return 0;
	}
}

/////////////////////////////////////////////////////////////////////////
// Define the exit wrapper
/////////////////////////////////////////////////////////////////////////
#ifndef TAU_DISABLE_SYSCALL_WRAPPER
void exit(int status) {
  dprintf("TAU: Inside %s: %s: status = %d\n", __FILE__, __func__, status);

  TAU_PROFILE_EXIT("EXITING from TAU...");

  dprintf("TAU: calling _internal_exit \n");
  _exit(status);
}
#endif /* TAU_DISABLE_SYSCALL_WRAPPER */

#ifdef TAU_LINUX
/////////////////////////////////////////////////////////////////////////
// Define the exit_group wrapper
/////////////////////////////////////////////////////////////////////////
void exit_group(int status) {
  dprintf("TAU: Inside %s: %s: status = %d\n", __FILE__, __func__, status);

  TAU_PROFILE_EXIT("EXIT_GROUPING from TAU...");

  dprintf("TAU: calling _internal_exit_group \n");
  _exit_group(status);
}
#endif

