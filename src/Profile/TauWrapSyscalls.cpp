/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauWrapSyscalls.cpp				  **
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

#include <dlfcn.h>

#include <TAU.h>
#include <stdlib.h>


#define dprintf TAU_VERBOSE 

/////////////////////////////////////////////////////////////////////////
// Define the exit wrapper
/////////////////////////////////////////////////////////////////////////
extern "C" void exit(int status) {

  static void (*_internal_exit) (int status) = NULL;

  int ret;
  dprintf("TAU: Inside tau_wrap.c: exit(): status = %d\n", status);

  TAU_PROFILE_EXIT("EXITING from TAU...");

  /* Search for exit */  
  if (_internal_exit == NULL) {
    _internal_exit = (void (*) (int status)) dlsym(RTLD_NEXT, "exit");
  }

  dprintf("TAU: calling _internal_exit \n");
  _internal_exit(status);
}

/////////////////////////////////////////////////////////////////////////
// Define the fork wrapper
/////////////////////////////////////////////////////////////////////////
extern "C" pid_t fork(void) {
  static pid_t (*_fork) (void) = NULL;

  pid_t pid_ret;
  

  if (_fork == NULL) {
    _fork = (pid_t (*) (void)) dlsym(RTLD_NEXT, "fork");
  }

  dprintf("TAU: calling _fork \n");
  pid_ret = _fork();

  if (pid_ret == 0) {
    TAU_REGISTER_FORK(getpid(), TAU_EXCLUDE_PARENT_DATA);
    dprintf ("[%d] Registered Fork!\n", getpid());
  }
  return pid_ret;

}

/***************************************************************************
 * $RCSfile: TauWrapSyscalls.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2010/06/06 03:25:05 $
 * TAU_VERSION_ID: $Id: TauWrapSyscalls.cpp,v 1.1 2010/06/06 03:25:05 sameer Exp $
 ***************************************************************************/
