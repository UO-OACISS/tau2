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

#include <dlfcn.h>

#include <TAU.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <Profile/TauEnv.h>


#define dprintf TAU_VERBOSE 

#if (defined (TAU_BGP) || defined(TAU_XLC))
#define TAU_DISABLE_SYSCALL_WRAPPER
#endif /* TAU_BGP || TAU_XLC */

/////////////////////////////////////////////////////////////////////////
// Define the exit wrapper
/////////////////////////////////////////////////////////////////////////
#ifndef TAU_DISABLE_SYSCALL_WRAPPER
void exit(int status) {

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
#endif /* TAU_DISABLE_SYSCALL_WRAPPER */

#ifdef TAU_LINUX
/////////////////////////////////////////////////////////////////////////
// Define the exit_group wrapper
/////////////////////////////////////////////////////////////////////////
void exit_group(int status) {

  static void (*_internal_exit_group) (int status) = NULL;

  int ret;
  dprintf("TAU: Inside tau_wrap.c: exit_group(): status = %d\n", status);

  TAU_PROFILE_EXIT("EXIT_GROUPING from TAU...");

  /* Search for exit_group */  
  if (_internal_exit_group == NULL) {
    _internal_exit_group = (void (*) (int status)) dlsym(RTLD_NEXT, "exit_group");
  }

  dprintf("TAU: calling _internal_exit_group \n");
  _internal_exit_group(status);
}
#endif

/////////////////////////////////////////////////////////////////////////
// Define the _exit wrapper
/////////////////////////////////////////////////////////////////////////
void _exit(int status) {

  static void (*_internal__exit) (int status) = NULL;

  int ret;
  dprintf("TAU: Inside tau_wrap.c: _exit(): status = %d\n", status);

  TAU_PROFILE_EXIT("_EXITING from TAU...");

  /* Search for _exit */  
  if (_internal__exit == NULL) {
    _internal__exit = (void (*) (int status)) dlsym(RTLD_NEXT, "_exit");
  }

  dprintf("TAU: calling _internal__exit \n");
  _internal__exit(status);
}

/////////////////////////////////////////////////////////////////////////
// Define the fork wrapper
/////////////////////////////////////////////////////////////////////////
#ifndef TAU_DISABLE_SYSCALL_WRAPPER
pid_t fork(void) {
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
#endif /* TAU_DISABLE_SYSCALL_WRAPPER */


/////////////////////////////////////////////////////////////////////////
// Define the kill wrapper
/////////////////////////////////////////////////////////////////////////
#ifndef TAU_DISABLE_SYSCALL_WRAPPER
int kill(pid_t pid, int sig) {

  static int (*_kill) (pid_t pid, int sig) = NULL;

  int ret;

  /* Search for kill */  
  if (_kill == NULL) {
    _kill = (int (*) (pid_t pid, int sig)) dlsym(RTLD_NEXT, "kill");
  }

  if(sig==SIGKILL||sig==SIGTERM){
  ret = _kill(pid, SIGUSR1);
  }
  else{
    ret = 0;
  }

  if(ret == 0) {
    dprintf("TAU: calling _kill \n");
    ret = _kill(pid, sig);
  }

  return ret;
}
#endif /* TAU_DISABLE_SYSCALL_WRAPPER */


/***************************************************************************
 * $RCSfile: TauWrapSyscalls.cpp,v $   $Author: sameer $
 * $Revision: 1.6 $   $Date: 2010/06/10 12:46:53 $
 * TAU_VERSION_ID: $Id: TauWrapSyscalls.cpp,v 1.6 2010/06/10 12:46:53 sameer Exp $
 ***************************************************************************/
