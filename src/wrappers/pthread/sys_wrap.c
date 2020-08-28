/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: sys_wrap.c				  **
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
#include <stdio.h>
#include <TAU.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <Profile/TauEnv.h>

extern int Tau_init_check_initialized();

#define dprintf TAU_VERBOSE

#if (defined (TAU_BGP) || defined(TAU_XLC) || defined(__APPLE__))
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
  static void (*_internal_exit) (int status) = NULL;

  int ret;
  dprintf("TAU: Inside %s: %s: status = %d\n", __FILE__, __func__, status);

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
  dprintf("TAU: Inside %s: %s: status = %d\n", __FILE__, __func__, status);

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
  dprintf("TAU: Inside %s: %s: status = %d\n", __FILE__, __func__, status);

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

  pid_ret = _fork();

	if (Tau_wrap_syscalls_checkPassThrough() == 1) {
		return pid_ret;
	}
  dprintf("TAU: calling _fork \n");

	if (pid_ret == 0) {
    TAU_REGISTER_FORK(RtsLayer::getPid(), TAU_EXCLUDE_PARENT_DATA);
    dprintf ("[%d] Registered Fork!\n", RtsLayer::getPid());

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

  TAU_PROFILE_TIMER(t,"sleep inside kill timer","" ,TAU_DEFAULT);
  /* Search for kill */
  if (_kill == NULL) {
    _kill = (int (*) (pid_t pid, int sig)) dlsym(RTLD_NEXT, "kill");
  }
  TAU_VERBOSE("TAU Kill Wrapper");
  if(sig==SIGKILL||sig==SIGTERM){
  ret = _kill(pid, SIGUSR1);
   TAU_PROFILE_START(t);
   sleep(5);
   TAU_PROFILE_STOP(t);
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
