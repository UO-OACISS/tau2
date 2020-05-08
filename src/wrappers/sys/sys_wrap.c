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

#define dprintf TAU_VERBOSE

#if (defined (TAU_BGP) || defined(TAU_XLC))
#define TAU_DISABLE_SYSCALL_WRAPPER
#endif /* TAU_BGP || TAU_XLC */

/////////////////////////////////////////////////////////////////////////
// Define the fork wrapper
/////////////////////////////////////////////////////////////////////////
#ifndef TAU_DISABLE_SYSCALL_WRAPPER
pid_t __real_fork(void);
pid_t __wrap_fork(void) {
  pid_t pid_ret;
  pid_ret = __real_fork();
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
int __real_kill(pid_t pid, int sig);
int __wrap_kill(pid_t pid, int sig) {

  int ret;

  TAU_PROFILE_TIMER(t,"sleep inside kill timer","" ,TAU_DEFAULT);
  TAU_VERBOSE("TAU Kill Wrapper");
  if(sig==SIGKILL||sig==SIGTERM){
  ret = __real_kill(pid, SIGUSR1);
   TAU_PROFILE_START(t);
   sleep(5);
   TAU_PROFILE_STOP(t);
  }
  else{
    ret = 0;
  }

  if(ret == 0) {
    dprintf("TAU: calling _kill \n");
    ret = kill(pid, sig);
  }

  return ret;
}
#endif /* TAU_DISABLE_SYSCALL_WRAPPER */

/////////////////////////////////////////////////////////////////////////
// Define the _exit wrapper
/////////////////////////////////////////////////////////////////////////
void __real__exit(int status);
void __wrap__exit(int status) {
  dprintf("TAU: Inside %s: %s: status = %d\n", __FILE__, __func__, status);

  TAU_PROFILE_EXIT("_EXITING from TAU...");

  dprintf("TAU: calling _internal__exit \n");
  __real__exit(status);
}


