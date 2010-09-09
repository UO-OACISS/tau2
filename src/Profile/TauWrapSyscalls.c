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
//const char *TauEnv_get_tracedir(void); 

//const char *TauEnv_get_profiledir(void);


pid_t fork(void) {
  static pid_t (*_fork) (void) = NULL;

  char newdirname[1024];

  pid_t pid_ret;
  

  if (_fork == NULL) {
    _fork = (pid_t (*) (void)) dlsym(RTLD_NEXT, "fork");
  }

  dprintf("TAU: calling _fork \n");
  pid_ret = _fork();

  if (pid_ret == 0) {
    TAU_REGISTER_FORK(getpid(), TAU_EXCLUDE_PARENT_DATA);

   int catch_fork = TauEnv_get_child_forkdirs();
   if(catch_fork!=0)
   {

     int flag=0;
     if(TauEnv_get_profiledir() != (char *)NULL){
        sprintf(newdirname, "%s/tau_child_data_%d",TauEnv_get_profiledir(),getpid());
        flag=1;
     } 
     TAU_VERBOSE("Tau Fork Wrapper");
     if(TauEnv_get_tracedir() != (char *)NULL){
        sprintf(newdirname, "%s/tau_child_data_%d",TauEnv_get_tracedir(),getpid());
        flag=1; 
     }


     if(flag==0)
         sprintf(newdirname, "./tau_child_data_%d",getpid());
     mkdir(newdirname, S_IRWXU | S_IRGRP | S_IXGRP);

    setenv("PROFILEDIR",newdirname,1);
    setenv("TRACEDIR",newdirname,1);
   }
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
  TAU_PROFILE_TIMER(t,"sleep inside kill timer","" ,TAU_DEFAULT);
  int ret;

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


#ifdef TAU_PTHREAD_PRELOAD

static int (*_pthread_create) (pthread_t* thread, const pthread_attr_t* attr, 
			       void *(*start_routine)(void*), void* arg) = NULL;
static void (*_pthread_exit) (void *value_ptr) = NULL;
extern void *tau_pthread_function (void *arg);
typedef struct tau_pthread_pack {
  void *(*start_routine) (void *);
  void *arg;
  int id;
} tau_pthread_pack;


#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
static int (*_pthread_barrier_wait) (pthread_barrier_t *barrier) = NULL;
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */

extern int pthread_create (pthread_t* thread, const pthread_attr_t* attr, 
		    void *(*start_routine)(void*), void* arg) {
  if (_pthread_create == NULL) {
    _pthread_create = (int (*) (pthread_t* thread, const pthread_attr_t* attr, void *(*start_routine)(void*), void* arg)) dlsym(RTLD_NEXT, "pthread_create");
  }

  tau_pthread_pack *pack = (tau_pthread_pack*) malloc (sizeof(tau_pthread_pack));
  pack->start_routine = start_routine;
  pack->arg = arg;
  pack->id = -1;
  return _pthread_create(thread, (pthread_attr_t*) attr, tau_pthread_function, (void*)pack);
}

extern void pthread_exit (void *value_ptr) {

  if (_pthread_exit == NULL) {
    _pthread_exit = (void (*) (void *value_ptr)) dlsym(RTLD_NEXT, "pthread_exit");
  }
  TAU_PROFILE_EXIT("pthread_exit");
  _pthread_exit(value_ptr);
}

#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
extern "C" int pthread_barrier_wait(pthread_barrier_t *barrier) {
  int retval;
  if (_pthread_barrier_wait == NULL) {
    _pthread_barrier_wait = (int (*) (pthread_barrier_t *barrier)) dlsym(RTLD_NEXT, "pthread_barrier_wait");
  }
  TAU_PROFILE_TIMER(timer, "pthread_barrier_wait", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  retval = _pthread_barrier_wait (barrier);
  TAU_PROFILE_STOP(timer);
  return retval;
}
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */
#endif /* TAU_PTHREAD_PRELOAD */


/***************************************************************************
 * $RCSfile: TauWrapSyscalls.cpp,v $   $Author: sameer $
 * $Revision: 1.6 $   $Date: 2010/06/10 12:46:53 $
 * TAU_VERSION_ID: $Id: TauWrapSyscalls.cpp,v 1.6 2010/06/10 12:46:53 sameer Exp $
 ***************************************************************************/
