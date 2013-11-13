/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: pthread_wrap.c				  **
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

typedef int (*sctk_user_thread_create_call_p) 
	(pthread_t *threadp,
	const pthread_attr_t *attr,
	void *(*start_routine) (void *),
	void *arg);

extern int tau_sctk_user_thread_create_wrapper (sctk_user_thread_create_call_p sctk_user_thread_create_call,
pthread_t *threadp, const pthread_attr_t *attr, void *(*start_routine) (void *),
void *arg);

/********************************/
/* LD_PRELOAD wrapper functions */
/********************************/

#ifdef TAU_PRELOAD_LIB
static int (*_sctk_user_thread_create) (pthread_t* thread, const pthread_attr_t* attr, 
			       void *(*start_routine)(void*), void* arg) = NULL;
static void (*_sctk_thread_exit) (void *value_ptr) = NULL;
static int (*_sctk_thread_join) (pthread_t thread, void ** retval) = NULL;
extern void *tau_pthread_function (void *arg);
typedef struct tau_pthread_pack {
  void *(*start_routine) (void *);
  void *arg;
  int id;
} tau_pthread_pack;


#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
static int (*_sctk_thread_barrier_wait) (pthread_barrier_t *barrier) = NULL;
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */

int sctk_user_thread_create (pthread_t* thread, const pthread_attr_t* attr, 
		    void *(*start_routine)(void*), void* arg) {
  if (_sctk_user_thread_create == NULL) {
    _sctk_user_thread_create = (int (*) (pthread_t* thread, const pthread_attr_t* attr, void *(*start_routine)(void*), void* arg)) dlsym(RTLD_NEXT, "sctk_user_thread_create");
  }
	/*
  tau_pthread_pack *pack = (tau_pthread_pack*) malloc (sizeof(tau_pthread_pack));
  pack->start_routine = start_routine;
  pack->arg = arg;
  pack->id = -1;
	*/
  return tau_sctk_user_thread_create_wrapper(_sctk_user_thread_create, thread, attr, start_routine, arg);
}

int sctk_thread_join (pthread_t thread, void **retval) {
  int ret;
  if (_sctk_thread_join == NULL) {
    _sctk_thread_join = (int (*) (pthread_t, void **)) dlsym(RTLD_NEXT, "sctk_thread_join"); 
  }
   TAU_PROFILE_TIMER(timer, "sctk_thread_join()", "", TAU_DEFAULT);
   TAU_PROFILE_START(timer);
   ret= _sctk_thread_join(thread, retval); 
   TAU_PROFILE_STOP(timer);
   return ret;
}
void sctk_thread_exit (void *value_ptr) {

  if (_sctk_thread_exit == NULL) {
    _sctk_thread_exit = (void (*) (void *value_ptr)) dlsym(RTLD_NEXT, "sctk_thread_exit");
  }

  TAU_PROFILE_EXIT("sctk_thread_exit");
  _sctk_thread_exit(value_ptr);
}

#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
extern "C" int sctk_thread_barrier_wait(pthread_barrier_t *barrier) {
  int retval;
  if (_sctk_thread_barrier_wait == NULL) {
    _sctk_thread_barrier_wait = (int (*) (pthread_barrier_t *barrier)) dlsym(RTLD_NEXT, "sctk_thread_barrier_wait");
  }
  TAU_PROFILE_TIMER(timer, "sctk_thread_barrier_wait", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  retval = _sctk_thread_barrier_wait (barrier);
  TAU_PROFILE_STOP(timer);
  return retval;
}
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */

#else // Wra via the the link line.
/*********************************/
/* LD wrappers                   */
/*********************************/
/////////////////////////////////////////////////////////////////////////
// Define PTHREAD wrappers
/////////////////////////////////////////////////////////////////////////

extern void *tau_pthread_function (void *arg);
typedef struct tau_pthread_pack {
  void *(*start_routine) (void *);
  void *arg;
  int id;
} tau_pthread_pack;


int __real_sctk_user_thread_create (pthread_t* thread, const pthread_attr_t* attr, 
		    void *(*start_routine)(void*), void* arg);
extern int __wrap_sctk_user_thread_create (pthread_t* thread, const pthread_attr_t* attr, 
		    void *(*start_routine)(void*), void* arg) {
	/*
  tau_pthread_pack *pack = (tau_pthread_pack*) malloc (sizeof(tau_pthread_pack));
  pack->start_routine = start_routine;
  pack->arg = arg;
  pack->id = -1;
	*/
  /* return tau_sctk_user_thread_create_wrapper(__real_sctk_user_thread_create, thread, attr, start_routine, arg);
   */
  printf("Inside __wrap_sctk_user_thread_create\n");
  return __real_sctk_user_thread_create(thread, attr, start_routine, arg);
}

int __real_sctk_thread_join (pthread_t thread, void **retval);
extern int __wrap_sctk_thread_join (pthread_t thread, void **retval) {
  int ret;
   TAU_PROFILE_TIMER(timer, "sctk_thread_join()", "", TAU_DEFAULT);
   TAU_PROFILE_START(timer);
   ret= __real_sctk_thread_join(thread, retval); 
   TAU_PROFILE_STOP(timer);
   return ret;
}
void __real_sctk_thread_exit (void *value_ptr);
extern void __wrap_sctk_thread_exit (void *value_ptr) {

  TAU_PROFILE_EXIT("sctk_thread_exit");
  __real_sctk_thread_exit(value_ptr);
}

#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
int __real_sctk_thread_barrier_wait(pthread_barrier_t *barrier);
int __wrap_sctk_thread_barrier_wait(pthread_barrier_t *barrier) {
  int retval;
  TAU_PROFILE_TIMER(timer, "sctk_thread_barrier_wait", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  retval = __real_sctk_thread_barrier_wait (barrier);
  TAU_PROFILE_STOP(timer);
  return retval;
}
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */


#include <sctk_config.h>
#include <sctk_inter_thread_comm.h>
#include <sctk_communicator.h>
#include <sctk_collective_communications.h>
#include <sctk_simple_collective_communications.h>
#include <sctk_messages_opt_collective_communications.h>
#include <sctk_messages_hetero_collective_communications.h>
#include <Profile/Profiler.h>
#include <stdio.h>


/**********************************************************
   sctk_perform_messages
 **********************************************************/

void  __real_sctk_perform_messages(sctk_request_t * a1) ;
void  __wrap_sctk_perform_messages(sctk_request_t * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_perform_messages(sctk_request_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_perform_messages(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_create_header
 **********************************************************/

sctk_thread_ptp_message_t *  __real_sctk_create_header(const int a1, sctk_message_type_t a2) ;
sctk_thread_ptp_message_t *  __wrap_sctk_create_header(const int a1, sctk_message_type_t a2)  {

  sctk_thread_ptp_message_t * retval;
  TAU_PROFILE_TIMER(t,"sctk_thread_ptp_message_t *sctk_create_header(const int, sctk_message_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_create_header(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_wait_message
 **********************************************************/

void  __real_sctk_wait_message(sctk_request_t * a1) ;
void  __wrap_sctk_wait_message(sctk_request_t * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_wait_message(sctk_request_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_wait_message(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_wait_all
 **********************************************************/

void  __real_sctk_wait_all(const int a1, const sctk_communicator_t a2) ;
void  __wrap_sctk_wait_all(const int a1, const sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_wait_all(const int, const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_wait_all(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_get_internal_ptp
 **********************************************************/

struct sctk_internal_ptp_s *  __real_sctk_get_internal_ptp(int a1) ;
struct sctk_internal_ptp_s *  __wrap_sctk_get_internal_ptp(int a1)  {

  struct sctk_internal_ptp_s * retval;
  TAU_PROFILE_TIMER(t,"struct sctk_internal_ptp_s *sctk_get_internal_ptp(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_internal_ptp(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_is_net_message
 **********************************************************/

int  __real_sctk_is_net_message(int a1) ;
int  __wrap_sctk_is_net_message(int a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_is_net_message(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_is_net_message(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_cancel_message
 **********************************************************/

void  __real_sctk_cancel_message(sctk_request_t * a1) ;
void  __wrap_sctk_cancel_message(sctk_request_t * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_cancel_message(sctk_request_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_cancel_message(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_ptp_per_task_init
 **********************************************************/

void  __real_sctk_ptp_per_task_init(int a1) ;
void  __wrap_sctk_ptp_per_task_init(int a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_ptp_per_task_init(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_ptp_per_task_init(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_message_copy
 **********************************************************/

void  __real_sctk_message_copy(sctk_message_to_copy_t * a1) ;
void  __wrap_sctk_message_copy(sctk_message_to_copy_t * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_message_copy(sctk_message_to_copy_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_message_copy(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_message_copy_pack
 **********************************************************/

void  __real_sctk_message_copy_pack(sctk_message_to_copy_t * a1) ;
void  __wrap_sctk_message_copy_pack(sctk_message_to_copy_t * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_message_copy_pack(sctk_message_to_copy_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_message_copy_pack(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_message_copy_pack_absolute
 **********************************************************/

void  __real_sctk_message_copy_pack_absolute(sctk_message_to_copy_t * a1) ;
void  __wrap_sctk_message_copy_pack_absolute(sctk_message_to_copy_t * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_message_copy_pack_absolute(sctk_message_to_copy_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_message_copy_pack_absolute(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_notify_idle_message
 **********************************************************/

void  __real_sctk_notify_idle_message() ;
void  __wrap_sctk_notify_idle_message()  {

  TAU_PROFILE_TIMER(t,"void sctk_notify_idle_message() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_notify_idle_message();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_notify_idle_message_inter
 **********************************************************/

void  __real_sctk_notify_idle_message_inter() ;
void  __wrap_sctk_notify_idle_message_inter()  {

  TAU_PROFILE_TIMER(t,"void sctk_notify_idle_message_inter() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_notify_idle_message_inter();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_init
 **********************************************************/

void  __real_sctk_init() ;
void  __wrap_sctk_init()  {

  TAU_PROFILE_TIMER(t,"void sctk_init(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_init();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_leave
 **********************************************************/

void  __real_sctk_leave() ;
void  __wrap_sctk_leave()  {

  TAU_PROFILE_TIMER(t,"void sctk_leave(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_leave();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_flush_version
 **********************************************************/

void  __real_sctk_flush_version() ;
void  __wrap_sctk_flush_version()  {

  TAU_PROFILE_TIMER(t,"void sctk_flush_version(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_flush_version();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_abort
 **********************************************************/

void  __real_sctk_abort() ;
void  __wrap_sctk_abort()  {

  TAU_PROFILE_TIMER(t,"void sctk_abort(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_abort();
  TAU_PROFILE_STOP(t);

}

#warning "TAU: Not generating wrapper for function sctk_error"
#warning "TAU: Not generating wrapper for function sctk_formated_assert_print"
#warning "TAU: Not generating wrapper for function sctk_debug_root"
#warning "TAU: Not generating wrapper for function sctk_silent_debug"
#warning "TAU: Not generating wrapper for function sctk_log"
#warning "TAU: Not generating wrapper for function sctk_warning"

/**********************************************************
   sctk_size_checking_eq
 **********************************************************/

void  __real_sctk_size_checking_eq(size_t a1, size_t a2, char * a3, char * a4, char * a5, int a6) ;
void  __wrap_sctk_size_checking_eq(size_t a1, size_t a2, char * a3, char * a4, char * a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void sctk_size_checking_eq(size_t, size_t, char *, char *, char *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_size_checking_eq(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_size_checking
 **********************************************************/

void  __real_sctk_size_checking(size_t a1, size_t a2, char * a3, char * a4, char * a5, int a6) ;
void  __wrap_sctk_size_checking(size_t a1, size_t a2, char * a3, char * a4, char * a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void sctk_size_checking(size_t, size_t, char *, char *, char *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_size_checking(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}

#warning "TAU: Not generating wrapper for function sctk_formated_dbg_print_abort"

/**********************************************************
   sctk_noalloc_fwrite
 **********************************************************/

size_t  __real_sctk_noalloc_fwrite(const void * a1, size_t a2, size_t a3, FILE * a4) ;
size_t  __wrap_sctk_noalloc_fwrite(const void * a1, size_t a2, size_t a3, FILE * a4)  {

  size_t retval;
  TAU_PROFILE_TIMER(t,"size_t sctk_noalloc_fwrite(const void *, size_t, size_t, FILE *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_noalloc_fwrite(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

#warning "TAU: Not generating wrapper for function sctk_noalloc_fprintf"

/**********************************************************
   sctk_noalloc_vfprintf
 **********************************************************/

void  __real_sctk_noalloc_vfprintf(FILE * a1, const char * a2, va_list a3) ;
void  __wrap_sctk_noalloc_vfprintf(FILE * a1, const char * a2, va_list a3)  {

  TAU_PROFILE_TIMER(t,"void sctk_noalloc_vfprintf(FILE *, const char *, va_list) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_noalloc_vfprintf(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}

#warning "TAU: Not generating wrapper for function sctk_noalloc_printf"

/**********************************************************
   sctk_print_version
 **********************************************************/

void  __real_sctk_print_version(char * a1, int a2, int a3) ;
void  __wrap_sctk_print_version(char * a1, int a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void sctk_print_version(char *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_print_version(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}

#warning "TAU: Not generating wrapper for function sctk_debug_print_backtrace"

/**********************************************************
   sctk_set_version_details
 **********************************************************/

void  __real_sctk_set_version_details() ;
void  __wrap_sctk_set_version_details()  {

  TAU_PROFILE_TIMER(t,"void sctk_set_version_details(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_set_version_details();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_unset_version_details
 **********************************************************/

void  __real_sctk_unset_version_details() ;
void  __wrap_sctk_unset_version_details()  {

  TAU_PROFILE_TIMER(t,"void sctk_unset_version_details(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_unset_version_details();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_communicator_world_init
 **********************************************************/

void  __real_sctk_communicator_world_init(int a1) ;
void  __wrap_sctk_communicator_world_init(int a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_communicator_world_init(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_communicator_world_init(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_communicator_self_init
 **********************************************************/

void  __real_sctk_communicator_self_init() ;
void  __wrap_sctk_communicator_self_init()  {

  TAU_PROFILE_TIMER(t,"void sctk_communicator_self_init() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_communicator_self_init();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_get_nb_task_local
 **********************************************************/

int  __real_sctk_get_nb_task_local(const sctk_communicator_t a1) ;
int  __wrap_sctk_get_nb_task_local(const sctk_communicator_t a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_nb_task_local(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_nb_task_local(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_first_task_local
 **********************************************************/

int  __real_sctk_get_first_task_local(const sctk_communicator_t a1) ;
int  __wrap_sctk_get_first_task_local(const sctk_communicator_t a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_first_task_local(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_first_task_local(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_last_task_local
 **********************************************************/

int  __real_sctk_get_last_task_local(const sctk_communicator_t a1) ;
int  __wrap_sctk_get_last_task_local(const sctk_communicator_t a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_last_task_local(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_last_task_local(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_nb_task_total
 **********************************************************/

int  __real_sctk_get_nb_task_total(const sctk_communicator_t a1) ;
int  __wrap_sctk_get_nb_task_total(const sctk_communicator_t a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_nb_task_total(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_nb_task_total(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_rank_size_total
 **********************************************************/

void  __real_sctk_get_rank_size_total(const sctk_communicator_t a1, int * a2, int * a3, int a4) ;
void  __wrap_sctk_get_rank_size_total(const sctk_communicator_t a1, int * a2, int * a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void sctk_get_rank_size_total(const sctk_communicator_t, int *, int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_get_rank_size_total(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_get_process_array
 **********************************************************/

int *  __real_sctk_get_process_array(const sctk_communicator_t a1) ;
int *  __wrap_sctk_get_process_array(const sctk_communicator_t a1)  {

  int * retval;
  TAU_PROFILE_TIMER(t,"int *sctk_get_process_array(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_process_array(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_process_nb_in_array
 **********************************************************/

int  __real_sctk_get_process_nb_in_array(const sctk_communicator_t a1) ;
int  __wrap_sctk_get_process_nb_in_array(const sctk_communicator_t a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_process_nb_in_array(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_process_nb_in_array(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_rank
 **********************************************************/

int  __real_sctk_get_rank(const sctk_communicator_t a1, const int a2) ;
int  __wrap_sctk_get_rank(const sctk_communicator_t a1, const int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_rank(const sctk_communicator_t, const int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_rank(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_comm_world_rank
 **********************************************************/

int  __real_sctk_get_comm_world_rank(const sctk_communicator_t a1, const int a2) ;
int  __wrap_sctk_get_comm_world_rank(const sctk_communicator_t a1, const int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_comm_world_rank(const sctk_communicator_t, const int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_comm_world_rank(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_delete_communicator
 **********************************************************/

sctk_communicator_t  __real_sctk_delete_communicator(const sctk_communicator_t a1) ;
sctk_communicator_t  __wrap_sctk_delete_communicator(const sctk_communicator_t a1)  {

  sctk_communicator_t retval;
  TAU_PROFILE_TIMER(t,"sctk_communicator_t sctk_delete_communicator(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_delete_communicator(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_communicator_delete
 **********************************************************/

void  __real_sctk_communicator_delete() ;
void  __wrap_sctk_communicator_delete()  {

  TAU_PROFILE_TIMER(t,"void sctk_communicator_delete() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_communicator_delete();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_duplicate_communicator
 **********************************************************/

sctk_communicator_t  __real_sctk_duplicate_communicator(const sctk_communicator_t a1, int a2, int a3) ;
sctk_communicator_t  __wrap_sctk_duplicate_communicator(const sctk_communicator_t a1, int a2, int a3)  {

  sctk_communicator_t retval;
  TAU_PROFILE_TIMER(t,"sctk_communicator_t sctk_duplicate_communicator(const sctk_communicator_t, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_duplicate_communicator(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_internal_collectives
 **********************************************************/

struct sctk_internal_collectives_struct_s *  __real_sctk_get_internal_collectives(const sctk_communicator_t a1) ;
struct sctk_internal_collectives_struct_s *  __wrap_sctk_get_internal_collectives(const sctk_communicator_t a1)  {

  struct sctk_internal_collectives_struct_s * retval;
  TAU_PROFILE_TIMER(t,"struct sctk_internal_collectives_struct_s *sctk_get_internal_collectives(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_internal_collectives(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_set_internal_collectives
 **********************************************************/

void  __real_sctk_set_internal_collectives(const sctk_communicator_t a1, struct sctk_internal_collectives_struct_s * a2) ;
void  __wrap_sctk_set_internal_collectives(const sctk_communicator_t a1, struct sctk_internal_collectives_struct_s * a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_set_internal_collectives(const sctk_communicator_t, struct sctk_internal_collectives_struct_s *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_set_internal_collectives(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_get_process_rank_from_task_rank
 **********************************************************/

int  __real_sctk_get_process_rank_from_task_rank(int a1) ;
int  __wrap_sctk_get_process_rank_from_task_rank(int a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_process_rank_from_task_rank(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_process_rank_from_task_rank(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_is_inter_comm
 **********************************************************/

int  __real_sctk_is_inter_comm(const sctk_communicator_t a1) ;
int  __wrap_sctk_is_inter_comm(const sctk_communicator_t a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_is_inter_comm(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_is_inter_comm(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_create_communicator
 **********************************************************/

sctk_communicator_t  __real_sctk_create_communicator(const sctk_communicator_t a1, const int a2, const int * a3, int a4) ;
sctk_communicator_t  __wrap_sctk_create_communicator(const sctk_communicator_t a1, const int a2, const int * a3, int a4)  {

  sctk_communicator_t retval;
  TAU_PROFILE_TIMER(t,"sctk_communicator_t sctk_create_communicator(const sctk_communicator_t, const int, const int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_create_communicator(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_barrier
 **********************************************************/

void  __real_sctk_barrier(const sctk_communicator_t a1) ;
void  __wrap_sctk_barrier(const sctk_communicator_t a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_barrier(const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_barrier(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_broadcast
 **********************************************************/

void  __real_sctk_broadcast(void * a1, const size_t a2, const int a3, const sctk_communicator_t a4) ;
void  __wrap_sctk_broadcast(void * a1, const size_t a2, const int a3, const sctk_communicator_t a4)  {

  TAU_PROFILE_TIMER(t,"void sctk_broadcast(void *, const size_t, const int, const sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_broadcast(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_all_reduce
 **********************************************************/

void  __real_sctk_all_reduce(const void * a1, void * a2, const size_t a3, const size_t a4, MPC_Op_f a5, const sctk_communicator_t a6, const sctk_datatype_t a7) ;
void  __wrap_sctk_all_reduce(const void * a1, void * a2, const size_t a3, const size_t a4, MPC_Op_f a5, const sctk_communicator_t a6, const sctk_datatype_t a7)  {

  TAU_PROFILE_TIMER(t,"void sctk_all_reduce(const void *, void *, const size_t, const size_t, MPC_Op_f, const sctk_communicator_t, const sctk_datatype_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_all_reduce(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_collectives_init
 **********************************************************/

void  __real_sctk_collectives_init(sctk_communicator_t a1, void (*a2)(sctk_internal_collectives_struct_t *, sctk_communicator_t), void (*a3)(sctk_internal_collectives_struct_t *, sctk_communicator_t), void (*a4)(sctk_internal_collectives_struct_t *, sctk_communicator_t)) ;
void  __wrap_sctk_collectives_init(sctk_communicator_t a1, void (*a2)(sctk_internal_collectives_struct_t *, sctk_communicator_t), void (*a3)(sctk_internal_collectives_struct_t *, sctk_communicator_t), void (*a4)(sctk_internal_collectives_struct_t *, sctk_communicator_t))  {

  TAU_PROFILE_TIMER(t,"void sctk_collectives_init(sctk_communicator_t, void (*)(sctk_internal_collectives_struct_t *, sctk_communicator_t), void (*)(sctk_internal_collectives_struct_t *, sctk_communicator_t), void (*)(sctk_internal_collectives_struct_t *, sctk_communicator_t)) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_collectives_init(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_terminaison_barrier
 **********************************************************/

void  __real_sctk_terminaison_barrier(const int a1) ;
void  __wrap_sctk_terminaison_barrier(const int a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_terminaison_barrier(const int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_terminaison_barrier(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_set_tls
 **********************************************************/

void  __real_sctk_set_tls(struct sctk_alloc_chain * a1) ;
void  __wrap_sctk_set_tls(struct sctk_alloc_chain * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_set_tls(struct sctk_alloc_chain *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_set_tls(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_alloc_posix_plug_on_egg_allocator
 **********************************************************/

void  __real_sctk_alloc_posix_plug_on_egg_allocator() ;
void  __wrap_sctk_alloc_posix_plug_on_egg_allocator()  {

  TAU_PROFILE_TIMER(t,"void sctk_alloc_posix_plug_on_egg_allocator(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_alloc_posix_plug_on_egg_allocator();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_get_current_alloc_chain
 **********************************************************/

struct sctk_alloc_chain *  __real_sctk_get_current_alloc_chain() ;
struct sctk_alloc_chain *  __wrap_sctk_get_current_alloc_chain()  {

  struct sctk_alloc_chain * retval;
  TAU_PROFILE_TIMER(t,"struct sctk_alloc_chain *sctk_get_current_alloc_chain(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_current_alloc_chain();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_alloc_posix_numa_migrate
 **********************************************************/

void  __real_sctk_alloc_posix_numa_migrate() ;
void  __wrap_sctk_alloc_posix_numa_migrate()  {

  TAU_PROFILE_TIMER(t,"void sctk_alloc_posix_numa_migrate(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_alloc_posix_numa_migrate();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_alloc_posix_numa_migrate_chain
 **********************************************************/

void  __real_sctk_alloc_posix_numa_migrate_chain(struct sctk_alloc_chain * a1) ;
void  __wrap_sctk_alloc_posix_numa_migrate_chain(struct sctk_alloc_chain * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_alloc_posix_numa_migrate_chain(struct sctk_alloc_chain *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_alloc_posix_numa_migrate_chain(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_alloc_posix_mmsrc_numa_init_phase_numa
 **********************************************************/

void  __real_sctk_alloc_posix_mmsrc_numa_init_phase_numa() ;
void  __wrap_sctk_alloc_posix_mmsrc_numa_init_phase_numa()  {

  TAU_PROFILE_TIMER(t,"void sctk_alloc_posix_mmsrc_numa_init_phase_numa(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_alloc_posix_mmsrc_numa_init_phase_numa();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_get_heap_size
 **********************************************************/

unsigned long  __real_sctk_get_heap_size() ;
unsigned long  __wrap_sctk_get_heap_size()  {

  unsigned long retval;
  TAU_PROFILE_TIMER(t,"unsigned long sctk_get_heap_size(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_heap_size();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __sctk_free
 **********************************************************/

void  __real___sctk_free(void * a1, struct sctk_alloc_chain * a2) ;
void  __wrap___sctk_free(void * a1, struct sctk_alloc_chain * a2)  {

  TAU_PROFILE_TIMER(t,"void __sctk_free(void *, struct sctk_alloc_chain *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___sctk_free(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __sctk_malloc
 **********************************************************/

void *  __real___sctk_malloc(size_t a1, struct sctk_alloc_chain * a2) ;
void *  __wrap___sctk_malloc(size_t a1, struct sctk_alloc_chain * a2)  {

  void * retval;
  TAU_PROFILE_TIMER(t,"void *__sctk_malloc(size_t, struct sctk_alloc_chain *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___sctk_malloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_free
 **********************************************************/

void  __real_sctk_free(void * a1) ;
void  __wrap_sctk_free(void * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_free(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_posix_memalign
 **********************************************************/

int  __real_sctk_posix_memalign(void ** a1, size_t a2, size_t a3) ;
int  __wrap_sctk_posix_memalign(void ** a1, size_t a2, size_t a3)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_posix_memalign(void **, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_posix_memalign(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __sctk_malloc_new
 **********************************************************/

void *  __real___sctk_malloc_new(size_t a1, struct sctk_alloc_chain * a2) ;
void *  __wrap___sctk_malloc_new(size_t a1, struct sctk_alloc_chain * a2)  {

  void * retval;
  TAU_PROFILE_TIMER(t,"void *__sctk_malloc_new(size_t, struct sctk_alloc_chain *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___sctk_malloc_new(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_env_init
 **********************************************************/

int  __real_sctk_env_init(int * a1, char *** a2) ;
int  __wrap_sctk_env_init(int * a1, char *** a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_env_init(int *, char ***) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_env_init(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_env_exit
 **********************************************************/

int  __real_sctk_env_exit() ;
int  __wrap_sctk_env_exit()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_env_exit(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_env_exit();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_launch_contribution
 **********************************************************/

void  __real_sctk_launch_contribution(FILE * a1) ;
void  __wrap_sctk_launch_contribution(FILE * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_launch_contribution(FILE *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_launch_contribution(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_user_main
 **********************************************************/

int  __real_sctk_user_main(int a1, char ** a2) ;
int  __wrap_sctk_user_main(int a1, char ** a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_user_main(int, char **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_user_main(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_process_nb
 **********************************************************/

int  __real_sctk_get_process_nb() ;
int  __wrap_sctk_get_process_nb()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_process_nb(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_process_nb();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_processor_nb
 **********************************************************/

int  __real_sctk_get_processor_nb() ;
int  __wrap_sctk_get_processor_nb()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_processor_nb(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_processor_nb();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_verbosity
 **********************************************************/

int  __real_sctk_get_verbosity() ;
int  __wrap_sctk_get_verbosity()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_verbosity() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_verbosity();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_total_tasks_number
 **********************************************************/

int  __real_sctk_get_total_tasks_number() ;
int  __wrap_sctk_get_total_tasks_number()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_total_tasks_number() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_total_tasks_number();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_spinlock_lock_yield
 **********************************************************/

int  __real_sctk_spinlock_lock_yield(sctk_spinlock_t * a1) ;
int  __wrap_sctk_spinlock_lock_yield(sctk_spinlock_t * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_spinlock_lock_yield(sctk_spinlock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_spinlock_lock_yield(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_spinlock_lock
 **********************************************************/

int  __real_sctk_spinlock_lock(sctk_spinlock_t * a1) ;
int  __wrap_sctk_spinlock_lock(sctk_spinlock_t * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_spinlock_lock(sctk_spinlock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_spinlock_lock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_spinlock_unlock
 **********************************************************/

int  __real_sctk_spinlock_unlock(sctk_spinlock_t * a1) ;
int  __wrap_sctk_spinlock_unlock(sctk_spinlock_t * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_spinlock_unlock(sctk_spinlock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_spinlock_unlock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_spinlock_trylock
 **********************************************************/

int  __real_sctk_spinlock_trylock(sctk_spinlock_t * a1) ;
int  __wrap_sctk_spinlock_trylock(sctk_spinlock_t * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_spinlock_trylock(sctk_spinlock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_spinlock_trylock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_spinlock_read_lock
 **********************************************************/

int  __real_sctk_spinlock_read_lock(sctk_spin_rwlock_t * a1) ;
int  __wrap_sctk_spinlock_read_lock(sctk_spin_rwlock_t * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_spinlock_read_lock(sctk_spin_rwlock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_spinlock_read_lock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_spinlock_write_lock
 **********************************************************/

int  __real_sctk_spinlock_write_lock(sctk_spin_rwlock_t * a1) ;
int  __wrap_sctk_spinlock_write_lock(sctk_spin_rwlock_t * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_spinlock_write_lock(sctk_spin_rwlock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_spinlock_write_lock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_spinlock_read_unlock
 **********************************************************/

int  __real_sctk_spinlock_read_unlock(sctk_spin_rwlock_t * a1) ;
int  __wrap_sctk_spinlock_read_unlock(sctk_spin_rwlock_t * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_spinlock_read_unlock(sctk_spin_rwlock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_spinlock_read_unlock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_spinlock_write_unlock
 **********************************************************/

int  __real_sctk_spinlock_write_unlock(sctk_spin_rwlock_t * a1) ;
int  __wrap_sctk_spinlock_write_unlock(sctk_spin_rwlock_t * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_spinlock_write_unlock(sctk_spin_rwlock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_spinlock_write_unlock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_tls_entry_add
 **********************************************************/

unsigned long  __real_sctk_tls_entry_add(unsigned long a1, void (*a2)(void *)) ;
unsigned long  __wrap_sctk_tls_entry_add(unsigned long a1, void (*a2)(void *))  {

  unsigned long retval;
  TAU_PROFILE_TIMER(t,"unsigned long sctk_tls_entry_add(unsigned long, void (*)(void *)) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_tls_entry_add(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_tls_init_key
 **********************************************************/

void  __real_sctk_tls_init_key(unsigned long a1, void (*a2)(void *)) ;
void  __wrap_sctk_tls_init_key(unsigned long a1, void (*a2)(void *))  {

  TAU_PROFILE_TIMER(t,"void sctk_tls_init_key(unsigned long, void (*)(void *)) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_tls_init_key(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_start_func
 **********************************************************/

void  __real_sctk_start_func(void *(*a1)(void *), void * a2) ;
void  __wrap_sctk_start_func(void *(*a1)(void *), void * a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_start_func(void *(*)(void *), void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_start_func(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_get_init_vp
 **********************************************************/

int  __real_sctk_get_init_vp(int a1) ;
int  __wrap_sctk_get_init_vp(int a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_get_init_vp(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_init_vp(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_is_restarted
 **********************************************************/

int  __real_sctk_is_restarted() ;
int  __wrap_sctk_is_restarted()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int sctk_is_restarted(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_is_restarted();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_get_check_point_key
 **********************************************************/

sctk_thread_key_t  __real_sctk_get_check_point_key() ;
sctk_thread_key_t  __wrap_sctk_get_check_point_key()  {

  sctk_thread_key_t retval;
  TAU_PROFILE_TIMER(t,"sctk_thread_key_t sctk_get_check_point_key(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_get_check_point_key();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_mpc_init_keys
 **********************************************************/

void  __real_sctk_mpc_init_keys() ;
void  __wrap_sctk_mpc_init_keys()  {

  TAU_PROFILE_TIMER(t,"void sctk_mpc_init_keys(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_mpc_init_keys();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_profiling_get_init_time
 **********************************************************/

double  __real_sctk_profiling_get_init_time() ;
double  __wrap_sctk_profiling_get_init_time()  {

  double retval;
  TAU_PROFILE_TIMER(t,"double sctk_profiling_get_init_time() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_profiling_get_init_time();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_profiling_get_dataused
 **********************************************************/

double  __real_sctk_profiling_get_dataused() ;
double  __wrap_sctk_profiling_get_dataused()  {

  double retval;
  TAU_PROFILE_TIMER(t,"double sctk_profiling_get_dataused() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_profiling_get_dataused();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_barrier_simple_init
 **********************************************************/

void  __real_sctk_barrier_simple_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_barrier_simple_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_barrier_simple_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_barrier_simple_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_broadcast_simple_init
 **********************************************************/

void  __real_sctk_broadcast_simple_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_broadcast_simple_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_broadcast_simple_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_broadcast_simple_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_allreduce_simple_init
 **********************************************************/

void  __real_sctk_allreduce_simple_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_allreduce_simple_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_allreduce_simple_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_allreduce_simple_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_collectives_init_simple
 **********************************************************/

void  __real_sctk_collectives_init_simple(sctk_communicator_t a1) ;
void  __wrap_sctk_collectives_init_simple(sctk_communicator_t a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_collectives_init_simple(sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_collectives_init_simple(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_barrier_opt_messages_init
 **********************************************************/

void  __real_sctk_barrier_opt_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_barrier_opt_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_barrier_opt_messages_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_barrier_opt_messages_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_broadcast_opt_messages_init
 **********************************************************/

void  __real_sctk_broadcast_opt_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_broadcast_opt_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_broadcast_opt_messages_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_broadcast_opt_messages_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_allreduce_opt_messages_init
 **********************************************************/

void  __real_sctk_allreduce_opt_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_allreduce_opt_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_allreduce_opt_messages_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_allreduce_opt_messages_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_collectives_init_opt_messages
 **********************************************************/

void  __real_sctk_collectives_init_opt_messages(sctk_communicator_t a1) ;
void  __wrap_sctk_collectives_init_opt_messages(sctk_communicator_t a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_collectives_init_opt_messages(sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_collectives_init_opt_messages(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_barrier_hetero_messages_init
 **********************************************************/

void  __real_sctk_barrier_hetero_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_barrier_hetero_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_barrier_hetero_messages_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_barrier_hetero_messages_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_broadcast_hetero_messages_init
 **********************************************************/

void  __real_sctk_broadcast_hetero_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_broadcast_hetero_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_broadcast_hetero_messages_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_broadcast_hetero_messages_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_allreduce_hetero_messages_init
 **********************************************************/

void  __real_sctk_allreduce_hetero_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2) ;
void  __wrap_sctk_allreduce_hetero_messages_init(struct sctk_internal_collectives_struct_s * a1, sctk_communicator_t a2)  {

  TAU_PROFILE_TIMER(t,"void sctk_allreduce_hetero_messages_init(struct sctk_internal_collectives_struct_s *, sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_allreduce_hetero_messages_init(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_collectives_init_hetero_messages
 **********************************************************/

void  __real_sctk_collectives_init_hetero_messages(sctk_communicator_t a1) ;
void  __wrap_sctk_collectives_init_hetero_messages(sctk_communicator_t a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_collectives_init_hetero_messages(sctk_communicator_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_collectives_init_hetero_messages(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_add_static_reorder_buffer
 **********************************************************/

void  __real_sctk_add_static_reorder_buffer(int a1) ;
void  __wrap_sctk_add_static_reorder_buffer(int a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_add_static_reorder_buffer(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_add_static_reorder_buffer(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_add_dynamic_reorder_buffer
 **********************************************************/

void  __real_sctk_add_dynamic_reorder_buffer(int a1) ;
void  __wrap_sctk_add_dynamic_reorder_buffer(int a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_add_dynamic_reorder_buffer(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_add_dynamic_reorder_buffer(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_set_dynamic_reordering_buffer_creation
 **********************************************************/

void  __real_sctk_set_dynamic_reordering_buffer_creation() ;
void  __wrap_sctk_set_dynamic_reordering_buffer_creation()  {

  TAU_PROFILE_TIMER(t,"void sctk_set_dynamic_reordering_buffer_creation() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_set_dynamic_reordering_buffer_creation();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_ib_create_remote
 **********************************************************/

struct sctk_route_table_s *  __real_sctk_ib_create_remote() ;
struct sctk_route_table_s *  __wrap_sctk_ib_create_remote()  {

  struct sctk_route_table_s * retval;
  TAU_PROFILE_TIMER(t,"struct sctk_route_table_s *sctk_ib_create_remote() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_sctk_ib_create_remote();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sctk_ib_init_remote
 **********************************************************/

void  __real_sctk_ib_init_remote(int a1, struct sctk_rail_info_s * a2, struct sctk_route_table_s * a3, int a4) ;
void  __wrap_sctk_ib_init_remote(int a1, struct sctk_rail_info_s * a2, struct sctk_route_table_s * a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void sctk_ib_init_remote(int, struct sctk_rail_info_s *, struct sctk_route_table_s *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_ib_init_remote(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_network_stats_ib
 **********************************************************/

void  __real_sctk_network_stats_ib(struct MPC_Network_stats_s * a1) ;
void  __wrap_sctk_network_stats_ib(struct MPC_Network_stats_s * a1)  {

  TAU_PROFILE_TIMER(t,"void sctk_network_stats_ib(struct MPC_Network_stats_s *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_network_stats_ib(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   sctk_network_deco_neighbors_ib
 **********************************************************/

void  __real_sctk_network_deco_neighbors_ib() ;
void  __wrap_sctk_network_deco_neighbors_ib()  {

  TAU_PROFILE_TIMER(t,"void sctk_network_deco_neighbors_ib() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_sctk_network_deco_neighbors_ib();
  TAU_PROFILE_STOP(t);

}

#endif //TAU_PRELOAD_LIB

/***************************************************************************
 * $RCSfile: TauWrapSyscalls.cpp,v $   $Author: sameer $
 * $Revision: 1.6 $   $Date: 2010/06/10 12:46:53 $
 * TAU_VERSION_ID: $Id: TauWrapSyscalls.cpp,v 1.6 2010/06/10 12:46:53 sameer Exp $
 ***************************************************************************/
