#include <Profile/Profiler.h>
#include <stdio.h>
#include <mpi.h>


/* This file uses the MPI Profiling Interface with TAU instrumentation.
   It has been adopted from the MPE Profiling interface wrapper generator
   wrappergen that is part of the MPICH distribution. It differs from MPE
   in where the calls are placed. For e.g., in TAU a send is traced before
   the MPI_Send and a receive after MPI_Recv. This avoids -ve time problems
   that can happen on a uniprocessor if a receive is traced before the send
   is traced. 

   To generate TauMpi.c use: 
   % <mpich>/mpe/profiling/wrappergen/wrappergen -w TauMpi.w -o TauMpi.c

   DO NOT EDIT TauMpi.c manually. Instead edit TauMpi.w and use wrappergen
*/

/* Requests */

typedef struct request_list_ {
    MPI_Request * request; /* SSS request should be a pointer */
    int         status, size, tag, otherParty;
    int         is_persistent;
    struct request_list_ *next;
} request_list;

#define RQ_SEND    0x1
#define RQ_RECV    0x2
#define RQ_CANCEL  0x4
/* if MPI_Cancel is called on a request, 'or' RQ_CANCEL into status.
** After a Wait* or Test* is called on that request, check for RQ_CANCEL.
** If the bit is set, check with MPI_Test_cancelled before registering
** the send/receive as 'happening'.
**
*/

#define rq_alloc( head_alloc, newrq ) {\
      if (head_alloc) {\
        newrq=head_alloc;head_alloc=newrq->next;\
	}else{\
      newrq = (request_list*) malloc(sizeof( request_list ));\
      }}

#define rq_remove_at( head, tail, head_alloc, ptr, last ) { \
  if (ptr) { \
    if (!last) { \
      head = ptr->next; \
    } else { \
      last->next = ptr->next; \
      if (tail == ptr) tail = last; \
    } \
	  ptr->next = head_alloc; head_alloc = ptr;}}

#define rq_remove( head, tail, head_alloc, rq ) { \
  request_list *ptr, *last; \
  ptr = head; \
  last = 0; \
  while (ptr && (ptr->request != rq)) { \
    last = ptr; \
    ptr = ptr->next; \
  } \
	rq_remove_at( head, tail, head_alloc, ptr, last );}

#define rq_add( head, tail, rq ) { \
  if (!head) { \
    head = tail = rq; \
  } else { \
    tail->next = rq; tail = rq; \
  }}

#define rq_find( head, req, rq ) { \
  rq = head; \
  while (rq && (rq->request != req)) rq = rq->next; }

#define rq_init( head_alloc ) {\
  int i; request_list *newrq; head_alloc = 0;\
  for (i=0;i<20;i++) {\
      newrq = (request_list*) malloc(sizeof( request_list ));\
      newrq->next = head_alloc;\
      head_alloc = newrq;\
  }}

#define rq_end( head_alloc ) {\
  request_list *rq; while (head_alloc) {\
	rq = head_alloc->next;free(head_alloc);head_alloc=rq;}}
static request_list *requests_head_{{fileno}}, *requests_tail_{{fileno}};
static int procid_{{fileno}};

/* MPI PROFILING INTERFACE WRAPPERS BEGIN HERE */

/* Message_prof keeps track of when sends and receives 'happen'.  The
** time that each send or receive happens is different for each type of
** send or receive.
**
** Check for MPI_PROC_NULL
**
**   definitely a send:
**     Before a call to MPI_Send.
**     Before a call to MPI_Bsend.
**     Before a call to MPI_Ssend.
**     Before a call to MPI_Rsend.
**
**
**   definitely a receive:
**     After a call to MPI_Recv.
**
**   definitely a send before and a receive after :
**     a call to MPI_Sendrecv
**     a call to MPI_Sendrecv_replace
**
**   maybe a send, maybe a receive:
**     Before a call to MPI_Wait.
**     Before a call to MPI_Waitany.
**     Before a call to MPI_Waitsome.
**     Before a call to MPI_Waitall.
**     After a call to MPI_Probe
**   maybe neither:
**     Before a call to MPI_Test.
**     Before a call to MPI_Testany.
**     Before a call to MPI_Testsome.
**     Before a call to MPI_Testall.
**     After a call to MPI_Iprobe
**
**   start request for a send:
**     After a call to MPI_Isend.
**     After a call to MPI_Ibsend.
**     After a call to MPI_Issend.
**     After a call to MPI_Irsend.
**     After a call to MPI_Send_init.
**     After a call to MPI_Bsend_init.
**     After a call to MPI_Ssend_init.
**     After a call to MPI_Rsend_init.
**
**   start request for a recv:
**     After a call to MPI_Irecv.
**     After a call to MPI_Recv_init.
**
**   stop watching a request:
**     Before a call to MPI_Request_free
**
**   mark a request as possible cancelled:
**     After a call to MPI_Cancel
**
*/

{{fn fn_name MPI_Init}}
  {{callfn}}
  PMPI_Comm_rank( MPI_COMM_WORLD, &procid_{{fileno}} );
  TAU_PROFILE_SET_NODE(procid_{{fileno}} ); 
  requests_head_{{fileno}} = requests_tail_{{fileno}} = 0;
{{endfn}}


{{fn fn_name MPI_Send MPI_Bsend MPI_Ssend MPI_Rsend}}
  {{vardecl int typesize}}
  TAU_PROFILE_TIMER(tautimer, "{{fn_name}}()",  " ", TAU_MESSAGE); 
  TAU_PROFILE_START(tautimer);
  if ({{dest}} != MPI_PROC_NULL) {
    PMPI_Type_size( {{datatype}}, &{{typesize}} );
    TAU_TRACE_SENDMSG({{tag}}, {{dest}}, {{typesize}}*{{count}}); 
    /*
    prof_send( procid_{{fileno}}, {{dest}}, {{tag}}, {{typesize}}*{{count}},
	       "{{fn_name}}" );
    */
  }
  {{callfn}}
  TAU_PROFILE_STOP(tautimer); 
{{endfn}}

{{fn fn_name MPI_Recv}}
  {{vardecl int size}}
  TAU_PROFILE_TIMER(tautimer, "{{fn_name}}()",  " ", TAU_MESSAGE); 
  TAU_PROFILE_START(tautimer);
  {{callfn}}
  if ({{source}} != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    PMPI_Get_count( {{status}}, MPI_BYTE, &{{size}} );
    TAU_TRACE_RECVMSG({{status}}->MPI_TAG, {{status}}->MPI_SOURCE, {{size}});
    /*
    prof_recv( procid_{{fileno}}, {{status}}->MPI_SOURCE,
	       {{status}}->MPI_TAG, {{size}}, "{{fn_name}}" );
    */
  }
  TAU_PROFILE_STOP(tautimer); 
{{endfn}}


{{fn fn_name MPI_Sendrecv}}
  {{vardecl int typesize, count}}
  TAU_PROFILE_TIMER(tautimer, "{{fn_name}}()",  " ", TAU_MESSAGE); 
  TAU_PROFILE_START(tautimer);
  if ({{dest}} != MPI_PROC_NULL) {
    MPI_Type_size( {{sendtype}}, &{{typesize}} );
    TAU_TRACE_SENDMSG({{sendtag}}, {{dest}}, {{typesize}}*{{sendcount}});
    /*         
    prof_send( procid_{{fileno}}, {{dest}}, {{sendtag}},
               {{typesize}}*{{sendcount}}, "{{fn_name}}" );
    */
  } 	
  {{callfn}}
  if ({{dest}} != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    PMPI_Get_count( {{status}}, MPI_BYTE, &{{count}} );
    TAU_TRACE_RECVMSG({{status}}->MPI_TAG, {{status}}->MPI_SOURCE, {{count}});
    /*
    prof_recv( {{dest}}, procid_{{fileno}}, {{recvtag}}, {{count}},
	       "{{fn_name}}" );
    NOTE: shouldn't we look at the status to get the tag and source?
    */
  }
  TAU_PROFILE_STOP(tautimer); 
{{endfn}}

  
{{fn fn_name MPI_Sendrecv_replace}}
  {{vardecl int size, typesize}}
  TAU_PROFILE_TIMER(tautimer, "{{fn_name}}()",  " ", TAU_MESSAGE); 
  TAU_PROFILE_START(tautimer);
  if ({{dest}} != MPI_PROC_NULL) {
    PMPI_Type_size( {{datatype}}, &{{typesize}} );
    TAU_TRACE_SENDMSG({{sendtag}}, {{dest}}, {{typesize}}*{{count}});
    /*         
    prof_send( procid_{{fileno}}, {{dest}}, {{sendtag}},
               {{typesize}}*{{count}}, "{{fn_name}}" );
    */
  }	
  {{callfn}}
  if ({{dest}} != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    PMPI_Get_count( {{status}}, MPI_BYTE, &{{size}} );
    TAU_TRACE_RECVMSG({{status}}->MPI_TAG, {{status}}->MPI_SOURCE, {{size}});
    /*
    prof_recv( {{dest}}, procid_{{fileno}}, {{recvtag}}, {{size}},
	       "{{fn_name}}" );
    */
  }
  TAU_PROFILE_STOP(tautimer); 
{{endfn}}

{{fn fn_name MPI_Isend MPI_Ibsend MPI_Issend MPI_Irsend
             MPI_Send_init MPI_Bsend_init MPI_Ssend_init MPI_Rsend_init}}
  {{vardecl request_list *newrq}}
  {{vardecl int typesize}}
/* fprintf( stderr, "{{fn_name}} call on %d\n", procid_{{fileno}} ); */
  {{callfn}}
  if ({{dest}} != MPI_PROC_NULL) {
    if ({{newrq}} = (request_list*) malloc(sizeof( request_list ))) {
      MPI_Type_size( {{datatype}}, &{{typesize}} );
      {{newrq}}->request = {{request}};
      {{newrq}}->status = RQ_SEND;
      {{newrq}}->size = {{count}} * {{typesize}};
      {{newrq}}->tag = {{tag}};
      {{newrq}}->otherParty = {{dest}};
      {{newrq}}->next = 0;
      rq_add( requests_head_{{fileno}}, requests_tail_{{fileno}}, {{newrq}} );
    }
  }
{{endfn}}


{{fn fn_name MPI_Irecv MPI_Recv_init}}
  {{vardecl request_list *newrq}}
  {{callfn}}
  if ({{source}} != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    if ({{newrq}} = (request_list*) malloc(sizeof( request_list ))) {
      {{newrq}}->request = {{request}};
      {{newrq}}->status = RQ_RECV;
      {{newrq}}->next = 0;
      rq_add( requests_head_{{fileno}}, requests_tail_{{fileno}}, {{newrq}} );
    }
  }
{{endfn}}


{{fn fn_name MPI_Request_free}}
  /* The request may have completed, may have not.  */
  /* We'll assume it didn't. */
#ifdef DIDNOT_COMPILE
  rq_remove( requests_head_{{fileno}}, {{request}} );
#endif
  {{callfn}}
{{endfn}}

{{fn fn_name MPI_Cancel}}
  {{vardecl request_list *rq}}
  {{callfn}}
  rq_find( requests_head_{{fileno}}, {{request}}, {{rq}} );
  if ({{rq}}) {{rq}}->status |= RQ_CANCEL;
  /* be sure to check on the Test or Wait if it was really cancelled */
{{endfn}}

void ProcessWaitTest_{{fileno}} ( request, status, note )
MPI_Request *request;
MPI_Status *status;
char *note;
{
  request_list *rq, *last;
  int flag, size;

  /* look for request */
  rq = requests_head_{{fileno}};
  last = 0;
  while ((rq != NULL) && (rq->request != request)) {
    last = rq;
    rq = rq->next;
  }

  if (!rq) {
#ifdef PRINT_PROBLEMS
    fprintf( stderr, "Request not found in '%s'.\n", note );
#endif
    return;		/* request not found */
  }

  if (status->MPI_TAG != MPI_ANY_TAG) {
    /* if the request was not invalid */

    if (rq->status & RQ_CANCEL) {
      PMPI_Test_cancelled( status, &flag );
      if (flag) return;		/* the request has been cancelled */
    }
    
    if (rq->status & RQ_SEND) {
      TAU_TRACE_SENDMSG(rq->tag, rq->otherParty, rq->size); 
      
      /*
      prof_send( procid_{{fileno}}, rq->otherParty, rq->tag, rq->size, note );
      */
    } else {
      PMPI_Get_count( status, MPI_BYTE, &size );
      TAU_TRACE_RECVMSG( status->MPI_TAG, status->MPI_SOURCE, size); 
      /*
      prof_recv( procid_{{fileno}}, status->MPI_SOURCE, status->MPI_TAG,
		size, note );
      */
    }
  }
  if (last) {
    last->next = rq->next;
  } else {
    requests_head_{{fileno}} = rq->next;
  }
  free( rq );
}

{{fn fn_name MPI_Wait}}
  {{callfn}}
  ProcessWaitTest_{{fileno}}( request, status, "{{fn_name}}" );
{{endfn}}




{{fn fn_name MPI_Waitany}}

  {{callfn}}
  ProcessWaitTest_{{fileno}}( &({{array_of_requests}}[*{{index}}]),
			{{status}}, "{{fn_name}}" );
{{endfn}}



{{fn fn_name MPI_Waitsome}}
  {{vardecl int i}}

  {{callfn}}
  for ({{i}}=0; {{i}} < *{{outcount}}; {{i}}++) {
    ProcessWaitTest_{{fileno}}( &({{array_of_requests}}
			          [{{array_of_indices}}[{{i}}]]),
			        &({{array_of_statuses}}
			          [{{array_of_indices}}[{{i}}]]),
			        "{{fn_name}}" );
  }
{{endfn}}


{{fn fn_name MPI_Waitall}}
  {{vardecl int i}}
/* fprintf( stderr, "{{fn_name}} call on %d\n", procid_{{fileno}} ); */
  {{callfn}}
  for ({{i}}=0; {{i}} < {{count}}; {{i}}++) {
    ProcessWaitTest_{{fileno}}( &({{array_of_requests}}[{{i}}]),
			        &({{array_of_statuses}}[{{i}}]),
			        "{{fn_name}}" );
  }
{{endfn}}


{{fn fn_name MPI_Test}}
  {{callfn}}
  if (*{{flag}}) 
    ProcessWaitTest_{{fileno}}( {{request}}, {{status}}, "{{fn_name}}" );
{{endfn}}

{{fn fn_name MPI_Testany}}
  {{callfn}}
  if (*{{flag}}) 
    ProcessWaitTest_{{fileno}}( &({{array_of_requests}}[*{{index}}]),
			        {{status}}, "{{fn_name}}" );
{{endfn}}

{{fn fn_name MPI_Testsome}}
  {{vardecl int i}}
  {{callfn}}
  for ({{i}}=0; {{i}} < *{{outcount}}; {{i}}++) {
    ProcessWaitTest_{{fileno}}( &({{array_of_requests}}
			          [{{array_of_indices}}[{{i}}]]),
			        &({{array_of_statuses}}
			          [{{array_of_indices}}[{{i}}]]),
			        "{{fn_name}}" );
  }
{{endfn}}


{{fn fn_name MPI_Testall}}
  {{vardecl int i}}
  {{callfn}}
  if (*{{flag}}) {
    for ({{i}}=0; {{i}} < {{count}}; {{i}}++) {
      ProcessWaitTest_{{fileno}}( &({{array_of_requests}}[{{i}}]),
				  &({{array_of_statuses}}[{{i}}]),
				  "{{fn_name}}" );
    }
  }
{{endfn}}

/* NOTE: MPI_Type_count was not implemented in mpich-1.2.0. Remove it from this
   list when it is implemented in libpmpich.a */
{{fnall fn_name MPI_Type_count MPI_Send MPI_Bsend MPI_Ssend MPI_Rsend MPI_Recv MPI_Sendrecv MPI_Sendrecv_replace }}
  TAU_PROFILE_TIMER(tautimer, "{{fn_name}}()",  " ", TAU_MESSAGE); 
  TAU_PROFILE_START(tautimer);
  {{callfn}}
  TAU_PROFILE_STOP(tautimer); 
{{endfnall}}








