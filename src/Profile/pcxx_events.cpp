/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1994                  */
/*  Indiana University  University of Oregon  University of Rennes   */
/*********************************************************************/

/*
 * pcxx_event.c: simple SW monitor routines
 *
 * (c) 1994 Jerry Manic Saftware
 *
 * Version 3.0
 */

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <sys/types.h>
# include <fcntl.h>
# include <signal.h>
# include <unistd.h>
# include <Profile/Profiler.h>


#   if !defined(__ksr__) || defined(UNIPROC)
#     define __private
#   endif

# define PCXX_EVENT_SRC
# define TRACING_ON 1
# include "Profile/pcxx_events.h"
/* # include "TulipTimers.h"
*/


extern "C" time_t time(time_t * t);

__private unsigned long int pcxx_ev_class = PCXX_EC_TRACER | PCXX_EC_TIMER;

# ifndef TRUE
# define FALSE 0
# define TRUE  1
# endif

/* -- event record buffer ------------------------------------ */
__private static PCXX_EV pcxx_buf[PCXX_BUFSIZE];

/* -- pointer to next free element of event record buffer ---- */
__private PCXX_EV *pcxx_ev_ptr = pcxx_buf;

/* -- pointer to last available element of event record buffer */
/* -- need one place for flush event => - 1 ------------------ */
__private PCXX_EV *pcxx_ev_max = pcxx_buf + PCXX_BUFSIZE - 1;

/* -- event trace file descriptor ---------------------------- */
__private static int pcxx_fd;

/* -- initialization status flags ---------------------------- */
__private static int pcxx_num_init[PCXX_MAXPROCS];
/*__private static int pcxx_not_first[PCXX_MAXPROCS]; */


/* -- Use Profiling interface for time -- */
long pcxx_GetUSecLong(void)
{
  return (long) RtsLayer::getUSecD();
}
/* -- write event to buffer only [without overflow check] ---- */
static void pcxx_EventOnly(long int ev,long int par)
{
  pcxx_ev_ptr->ev   = ev;
  pcxx_ev_ptr->ti   = pcxx_GetUSecLong();
  pcxx_ev_ptr->par  = par;
  pcxx_ev_ptr->nid  = PCXX_MYNODE;
  pcxx_ev_ptr->tid  = PCXX_MYTHREAD;
  pcxx_ev_ptr++;
}

/* -- write event buffer to file ----------------------------- */
void pcxx_EvFlush()
{
  static PCXX_EV flush_end = { PCXX_EV_FLUSH_EXIT, 0, 0, 0L };

  if ( pcxx_ev_ptr != pcxx_buf )
  {
    if ( pcxx_ev_class & PCXX_EC_TRACER )
      pcxx_EventOnly (PCXX_EV_FLUSH_ENTER, pcxx_ev_ptr - pcxx_buf);

    write (pcxx_fd, pcxx_buf, (pcxx_ev_ptr - pcxx_buf) * sizeof(PCXX_EV));
    if ( pcxx_ev_class & PCXX_EC_TRACER )
    {
      flush_end.nid = PCXX_MYNODE;
      flush_end.ti  = pcxx_GetUSecLong();
      write (pcxx_fd, &flush_end, sizeof(PCXX_EV));
    }
    pcxx_ev_ptr = pcxx_buf;
  }
}

/* -- signal catching to flush event buffers ----------------- */
# ifndef NSIG
#   define NSIG 32
# endif
static SIGNAL_TYPE (*sighdlr[NSIG])(SIGNAL_ARG_TYPE);

static void wrap_up(int sig)
{
  fprintf (stderr, "signal %d on %d - flushing event buffer...\n", sig, PCXX_MYNODE);
  pcxx_EvFlush ();
  fprintf (stderr, "done.\n");
  if ( sighdlr[sig] != SIG_IGN ) (* sighdlr)(sig);
  exit (1);
}

static void init_wrap_up()
{
# ifdef SIGINT
  sighdlr[SIGINT ] = signal (SIGINT , wrap_up);
# endif
# ifdef SIGQUIT
  sighdlr[SIGQUIT] = signal (SIGQUIT, wrap_up);
# endif
# ifdef SIGILL
  sighdlr[SIGILL ] = signal (SIGILL , wrap_up);
# endif
# ifdef SIGFPE
  sighdlr[SIGFPE ] = signal (SIGFPE , wrap_up);
# endif
# ifdef SIGBUS
  sighdlr[SIGBUS ] = signal (SIGBUS , wrap_up);
# endif
# ifdef SIGTERM
  sighdlr[SIGTERM] = signal (SIGTERM, wrap_up);
# endif
# ifdef SIGABRT
  sighdlr[SIGABRT] = signal (SIGABRT, wrap_up);
# endif
# ifdef SIGSEGV
  sighdlr[SIGSEGV] = signal (SIGSEGV, wrap_up);
# endif
}

/* -- initialize SW monitor and open trace file(s) ----------- */
void pcxx_EvInit(char *name)
{
  char *ptr;
  char *ptr1;
  PCXX_EV *pcxx_iter = pcxx_buf;
  
  static int first_time = 0; /* Sameer's fix to pcxx_num_init[] = 0 - 
				error on mpi in sgi8k*/

  if (first_time == 0)
  { 
    first_time = 1;
# ifdef UNIPROC
    ptr = name;
# else
    ptr = (char *) PCXX_MALLOC (strlen(name) + 1);
    strcpy (ptr, name);
    if ( ptr1 = strchr (ptr, '#') )
    {
      *ptr1++ = PCXX_MYNODE / 1000 + '0';
      *ptr1++ = PCXX_MYNODE % 1000 / 100 + '0';
      *ptr1++ = PCXX_MYNODE % 100 / 10 + '0';
      *ptr1   = PCXX_MYNODE % 10 + '0';
    }
    else
    {
      fprintf (stderr, "%s: trace file name does not contain '####'\n", name);
      exit (1);
    }
# endif

      init_wrap_up ();

      if ((pcxx_fd = open (ptr, O_WRONLY|O_CREAT|O_TRUNC|O_APPEND, 0600)) < 0)
      {
        fprintf (stderr, "pcxx_EvInit[open]: ");
        perror (ptr);
        exit (1);
      }
/* there may be some records in pcxx_ev_ptr already. Make sure that the
   first record has node id set properly */
      if ((&pcxx_buf[0])->ev == PCXX_EV_INIT) 
      { /* first record is init */
	for(pcxx_iter = pcxx_buf; pcxx_iter != pcxx_ev_ptr ; pcxx_iter++)
	{ 
	  pcxx_iter->nid = PCXX_MYNODE;
	}
      }
      else 
      { /* either the first record is blank - in which case we should 
	   put INIT record, or it is an error */
	if (pcxx_ev_ptr == pcxx_buf) 
	{ /* no records in here */
	  pcxx_Event(PCXX_EV_INIT, pcxx_ev_class);
	}
	else 
	{
	  printf("Warning: pcxx_EvInit(): First record is not INIT\n");
	}
      } 

    if ( pcxx_ev_class & PCXX_EC_TRACER )
      pcxx_Event (PCXX_EV_WALL_CLOCK, time((time_t *)0));
  }
  pcxx_num_init[PCXX_MYNODE]++;
  /* pcxx_not_first[PCXX_MYNODE] = TRUE; */
}

/* -- write event to buffer ---------------------------------- */
void pcxx_Event(long int ev, long int par)
{
  static int first_time = 0;
  if (first_time == 0) 
  { 
    if (ev != PCXX_EV_INIT) 
    {
      pcxx_ev_ptr = pcxx_buf;
	/* we need to ensure that INIT is the first event */
      pcxx_ev_ptr->ev = PCXX_EV_INIT; 
      pcxx_ev_ptr->ti   = pcxx_GetUSecLong();
      pcxx_ev_ptr->par  = pcxx_ev_class; /* init event */ 
      /* probably the nodeid is not set yet */
      pcxx_ev_ptr->nid  = PCXX_MYNODE;
      pcxx_ev_ptr->tid  = PCXX_MYTHREAD;
 
      pcxx_ev_ptr++; /* proceed to add this record, nodeid will be set later */
    } 
    first_time = 1;
  } 
        
  pcxx_ev_ptr->ev   = ev;
  pcxx_ev_ptr->ti   = pcxx_GetUSecLong();
  pcxx_ev_ptr->par  = par;
  pcxx_ev_ptr->nid  = PCXX_MYNODE;
  pcxx_ev_ptr->tid  = PCXX_MYTHREAD;
  pcxx_ev_ptr++;

  if ( pcxx_ev_ptr >= pcxx_ev_max ) pcxx_EvFlush (); 
}

/* -- terminate SW tracing ----------------------------------- */
void pcxx_EvClose()
{
  pcxx_num_init[PCXX_MYNODE]--;
  if ( pcxx_num_init[PCXX_MYNODE] == 0 )
  {
    if ( pcxx_ev_class & PCXX_EC_TRACER )
    {
      pcxx_Event (PCXX_EV_CLOSE, 0);
      pcxx_Event (PCXX_EV_WALL_CLOCK, time((time_t *)0));
    }
    pcxx_EvFlush ();
    close (pcxx_fd);
  }
}

/* -- write long event to buffer ----------------------------- */
# define CONT_EV_LEN (sizeof(PCXX_EV) - sizeof(long int))

void pcxx_LongEvent(long int ev, int ln, char *par)
{
  char *buf;
  int i, j, cev_no;

  cev_no = ln / CONT_EV_LEN + ((ln % CONT_EV_LEN) > 0);

  /* -- inlined pcxx_Event (ev, cev_no); ----------------------- */
  pcxx_ev_ptr->ev   = ev;
  pcxx_ev_ptr->ti   = pcxx_GetUSecLong();
  pcxx_ev_ptr->par  = ln;
  pcxx_ev_ptr->nid  = PCXX_MYNODE;
  pcxx_ev_ptr->tid  = PCXX_MYTHREAD;
  pcxx_ev_ptr++;
  if ( pcxx_ev_ptr >= pcxx_ev_max ) pcxx_EvFlush (); 

  /* -- inlined pcxx_Event (ev, cev_no); ----------------------- */
  if ( ln )
  {
    for (i=0; i<cev_no; i++)
    {
      pcxx_ev_ptr->ev = PCXX_EV_CONT_EVENT;
      buf = ((char *) pcxx_ev_ptr) + sizeof(short unsigned int);
      for (j=0; j<CONT_EV_LEN; j++)
        *buf++ = (i*CONT_EV_LEN+j) < ln ? *par++ : '\0';
      pcxx_ev_ptr++;
      if ( pcxx_ev_ptr >= pcxx_ev_max ) pcxx_EvFlush (); 
    }
  }
} 

#if defined( TRACING_ON ) && defined( ARIADNE_SUPPORT )
/* Function to trace the events of Ariadne. */
void pcxx_AriadneTrace (long int event_class, long int event, int pid, int oid, int rwtype, int mtag, int par)
{
/* This routine writes the ariadne events to the trace file */
long int trace_value = 0L; /* the first parameter to be traced */
long int parameter = 0L; /* dummy to shift the par by 32 bits */ 
/* Even for pC++ events we use U as the event rwtype and PCXX_... as the utag */
/* This way we can keep the old format for tracing :
	parameter (32), pid (10), oid (10), rwtype (4) , utag (8) 
for 64 bit long int */ 
  parameter = (long int) par; 

  if (sizeof (long int) == 8) 
  { /* This is true of SGI8K  */

    /* care has to be taken to ensure that mtag is 8 bits long */
  trace_value = (parameter << 32) | (pid << 22) | (oid << 12) | (rwtype << 8) | mtag;

  /*
  printf("Tracing ec = %lx, ev = %lx, pid = %d, oid = %d, mtag = %d, rwtype = %d, parameter = %d, trace_value = %ld\n", event_class, event, pid, oid, mtag, rwtype, parameter, trace_value);	
  */

  PCXX_EVENT(event_class, event, trace_value);
  } 
	
}

#endif  /* defined( TRACING_ON ) && defined( ARIADNE_SUPPORT ) */

/* eof */
