/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1994                  */
/*  Indiana University  University of Oregon  University of Rennes   */
/*********************************************************************/

/*
 * tau_convert.c : convert software event traces to other formats
 *
 * (c) 1994 Jerry Manic Saftware
 *
 * Version 3.0
 * Revision Jan 1998 Author Sameer Shende, (sameer@cs.uoregon.edu)
 */

# include <stdio.h>
# include <stdlib.h>
# include <sys/types.h>
# include <fcntl.h>
# include <unistd.h>

#include <sys/time.h>

# include <string.h>

# define TRACING_ON
# define PCXX_EVENT_SRC

# ifdef __PCXX__
#   include "Profile/pcxx_events_def.h"
# else
#   include "Profile/pcxx_events.h"
# endif
# include "Profile/pcxx_ansi.h"

# ifndef TRUE
#   define FALSE  0
#   define TRUE   1
# endif

# define F_EXISTS    0

# define SEND_EVENT -7
# define RECV_EVENT -8

/* The following three decl apply to -vampir (multi-node, multi-threaded) */
# define TAU_MAX_NODES 32*1024 /* max nodes, within nodes are threads */
static int offset[TAU_MAX_NODES] = {  0 }; /* offset to calculate cpuid */
static int maxtid[TAU_MAX_NODES] = {  0 }; /* max tid encountered for a node */

static struct trcdescr
{
  int     fd;              /* -- input file descriptor                     -- */
  char   *name;            /* -- corresponding file name                   -- */
  int     overflows;       /* -- clock overflows in that trace             -- */
  int     numevent;        /* -- number of event types                     -- */
  int     numproc;         /* -- number of processors                      -- */
  long    numrec;          /* -- number of event records already processed -- */
  unsigned long long  firsttime; /* -- timestamp of first event record           -- */
  unsigned long long  lasttime;  /* -- timestamp of previous event record        -- */

  PCXX_EV  *buffer;   /* -- input buffer                              -- */
  PCXX_EV  *erec;     /* -- current event record                      -- */
  PCXX_EV  *next;     /* -- next available event record in buffer     -- */
  PCXX_EV  *last;     /* -- last event record in buffer               -- */
} intrc;


struct trcrecv
{
  int      fd;        			  /* -- input file descriptor         			-- */
  PCXX_EV  *buffer;   		      /* -- input buffer                  			-- */
  PCXX_EV  *erec;     			  /* -- current event record         			-- */
  PCXX_EV  *prev;       		  /* -- prev available event record in buffer   -- */
  PCXX_EV  *first;    			  /* -- first event record in buffer            -- */
};

PCXX_EV *tmpbuffer; /* for threaded program */


static enum format_t { alog, SDDF, pv, dump, paraver } outFormat = pv;
static enum pvmode_t { user, pvclass, all } pvMode = user;
static int pvCompact = FALSE;
static int pvComm = TRUE;
static int threads = FALSE;
static int pvLongSymbolBugFix = FALSE; /* if symbol name > 200 prepend id */
static int dynamictrace = FALSE;

static char *barrin, *barrout;  /* -- for barrier checking -- */
#ifdef __PCXX__
static int numin, numout;       /* -- for barrier checking -- */
#endif /* __PCXX__ */

static struct stkitem
{
  char *state;  /* -- state name ---- */
  int tag;      /* -- activity tag -- */
}
**statestk,     /* -- state stacks -- */
**stkptr;       /* -- stack pointers -- */

# define STACKSIZE 1024

/* -------------------------------------------------------------------------- */
/* -- event type descriptor handling                                       -- */
/* -------------------------------------------------------------------------- */
typedef struct evdescr {
  int no;
  int id;
  int tag;
  int used;
  char *name;
  char *param;
  char *state;
  struct evdescr *next;
} EVDESCR;

static EVDESCR **evtable;
static int evno;
static int numEvent;
static int numUsedEvent;

int GetNodeId(PCXX_EV *rec);
static void InitEvent (int numev)
{
  int i;

  evtable = (EVDESCR **) malloc (numev * sizeof(EVDESCR));
  for (i=0; i<numev; i++) evtable[i] = (EVDESCR *) NULL;
  evno     = 1;
  numEvent = numev;
  numUsedEvent = 0;
}

static void AddEvent (int id, char *name, char *p, char *state, int tag)
{
  int h;
  char *ptr;
  EVDESCR *newev;

  newev = (EVDESCR *) malloc (sizeof(EVDESCR));
  newev->id   = id;
  newev->tag  = tag;
  newev->no   = evno++;
  newev->used = FALSE;
  newev->name = (char *) malloc (strlen(name) + 1); strcpy (newev->name, name);
  if ( *p && outFormat != pv )
  {
    newev->param = (char *) malloc (strlen(p) + 5);
    if ( outFormat == alog )
      sprintf (newev->param, "%s: %%d", p);
    else
      sprintf (newev->param, "%s", p);
  }
  else
    newev->param = "";
  if ( state[0] != '-' )
  {
    if ( tag > 0 && pvMode == all )
    {
      newev->name[strlen(newev->name)-6] = '\0';
      if ( (ptr = strchr(newev->name, ':')) == NULL )
        ptr = newev->name;
      else
        ptr += 2;
      newev->state = (char *) malloc (strlen(ptr) + 1);
      strcpy (newev->state, ptr);
      newev->name[strlen(newev->name)] = '-';
    }
    else
    {
      newev->state = (char *) malloc (strlen(state) + 1);
      strcpy (newev->state, state);
    }
  }
  else
    newev->state = "";

  h = id % numEvent;
  newev->next = evtable[h];
  evtable[h] = newev;
}

static void AddEventDynamic (int id, char *name, char *p, char *state, int tag)
{
  int h;
  EVDESCR *newev;

#ifdef DEBUG
  printf("Adding id %d name %s par %s state %s tag %d\n",
	id, name, p, state, tag);
#endif /* DEBUG */
  newev = (EVDESCR *) malloc (sizeof(EVDESCR));
  if (newev == (EVDESCR *) NULL) {
    fprintf(stderr,"AddEventDynamic : out of memory malloc returns NULL\n");
    exit(1);
  }
  newev->id   = id;
  newev->tag  = tag;
  newev->no   = evno++;
  newev->used = FALSE;
  newev->name = (char *) malloc (strlen(name) + 10);
  /* add 10 not 1, so if we need to add an id to the name, there is space */

  if (pvLongSymbolBugFix) /* For a long name, prepend its id to its name */
  {
    if (strlen(name) > 200)
      sprintf(newev->name, "\"%d-%s", newev->no, &name[1]);
    else
      strcpy (newev->name, name);
  }
  else
    strcpy (newev->name, name);

  if ( *p && outFormat != pv )
  {
    newev->param = (char *) malloc (strlen(p) + 5);
    if ( outFormat == alog )
      sprintf (newev->param, "%s: %%d", p);
    else
      sprintf (newev->param, "%s", p);
  }
  else
    newev->param = "";
  if ( state[0] != '-' )
  {
    newev->state = (char *) malloc (strlen(state) + 1);
    strcpy (newev->state, state);
  }
  else
    newev->state = "";

  h = id % numEvent;
  newev->next = evtable[h];
  evtable[h] = newev;
}

static EVDESCR *GetEventStruct (int id)
{
  int h;
  EVDESCR *ev;

  h = id % numEvent;
  ev = evtable[h];

  while ( ev )
  {
    if ( ev->id == id ) return (ev);
    ev = ev->next;
  }
  return (0);
}

static int GetEvent (int id)
{
  int h;
  EVDESCR *ev;

  h = id % numEvent;
  ev = evtable[h];

  while ( ev )
  {
    if ( ev->id == id )
    {
      if ( !ev->used ) { ev->used = TRUE; numUsedEvent++; }
      return (ev->no);
    }
    ev = ev->next;
  }
  return (0);
}

static char *GetEventName (int id, int *hasParam)
{
  int h;
  EVDESCR *ev;

  h = id % numEvent;
  ev = evtable[h];

  while ( ev )
  {
    if ( ev->id == id ) { *hasParam = ev->param[0]; return (ev->name); }
    ev = ev->next;
  }
  *hasParam = FALSE;
  return (0);
}

static void PrintEventDescr (FILE *out)
{
  int i;
  EVDESCR *ev;

  for (i=0; i<numEvent; i++)
  {
    ev = evtable[i];
    while ( ev )
    {
      if ( ev->used )
      {
        if ( outFormat == alog )
        {
          fprintf (out, " -9   0 0 %10d 0          0 %s\n", ev->no, ev->name);
          fprintf (out, "-10   0 0 %10d 0          0 %s\n", ev->no, ev->param);

        }
        else if ( outFormat == SDDF )
        {
          fprintf (out, "#%d:\n", ev->no);
          fprintf (out, "%s {\n", ev->name);
          fprintf (out, "\tunsigned long long\t\"Timestamp\";\n");
          fprintf (out, "\tint\t\"Processor Number\";\n");
          fprintf (out, "\tint\t\"Thread Id\";\n");

          if ( ev->param[0] ) fprintf (out, "\tint\t\"%s\";\n", ev->param);
	  /* OLD Code. Why have the name? There's no %s. */
	  /*
          fprintf (out, "};;\n\n", ev->name);
	  */
          fprintf (out, "};;\n\n");
        }
	  else if ( outFormat == pv ){
	  if (! dynamictrace)
	  {
            if ( ev->tag > 0 || ev->tag < -9 )
            {
              ev->name[strlen(ev->name)-6] = '\0';
              /*fprintf (out, "SYMBOL %s %d %s\n", ev->state, ev->tag, ev->name);*/
              fprintf (out, "SYMBOL %s %d %s\n", ev->state, ev->no, ev->name);
              ev->name[strlen(ev->name)] = '-';
            }
	  }
	  else /* don't do anything special for dynamic trace */
	  {
              fprintf (out, "SYMBOL %s %d %s\n", ev->state, ev->no, ev->name);

	  }

        }
      }
      ev = ev->next;
    }
  }
}

/* -------------------------------------------------------------------------- */
/* -- is it an INIT event ? 						   -- */
/* -------------------------------------------------------------------------- */

int isInitEvent(PCXX_EV *erecord)
{
  char *eventName;
  int hasParam;

  eventName = GetEventName(erecord->ev, &hasParam);

  if (dynamictrace)
  {
    if (strcmp(eventName, "\"EV_INIT\"") == 0)
    {
      return TRUE;
    }
    else
    {
      return FALSE;
    }
  }
  else
  { /* old traces use events determined by fixed #define nos. */
    if  ( (erecord->ev == PCXX_EV_INIT) || (erecord->ev == PCXX_EV_INITM) )
    {
      return TRUE;
    }
    else
    {
      return FALSE;
    }
  } /* old format */
}



/* -------------------------------------------------------------------------- */
/* -- get today's date                                                     -- */
/* -------------------------------------------------------------------------- */
# include <time.h>

static char tibuf[12];
static char *Months[12] =
{"Jan","Feb","Mar","Apr","May","Jun","jul","Aug","Sep","Oct","Nov","Dec"};

static char *Today (void)
{
  time_t t;
  struct tm *tm;

  t = time ((time_t *) 0);
  tm = localtime (&t);
  sprintf (tibuf, "%s-%02d-%02d", Months[tm->tm_mon], tm->tm_mday, 1900+tm->tm_year);
  return (tibuf);
}


/* -------------------------------------------------------------------------- */
/* -- input buffer handling                                                -- */
/* -------------------------------------------------------------------------- */
# define INMAX    BUFSIZ /* records */

static PCXX_EV *get_next_rec (struct trcdescr *tdes)
{
  long no;

  if ( (tdes->last == NULL) || (tdes->next > tdes->last) )
  {
    /* -- input buffer empty: read new records from file -------------------- */
    if ( (no = read (tdes->fd, tdes->buffer, INMAX * sizeof(PCXX_EV)))
         != (INMAX * sizeof(PCXX_EV)) )
    {
      if ( no == 0 )
      {
        /* -- no more event record: ----------------------------------------- */
	/* when this is called by GetMatchingRecv it shouldn't close the file. */
        /*
        close (tdes->fd);
        tdes->fd = -1;
	*/
        return ((PCXX_EV *) NULL);
      }
      else if ( (no % sizeof(PCXX_EV)) != 0 )
      {
        /* -- read error: --------------------------------------------------- */
        fprintf (stderr, "%s: read error\n", tdes->name);
        exit (1);
      }
    }

    /* -- we got some event records ----------------------------------------- */
    tdes->next = tdes->buffer;
    tdes->last = tdes->buffer + (no / sizeof(PCXX_EV)) - 1;
  }
  return (tdes->erec = tdes->next++);
}

static PCXX_EV *get_prev_rec (struct trcrecv *tdes)
{
 /* Before calling this the first time set first properly. */
long no;
off_t last_position;

  last_position = lseek(tdes->fd, 0, SEEK_CUR);

/* We reuse last and next to actually mean first and prev respectively */
/* i.e., before calling this the first time set tdes->last = tdes->buffer */
  /* if prev < first, go fetch more records */
/* to debug: print each record */
  if (( last_position == 0) || (tdes->prev < tdes->first))
  {
    /* move the pointer 2*INMAX*sizeof(PCXX_EV) earlier */
    last_position -= 2*INMAX*sizeof(PCXX_EV);
#ifdef DEBUG
    printf("last_position = %d\n", last_position);
#endif /* DEBUG */
    if (last_position < 0) return NULL;
    lseek(tdes->fd, last_position, SEEK_SET);
    /* -- input buffer empty: read new records from file -------------------- */
    if ( (no = read (tdes->fd, tdes->buffer, INMAX * sizeof(PCXX_EV)))
         != (INMAX * sizeof(PCXX_EV)) )
    {
      if ( no == 0 )
      {
        /* -- no more event record: ----------------------------------------- */
	/* when this is called by GetMatchingRecv it shouldn't close the file. */
	/*
        close (tdes->fd);
        tdes->fd = -1;
	*/
        return ((PCXX_EV *) NULL);
      }
      else if ( (no % sizeof(PCXX_EV)) != 0 )
      {
        /* -- read error: --------------------------------------------------- */
        fprintf (stderr, "read error in get_prev_rec\n");
        exit (1);
      }
    }

    /* -- we got some event records ----------------------------------------- */
    tdes->prev =  tdes->buffer + (no / sizeof(PCXX_EV)) - 1;
    tdes->first = tdes->buffer ;
  }
  return (tdes->erec = tdes->prev--);

}

int GetNodeId(PCXX_EV *rec)
{
  if (threads)
  {
    /* OLD
    return rec->tid;
    */
    return offset[rec->nid] + rec->tid;
	/* CPUID ranges from 0..N-1: N is sum(threads on all nodes ) */
  }
  else
    return rec->nid;
}

long long GetNextStateBurst(struct trcdescr trcdes, int myNid, int myTid,int hasParam){
	off_t last_position;
	PCXX_EV *curr_rec;
	int currNid, currTid;
	trcdes.buffer = tmpbuffer;
	last_position = lseek(trcdes.fd, 0, SEEK_CUR);
	if(last_position < 0){
		perror("lseek ERROR: GetNextStateBurst() routing that get next state burst");
		exit(1);
	}
	  while (( curr_rec = get_next_rec(&trcdes)) != NULL){
	    currNid = GetNodeId(curr_rec);
	    currTid = curr_rec->tid;
	    if((curr_rec->par == 0) && (currNid == myNid) && (currTid == myTid)){
	      return curr_rec->ti-intrc.firsttime;
	    }
	    else if((strcmp((GetEventName(curr_rec->ev,&hasParam)), "\"MPI_Send()  \"") ==0) && (currNid == myNid) && (currTid == myTid)){
	      return curr_rec->ti-intrc.firsttime;
	    }
	    else if((strcmp((GetEventName(curr_rec->ev,&hasParam)), "\"MPI_Recv()  \"") ==0) && (currNid == myNid) && (currTid == myTid)){
	      return curr_rec->ti-intrc.firsttime;
	    }
	  }
	  return -1;
}


int GetMatchingRecv(struct trcdescr trcdes, int msgtag,
    int myid, int otherid, int msglen, int *other_tid, int *other_nodeid)
/* parameters: trcdes:    input trace file descriptor.
	       msgtag:     message tag that we're searching for
	       msglen: 	   message length that we're searching for
	       myid:       id encoded in the parameter. Not +1.
	       otherid:    rank of the other process
	       other_tid:  thread id of the matching ipc call
	       other_nodeid: node id of the matching ipc call
*/
{
  off_t last_position;
  PCXX_EV *curr_rec;
  EVDESCR *curr_ev;
  int curr_tag, curr_len, curr_nid;
  trcdes.buffer    = tmpbuffer;
#ifdef DEBUG
  printf("GetMatchingRecv: SEND, tag=%d, len=%d, myid=%d, otherid=%d\n",
	msgtag, msglen, myid, otherid);
#endif /* DEBUG */


  /* get the current position from the trace file descriptor */
  last_position = lseek(trcdes.fd, 0, SEEK_CUR);
  if (last_position < 0) {
    perror("lseek ERROR: GetMatchingRecv() routine that matches sends/receives");
    exit(1);
  }

#ifdef DEBUG
  printf("last_position = %d\n", last_position);
#endif /* DEBUG */
  /* now get the records one by one */
  /* We've made a copy of intrc in the trcdes descriptor. So, even if this
     changes the state of intrc remains the same. We need to do an lseek
     with the original position, of course. */
  while (( curr_rec = get_next_rec(&trcdes)) != NULL)
  {
    /* Get the event type for this record */
    curr_ev = GetEventStruct (curr_rec->ev);

   /* Find the matching send and receive */
   /* is the current record of the complementary IPC type? */
   if (curr_ev->tag == RECV_EVENT)
   {
     /* possible match */
     curr_tag = (curr_rec->par>>16) & 0x000000FF;
     curr_len = curr_rec->par & 0x0000FFFF;
     curr_nid = curr_rec->nid;
#ifdef DEBUG
     printf("Possible match... tag=%d, len=%d, nid=%d\n", curr_tag, curr_len, curr_nid);
#endif /* DEBUG */
     if ((curr_tag == msgtag) && (curr_len == msglen) && (curr_nid == otherid ))
     {
       *other_tid = curr_rec->tid;
       *other_nodeid = curr_rec->nid;
#ifdef DEBUG
       printf("PERFECT MATCH! other tid = %d, nid = %d\n",
	 *other_tid, *other_nodeid);
#endif /* DEBUG */
       /* Reset trace file */
       lseek(trcdes.fd, last_position, SEEK_SET);
       return 1;
     }
     /* This only applies to Send! */
   }
  }
  /* EOF : didn't find the matching send. Reset and leave */
#ifdef DEBUG
  printf("Didn't find matching ipc...\n");
#endif /* DEBUG */
  lseek(trcdes.fd, last_position, SEEK_SET);
  return 0;

}

int GetMatchingSend(struct trcdescr trcdes, int msgtag,
    int myid, int otherid, int msglen, int *other_tid, int *other_nodeid)
/* parameters: trcdes:    input trace file descriptor.
               msgtag:     message tag that we're searching for
               msglen:     message length that we're searching for
               myid:       id encoded in the parameter. Not +1.
               otherid:    rank of the other process
               other_tid:  thread id of the matching ipc call
               other_nodeid: node id of the matching ipc call
*/
{
  off_t last_position;
  PCXX_EV *curr_rec;
  EVDESCR *curr_ev;
  int curr_tag, curr_len, curr_nid;
  struct trcrecv rcvdes;

  rcvdes.buffer    = tmpbuffer;
  rcvdes.erec	   = trcdes.erec;
  rcvdes.fd 	   = trcdes.fd;
  rcvdes.first 	   = trcdes.buffer;
  rcvdes.prev	   = trcdes.erec - 1;
  /* initialize the first and prev pointers */
#ifdef DEBUG
  printf("GetMatchingSend: RECV, tag=%d, len=%d, myid=%d, otherid=%d\n",
        msgtag, msglen, myid, otherid);
#endif /* DEBUG */


  /* get the current position from the trace file descriptor */
  last_position = lseek(rcvdes.fd, 0, SEEK_CUR);
  if (last_position < 0) {
    perror("lseek ERROR: GetMatchingSend() routine that matches sends/receives");
    exit(1);
  }

#ifdef DEBUG
  printf("last_position = %d\n", last_position);
#endif /* DEBUG */
  /* now get the records one by one */
  /* We've made a copy of intrc in the trcdes descriptor. So, even if this
     changes the state of intrc remains the same. We need to do an lseek
     with the original position, of course. */
  while (( curr_rec = get_prev_rec(&rcvdes)) != NULL)
  {
    /* Get the event type for this record */
    curr_ev = GetEventStruct (curr_rec->ev);

   /* Find the matching send and receive */
   /* is the current record of the complementary IPC type? */
   if (curr_ev->tag == SEND_EVENT)
   {
     /* possible match */
     curr_tag = (curr_rec->par>>16) & 0x000000FF;
     curr_len = curr_rec->par & 0x0000FFFF;
     curr_nid = curr_rec->nid;
#ifdef DEBUG
     printf("Possible match... tag=%d, len=%d, nid=%d\n", curr_tag, curr_len, curr_nid);
#endif /* DEBUG */

     if ((curr_tag == msgtag) && (curr_len == msglen) && (curr_nid == otherid ))
     {
       *other_tid = curr_rec->tid;
       *other_nodeid = curr_rec->nid;
#ifdef DEBUG
       printf("PERFECT MATCH! other tid = %d, nid = %d\n",
         *other_tid, *other_nodeid);
#endif /* DEBUG */
       /* Reset trace file */
       lseek(rcvdes.fd, last_position, SEEK_SET);
       return 1;
     }
     /* This only applies to Send! */
   }
  }
  /* EOF : didn't find the matching send. Reset and leave */
#ifdef DEBUG
  printf("Didn't find matching ipc...\n");
#endif /* DEBUG */
  lseek(trcdes.fd, last_position, SEEK_SET);
  return 0;
}
/* -------------------------------------------------------------------------- */
/* -- PCXX_CONVERT MAIN PROGRAM --------------------------------------------- */
/* -------------------------------------------------------------------------- */

# define LINEMAX 64*1024

int main (int argc, char *argv[])
{
  FILE *outfp, *inev,*pcffp;
  PCXX_EV *erec;
  int firstScan = TRUE;
  int i,j,k,l;
  int nodeId, totalnodes = 0;
  int num;
  int tag;
  int myid, otherid, msglen, msgtag;
  int other_tid, other_nodeid; /* for threaded programs */
  int hasParam;
  int fileIdx;
  int *prvPCF;
  int numproc = 0;
  char name[LINEMAX], state[80], param[80], linebuf[LINEMAX];
  char traceflag[32];
  char *inFile, *edfFile, *outFile, *ptr,*pcfFile;
  EVDESCR *ev;
  struct trcrecv rcvdes;

  /* ------------------------------------------------------------------------ */
  /* -- scan command line arguments                                        -- */
  /* ------------------------------------------------------------------------ */
  if ( strcmp (argv[0]+strlen(argv[0])-4, "sddf") == 0 )
    outFormat = SDDF;
  else if ( strcmp (argv[0]+strlen(argv[0])-4, "alog") == 0 )
    outFormat = alog;
  else if ( strcmp (argv[0]+strlen(argv[0])-4, "dump") == 0 )
    outFormat = dump;
  else if ( strcmp (argv[0]+strlen(argv[0])-2, "pv") == 0 )
    outFormat = pv;
  else if ( strcmp (argv[0]+strlen(argv[0])-7, "paraver") == 0){
    outFormat = paraver;
  }
  else if ( strcmp (argv[0]+strlen(argv[0])-6, "vampir") == 0 )
    threads = TRUE;

  fileIdx = 1;

  if ( argc < 3 )
  {
    fprintf (stderr, "usage: %s [-alog | -SDDF | -dump | -paraver |", argv[0]);
    fprintf (stderr, " -pv | -vampir [-longsymbolbugfix] [-compact] [-user|-class|-all] [-nocomm]]");
    fprintf (stderr, " inputtrc edffile [outputtrc]\n");
    fprintf (stderr, " Note: -vampir option assumes multiple threads/node\n");
    fprintf (stderr, " Note: -paraver option with -t option should be used for multiple threads\n");
    exit (1);
  }
  else if ( strcmp (argv[1], "-alog") == 0 || strcmp (argv[1], "-a") == 0 )
  {
    outFormat = alog;
    fileIdx = 2;
  }
  else if ( strcmp (argv[1], "-SDDF") == 0 || strcmp (argv[1], "-S") == 0 )
  {
    outFormat = SDDF;
    fileIdx = 2;
  }
  else if ( strcmp (argv[1], "-pv") == 0 || strcmp (argv[1], "-p") == 0 )
  {
    outFormat = pv;
    i = 2;
    while ( argv[i][0] == '-' )
    {
      if ( strcmp (argv[i], "-longsymbolbugfix") == 0 )
        pvLongSymbolBugFix = TRUE;
      else if ( strcmp (argv[i], "-compact") == 0 )
        pvCompact = TRUE;
      else if ( strcmp (argv[i], "-user") == 0 )
        pvMode = user;
      else if ( strcmp (argv[i], "-class") == 0 )
        pvMode = pvclass;
      else if ( strcmp (argv[i], "-all") == 0 )
        pvMode = all;
      else if ( strcmp (argv[i], "-nocomm") == 0 )
        pvComm = FALSE;
      else
        break;
      i++;
    }
    fileIdx = i;
  }
  else if ( strcmp (argv[1], "-dump") == 0 || strcmp (argv[1], "-d") == 0 )
  {
    outFormat = dump;
    fileIdx = 2;
  }
  else if ( strcmp (argv[1], "-paraver") == 0 || strcmp (argv[1], "-prv") == 0 ) {
    outFormat = paraver;
    fileIdx = 2;
  }

  else if ( strcmp (argv[1], "-vampir") == 0 || strcmp (argv[1], "-v") == 0)
  {
    outFormat = pv;
    threads   = TRUE;
#ifdef DEBUG
    printf("Using Vampir with threads");
#endif
    i = 2;
    while ( argv[i][0] == '-' )
    {
      if ( strcmp (argv[i], "-longsymbolbugfix") == 0 )
        pvLongSymbolBugFix = TRUE;
      else if ( strcmp (argv[i], "-compact") == 0 )
        pvCompact = TRUE;
      else if ( strcmp (argv[i], "-user") == 0 )
        pvMode = user;
      else if ( strcmp (argv[i], "-class") == 0 )
        pvMode = pvclass;
      else if ( strcmp (argv[i], "-all") == 0 )
        pvMode = all;
      else if ( strcmp (argv[i], "-nocomm") == 0 )
        pvComm = FALSE;
      else
        break;
      i++;
    }
    fileIdx = i;
  }

#ifdef DEBUG
  if (pvLongSymbolBugFix)
    printf("LONG_SYMBOL_BUG_FIX is in effect ! \n");
#endif /* DEBUG */


  inFile  = argv[fileIdx];
  edfFile = argv[fileIdx+1];
  if ( (fileIdx+2) == argc )
    outFile = (char *) NULL;
  else
    outFile = argv[fileIdx+2];

  /* ------------------------------------------------------------------------ */
  /* -- open input trace                                                   -- */
  /* ------------------------------------------------------------------------ */
  if ( (intrc.fd = open (inFile, O_RDONLY)) < 0 )
  {
    perror (inFile);
    exit (1);
  }
  else
  {
    intrc.name      = inFile;
    intrc.buffer    = (PCXX_EV *) malloc (INMAX * sizeof(PCXX_EV));
    intrc.erec      = (PCXX_EV *) NULL;
    intrc.next      = (PCXX_EV *) NULL;
    intrc.last      = (PCXX_EV *) NULL;
    intrc.overflows = 0;

    if(threads)
      tmpbuffer = (PCXX_EV *) malloc (INMAX * sizeof(PCXX_EV));

    /* -- read first event record ------------------------------------------- */
    if ( (erec = get_next_rec (&intrc)) == NULL )
    {
      /* -- no event record: ------------------------------------------------ */
      fprintf (stderr, "%s: warning: trace empty - ignored\n",
               intrc.name);
      intrc.numrec = 0L;
    }

    /* -- check first event record ------------------------------------------ */
/* Can't check this as isInitEvent requires event .edf file to be read in */
/* and at this stage the edf file has not been opened */
/*
    else if (!isInitEvent(erec))
    {
      fprintf (stderr, "%s: no valid event trace\n", intrc.name);
      exit (1);
    }
*/
    else
    {
      intrc.numrec    = 1L;
      if (threads)
      { /* Don't call GetNodeId here as it uses offset, maxthreads */
	intrc.numproc = erec->nid;
      }
      else
      { /* No threads */
        intrc.numproc   = GetNodeId(erec);
      }
      intrc.firsttime = erec->ti;
      intrc.lasttime  = erec->ti;
    }
  }

  /* ------------------------------------------------------------------------ */
  /* -- open output trace file                                             -- */
  /* ------------------------------------------------------------------------ */
  if ( outFile )
  {
    if ( access (outFile, F_EXISTS) == 0  && isatty(2) )
    {
      fprintf (stderr, "%s exists; override [y]? ", outFile);
      if ( getchar() == 'n' ) exit (1);
    }

    if ( (outfp = fopen (outFile, "w")) == NULL )
    {
      perror (outFile);
      exit (1);
    }
  }
  else
    outfp = stdout;

  /* ------------------------------------------------------------------------ */
  /* -- initialize event description database ------------------------------- */
  /* ------------------------------------------------------------------------ */

  /* -- read event descriptor file and write event descriptor part header --- */
  if ( (inev = fopen (edfFile, "r")) == NULL )
  {
    perror (edfFile);
    exit (1);
  }
  fgets (linebuf, LINEMAX, inev);
  sscanf (linebuf, "%d %s", &intrc.numevent, traceflag);

  if (strcmp(traceflag, "dynamic_trace_events") == 0)
  {
    dynamictrace = TRUE;
  }

  InitEvent (intrc.numevent);

  for (i=0; i<intrc.numevent; i++)
  {
    fgets (linebuf, LINEMAX, inev);
    if ( (linebuf[0] == '\n') || (linebuf[0] == '#') )
    {
      /* -- skip empty and comment lines -- */
      i--;
      continue;
    }

    num = -1;
    name[0]  = '\0';
    param[0] = '\0';
    if (dynamictrace) /* get name in quotes */
    {
      sscanf (linebuf, "%d %s %d", &num, state, &tag);
#if DEBUG
      printf("Got num %d state %s tag %d\n", num, state, tag);
#endif /* DEBUG */
      for(j=0; linebuf[j] !='"'; j++)
	;
      name[0] = linebuf[j];
      j++;
      /* skip over till name begins */
      for (k=j; linebuf[k] != '"'; k++)
      {
	name[k-j+1] = linebuf[k];
      }
      name[k-j+1] = '"';
      name[k-j+2] = '\0'; /* terminate name */

      strcpy(param, &linebuf[k+2]);

#ifdef DEBUG
      printf(" Got name=%s param=%s\n", name, param);
#endif /* DEBUG */

    }
    else
    {
      sscanf (linebuf, "%d %s %d %s %s", &num, state, &tag, name, param);
    }

    if ( (num < 0) || !*name )
    {
      fprintf (stderr, "%s: blurb in line %d\n", edfFile, i+2);
      exit (1);
    }

    if (dynamictrace)
    {
      AddEventDynamic (num, name, param, state, tag);
    }
    else
    {
      AddEvent (num, name, param, state, tag);
    }
  }
  fclose (inev);

#ifdef DEBUG
  printf("After closing edf file ... \n");
#endif /* DEBUG */

  /* ------------------------------------------------------------------------ */
  /* -- skip through trace file to determine trace parameters --------------- */
  /* ------------------------------------------------------------------------ */
  do
  {
    if ( (i = GetEvent (erec->ev)) == 0 )
    {
      fprintf (stderr, "%s: unknown event type %d in event record %ld\n",
               intrc.name, erec->ev, intrc.numrec);
    }
    else if ( isInitEvent(erec))
    {
      numproc++;
    }

    if ( (erec = get_next_rec (&intrc)) == NULL )
      break;
    else
    {
      intrc.numrec++;

      /* -- check clock overflow ---------------------------------------- */
      if ( erec->ti < intrc.lasttime ) intrc.overflows++;
      intrc.lasttime = erec->ti;


      /* -- check thread id -------------------------------------------- */
      if (threads)
      {
        /* -- check node id -------------------------------------------- */
        if ( erec->nid > totalnodes ) totalnodes = erec->nid;

        /* totalnodes has node id in the range 0..N-1 */

	if ( maxtid[erec->nid] < erec->tid ) maxtid[erec->nid] = erec->tid;
	/* Update the max thread id vector for this node for each record */
 	/* printf("maxtid[%d] = %d\n", totalnodes, maxtid[totalnodes]); */
      }
      else
      { /* no threads */
        if ( GetNodeId(erec) > intrc.numproc ) intrc.numproc = GetNodeId(erec);
      }

    }
  }
  while ( erec != NULL );


#ifdef DEBUG
  printf("After parsing the trace file...\n");
#endif /* DEBUG */

  if (threads)
  { /* We've gone through the whole trace, now make the offset vector */
    offset[0] = 0;
    for(nodeId = 1; nodeId <= totalnodes; nodeId++)
    {
      offset[nodeId] = offset[nodeId - 1] + maxtid[nodeId - 1] + 1;
      /* printf("offset[%d] = %d\n", nodeId, offset[nodeId]); */
      /* So if node 0 has 2 threads, 1 has 3 and 2 has 3, then
	 maxtid[0] = 1, maxtid[1] = 2, maxtid[2] = 2 and
	 offset[0] = 0, offset[1] = 2, offset[2] = 5 */
    }
    intrc.numproc = offset[totalnodes] + maxtid[totalnodes];
  }
  /* printf("Done with offset! numproc = %d\n", intrc.numproc); */

  /* ------------------------------------------------------------------------ */
  /* -- write trace file header --------------------------------------------- */
  /* ------------------------------------------------------------------------ */

  /* -- write fixed header -------------------------------------------------- */
  if ( outFormat == alog )
  {
    fprintf (outfp, " -2   0 0 %10ld 0          0\n", intrc.numrec);
    fprintf (outfp, " -3   0 0 %10d 0          0\n", intrc.numproc+1);
    fprintf (outfp, " -4   0 0          1 0          0\n");
    fprintf (outfp, " -5   0 0 %10d 0          0\n", numUsedEvent);
    fprintf (outfp, " -6   0 0          0 0 %10llu\n", intrc.firsttime);
    fprintf (outfp, " -7   0 0          0 0 %10llu\n", intrc.lasttime);
    fprintf (outfp, " -8   0 0 %10d 0          0\n", intrc.overflows+1);
    fprintf (outfp, " -1   0 0          0 0          0 tau_convert -alog %s\n",
             Today());
    fprintf (outfp, "-11   0 0          0 0 4294967295\n");
  }
  else if ( outFormat == SDDF )
  {
    fprintf (outfp, "/*\n");
    fprintf (outfp, " * \"creation program\" \"tau_convert -SDDF\"\n");
    fprintf (outfp, " * \"creation date\" \"%s\"\n", Today());
    fprintf (outfp, " * \"number records\" \"%ld\"\n", intrc.numrec);
    fprintf (outfp, " * \"number processors\" \"%d\"\n", intrc.numproc+1);
    fprintf (outfp, " * \"first timestamp\" \"%llu\"\n", intrc.firsttime);
    fprintf (outfp, " * \"last timestamp\" \"%llu\"\n", intrc.lasttime);
    fprintf (outfp, " */\n\n");
  }
  else if ( outFormat == pv )
  {
/* The time in the record comes from GetUSecD() converted to long - so it
   should be in microseconds. */
/* old
    fprintf (outfp, "CLKPERIOD 0.1000E-06\n");
*/
    fprintf (outfp, "CLKPERIOD 1.0E-06\n");
    if (threads)
    {
/* PUT CPUS HERE !*/
      fprintf(outfp,"NCPUS");
      for(l=0; l < totalnodes+1; l++)
      {
	fprintf(outfp," %d", maxtid[l]+1);
      }
      fprintf(outfp,"\n");
      fprintf(outfp, "CPUNAMES");
      for(l=0; l < totalnodes+1; l++)
      {
	fprintf(outfp," \"Node %d\"", l);
      }
      fprintf(outfp,"\n");
    }
    else
    { /* just report number of processors */
      fprintf (outfp, "NCPUS %d\n", intrc.numproc+1);
    }
    fprintf (outfp, "C CREATION PROGRAM tau_convert -pv\n");
    fprintf (outfp, "C CREATION DATE %s\n", Today());
    fprintf (outfp, "C NUMBER RECORDS %ld\n", intrc.numrec);
    fprintf (outfp, "C FIRST TIMESTAMP %llu\n", intrc.firsttime);
    fprintf (outfp, "C LAST TIMESTAMP %llu\n", intrc.lasttime);
    fprintf (outfp, "C\n");

    /* -- initialize state stacks -- */
    statestk = (struct stkitem **)
               malloc ((intrc.numproc+1)*sizeof(struct stkitem *));
    stkptr   = (struct stkitem **)
               malloc ((intrc.numproc+1)*sizeof(struct stkitem *));


    for (i=0; i<=intrc.numproc; i++)
    {
      stkptr[i] = statestk[i] = (struct stkitem *)
                                malloc (STACKSIZE * sizeof(struct stkitem));
      stkptr[i]->state = "IDLE";
      stkptr[i]->tag = -99;
    }
  }
  else if ( outFormat == dump )
  {
    fprintf (outfp, "#  creation program: tau_convert -dump\n");
    fprintf (outfp, "#     creation date: %s\n", Today());
    fprintf (outfp, "#    number records: %ld\n", intrc.numrec);
    fprintf (outfp, "# number processors: %d\n", numproc);
    fprintf (outfp, "# max processor num: %d\n", intrc.numproc);
    fprintf (outfp, "#   first timestamp: %llu\n", intrc.firsttime);
    fprintf (outfp, "#    last timestamp: %llu\n\n", intrc.lasttime);

    fprintf (outfp, "#=NO= =======================EVENT==");
    fprintf (outfp, " ==TIME [us]= =NODE= =THRD= ==PARAMETER=\n");
  }
  else if ( outFormat == paraver )
    {
      /* The code from libseqparaver.c is used here */
      char date[50];
      struct timeval tp;
      struct timezone tzp;
      time_t clock;
      struct tm *ptm;
      int numThreads = 0;
      int i;

      gettimeofday (&tp, &tzp);
      clock = tp.tv_sec;
      ptm = localtime (&clock);
      strftime (date, 50, "%d/%m/%y at %H:%M", ptm);

      for(i = 0; i < numproc; i++){
	if(maxtid[i] >= 0){
	  numThreads = numThreads + (maxtid[i] + 1);
	}
      }

      
      /* create the first part of the pcf File
         change the file extension of outFile from prv to pcf
      */
      
      if(! outFile){
	outfp = stdout;
	pcfFile = (char*)malloc(6 * sizeof(char));
	pcfFile = (strcat(pcfFile,"config.pcf"));
      }
      else{
	if(strlen(outFile) < 5){
	  printf("Outfile must be of form *.prv\n");
	  exit(1);
	}
     

	pcfFile = (char*)malloc((strlen(outFile)) * sizeof (char));  
	pcfFile = (strcpy(pcfFile,outFile));
	pcfFile[strlen(outFile) - 3] = '\0';
	pcfFile = (strcat(pcfFile,"pcf"));
      }  
      if ( pcfFile ){
	if ( access ("pcfFile", F_EXISTS) == 0  && isatty(2) ){
	  fprintf (stderr, "%s exists; override [y]? ", pcfFile);
	  if ( getchar() == 'n' ) exit (1);
	}
	
	if ( (pcffp = fopen (pcfFile, "w")) == NULL ){
	  perror (pcfFile);
	  exit (1);
	}
      }
      else{
	pcffp = stdout;
      }
      fprintf(pcffp,"DEFAULT_OPTIONS\n\n");
      fprintf(pcffp,"LEVEL               THREAD\n");
      fprintf(pcffp,"UNITS               MICROSEC\n");
      fprintf(pcffp,"LOOK_BACK           100\n");
      fprintf(pcffp,"SPEED               1\n");
      fprintf(pcffp,"FLAG_ICONS          ENABLED\n");
      fprintf(pcffp,"NUM_OF_STATE_COLORS 5\n");
      fprintf(pcffp,"YMAX_SCALE          20\n\n\n");
      fprintf(pcffp,"DEFAULT_SEMANTIC\n\n");
      fprintf(pcffp,"COMPOSE1_FUNC       Stacked Val\n");
      fprintf(pcffp,"THREAD_FUNC         Last Evt Val\n\n\n");
      fprintf(pcffp,"STATES\n");
      fprintf(pcffp,"0    IDLE\n");
      fprintf(pcffp,"1    RUNNING\n");
      fprintf(pcffp,"2    STOPPED\n");
      fprintf(pcffp,"3    WAITING FOR MESSAGE\n");
      fprintf(pcffp,"4    NOT USED\n");
      fprintf(pcffp,"5    NOT USED\n");
      fprintf(pcffp,"6    NOT USED\n");
      fprintf(pcffp,"7    NOT USED\n");
      fprintf(pcffp,"8    NOT USED\n");
      fprintf(pcffp,"9    SEND OVERHEAD\n");
      fprintf(pcffp,"4    NOT USED\n");
      fprintf(pcffp,"10    NOT USED\n");
      fprintf(pcffp,"11    NOT USED\n");
      fprintf(pcffp,"12    NOT USED\n");
      fprintf(pcffp,"13    NOT USED\n\n");
      fprintf(pcffp,"STATES_COLOR\n");
      fprintf(pcffp,"0    {117,195,255}\n");
      fprintf(pcffp,"1    {0,0,255}\n");
      fprintf(pcffp,"2    {255,255,255}\n");
      fprintf(pcffp,"3    {0,255,0}\n");
      fprintf(pcffp,"9    {255,0,0}\n\n");
      fprintf(pcffp,"EVENT_TYPE\n");
      fprintf(pcffp,"0    5    METHOD entry/exit:\n");
      fprintf(pcffp,"VALUES\n");
      
    
      prvPCF = (int*)malloc(intrc.numevent * sizeof(int));
      for(i = 0; i < intrc.numevent; i++){
	prvPCF[i] = 0;
      }
      /*
   
      int size = (numproc * 4);
      char *taskList = (char*)malloc(size * sizeof (char));
      char *tempList = (char*)malloc(size * sizeof (char));
      taskList[0] = '\0';


      for(int i = 0; i < numproc; i++){
	if(numproc - i != 1){
	  sprintf(tempList,"%d:%d,",maxtid[i]+1,i+1);
	  taskList = strcat(taskList,tempList);
	}
	else{
	  sprintf(tempList,"%d:%d",maxtid[i]+1,i+1);
	  taskList = strcat(taskList,tempList);
	}
      */

    
      fprintf (outfp, "#Paraver (%s):%d:1:1:1(%d:1)\n", date,
	       (intrc.lasttime - intrc.firsttime),numThreads);
    }



  PrintEventDescr (outfp);


  /* ------------------------------------------------------------------------ */
  /* -- re-open input trace                                                -- */
  /* ------------------------------------------------------------------------ */
  if ( (intrc.fd = open (inFile, O_RDONLY)) < 0 )
  {
    perror (inFile);
    exit (1);
  }
  else
  {
    intrc.erec      = (PCXX_EV *) NULL;
    intrc.next      = (PCXX_EV *) NULL;
    intrc.last      = (PCXX_EV *) NULL;
    intrc.overflows = 0;
    intrc.numrec    = 1;

    erec = get_next_rec (&intrc);
    intrc.firsttime = erec->ti;
    intrc.lasttime  = erec->ti;
  }

  /* ------------------------------------------------------------------------ */
  /* -- initialize barrier check variables ---------------------------------- */
  /* ------------------------------------------------------------------------ */
#ifdef __PCXX__
  numin   = 0;
  numout  = numproc;
#endif /* __PCXX__ */
  barrin  = (char *) malloc (intrc.numproc + 1);
  barrout = (char *) malloc (intrc.numproc + 1);
  for (i=0; i<=intrc.numproc; i++)
  {
    barrin[i]  = FALSE;
    barrout[i] = TRUE;
  }

  /* ------------------------------------------------------------------------ */
  /* -- convert trace file -------------------------------------------------- */
  /* ------------------------------------------------------------------------ */
  do
  {
# ifdef __PCXX__
    /* -- check barrier order ----------------------------------------------- */
    if ( erec->ev == PCXX_BARRIER_ENTER )
    {
      if ( !barrout[GetNodeId(erec)] )
      {
        fprintf (stderr, "%s:%d: [%d] not yet out of barrier\n",
                 intrc.name, intrc.numrec, GetNodeId(erec));
      }
      if ( barrin[GetNodeId(erec)] )
      {
        fprintf (stderr, "%s:%d: [%d] already in barrier\n",
                 intrc.name, intrc.numrec, GetNodeId(erec));
      }
      else
      {
        barrin[GetNodeId(erec)] = TRUE;
        numin++;

        if ( numin == numproc )
        {
          if ( numout != numproc )
          {
            fprintf (stderr, "%s:%d: barrier event count error\n",
                     intrc.name, intrc.numrec);
          }
          numin = numout = 0;
          for (i=0; i<=intrc.numproc; i++) barrin[i] = barrout[i] = FALSE;
        }
      }
    }
    else if ( erec->ev == PCXX_BARRIER_EXIT )
    {
      if ( barrin[GetNodeId(erec)] )
      {
        fprintf (stderr, "%s:%d: [%d] not yet in barrier\n",
                 intrc.name, intrc.numrec, GetNodeId(erec));
      }
      if ( barrout[GetNodeId(erec)] )
      {
        fprintf (stderr, "%s:%d: [%d] already out of barrier\n",
                 intrc.name, intrc.numrec, GetNodeId(erec));
      }
      else
      {
        barrout[GetNodeId(erec)] = TRUE;
        numout++;
      }
    }
# endif

    if ( outFormat == alog )
    {
      i = GetEvent (erec->ev);
      fprintf (outfp, "%3d %3d 0 %10lld %d %10llu\n",
        i,                /* event type */
        GetNodeId(erec),        /* process id */
        erec->par,        /* integer parameter */
        intrc.overflows,  /* clock cycle */
        erec->ti);        /* timestamp */
    }
    else if ( outFormat == SDDF )
    {
      ptr = GetEventName (erec->ev, &hasParam);
      if ( hasParam )
        fprintf (outfp, "%s { %llu, %d, %d, %lld };;\n\n",
                 ptr, erec->ti, GetNodeId(erec), erec->tid, erec->par);
      else
        fprintf (outfp, "%s { %llu, %d, %d };;\n\n",
                 ptr, erec->ti, GetNodeId(erec), erec->tid);
    }

    else if( outFormat == paraver ){
      long long logSend= 0;
      long long logRecv= 0;
      long long phSend= 0;
      long long phRecv= 0;
      long long tempTid = 0;
      long long tempNid = 0;
      long long endBurstTime = 0;
      int sendTid,recvTid;
      int tempTag, tempLen;
      int looking;
      off_t last_position;
      PCXX_EV *curr_rec;
      EVDESCR *curr_ev;
      int curr_tag, curr_len, curr_nid;
      struct trcdescr trcdes = intrc;
      
      
      
      /* Check for Logical Send */
      if(((strcmp((GetEventName(erec->ev,&hasParam)), "\"MPI_Send()  \"") ==0)) && (erec->par == 1)){
	logSend = erec->ti - intrc.firsttime;
	tempTid = erec->tid;
	tempNid = GetNodeId(erec);
	looking = TRUE;
	
	trcdes.buffer    = tmpbuffer;
	/* get the current position from the trace file descriptor */
	last_position = lseek(trcdes.fd, 0, SEEK_CUR);
	if (last_position < 0) {
	  perror("lseek ERROR: GetMatchingSend(), routing that matches logical & physical Send/Recv");
	  exit(1);
	}
	
	/* Find Physical Send */
	while(looking){
	  if ((curr_rec = get_next_rec (&trcdes)) == NULL ){
	    looking = FALSE;
	  }
	  else{
	    curr_ev = GetEventStruct(curr_rec->ev);
	    if((curr_ev->tag == SEND_EVENT) && (GetNodeId(curr_rec) == tempNid) && (curr_rec->tid == tempTid)){
	      phSend = curr_rec->ti - intrc.firsttime;
	      looking = FALSE;
	      sendTid = curr_rec->tid;
	      msgtag 	= (curr_rec->par>>16) & 0x000000FF;
	      myid 		= GetNodeId(curr_rec);
	      otherid 	= ((curr_rec->par>>24) & 0x000000FF);
	      msglen  	= curr_rec->par & 0x0000FFFF;
	    }
	  }
	}/* while */
	
	/* Find Physical Receive */
	looking = TRUE;
	while(looking){
	  if ((curr_rec = get_next_rec (&trcdes)) == NULL ){
	    looking = FALSE;
	  }
	  else{
	    curr_ev = GetEventStruct(curr_rec->ev);
	    if(curr_ev->tag == RECV_EVENT){
	      tempTag = (curr_rec->par>>16) & 0x000000FF;
	      tempLen = curr_rec->par & 0x0000FFFF;
	      tempNid = curr_rec->nid;
	      tempTid = curr_rec->tid;
	      if ((tempTag == msgtag) && (tempLen == msglen) && (tempNid == otherid )){
		phRecv = curr_rec->ti - intrc.firsttime;
		looking = FALSE;
	      }
	    }
	  }
	}/* while */
	
	/* Find logRecv */
	looking = TRUE;
	
	rcvdes.buffer   = trcdes.buffer;
	rcvdes.erec	   = curr_rec;
	rcvdes.fd 	   = intrc.fd;
	rcvdes.first    = trcdes.buffer;
	rcvdes.prev	   = curr_rec - 1;
	
	/* get the current position from the trace file descriptor */
	
	if (last_position < 0) {
	  perror("lseek ERROR: Get matching logical Receive");
	  exit(1);
	}
	while(looking){
	  if((curr_rec = get_prev_rec(&rcvdes)) != NULL){
	    if(strcmp((GetEventName(curr_rec->ev,&hasParam)), "\"MPI_Recv()  \"") ==0){
	      if((curr_rec->par == 1) && (GetNodeId(curr_rec) == tempNid) && (curr_rec->tid == tempTid)){
		logRecv = curr_rec->ti - intrc.firsttime;
		fprintf(outfp,"3:%d:1:%d:%d:%llu:%llu:%d:1:%d:%d:%llu:%llu:%lld:%lld\n",myid+1,myid+1,sendTid+1,logSend,phSend,otherid+1,otherid+1,recvTid+1,logRecv,phRecv,msglen,msgtag);
		fprintf(outfp,"2:%d:1:%d:%d:%llu:5:%d\n",myid+1,myid+1,sendTid+1,logSend,erec->ev);
		if(prvPCF[erec->ev-1] == 0){
		  fprintf(pcffp,"%d       %s\n",erec->ev,GetEventName(erec->ev,&hasParam));
		  prvPCF[erec->ev-1] = 1;
		}
		fprintf(outfp,"1:%d:1:%d:%d:%llu:%llu:9\n",myid+1,myid+1,sendTid+1,logSend,phSend);
		lseek(rcvdes.fd, last_position, SEEK_SET);
		lseek(trcdes.fd,last_position,SEEK_SET);
		looking = FALSE;
	      }
	    }
	  }
	  else{
	    lseek(rcvdes.fd, last_position, SEEK_SET);
	    lseek(trcdes.fd,last_position,SEEK_SET);
	    looking = FALSE;
	  }
 	}/* while */
      }
      else{
        int parameter = erec->par;
	/* Print Event Records for entry and exit of methods */
	switch(parameter){
	case 1:
	  fprintf (outfp, "2:%d:1:%d:%d:%llu:5:%d\n",GetNodeId(erec)+1,GetNodeId(erec)+1,erec->tid+1,erec->ti-intrc.firsttime,erec->ev);
	  if(prvPCF[erec->ev-1] == 0){
	    fprintf(pcffp,"%d       %s\n",erec->ev,GetEventName(erec->ev,&hasParam));
	    prvPCF[erec->ev-1] = 1;
	  }
	  if(strcmp((GetEventName(erec->ev,&hasParam)), "\"MPI_Recv()  \"") ==0){
	    endBurstTime = GetNextStateBurst(intrc,GetNodeId(erec),erec->tid,hasParam);
	    fprintf(outfp,"1:%d:1:%d:%d:%llu:%llu:3\n",GetNodeId(erec)+1,GetNodeId(erec)+1,erec->tid+1,erec->ti-intrc.firsttime,endBurstTime);
	  }
	  break;
       	case -1:
	  fprintf (outfp, "2:%d:1:%d:%d:%llu:5:0\n",GetNodeId(erec)+1,GetNodeId(erec)+1,erec->tid+1,erec->ti-intrc.firsttime);
	  if(prvPCF[erec->ev-1] == 0){
	    fprintf(pcffp,"%d       %s\n",erec->ev,GetEventName(erec->ev,&hasParam));
	    prvPCF[erec->ev-1] = 1;
	  }
	  if(strcmp((GetEventName(erec->ev,&hasParam)), "\"MPI_Send()  \"") ==0){
	    endBurstTime = GetNextStateBurst(intrc,GetNodeId(erec),erec->tid,hasParam);
	    fprintf(outfp,"1:%d:1:%d:%d:%llu:%llu:1\n",GetNodeId(erec)+1,GetNodeId(erec)+1,erec->tid+1,erec->ti-intrc.firsttime,endBurstTime);
	  }
	  else if(strcmp((GetEventName(erec->ev,&hasParam)),"\"MPI_Recv()  \"") ==0){
	    endBurstTime = GetNextStateBurst(intrc,GetNodeId(erec),erec->tid,hasParam);
	    fprintf(outfp,"1:%d:1:%d:%d:%llu:%llu:1\n",GetNodeId(erec)+1,GetNodeId(erec)+1,erec->tid+1,erec->ti-intrc.firsttime,endBurstTime);
	  }
	  break;
	case 3:
	  endBurstTime = GetNextStateBurst(intrc,GetNodeId(erec),erec->tid,hasParam);
	  fprintf(outfp, "1:%d:1:%d:%d:%llu:%llu:1\n",GetNodeId(erec)+1,GetNodeId(erec)+1,erec->tid+1,erec->ti-intrc.firsttime,endBurstTime);
	  break;
	  
	case 0:
	  fprintf(outfp,"1:%d:1:%d:%d:%llu:%llu:2\n",GetNodeId(erec)+1,GetNodeId(erec)+1,erec->tid+1,erec->ti-intrc.firsttime,intrc.lasttime-intrc.firsttime);
	  break;
 	}
      }
    }
    else if ( outFormat == pv )
      {
	ev = GetEventStruct (erec->ev);
	if (( ev->tag != 0 ) || (dynamictrace)) /* dynamic trace doesn't use tag*/
	  {
	    if ( (ev->tag == SEND_EVENT) && pvComm )

	      {
		/* send message */
		/* In dynamic trace the format for par is
		   31 ..... 24 23 ......16 15..............0
       	           other       type          length
		So, mynode is the sender and its in GetNodeId(erec)
		SENDMSG <type> FROM <sender> TO <receiver> LEN <length>
		*/
		/* extract the information from the parameter */
		msgtag 	= (erec->par>>16) & 0x000000FF;
		myid 		= GetNodeId(erec) + 1;
		otherid 	= ((erec->par>>24) & 0x000000FF) + 1;
		msglen  	= erec->par & 0x0000FFFF;

		if (threads)
		  {
		    if (GetMatchingRecv(intrc, msgtag, GetNodeId(erec),
					otherid -1 , msglen, &other_tid, &other_nodeid))
		      { /* call was successful, we've the other_tid and other_nodeid */
			otherid = offset[other_nodeid] + other_tid + 1;
#ifdef DEBUG
			printf("Calculated otherid = %d\n", otherid);
#endif /* DEBUG */

		      }
		    else
		      { /* call was unsuccessful, we couldn't locate a matching ipc call */
			printf("Matching IPC call not found. Assumption in place...\n");

			/* ASSUMPTION: Thread 4 in a node can comm with thread 4 on another
			   node. True for MPI+JAVA. In future, do a matching algo. */
			otherid	= offset[otherid-1] + erec->tid + 1;
			/* THIS ABOVE IS TRUE ONLY WHEN SAME THREADS COMMUNICATE !! */
#ifdef DEBUG
			printf("ASSUMPTION: SAME THREADIDS ON DIFF NODES COMMUNICATE!!\n");
			printf("SEND: OTHER %d, myid %d len %d tag %d: PAR: %lx\n",
			       otherid, myid, msglen, msgtag, erec->par);
#endif /* DEBUG */
		      }
		  }

#ifdef DEBUG
		printf ("\n\n%llu SENDMSG %d FROM %d TO %d LEN %d\n\n\n",
			erec->ti - intrc.firsttime,
			msgtag, myid , otherid , msglen);
#endif /* DEBUG */

		fprintf (outfp, "%llu SENDMSG %d FROM %d TO %d LEN %d\n",
			 erec->ti - intrc.firsttime,
			 msgtag, myid , otherid , msglen);
	      }
	    else if ( (ev->tag == RECV_EVENT ) && pvComm )
	      {
		/* receive message */
		/* In dynamic trace the format for par is
		   31 ..... 24 23 ......16 15..............0
       	           other       type          length
		   So, mynode is the receiver and its in GetNodeId(erec)
		   RECVMSG <type> BY <receiver> FROM <sender> LEN <length>
		*/
		/* extract the information from the parameter */
		msgtag 	= (erec->par>>16) & 0x000000FF;
		myid 		= GetNodeId(erec)+1;
		otherid       = ((erec->par>>24) & 0x000000FF) + 1;
		msglen	= erec->par & 0x0000FFFF;

		if (threads)
		  {
		    if (GetMatchingSend(intrc, msgtag, GetNodeId(erec),
					otherid - 1 , msglen, &other_tid, &other_nodeid))
		      { /* call was successful, we've the other_tid and other_nodeid */
			otherid = offset[other_nodeid] + other_tid + 1;
#ifdef DEBUG
			printf("Calculated senderid = %d\n", otherid);
#endif /* DEBUG */

		      }
		    else
		      { /* call was unsuccessful, we couldn't locate a matching ipc call */
			printf("Matching IPC call not found. Assumption in place...\n");
			/* ASSUMPTION: Thread 4 in a node can comm with thread 4 on another
			   node. True for MPI+JAVA. In future, do a matching algo. */
			otherid	= offset[otherid-1] + erec->tid + 1;
			/* THIS ABOVE IS TRUE ONLY WHEN SAME THREADS COMMUNICATE !! */
#ifdef DEBUG
			printf("ASSUMPTION: SAME THREADIDS ON DIFF NODES COMMUNICATE!!\n");
			printf("RECV: OTHER %d, myid %d len %d tag %d: PAR: %lx\n",
			       otherid, myid, msglen, msgtag, erec->par);
#endif /* DEBUG */
		      }
		  }

#ifdef DEBUG
		printf ("%llu RECVMSG %d BY %d FROM %d LEN %d\n",
			erec->ti - intrc.firsttime,
			msgtag, myid , otherid , msglen);
#endif /* DEBUG */

		fprintf (outfp, "%llu RECVMSG %d BY %d FROM %d LEN %d\n",
			 erec->ti - intrc.firsttime,
			 msgtag, myid , otherid , msglen);
	      }
	    else if (( ev->tag == -9 ) || ( erec->par == -1))
	      { /* In dynamic tracing, 1/-1 par values are for Entry/Exit resp. */
		/* exit state */
		/* PARVis needs time values relative to the start of the program! */
		stkptr[GetNodeId(erec)]--;
		if ( stkptr[GetNodeId(erec)] < statestk[GetNodeId(erec)] )
		  {
		    fprintf (stderr, "ERROR: stack underflow on node %d\n", GetNodeId(erec));
		    fprintf (stderr, "       event %s at %llu\n", ev->name, erec->ti);
		    exit (1);
		  }
		if ( pvCompact )
		  fprintf (outfp, "%llu EXCH %d 1 1 %s %d\n",
			   erec->ti - intrc.firsttime, GetNodeId(erec)+1,
			   stkptr[GetNodeId(erec)]->state, stkptr[GetNodeId(erec)]->tag);
		else
		  fprintf (outfp, "%llu EXCHANGE ON CPUID %d TO %s %d CLUSTER 1\n",
			   erec->ti - intrc.firsttime, GetNodeId(erec)+1,
			   stkptr[GetNodeId(erec)]->state, stkptr[GetNodeId(erec)]->tag);
	      }
	    else if (erec->par == 1)
	      {
		/* enter new state */
		stkptr[GetNodeId(erec)]++;
		if ( stkptr[GetNodeId(erec)] > (statestk[GetNodeId(erec)] + STACKSIZE) )
		  {
		    fprintf (stderr, "ERROR: stack overflow on node %d\n", GetNodeId(erec));
		    fprintf (stderr, "       event %s at %llu\n", ev->name, erec->ti);
		    exit (1);
		  }
		if ( pvCompact )
		  fprintf (outfp, "%llu EXCH %d 1 1 %s %d\n",
			   /*???erec->ti, GetNodeId(erec)+1, ev->state, ev->tag);*/
			   erec->ti - intrc.firsttime, GetNodeId(erec)+1, ev->state, ev->no);
		else
		  fprintf (outfp, "%llu EXCHANGE ON CPUID %d TO %s %d CLUSTER 1\n",
			   /*???erec->ti, GetNodeId(erec)+1, ev->state, ev->tag);*/
			   erec->ti - intrc.firsttime, GetNodeId(erec)+1, ev->state, ev->no);
		stkptr[GetNodeId(erec)]->state = ev->state;
		/*???stkptr[GetNodeId(erec)]->tag = ev->tag;*/
		stkptr[GetNodeId(erec)]->tag = ev->no;
	      }
	  }
      }
    else if ( outFormat == dump )
    {
      ptr = GetEventName (erec->ev, &hasParam);
      if ( hasParam )
        fprintf (outfp, "%5ld %30.30s %12llu %6d %6d %12lld\n",
                 intrc.numrec, ptr, erec->ti, GetNodeId(erec), erec->tid, erec->par);
      else
        fprintf (outfp, "%5ld %30.30s %12llu %6d %6d\n",
                 intrc.numrec, ptr, erec->ti, GetNodeId(erec), erec->tid);
	/* Changed 12lu to 12llu for unsigned long long time */
    }

    if ( (erec = get_next_rec (&intrc)) == NULL )
      break;
    else
    {
      intrc.numrec++;

      /* -- check clock overflow ---------------------------------------- */
      if ( erec->ti < intrc.lasttime ) intrc.overflows++;
      intrc.lasttime = erec->ti;
    }
  }
  while ( erec != NULL );

  if ( outFormat == pv )
  {
    for (i=0; i<=intrc.numproc; i++)
    {
      if ( stkptr[i] != statestk[i] )
      {
        fprintf (stderr, "ERROR: stack not empty on node %d\n", i);
        exit (1);
      }
    }
  }

  fclose (outfp);
  exit (0);
}
