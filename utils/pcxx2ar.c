/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1994                  */
/*  Indiana University  University of Oregon  University of Rennes   */
/*  Author : Sameer Shende, 				             */ 
/*           sameer@cs.uoregon.edu 	     			     */
/*********************************************************************/

/*
 * pcxx_convert.c : convert software event traces to other formats
 *
 * (c) 1994 Jerry Manic Saftware
 *
 * Version 3.0
 */

/* Sameer Shende - for Ariadne */
# include <stdio.h>
# include <stdlib.h>
# include <sys/types.h>
# include <fcntl.h>

# include <string.h>

# define TRACING_ON
# define PCXX_EVENT_SRC

# ifdef __PCXX__
#   include "pcxx_events_def.h"
# else
#   include "pcxx_events.h"
# endif
# include "pcxx_ansi.h"
# include "pcxx_ariadne.h" 

# ifndef TRUE
#   define FALSE  0
#   define TRUE   1
# endif

# define F_EXISTS    0
/* get the types as defined in ../tulip/passive.h */
/* change this portion if the passive.h is changed */
/*****************************/
#define BarrierFanOutMsg 100 /* Message to notify that work can resume */
#define BarrierFanInMsg  101 /* Fan into one node to sync */
#define BroadcastMsg     110 /* All processors listen and receive... */
#define ReduceMsg        120 /* Reduce a value across all processors */

#define AskForDataMsg    130 /* Send this type to remote node to get data */
#define FetchBlockMsg    140 /* Partial reply to earlier AskForDataMsg */
#define EndOfFetchMsg    141 /* Completed reply to earlier AskForDataMsg */

#define PrefixStoreMsg   150 /* Send this type to prepare remote for store */
#define StoreBlockMsg    160 /* Partial block of store */
#define EndOfStoreMsg    161 /* Final block of a store message */

#define RemoteActionMsg  170 /* Send to remote to execute handler */

#define AckReplyMsg      180 /* Reply from remote that store has completed */

/* GLOBAL array definition */
char mtag_string_array[256][64]; /* 64 chars in each string. 256 indices */

static struct trcdescr
{
  int     fd;              /* -- input file descriptor                     -- */
  char   *name;            /* -- corresponding file name                   -- */
  int     overflows;       /* -- clock overflows in that trace             -- */
  int     numevent;        /* -- number of event types                     -- */
  int     numproc;         /* -- number of processors                      -- */
  long    numrec;          /* -- number of event records already processed -- */
  unsigned long firsttime; /* -- timestamp of first event record           -- */
  unsigned long lasttime;  /* -- timestamp of previous event record        -- */

  PCXX_EV  *buffer;   /* -- input buffer                              -- */
  PCXX_EV  *erec;     /* -- current event record                      -- */
  PCXX_EV  *next;     /* -- next available event record in buffer     -- */
  PCXX_EV  *last;     /* -- last event record in buffer               -- */
} intrc;

static enum format_t { alog, SDDF, pv, dump } outFormat = pv;
static enum mode_t { user, class, all } pvMode = user;
static int pvCompact = FALSE;
static int pvComm = TRUE;

static char *barrin, *barrout;  /* -- for barrier checking -- */
static int numin, numout;       /* -- for barrier checking -- */

static struct stkitem
{
  char *state;  /* -- state name ---- */
  int tag;      /* -- activity tag -- */
}
**statestk,     /* -- state stacks -- */
**stkptr;       /* -- stack pointers -- */

# define STACKSIZE 25

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

static void InitEvent (int numev)
{
  int i;

  evtable = (EVDESCR **) malloc (numev * sizeof(EVDESCR));
  for (i=0; i<numev; i++) evtable[i] = NULL;
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
          fprintf (out, "\"%s\" {\n", ev->name);
          fprintf (out, "\tint\t\"Timestamp\";\n");
          fprintf (out, "\tint\t\"Processor Number\";\n");
          fprintf (out, "\tint\t\"Thread Id\";\n");
          if ( ev->param[0] ) fprintf (out, "\tint\t\"%s\";\n", ev->param);
          fprintf (out, "};;\n\n", ev->name);
        }
        else if ( outFormat == pv )
        {
          if ( ev->tag > 0 || ev->tag < -9 )
          {
            ev->name[strlen(ev->name)-6] = '\0';
            /*fprintf (out, "SYMBOL %s %d %s\n", ev->state, ev->tag, ev->name);*/
            fprintf (out, "SYMBOL %s %d %s\n", ev->state, ev->no, ev->name);
            ev->name[strlen(ev->name)] = '-';
          }
        }
      }
      ev = ev->next;
    }
  }
}

/* -------------------------------------------------------------------------- */
/* -- get today's date                                                     -- */
/* -------------------------------------------------------------------------- */
# include <time.h>

static char tibuf[12];
static char *Months[12] =
{"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};

static char *Today (void)
{
  time_t t;
  struct tm *tm;

  t = time ((time_t *) 0);
  tm = localtime (&t);
  sprintf (tibuf, "%s-%02d-%02d", Months[tm->tm_mon], tm->tm_mday, tm->tm_year);
  return (tibuf);
}

/* -------------------------------------------------------------------------- */
/* -- input buffer handling                                                -- */
/* -------------------------------------------------------------------------- */
# define INMAX    BUFSIZ   /* records */

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
        close (tdes->fd);
        tdes->fd = -1;
        return (NULL);
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

/* -------------------------------------------------------------------------- */
/* -- PCXX_CONVERT MAIN PROGRAM --------------------------------------------- */
/* -------------------------------------------------------------------------- */

# define LINEMAX 256

int main (int argc, char *argv[])
{
  FILE *outfp, *inev;
  PCXX_EV *erec;
  int i;
  int num;
  int tag;
  int hasParam;
  int fileIdx;
  int numproc = 0;
  char name[80], state[80], param[80], linebuf[LINEMAX];
  char mtag_string[64];
  char *inFile, *edfFile, *outFile, *ptr;
  EVDESCR *ev;
  long int parameter, pid, oid, mtag, x,  aa_poll_cnt; /* Ariadne */
  unsigned char type;
  char message_type; /* RWMU */
  /* for PCXX_user_event -  additional variables for tracing */
  int event_no, no_of_vars, var1, var2, var3;

  /* ------------------------------------------------------------------------ */
  /* -- scan command line arguments                                        -- */
  /* ------------------------------------------------------------------------ */

/* No need to do this. Its always dump - Sameer */
/*
  if ( strcmp (argv[0]+strlen(argv[0])-4, "sddf") == 0 )
    outFormat = SDDF;
  else if ( strcmp (argv[0]+strlen(argv[0])-4, "alog") == 0 )
    outFormat = alog;
  else if ( strcmp (argv[0]+strlen(argv[0])-4, "dump") == 0 )
    outFormat = dump;
  else if ( strcmp (argv[0]+strlen(argv[0])-2, "pv") == 0 )
    outFormat = pv;
*/

  outFormat = dump; /* forced it for pcxx2ar */

  fileIdx = 1;

  if ( argc < 3 )
  {
    fprintf (stderr, "usage: %s [-alog | -SDDF | -dump |", argv[0]);
    fprintf (stderr, " -pv [-compact] [-user|-class|-all] [-nocomm]]");
    fprintf (stderr, " inputtrc edffile [outputtrc]\n");
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
      if ( strcmp (argv[i], "-compact") == 0 )
        pvCompact = TRUE;
      else if ( strcmp (argv[i], "-user") == 0 )
        pvMode = user;
      else if ( strcmp (argv[i], "-class") == 0 )
        pvMode = class;
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

  inFile  = argv[fileIdx];
  edfFile = argv[fileIdx+1];
  if ( (fileIdx+2) == argc )
    outFile = NULL;
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
    intrc.erec      = NULL;
    intrc.next      = NULL;
    intrc.last      = NULL;
    intrc.overflows = 0;

    /* -- read first event record ------------------------------------------- */
    if ( (erec = get_next_rec (&intrc)) == NULL )
    {
      /* -- no event record: ------------------------------------------------ */
      fprintf (stderr, "%s: warning: trace empty - ignored\n",
               intrc.name);
      intrc.numrec = 0L;
    }

    /* -- check first event record ------------------------------------------ */
    else if ( (erec->ev != PCXX_EV_INIT) && (erec->ev != PCXX_EV_INITM) )
    {
      fprintf (stderr, "%s: no valid event trace\n", intrc.name);
      exit (1);
    }
    else
    {
      intrc.numrec    = 1L;
      intrc.numproc   = erec->nid;
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
  sscanf (linebuf, "%d", &intrc.numevent);

  InitEvent (intrc.numevent);
  init_mtag_string_array();

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
    sscanf (linebuf, "%d %s %d %s %s", &num, state, &tag, name, param);
    if ( (num < 0) || !*name )
    {
      fprintf (stderr, "%s: blurb in line %d\n", edfFile, i+2);
      exit (1);
    }
    AddEvent (num, name, param, state, tag);
  }
  fclose (inev);

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
    else if ( (erec->ev == PCXX_EV_INIT) || (erec->ev == PCXX_EV_INITM) )
    {
      if ( erec->nid > PCXX_MAXPROCS )
        fprintf (stderr,
           "%s: warning: node id %d too big for this machine (max. %d nodes)\n",
           intrc.name, erec->nid, PCXX_MAXPROCS);
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

      /* -- check process id -------------------------------------------- */
      if ( erec->nid > intrc.numproc ) intrc.numproc = erec->nid;
    }
  }
  while ( erec != NULL );

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
    fprintf (outfp, " -6   0 0          0 0 %10lu\n", intrc.firsttime);
    fprintf (outfp, " -7   0 0          0 0 %10lu\n", intrc.lasttime);
    fprintf (outfp, " -8   0 0 %10d 0          0\n", intrc.overflows+1);
    fprintf (outfp, " -1   0 0          0 0          0 pcxx_convert -alog %s\n",
             Today());
    fprintf (outfp, "-11   0 0          0 0 4294967295\n");
  }
  else if ( outFormat == SDDF )
  {
    fprintf (outfp, "/*\n");
    fprintf (outfp, " * \"creation program\" \"pcxx_convert -SDDF\"\n");
    fprintf (outfp, " * \"creation date\" \"%s\"\n", Today());
    fprintf (outfp, " * \"number records\" \"%ld\"\n", intrc.numrec);
    fprintf (outfp, " * \"number processors\" \"%d\"\n", intrc.numproc+1);
    fprintf (outfp, " * \"first timestamp\" \"%ld\"\n", intrc.firsttime);
    fprintf (outfp, " * \"last timestamp\" \"%ld\"\n", intrc.lasttime);
    fprintf (outfp, " */\n\n");
  }
  else if ( outFormat == pv )
  {
    fprintf (outfp, "CLKPERIOD 0.1000E-06\n");
    fprintf (outfp, "NCPUS %d\n", intrc.numproc+1);
    fprintf (outfp, "C CREATION PROGRAM pcxx_convert -pv\n");
    fprintf (outfp, "C CREATION DATE %s\n", Today());
    fprintf (outfp, "C NUMBER RECORDS %ld\n", intrc.numrec);
    fprintf (outfp, "C FIRST TIMESTAMP %ld\n", intrc.firsttime);
    fprintf (outfp, "C LAST TIMESTAMP %ld\n", intrc.lasttime);
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
/*
    fprintf (outfp, "#  creation program: pcxx_convert -dump\n");
    fprintf (outfp, "#     creation date: %s\n", Today());
    fprintf (outfp, "#    number records: %ld\n", intrc.numrec);
    fprintf (outfp, "# number processors: %d\n", numproc);
    fprintf (outfp, "# max processor num: %d\n", intrc.numproc);
    fprintf (outfp, "#   first timestamp: %ld\n", intrc.firsttime);
    fprintf (outfp, "#    last timestamp: %ld\n\n", intrc.lasttime);

    fprintf (outfp, "#=NO= =======================EVENT==");
    fprintf (outfp, " ==TIME [us]= =NODE= =THRD= ==PARAMETER=\n");
*/
    fprintf (outfp, "1 %d\n", numproc);
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
    intrc.erec      = NULL;
    intrc.next      = NULL;
    intrc.last      = NULL;
    intrc.overflows = 0;
    intrc.numrec    = 1;

    erec = get_next_rec (&intrc);
    intrc.firsttime = erec->ti;
    intrc.lasttime  = erec->ti;
  }

  /* ------------------------------------------------------------------------ */
  /* -- initialize barrier check variables ---------------------------------- */
  /* ------------------------------------------------------------------------ */
  numin   = 0;
  numout  = numproc;
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
      if ( !barrout[erec->nid] )
      {
        fprintf (stderr, "%s:%d: [%d] not yet out of barrier\n",
                 intrc.name, intrc.numrec, erec->nid);
      }
      if ( barrin[erec->nid] )
      {
        fprintf (stderr, "%s:%d: [%d] already in barrier\n",
                 intrc.name, intrc.numrec, erec->nid);
      }
      else
      {
        barrin[erec->nid] = TRUE;
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
      if ( barrin[erec->nid] )
      {
        fprintf (stderr, "%s:%d: [%d] not yet in barrier\n",
                 intrc.name, intrc.numrec, erec->nid);
      }
      if ( barrout[erec->nid] )
      {
        fprintf (stderr, "%s:%d: [%d] already out of barrier\n",
                 intrc.name, intrc.numrec, erec->nid);
      }
      else
      {
        barrout[erec->nid] = TRUE;
        numout++;
      }
    }
# endif

    if ( outFormat == alog )
    {
      i = GetEvent (erec->ev);
      fprintf (outfp, "%3d %3d 0 %10ld %d %10lu\n",
        i,                /* event type */
        erec->nid,        /* process id */
        erec->par,        /* integer parameter */
        intrc.overflows,  /* clock cycle */
        erec->ti);        /* timestamp */
    }
    else if ( outFormat == SDDF )
    {
      ptr = GetEventName (erec->ev, &hasParam);
      if ( hasParam )
        fprintf (outfp, "\"%s\" { %lu, %d, %d, %ld };;\n\n",
                 ptr, erec->ti, erec->nid, erec->tid, erec->par);
      else
        fprintf (outfp, "\"%s\" { %lu, %d, %d };;\n\n",
                 ptr, erec->ti, erec->nid, erec->tid);
    }
    else if ( outFormat == pv )
    {
      ev = GetEventStruct (erec->ev);
      if ( ev->tag != 0 )
      {
        if ( (ev->tag == -7) && pvComm )
        {
          /* send message */
          fprintf (outfp, "%lu SENDMSG %d FROM %d TO %d LEN %d\n", erec->ti,
                  ((erec->par>>16) & 0x000000FF),
                  ((erec->par>>24) & 0x000000FF) + 1, erec->nid+1,
                  erec->par & 0x0000FFFF);
        }
        else if ( (ev->tag == -8) && pvComm )
        {
          /* receive message */
          fprintf (outfp, "%lu RECVMSG %d BY %d FROM %d LEN %d\n", erec->ti,
                  ((erec->par>>16) & 0x000000FF),
                  erec->nid+1, ((erec->par>>24) & 0x000000FF) + 1,
                  erec->par & 0x0000FFFF);
        }
        else if ( ev->tag == -9 )
        {
          /* exit state */
          stkptr[erec->nid]--;
          if ( stkptr[erec->nid] < statestk[erec->nid] )
          {
            fprintf (stderr, "ERROR: stack underflow on node %d\n", erec->nid);
            fprintf (stderr, "       event %s at %lu\n", ev->name, erec->ti);
            exit (1);
          }
          if ( pvCompact )
            fprintf (outfp, "%lu EXCH %d 1 1 %s %d\n",
                    erec->ti, erec->nid+1,
                    stkptr[erec->nid]->state, stkptr[erec->nid]->tag);
          else
            fprintf (outfp, "%lu EXCHANGE ON CPUID %d TO %s %d CLUSTER 1\n",
                    erec->ti, erec->nid+1,
                    stkptr[erec->nid]->state, stkptr[erec->nid]->tag);
        }
        else
        {
          /* enter new state */
          stkptr[erec->nid]++;
          if ( stkptr[erec->nid] > (statestk[erec->nid] + STACKSIZE) )
          {
            fprintf (stderr, "ERROR: stack overflow on node %d\n", erec->nid);
            fprintf (stderr, "       event %s at %lu\n", ev->name, erec->ti);
            exit (1);
          }
          if ( pvCompact )
            fprintf (outfp, "%lu EXCH %d 1 1 %s %d\n",
                    /*???erec->ti, erec->nid+1, ev->state, ev->tag);*/
                    erec->ti, erec->nid+1, ev->state, ev->no);
          else
            fprintf (outfp, "%lu EXCHANGE ON CPUID %d TO %s %d CLUSTER 1\n",
                    /*???erec->ti, erec->nid+1, ev->state, ev->tag);*/
                    erec->ti, erec->nid+1, ev->state, ev->no);
          stkptr[erec->nid]->state = ev->state;
          /*???stkptr[erec->nid]->tag = ev->tag;*/
          stkptr[erec->nid]->tag = ev->no;
        }
      }
    }
    else if ( outFormat == dump )
    {
      ptr = GetEventName (erec->ev, &hasParam);
      if((strcmp(ptr,"ariadne_ipc") == 0) || (strcmp(ptr,"ariadne_pcxx_ev") == 0))
	/* ariadne event extract params */
      {
       
	/* Now to extract the fields. */
	parameter = erec->par; 
	aa_poll_cnt = (parameter >> 32); /* get 32 bits of poll count */	
	
	parameter = erec->par;
	pid = (parameter << 32) >> (22+32); /* get my pid 10 bits */
	parameter = erec->par;
	oid = (parameter << (10+32)) >> (22+32); /* the other process pid 10 bits */
 	parameter = erec->par;
	mtag = (parameter << (20+32)) >> (28+32); /* mtag RWMU... 4 bits */
	switch(mtag) 
	{
		case PCXX_AA_MTAG_READ : 
			message_type = 'R';
			break;
		case PCXX_AA_MTAG_WRITE :
			message_type = 'W';
			break;
		case PCXX_AA_MTAG_MCAST : 
			message_type = 'M';
			break;
		case PCXX_AA_MTAG_USER :
			message_type = 'U';
			break;
		default :
			message_type = mtag;
			break; 
	}
	parameter = erec->par;
	x = (parameter << (24+32)) >> (24+32); 
	type = (unsigned char ) x ;
	/* type 107 etc. 8 bits */
	convert_mtag_to_string(type, mtag_string);
        if(strncmp(mtag_string,"PCXX_user_event",strlen("PCXX_user_event")) != 0)
	{ /* not a user event */
	  fprintf(outfp,"%d %d %c %s %d 0 0\n", pid, oid, message_type, mtag_string, aa_poll_cnt);
        } 
	else 
        {
	   /* We have with us a user event - we need to extract the parameters
	      from here - from aa_poll_cnt which no longer holds the poll count
	      but contains the following description for user event */
/* This sets puts the value of the parameter in the trace for Ariadne */
/* This is used for logging user event in pc++ */
/* out of 32 bits for the integer parameter we use
    xx yyyy vv var1 var2 var3
    2   4    2  8    8    8
    xx - 2 bits unused - signed etc.
    yyyy - event no. goes from 0 to 15 - different types  of events- merge
                and record - values 0 1 ?
    vv - how many variables do you want to store.
    var1 - one variable,
    var2 - second variable - 8 bits.
    var3 - third variable - 8 bits.
*/
        extract_parameters(aa_poll_cnt, &event_no, &no_of_vars, &var1, &var2, &var3);
	  switch(no_of_vars)  
          {
	    case 0 : fprintf(outfp,"%d %d %c %s%d 0 0 0\n", pid, oid, 
                   message_type, mtag_string, event_no); 
		/* nothing to log even the poll_cnt makes no sense here */
		   break;
	    case 1 : fprintf(outfp,"%d %d %c %s%d 0 0 1 %d\n", pid, oid, 
		   message_type, mtag_string, event_no, var1);
		/* store event_no with user_event and log one variable */
	           break;
	    case 2 : fprintf(outfp,"%d %d %c %s%d 0 0 2 %d %d\n", pid, oid, 
		   message_type, mtag_string, event_no, var1, var2);
		/* store two variables */
		   break;
	    case 3 : fprintf(outfp,"%d %d %c %s%d 0 0 3 %d %d %d\n", pid, oid, 
		   message_type, mtag_string, event_no, var1, var2, var3);
		/* store two variables */
		   break;
	    default : fprintf(outfp,"%d %d %c %s %d 0 0\n", pid, oid, 
		   message_type, mtag_string, aa_poll_cnt);
		/* the default - but this will not happen */
		printf("%s - read a default PCXX_user_event - shouldn't happen\n");
		   break;
	  }

         } 
       }
    }






/*
      if ( hasParam )
        fprintf (outfp, "%5ld %30.30s %12lu %6d %6d %12ld\n",
                 intrc.numrec, ptr, erec->ti, erec->nid, erec->tid, erec->par);
      else
        fprintf (outfp, "%5ld %30.30s %12lu %6d %6d\n",
                 intrc.numrec, ptr, erec->ti, erec->nid, erec->tid);
    }
*/

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

int init_mtag_string_array(void)
{ /* we have a global array - mtag_string_array which holds all the tags and
     their names - it is initialized here */
     
    /* first the regular RW messages with pC++ types defined in passive.h */
    strcpy(mtag_string_array[ BarrierFanOutMsg ], "BarrierFanOutMsg");	
    strcpy(mtag_string_array[ BarrierFanInMsg ],  "BarrierFanInMsg");	
    strcpy(mtag_string_array[ BroadcastMsg ],     "BroadcastMsg");	
    strcpy(mtag_string_array[ ReduceMsg ],        "ReduceMsg");	
    strcpy(mtag_string_array[ AskForDataMsg ],    "AskForDataMsg");	
    strcpy(mtag_string_array[ FetchBlockMsg ],    "FetchBlockMsg");	
    strcpy(mtag_string_array[ EndOfFetchMsg ],    "EndOfFetchMsg");	
    strcpy(mtag_string_array[ PrefixStoreMsg ],   "PrefixStoreMsg");	
    strcpy(mtag_string_array[ StoreBlockMsg ],    "StoreBlockMsg");	
    strcpy(mtag_string_array[ EndOfStoreMsg ],    "EndOfStoreMsg");	
    strcpy(mtag_string_array[ RemoteActionMsg ],  "RemoteActionMsg");	
    strcpy(mtag_string_array[ AckReplyMsg ],      "AckReplyMsg");	
    strcpy(mtag_string_array[ BarrierFanInMsg ],  "BarrierFanInMsg");	

    /* next the pC++ Ariadne events defined in pcxx_ariadne.h */
    strcpy(mtag_string_array[ PCXX_error ],  
			     "PCXX_error");	
    strcpy(mtag_string_array[ PCXX_thread_terminate ],  
			     "PCXX_thread_terminate");	
    strcpy(mtag_string_array[ PCXX_create_collection ],  
		             "PCXX_create_collection");	
    strcpy(mtag_string_array[ PCXX_delete_collection ],  
			     "PCXX_delete_collection");	
    strcpy(mtag_string_array[ PCXX_begin_barrier ],  
			     "PCXX_begin_barrier");	
    strcpy(mtag_string_array[ PCXX_end_barrier ],  
		             "PCXX_end_barrier");	
    strcpy(mtag_string_array[ PCXX_begin_fetch ],  
			     "PCXX_begin_fetch");	
    strcpy(mtag_string_array[ PCXX_end_fetch ],  
			     "PCXX_end_fetch");	
    strcpy(mtag_string_array[ PCXX_begin_put ],  
			     "PCXX_begin_put");	
    strcpy(mtag_string_array[ PCXX_end_put ],  
			     "PCXX_end_put");	
    strcpy(mtag_string_array[ PCXX_begin_rpc ],  
			     "PCXX_begin_rpc");	
    strcpy(mtag_string_array[ PCXX_end_rpc ],  
			     "PCXX_end_rpc");	
    strcpy(mtag_string_array[ PCXX_begin_parallel ],  
			     "PCXX_begin_parallel");	
    strcpy(mtag_string_array[ PCXX_enter_element ],  
			     "PCXX_enter_element");	
    strcpy(mtag_string_array[ PCXX_exit_element ],  
			     "PCXX_exit_element");	
    strcpy(mtag_string_array[ PCXX_begin_service_rpc ],  
			     "PCXX_begin_service_rpc");	
    strcpy(mtag_string_array[ PCXX_end_service_rpc ],  
			     "PCXX_end_service_rpc");	
    strcpy(mtag_string_array[ PCXX_user_event ],  
			     "PCXX_user_event");	
    strcpy(mtag_string_array[ PCXX_begin_create_collection ],  
			     "PCXX_begin_create_collection");	
    strcpy(mtag_string_array[ PCXX_end_create_collection ],  
			     "PCXX_end_create_collection");	
    strcpy(mtag_string_array[ PCXX_begin_delete_collection ],  
			     "PCXX_begin_delete_collection");	
    strcpy(mtag_string_array[ PCXX_end_delete_collection ],  
			     "PCXX_end_delete_collection");	
    strcpy(mtag_string_array[ PCXX_end_parallel ], 
			     "PCXX_end_parallel");	

    return 1;
}


int convert_mtag_to_string(unsigned char mtag, char *mtag_string)
{
  memset(mtag_string, '\0', 64);
  strcpy(mtag_string,mtag_string_array[mtag]);
  if (mtag_string == (char *) NULL) 
    sprintf(mtag_string,"a%d",mtag);
  return 1;
}


unsigned int unpack_var(unsigned int par, int length, int starting_lsb_position)
{
/* packing routines for pcxx_AriadneRecordUserEvent */
   /* it starts at position lsb */
int value;
	value = ((par << (32 - (length + starting_lsb_position))) >> (32 - (length + starting_lsb_position) + starting_lsb_position)) ;
	
       return value;
}

unsigned int pack_var(unsigned int var, int length, int starting_lsb_position)
{
    /* make sure its of the right length */

    var = ((var << (32 - length)) >> (32 - length)); 
	/* its of length - won't interfere with other bits */
    /* now shift it starting at point position */
    /* for eg. if we want to put x (8 bits length at position lsb 16) */
    /* then 8........16  x goes in the dots */
    return (var << starting_lsb_position);
} 

int extract_parameters(int par, int *event_no, int *no_of_vars, int *var1, int *var2, int *var3)
{
    /* get the values out of the parameter */
    *event_no = (int) unpack_var(par, 4, 26);
    *no_of_vars = (int) unpack_var(par, 2, 24);
    *var1 = (int) unpack_var(par, 8, 0);
    *var2 = (int) unpack_var(par, 8, 8);
    *var3 = (int) unpack_var(par, 8, 16);
    return 1;
}
