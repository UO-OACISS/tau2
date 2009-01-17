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

#ifdef TAU_WINDOWS
  #include <io.h>
#else
  #include <unistd.h>
  #include <sys/time.h>
  #define O_BINARY 0
#endif


#ifdef TAU_LARGEFILE
  #define LARGEFILE_OPTION O_LARGEFILE
#else
  #define LARGEFILE_OPTION 0
#endif


#include <string.h>

#define TRACING_ON
#include <Profile/TauTrace.h>

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
  x_uint64  firsttime; /* -- timestamp of first event record           -- */
  x_uint64  lasttime;  /* -- timestamp of previous event record        -- */

/*   PCXX_EV  *buffer;   -- input buffer                              -- */
/*   PCXX_EV  *erec;     -- current event record                      -- */
/*   PCXX_EV  *next;     -- next available event record in buffer     -- */
/*   PCXX_EV  *last;     -- last event record in buffer               -- */


  void  *buffer;    /* -- input buffer                              -- */
  void  *erec;      /* -- current event record                      -- */
  void  *next;      /* -- next available event record in buffer     -- */
  void  *last;      /* -- last event record in buffer               -- */


  int           format;    /* see above */
  int           eventSize; /* sizeof() the corresponding format struct */



} intrc;


struct trcrecv
{
  int      fd;        			  /* -- input file descriptor         			-- */
/*   PCXX_EV  *buffer;   		      /\* -- input buffer                  			-- *\/ */
/*   PCXX_EV  *erec;     			  /\* -- current event record         			-- *\/ */
/*   PCXX_EV  *prev;       		  /\* -- prev available event record in buffer   -- *\/ */
/*   PCXX_EV  *first;    			  /\* -- first event record in buffer            -- *\/ */


  void  *buffer;    /* -- input buffer                              -- */
  void  *erec;      /* -- current event record                      -- */
  void  *prev;      /* -- prev available event record in buffer     -- */
  void  *first;     /* -- first event record in buffer               -- */


  int           format;    /* see above */
  int           eventSize; /* sizeof() the corresponding format struct */

};

void *tmpbuffer; /* for threaded program */


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



/* copied from TAU_tf_decl.h  */

#include "Profile/tau_types.h"



#define FORMAT_NATIVE  0   /* as a fallback */
#define FORMAT_32      1
#define FORMAT_64      2
#define FORMAT_32_SWAP 3
#define FORMAT_64_SWAP 4


/* for 32 bit platforms */
typedef struct {
  x_int32            ev;    /* -- event id        -- */
  x_uint16           nid;   /* -- node id         -- */
  x_uint16           tid;   /* -- thread id       -- */
  x_int64            par;   /* -- event parameter -- */
  x_uint64           ti;    /* -- time [us]?      -- */
} PCXX_EV32;

/* for 64 bit platforms */
typedef struct {
  x_int64            ev;    /* -- event id        -- */
  x_uint16           nid;   /* -- node id         -- */
  x_uint16           tid;   /* -- thread id       -- */
  x_uint32           padding; /*  space wasted for 8-byte aligning the next item */ 
  x_int64            par;   /* -- event parameter -- */
  x_uint64           ti;    /* -- time [us]?      -- */
} PCXX_EV64;


typedef PCXX_EV PCXX_EV_NATIVE;



#define swap16(A)  ((((x_uint16)(A) & 0xff00) >> 8) | \
                   (((x_uint16)(A) & 0x00ff) << 8))
#define swap32(A)  ((((x_uint32)(A) & 0xff000000) >> 24) | \
                   (((x_uint32)(A) & 0x00ff0000) >> 8)  | \
                   (((x_uint32)(A) & 0x0000ff00) << 8)  | \
                   (((x_uint32)(A) & 0x000000ff) << 24))
#define swap64(A)  ((((x_uint64)(A) & 0xff00000000000000ull) >> 56) | \
                    (((x_uint64)(A) & 0x00ff000000000000ull) >> 40) | \
                    (((x_uint64)(A) & 0x0000ff0000000000ull) >> 24) | \
                    (((x_uint64)(A) & 0x000000ff00000000ull) >> 8) | \
                    (((x_uint64)(A) & 0x00000000ff000000ull) << 8) | \
                    (((x_uint64)(A) & 0x0000000000ff0000ull) << 24)  | \
                    (((x_uint64)(A) & 0x000000000000ff00ull) << 40)  | \
                    (((x_uint64)(A) & 0x00000000000000ffull) << 56))
  


  
/* copied from TAU_tf_decl.h  */


x_int32 event_GetEv(struct trcdescr *tFile, void *event, int index) {
  PCXX_EV_NATIVE *nativeEvent;
  PCXX_EV32 *event32;
  PCXX_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (PCXX_EV_NATIVE*)event;
    return nativeEvent[index].ev;
    
  case FORMAT_32:
    event32 = (PCXX_EV32*) event;
    return event32[index].ev;
  case FORMAT_32_SWAP:
    event32 = (PCXX_EV32*) event;
    return swap32(event32[index].ev);

  case FORMAT_64:
    event64 = (PCXX_EV64*) event;
    return event64[index].ev;
  case FORMAT_64_SWAP:
    event64 = (PCXX_EV64*) event;
    return swap64(event64[index].ev);
  }
  return 0;
}

void event_SetEv(struct trcdescr *tFile, void *event, int index, x_int32 value) {
  PCXX_EV_NATIVE *nativeEvent;
  PCXX_EV32 *event32;
  PCXX_EV64 *event64;
  x_int64 tmpValue;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (PCXX_EV_NATIVE*)event;
    nativeEvent[index].ev = value;
    break;

  case FORMAT_32:
    event32 = (PCXX_EV32*) event;
    event32[index].ev = value;
    break;
  case FORMAT_32_SWAP:
    event32 = (PCXX_EV32*) event;
    event32[index].ev = swap32(value);
    break;

  case FORMAT_64:
    event64 = (PCXX_EV64*) event;
    event64[index].ev = value;
    break;
  case FORMAT_64_SWAP:
    tmpValue = value;
    event64 = (PCXX_EV64*) event;
    event64[index].ev = swap64(tmpValue);
    break;
  }
  return;
}

x_uint64 event_GetTi(struct trcdescr *tFile, void *event, int index) {
  PCXX_EV_NATIVE *nativeEvent;
  PCXX_EV32 *event32;
  PCXX_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (PCXX_EV_NATIVE*)event;
    return nativeEvent[index].ti;
    
  case FORMAT_32:
    event32 = (PCXX_EV32*) event;
    return event32[index].ti;
  case FORMAT_32_SWAP:
    event32 = (PCXX_EV32*) event;
    return swap64(event32[index].ti);

  case FORMAT_64:
    event64 = (PCXX_EV64*) event;
    return event64[index].ti;
  case FORMAT_64_SWAP:
    event64 = (PCXX_EV64*) event;
    return swap64(event64[index].ti);
  }
  return 0;
}

void event_SetTi(struct trcdescr *tFile, void *event, int index, x_uint64 value) {
  PCXX_EV_NATIVE *nativeEvent;
  PCXX_EV32 *event32;
  PCXX_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (PCXX_EV_NATIVE*)event;
    nativeEvent[index].ti = value;
    break;
  case FORMAT_32:
    event32 = (PCXX_EV32*) event;
    event32[index].ti = value;
    break;
  case FORMAT_32_SWAP:
    event32 = (PCXX_EV32*) event;
    event32[index].ti = swap64(value);
    break;
  case FORMAT_64:
    event64 = (PCXX_EV64*) event;
    event64[index].ti = value;
    break;
  case FORMAT_64_SWAP:
    event64 = (PCXX_EV64*) event;
    event64[index].ti = swap64(value);
    break;
  }
  return;
}


x_uint16 event_GetNid(struct trcdescr *tFile, void *event, int index) {
  PCXX_EV_NATIVE *nativeEvent;
  PCXX_EV32 *event32;
  PCXX_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (PCXX_EV_NATIVE*)event;
    return nativeEvent[index].nid;
    
  case FORMAT_32:
    event32 = (PCXX_EV32*) event;
    return event32[index].nid;
  case FORMAT_32_SWAP:
    event32 = (PCXX_EV32*) event;
    return swap16(event32[index].nid);

  case FORMAT_64:
    event64 = (PCXX_EV64*) event;
    return event64[index].nid;
  case FORMAT_64_SWAP:
    event64 = (PCXX_EV64*) event;
    return swap16(event64[index].nid);
  }
  return 0;
}

void event_SetNid(struct trcdescr *tFile, void *event, int index, x_uint16 value) {
  PCXX_EV_NATIVE *nativeEvent;
  PCXX_EV32 *event32;
  PCXX_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (PCXX_EV_NATIVE*)event;
    nativeEvent[index].nid = value;
    break;

  case FORMAT_32:
    event32 = (PCXX_EV32*) event;
    event32[index].nid = value;
    break;
  case FORMAT_32_SWAP:
    event32 = (PCXX_EV32*) event;
    event32[index].nid = swap16(value);
    break;

  case FORMAT_64:
    event64 = (PCXX_EV64*) event;
    event64[index].nid = value;
    break;
  case FORMAT_64_SWAP:
    event64 = (PCXX_EV64*) event;
    event64[index].nid = swap16(value);
    break;
  }
  return;
}


x_uint16 event_GetTid(struct trcdescr *tFile, void *event, int index) {
  PCXX_EV_NATIVE *nativeEvent;
  PCXX_EV32 *event32;
  PCXX_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (PCXX_EV_NATIVE*)event;
    return nativeEvent[index].tid;
    
  case FORMAT_32:
    event32 = (PCXX_EV32*) event;
    return event32[index].tid;
  case FORMAT_32_SWAP:
    event32 = (PCXX_EV32*) event;
    return swap16(event32[index].tid);

  case FORMAT_64:
    event64 = (PCXX_EV64*) event;
    return event64[index].tid;
  case FORMAT_64_SWAP:
    event64 = (PCXX_EV64*) event;
    return swap16(event64[index].tid);
  }
  return 0;
}


x_uint64 event_GetPar(struct trcdescr *tFile, void *event, int index) {
  PCXX_EV_NATIVE *nativeEvent;
  PCXX_EV32 *event32;
  PCXX_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (PCXX_EV_NATIVE*)event;
    return nativeEvent[index].par;
    
  case FORMAT_32:
    event32 = (PCXX_EV32*) event;
    return event32[index].par;
  case FORMAT_32_SWAP:
    event32 = (PCXX_EV32*) event;
    return swap64(event32[index].par);

  case FORMAT_64:
    event64 = (PCXX_EV64*) event;
    return event64[index].par;
  case FORMAT_64_SWAP:
    event64 = (PCXX_EV64*) event;
    return swap64(event64[index].par);
  }
  return 0;
}


void determineFormat(struct trcdescr *tdes) {
  int bytesRead;
  int formatFound;
  PCXX_EV32 event32;
  PCXX_EV64 event64;

  formatFound = 0;
/*   printf ("determining format!\n"); */
/*   printf ("sizeof(PCXX_EV32) = %d\n", sizeof(PCXX_EV32)); */
/*   printf ("sizeof(PCXX_EV64) = %d\n", sizeof(PCXX_EV64)); */


/*   printf ("par32 : %d\n", (long)&event32.par - (long)&event32); */
/*   printf ("par64 : %d\n", (long)&event64.par - (long)&event64); */


/*   lseek(tdes->fd, 0, SEEK_SET); */
  bytesRead = read(tdes->fd, &event32, sizeof(PCXX_EV32));
  lseek(tdes->fd, 0, SEEK_SET);
  bytesRead = read(tdes->fd, &event64, sizeof(PCXX_EV64));
  lseek(tdes->fd, 0, SEEK_SET);

  /* 32 bit regular */
  if (event32.par == 3) {
    tdes->format = FORMAT_32;
    tdes->eventSize = sizeof(PCXX_EV32);
    formatFound = 1;
/*     printf ("32 regular!\n"); */
  }


  /* 32 bit swapped */
  if (swap64(event32.par) == 3) {
    if (formatFound == 1) { /* shouldn't happen, if it does, go to native */
      tdes->format = FORMAT_NATIVE;
      tdes->eventSize = sizeof(PCXX_EV_NATIVE);
      return;
    }
    tdes->format = FORMAT_32_SWAP;
    tdes->eventSize = sizeof(PCXX_EV32);
    formatFound = 1;
/*     printf ("32 swapped!\n"); */
  }

  /* 64 bit regular */
  if (event64.par == 3) {
    if (formatFound == 1) { /* shouldn't happen, if it does, go to native */
      tdes->format = FORMAT_NATIVE;
      tdes->eventSize = sizeof(PCXX_EV_NATIVE);
      return;
    }
    tdes->format = FORMAT_64;
    tdes->eventSize = sizeof(PCXX_EV64);
    formatFound = 1;
/*     printf ("64 regular!\n"); */
  }


  /* 64 bit swapped */
  if (swap64(event64.par) == 3) {
    if (formatFound == 1) { /* shouldn't happen, if it does, go to native */
      tdes->format = FORMAT_NATIVE;
      tdes->eventSize = sizeof(PCXX_EV_NATIVE);
      return;
    }
    tdes->format = FORMAT_64_SWAP;
    tdes->eventSize = sizeof(PCXX_EV64);
    formatFound = 1;
/*     printf ("64 swapped!\n"); */
  }

  if (formatFound == 0) {
    fprintf (stderr, "Warning: couldn't determine format, using native!\n");
    tdes->format = FORMAT_NATIVE;
    tdes->eventSize = sizeof(PCXX_EV_NATIVE);
  }

/*   printf ("event32.par = 0x%x\n", event32.par); */
/*   printf ("swap32(event32.par) = 0x%x\n", swap32(event32.par)); */
/*   printf ("event64.par = 0x%x\n", event64.par); */
/*   printf ("swap64(event64.par) = 0x%x\n", swap64(event64.par)); */

/*   printf ("---------------\n"); */
/*   printf ("event64.ev = 0x%llx\n", swap64(event64.ev)); */
/*   printf ("event64.nid = 0x%x\n",  swap64(event64.nid)); */
/*   printf ("event64.tid = 0x%x\n",  swap64(event64.tid)); */
/*   printf ("event64.padding = 0x%x\n",  swap64(event64.padding)); */
/*   printf ("event64.par = 0x%x\n",  swap64(event64.par)); */
/*   printf ("event64.ti = 0x%llx\n",  swap64(event64.ti)); */

/*   printf ("---------------\n"); */
/*   printf ("event32.ev = 0x%llx\n", swap32(event32.ev)); */
/*   printf ("event32.nid = 0x%x\n",  swap32(event32.nid)); */
/*   printf ("event32.tid = 0x%x\n",  swap32(event32.tid)); */
/*   printf ("event32.par = 0x%x\n",  swap32(event32.par)); */
/*   printf ("event32.ti = 0x%llx\n",  swap32(event32.ti)); */

/*   printf ("---------------\n"); */
/*   printf ("swap(event32.ev) = %lu\n", swap32(event32.ev)); */
/*   printf ("swap(event32.nid) = %u\n",  swap32(event32.nid)); */
/*   printf ("swap(event32.tid) = %u\n",  swap32(event32.tid)); */
/*   printf ("swap(event32.par) = %u\n",  swap32(event32.par)); */
/*   printf ("swap(event32.ti) = %llu\n",  swap64(event32.ti)); */
}









void decodeParameter(x_uint64 xpar,  int *otherNid, int *msgLen, int *msgTag, int *comm) {
  /* See RtsLayer::TraceSendMsg for documentation on the bit patterns of "parameter" */
  *msgTag   = ((xpar>>16) & 0x000000FF) | (((xpar >> 48) & 0xFF) << 8);
  *otherNid = ((xpar>>24) & 0x000000FF) | (((xpar >> 56) & 0xFF) << 8);
  *msgLen   = xpar & 0x0000FFFF | (xpar << 22 >> 54 << 16);
  *comm     = xpar << 16 >> 58;
}



int GetNodeId(struct trcdescr *trc, void *rec);
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
    if ( outFormat == alog ) {
      /* sprintf (newev->param, "%s: %%d", p); */
    } else {
      sprintf (newev->param, "%s", p);
    }
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
  {    newev->state = (char *) malloc (strlen(state) + 1);
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

static int IsDoubleParam(int id) {
  int h;
  EVDESCR *ev;

  /* TAU uses doubles for internal representation but for some reason writes 64-bit unsigned integers to traces
     I had added this IsDoubleParam and it is used below when I changed it to write doubles for the traces, but I
     never commited the changes since I didn't want to break compatibility */
  
  /* comment this out if TAU stores user events as doubles */
  return 0;

  h = id % numEvent;
  ev = evtable[h];

  while ( ev ) {
    if ( ev->id == id ) { 
      if (strncmp(ev->param,"TriggerValue",strlen("TriggerValue")) == 0) {
	return 1;
      } else {
	return 0;
      }
    }
    ev = ev->next;
  }

  return 0;
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

int isInitEvent(struct trcdescr *trc, void *erecord)
{
  char *eventName;
  int hasParam;

  eventName = GetEventName(event_GetEv(trc,erecord,0), &hasParam);

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
    if  ( (event_GetEv(trc,erecord,0) == PCXX_EV_INIT) || 
	  (event_GetEv(trc,erecord,0) == PCXX_EV_INITM) )
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

static void *get_next_rec (struct trcdescr *tdes) {
  long no;

  if ( (tdes->last == NULL) || (tdes->next > tdes->last) ) {
    /* -- input buffer empty: read new records from file -------------------- */
    if ( (no = read (tdes->fd, tdes->buffer, INMAX * tdes->eventSize))
         != (INMAX * tdes->eventSize) ) {
      if ( no == 0 ) {
	/* no more records */
        return NULL;
      } else if ( (no % tdes->eventSize) != 0 ) {
        /* -- read error: --------------------------------------------------- */
	printf("Read error: %ld and %ld\n",no,tdes->eventSize);
        fprintf (stderr, "%s: read error\n", tdes->name);
        exit (1);
      }
    }

    /* -- we got some event records ----------------------------------------- */
    tdes->next = tdes->buffer;
    tdes->last = (char*)tdes->buffer + no - tdes->eventSize;
  }

  tdes->erec = tdes->next;
  tdes->next = (void*)(((char*)tdes->next) + tdes->eventSize);
  return tdes->erec;
}

static void *get_prev_rec (struct trcrecv *tdes) {
 /* Before calling this the first time set first properly. */
  long no;
  off_t last_position;

  last_position = lseek(tdes->fd, 0, SEEK_CUR);

/* We reuse last and next to actually mean first and prev respectively */
/* i.e., before calling this the first time set tdes->last = tdes->buffer */
  /* if prev < first, go fetch more records */
/* to debug: print each record */

  if (( last_position == 0) || (tdes->prev < tdes->first)) {
    /* move the pointer 2*INMAX*sizeof(PCXX_EV) earlier */
    last_position -= 2*INMAX*tdes->eventSize;
#ifdef DEBUG
    printf("last_position = %d\n", last_position);
#endif /* DEBUG */
    if (last_position < 0) return NULL;
    lseek(tdes->fd, last_position, SEEK_SET);
    /* -- input buffer empty: read new records from file -------------------- */
    if ( (no = read (tdes->fd, tdes->buffer, INMAX * tdes->eventSize))
         != (INMAX * tdes->eventSize) ) {
      if ( no == 0 ) {
        /* -- no more event record: ----------------------------------------- */
        return NULL;
      } else if ( no % tdes->eventSize != 0 ) {
        /* -- read error: --------------------------------------------------- */
        fprintf (stderr, "read error in get_prev_rec\n");
        exit (1);
      }
    }
    
    /* -- we got some event records ----------------------------------------- */
    tdes->prev =  (char*)tdes->buffer + no - tdes->eventSize;
    tdes->first = tdes->buffer ;
  }
  
  tdes->erec = tdes->prev;
  tdes->prev = (void*)(((char*)tdes->prev) - tdes->eventSize);
  return tdes->erec;
}

int GetNodeId(struct trcdescr *trc, void *rec)
{
  if (threads)
  {
    /* OLD
    return rec->tid;
    */
    return offset[event_GetNid(trc,rec,0)] + event_GetTid(trc,rec,0);
	/* CPUID ranges from 0..N-1: N is sum(threads on all nodes ) */
  }
  else {
    return event_GetNid(trc,rec,0);
  }
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
  /* PCXX_EV *curr_rec; */
  void *curr_rec;
  EVDESCR *curr_ev;
  int curr_tag, curr_len, curr_nid, dummy, comm;
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
/*     curr_ev = GetEventStruct (curr_rec->ev); */
    curr_ev = GetEventStruct (event_GetEv(&trcdes,curr_rec,0));

   /* Find the matching send and receive */
   /* is the current record of the complementary IPC type? */
   if (curr_ev->tag == RECV_EVENT)
   {
     /* possible match */

     decodeParameter(event_GetPar(&trcdes,curr_rec,0), &dummy, &curr_len, &curr_tag, &comm);
     curr_nid = event_GetNid(&trcdes,curr_rec,0);
#ifdef DEBUG
     printf("Possible match... tag=%d, len=%d, nid=%d\n", curr_tag, curr_len, curr_nid);
#endif /* DEBUG */
     if ((curr_tag == msgtag) && (curr_len == msglen) && (curr_nid == otherid ))
     {
       *other_tid = event_GetTid(&trcdes,curr_rec,0);
       *other_nodeid = event_GetNid(&trcdes,curr_rec,0);
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
  void *curr_rec;
  EVDESCR *curr_ev;
  int curr_tag, curr_len, curr_nid, dummy, comm;
  struct trcrecv rcvdes;
  struct trcrecv *ted; 

  rcvdes.buffer    = tmpbuffer;
  rcvdes.erec	   = trcdes.erec;
  rcvdes.fd 	   = trcdes.fd;
  rcvdes.first 	   = trcdes.buffer;
  rcvdes.prev	   = (char*)trcdes.erec - trcdes.eventSize;
  rcvdes.eventSize = trcdes.eventSize;


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

  ted = &rcvdes;



  while (( curr_rec = get_prev_rec(&rcvdes)) != NULL)
  {
    /* Get the event type for this record */
    curr_ev = GetEventStruct (event_GetEv(&trcdes,curr_rec,0));

   /* Find the matching send and receive */
   /* is the current record of the complementary IPC type? */
   if (curr_ev->tag == SEND_EVENT)
   {
     /* possible match */
     decodeParameter(event_GetPar(&trcdes,curr_rec,0), &dummy, &curr_len, &curr_tag, &comm);
     curr_nid = event_GetNid(&trcdes,curr_rec,0);
#ifdef DEBUG
     printf("Possible match... tag=%d, len=%d, nid=%d\n", curr_tag, curr_len, curr_nid);
#endif /* DEBUG */

     if ((curr_tag == msgtag) && (curr_len == msglen) && (curr_nid == otherid ))
     {
       *other_tid = event_GetTid(&trcdes,curr_rec,0);
       *other_nodeid = event_GetNid(&trcdes,curr_rec,0);
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

x_int64 GetMatchingRecvPRV(struct trcdescr trcdes, int msgtag,
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
  x_int64 phRecv = 0;
  off_t last_position;
  void *curr_rec;
  EVDESCR *curr_ev;
  int curr_tag, curr_len, curr_nid, dummy, comm;
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
    curr_ev = GetEventStruct (event_GetEv(&trcdes,curr_rec,0));

   /* Find the matching send and receive */
   /* is the current record of the complementary IPC type? */
   if (curr_ev->tag == RECV_EVENT)
   {
     /* possible match */
     decodeParameter(event_GetPar(&trcdes,curr_rec,0), &dummy, &curr_len, &curr_tag, &comm);
     curr_nid = event_GetNid(&trcdes,curr_rec,0);
#ifdef DEBUG
     printf("Possible match... tag=%d, len=%d, nid=%d\n", curr_tag, curr_len, curr_nid);
#endif /* DEBUG */
     if ((curr_tag == msgtag) && (curr_len == msglen) && (curr_nid == otherid ))
     {
       *other_tid = event_GetTid(&trcdes,curr_rec,0);
       *other_nodeid = event_GetNid(&trcdes,curr_rec,0);
       phRecv = event_GetTi(&trcdes,curr_rec,0) - intrc.firsttime;
#ifdef DEBUG
       printf("PERFECT MATCH! other tid = %d, nid = %d\n",
	 *other_tid, *other_nodeid);
#endif /* DEBUG */
       /* Reset trace file */
       lseek(trcdes.fd, last_position, SEEK_SET);
       return phRecv;
     }
     /* This only applies to Send! */
   }
  }
  /* EOF : didn't find the matching send. Reset and leave */
#ifdef DEBUG
  printf("Didn't find matching ipc...\n");
#endif /* DEBUG */
  lseek(trcdes.fd, last_position, SEEK_SET);
  return phRecv;

}

/* -------------------------------------------------------------------------- */
/* -- PCXX_CONVERT MAIN PROGRAM --------------------------------------------- */
/* -------------------------------------------------------------------------- */

# define LINEMAX 64*1024

int main (int argc, char *argv[])
{
  FILE *outfp, *inev,*pcffp;
  void *erec;
  int i,j,k,l;
  int nodeId, totalnodes = 0;
  int num;
  int tag;
  int myid, otherid, msglen, msgtag, comm;
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

  myid = 0;
  otherid = 0;
  msglen = 0;
  msgtag = 0;
  comm = 0;
  other_tid = 0;
  other_nodeid = 0;

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

  if ( argc < 4 )
  {
    fprintf (stderr, "usage: %s [-alog | -SDDF | -dump | -paraver [-t] |", argv[0]);
    fprintf (stderr, " -pv | -vampir [-longsymbolbugfix] [-compact] [-user|-class|-all] [-nocomm]]");
    fprintf (stderr, " inputtrc edffile [outputtrc]\n");
    fprintf (stderr, " Note: -vampir option assumes multiple threads/node\n");
    fprintf (stderr, " Note: -t option used in conjunction with -paraver option assumes multiple threads/node\n");
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
    i = 2;
    while( argv[i][0] == '-' )
      {
	if( strcmp (argv[i], "-t") == 0 ){
	  threads = TRUE;
	}
	else break;
	i++;
      }
    fileIdx = i;

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
  if ( (intrc.fd = open (inFile, O_RDONLY | O_BINARY | LARGEFILE_OPTION)) < 0 )
  {
    perror (inFile);
    exit (1);
  }
  else
  {
    determineFormat(&intrc);
    intrc.name      = inFile;
    intrc.buffer    = malloc (INMAX * intrc.eventSize);
    intrc.erec      = NULL;
    intrc.next      = NULL;
    intrc.last      = NULL;
    intrc.overflows = 0;

    if((threads) || (outFormat == paraver))
      tmpbuffer = malloc (INMAX * intrc.eventSize);

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
	intrc.numproc = event_GetNid(&intrc,erec,0);
      }
      else
      { /* No threads */
        intrc.numproc   = GetNodeId(&intrc,erec);
      }
      intrc.firsttime = event_GetTi(&intrc,erec,0);
      intrc.lasttime  = event_GetTi(&intrc,erec,0);
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

    if ( (outfp = fopen (outFile, "wb")) == NULL )
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
  if ( (inev = fopen (edfFile, "rb")) == NULL )
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
#ifdef DEBUG
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

      /* Fix 13/10 to 10 for event files generated with windows */
/*       if (param[strlen(param)-2] == 13) { */
/*  	param[strlen(param)-2] = 10; */
/*  	param[strlen(param)-1] = 0; */
/*       } */

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
    if ( (i = GetEvent (event_GetEv(&intrc,erec,0))) == 0 )
    {
      fprintf (stderr, "%s: unknown event type %d in event record %ld\n",
               intrc.name, event_GetEv(&intrc,erec,0), intrc.numrec);
    }
    else if ( isInitEvent(&intrc,erec))
    {
      numproc++;
    }

    if ( (erec = get_next_rec (&intrc)) == NULL )
      break;
    else
    {
      intrc.numrec++;

      /* -- check clock overflow ---------------------------------------- */
      if ( event_GetTi(&intrc,erec,0) < intrc.lasttime ) intrc.overflows++;
      intrc.lasttime = event_GetTi(&intrc,erec,0);


      /* -- check thread id -------------------------------------------- */
      if (threads)
      {
        /* -- check node id -------------------------------------------- */
        if ( event_GetNid(&intrc,erec,0) > totalnodes ) totalnodes = event_GetNid(&intrc,erec,0);

        /* totalnodes has node id in the range 0..N-1 */

	if ( maxtid[event_GetNid(&intrc,erec,0)] < event_GetTi(&intrc,erec,0) ) maxtid[event_GetNid(&intrc,erec,0)] = event_GetTi(&intrc,erec,0);
	/* Update the max thread id vector for this node for each record */
 	/* printf("maxtid[%d] = %d\n", totalnodes, maxtid[totalnodes]); */
      }
      else
      { /* no threads */
        if ( GetNodeId(&intrc,erec) > intrc.numproc ) intrc.numproc = GetNodeId(&intrc,erec);
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

#ifdef TAU_WINDOWS
    fprintf (outfp, "#   first timestamp: %I64u\n", intrc.firsttime);
    fprintf (outfp, "#    last timestamp: %I64u\n\n", intrc.lasttime);
#else
    fprintf (outfp, "#   first timestamp: %llu\n", intrc.firsttime);
    fprintf (outfp, "#    last timestamp: %llu\n\n", intrc.lasttime);
#endif
    fprintf (outfp, "#=NO= =======================EVENT==");
    fprintf (outfp, " ==TIME [us]= =NODE= =THRD= ==PARAMETER=\n");
  }
  else if ( outFormat == paraver ){
      char date[50];
      
#ifndef TAU_WINDOWS      
      struct timeval tp;
      struct timezone tzp;
      time_t clock;
      struct tm *ptm;
#endif      
      int i;
      int j;
      int size;
      char *taskList;
      char *tempList;
      int myNodeNum = 0;

      /* The code from libseqparaver.c is used here */
      /* Get the date in the form dd/mm/yy */

#ifdef TAU_WINDOWS
      sprintf (date, "Unsupported in Win32");
#else
      gettimeofday (&tp, &tzp);
      clock = tp.tv_sec;
      ptm = localtime (&clock);
      strftime (date, 50, "%d/%m/%y at %H:%M", ptm);
#endif
      /*Allocate space for the taskList portion of the header
	/if using threads, then numCpus = totalnodes.  Otherwise
	/use intrc.numproc.*/
      if(threads){
	myNodeNum = totalnodes + 1;
      }
      else{
	myNodeNum = intrc.numproc + 1;
      }
      
      size = ((myNodeNum) * 5);
    
      taskList = (char*)malloc(size * sizeof (char));
      tempList = (char*)malloc(size * sizeof (char));
      if(taskList == (char*)NULL){
	perror("Malloc error\n");
      }
      if(tempList == (char*)NULL){
	perror("Malloc error\n");
      }
      taskList[0] = '\0';
      
      /*Write the taskList portion of the header in the form
	/ "numThreads:onCpuId n,numThreads:onCpuId n+1"*/
      for(i = 0; i < myNodeNum; i++){
	if(myNodeNum - i != 1){
	  if(tempList == (char*)NULL){
	    perror("Malloc error-Paraver Header Var. tempList");
	  }
	  sprintf(tempList,"%d:%d,",maxtid[i]+1,i+1);
	  if(taskList == (char*)NULL){
	    perror("Malloc error-Paraver Header Var. taskList");
	  }	  
	  taskList = strcat(taskList,tempList);
	}
	
	else{
	  sprintf(tempList,"%d:%d",maxtid[i]+1,i+1);
	  taskList = strcat(taskList,tempList);
	}
	
      }
     

      /* create the first part of the pcf File to be loaded
	 automatically with the .prv file in paraver
         change the file extension of outFile from prv to pcf
      */

      if(! outFile){
	outfp = stdout;
	pcfFile = (char*)malloc(30 * sizeof(char));
	strcpy(pcfFile,"config.pcf");
      }
      else{
	if(strlen(outFile) < 5){
	  printf("Outfile must be of form *.prv\n");
	  exit(1);
	}


	pcfFile = (char*)malloc((strlen(outFile)+1) * sizeof (char));
	pcfFile = (strcpy(pcfFile,outFile));
	pcfFile[strlen(outFile) - 3] = '\0';
	pcfFile = (strcat(pcfFile,"pcf"));
      }
      if ( pcfFile ){
	if ( access ("pcfFile", F_EXISTS) == 0  && isatty(2) ){
	  fprintf (stderr, "%s exists; override [y]? ", pcfFile);
	  if ( getchar() == 'n' ) exit (1);
	}

	if ( (pcffp = fopen (pcfFile, "wb")) == NULL ){
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
      fprintf(pcffp,"NUM_OF_STATE_COLORS 1\n");
      fprintf(pcffp,"YMAX_SCALE          %d\n\n\n",intrc.numrec);
      fprintf(pcffp,"DEFAULT_SEMANTIC\n\n");
      fprintf(pcffp,"COMPOSE1_FUNC       As Is\n");
      fprintf(pcffp,"THREAD_FUNC         Last Evt Val\n\n\n");
      fprintf(pcffp,"STATES\n");
      fprintf(pcffp,"0    NOT USED\n");
      fprintf(pcffp,"1    RUNNING\n");
      fprintf(pcffp,"2    NOT USED\n");
      fprintf(pcffp,"3    NOT USED\n");
      fprintf(pcffp,"4    NOT USED\n");
      fprintf(pcffp,"5    NOT USED\n");
      fprintf(pcffp,"6    NOT USED\n");
      fprintf(pcffp,"7    NOT USED\n");
      fprintf(pcffp,"8    NOT USED\n");
      fprintf(pcffp,"9    NOT USED\n");
      fprintf(pcffp,"4    NOT USED\n");
      fprintf(pcffp,"10    NOT USED\n");
      fprintf(pcffp,"11    NOT USED\n");
      fprintf(pcffp,"12    NOT USED\n");
      fprintf(pcffp,"13    NOT USED\n\n");
      fprintf(pcffp,"STATES_COLOR\n");
      fprintf(pcffp,"1    {117,195,255}\n");
      fprintf(pcffp,"EVENT_TYPE\n");
      fprintf(pcffp,"0    5    METHOD entry/exit:\n");
      fprintf(pcffp,"VALUES\n");
      

      prvPCF = (int*)malloc(intrc.numevent * sizeof(int));
      for(i = 0; i < intrc.numevent; i++){
	prvPCF[i] = 0;
      }
      
      fprintf (outfp, "#Paraver (%s):%llu:%d:1:%d(%s)\n", date,
	       (intrc.lasttime - intrc.firsttime),myNodeNum,myNodeNum,taskList);
      /*for each thread on each node write at the beginning of the
	/ trace file a state record indicating "RUNNING" state*/
      for(i = 0; i < myNodeNum; i++){
	for(j = 0; j <= maxtid[i]; j++){
	  fprintf(outfp,"1:%d:1:%d:%d:0:%llu:1\n",i+1,i+1,j+1,(intrc.lasttime-intrc.firsttime));
	}
      }
    
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

  PrintEventDescr (outfp);

  if (outFormat == pv)
  {
    /* print idle records */
    for (i=0; i<=intrc.numproc; i++)
    {
      fprintf (outfp, "0 EXCHANGE ON CPUID %d DOWNTO IDLE -99 CLUSTER 1\n",
		      i+1);
    }
  }

  /* ------------------------------------------------------------------------ */
  /* -- re-open input trace                                                -- */
  /* ------------------------------------------------------------------------ */
  if ( (intrc.fd = open (inFile, O_RDONLY | O_BINARY | LARGEFILE_OPTION)) < 0 )
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
    intrc.firsttime = event_GetTi(&intrc,erec,0);
    intrc.lasttime  = event_GetTi(&intrc,erec,0);
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
    if ( event_GetEv(&intrc,erec,0) == PCXX_BARRIER_ENTER )
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
    else if ( event_GetEv(&intrc,erec,0) == PCXX_BARRIER_EXIT )
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
      i = GetEvent (event_GetEv(&intrc,erec,0));
      fprintf (outfp, "%3d %3d 0 %10lld %d %10llu\n",
        i,                /* event type */
        GetNodeId(&intrc,erec),        /* process id */
        event_GetPar(&intrc,erec,0),        /* integer parameter */
        intrc.overflows,  /* clock cycle */
        event_GetTi(&intrc,erec,0));        /* timestamp */
    }
    else if ( outFormat == SDDF )
    {
      ptr = GetEventName (event_GetEv(&intrc,erec,0), &hasParam);
      if ( hasParam )
        fprintf (outfp, "%s { %llu, %d, %d, %lld };;\n\n",
                 ptr, event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec), event_GetTid(&intrc,erec,0), event_GetPar(&intrc,erec,0));
      else
        fprintf (outfp, "%s { %llu, %d, %d };;\n\n",
                 ptr, event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec), event_GetTid(&intrc,erec,0));
    }

    else if( outFormat == paraver ){
      x_int64 phRecv = 0;
      ev = GetEventStruct (event_GetEv(&intrc,erec,0));
      if ((ev->tag != 0) || (dynamictrace))
	{
	  if( (ev->tag == SEND_EVENT) && pvComm )
	    {
	      myid 	= GetNodeId(&intrc,erec) + 1;

	      decodeParameter(event_GetPar(&intrc,erec,0), &otherid, &msglen, &msgtag, &comm);
	      otherid = otherid+1;
	      
	      phRecv = GetMatchingRecvPRV(intrc, msgtag, GetNodeId(&intrc,erec),
					  otherid -1 , msglen, &other_tid, &other_nodeid);

	      
	      
#ifdef DEBUG
	      printf ("\n\n%llu SENDMSG %d FROM %d TO %d LEN %d\n\n\n",
		      event_GetTi(&intrc,erec,0) - intrc.firsttime,
		      msgtag, myid , otherid , msglen);
#endif /* DEBUG */
	      
	      /*Print the record in the form:
		RecordType(3)for Communication:SendNodeId:SendPTaskID:
		SendTaskID:SendThreadID:logicalSendTime:PhysicalSendTime:
		RecvNodeID:RecvPTaskID:RecvTaskID:RecvThreadID:
		:LogicalRecvTime:PhysicalRecvTime:MessageLength:MessageTag
	      */

/* 	      fprintf(outfp,"3:%d:1:%d:%d:%llu:%llu:%d:1:%d:%d:%llu:%llu:%lld:%lld\n",myid,myid,event_GetTid(&intrc,erec,0)+1,event_GetTi(&intrc,erec,0)-intrc.firsttime,event_GetTi(&intrc,erec,0) - intrc.firsttime,otherid,otherid,other_tid+1,phRecv,phRecv,msglen,msgtag); */


	      fprintf(outfp,"3:%d:1:%d:%d:%llu:%llu:%d:1:%d:%d:%llu:%llu:%d:%d\n",
		      myid,
		      myid,
		      event_GetTid(&intrc,erec,0),
		      (x_uint64)(event_GetTi(&intrc,erec,0) - intrc.firsttime),
		      (x_uint64)(event_GetTi(&intrc,erec,0) - intrc.firsttime),
		      otherid,
		      otherid,
		      other_tid+1,
		      phRecv,
		      phRecv,
		      msglen,
		      msgtag);

	   
	    }
	  else if ( (ev->tag == RECV_EVENT ) && pvComm )
	    {
	      /* receive message */
	      /* In dynamic trace the format for par is
		 31 ..... 24 23 ......16 15..............0
		 other       type          length
		 So, mynode is the receiver and its in GetNodeId(&intrc,erec)
		 RECVMSG <type> BY <receiver> FROM <sender> LEN <length>
	      */
	      /* extract the information from the parameter */
	      myid 		= GetNodeId(&intrc,erec)+1;

	      decodeParameter(event_GetPar(&intrc,erec,0), &otherid, &msglen, &msgtag, &comm);
	      otherid = otherid+1;
	      
	      
		  if (GetMatchingSend(intrc, msgtag, GetNodeId(&intrc,erec),
				      otherid - 1 , msglen, &other_tid, &other_nodeid))
		    { /* call was successful, we've the other_tid and other_nodeid */
		   
#ifdef DEBUG
		      printf("Calculated senderid = %d\n", otherid);
#endif /* DEBUG */
		      
		    }

	      
#ifdef DEBUG
	      printf ("%llu RECVMSG %d BY %d FROM %d LEN %d\n",
		      event_GetTi(&intrc,erec,0) - intrc.firsttime,
		      msgtag, myid , otherid , msglen);
#endif /* DEBUG */
	      
	    }
	  else if (( ev->tag == -9 ) || ( event_GetPar(&intrc,erec,0) == -1))
	    { /* In dynamic tracing, 1/-1 par values are for Entry/Exit resp. */
	      /* exit state */
	      /* PARVis needs time values relative to the start of the program! */
	      stkptr[GetNodeId(&intrc,erec)]--;
	      if ( stkptr[GetNodeId(&intrc,erec)] < statestk[GetNodeId(&intrc,erec)] )
		{
		  fprintf (stderr, "ERROR: stack underflow on node %d\n", GetNodeId(&intrc,erec));
		  fprintf (stderr, "       event %s at %llu\n", ev->name, event_GetTi(&intrc,erec,0));
		  exit (1);
		}

	      /*Print the record in the form:
		RecordType(2)for event:NodeID:PTaskID:TaskID:ThreadID:
		:event time:eventType(5) for method entry/exit:event tag(to
		distinguish between events of the same type, 0 for exit and
		event_GetEv(&intrc,erec,0) for entry
	      */
	 
	      fprintf (outfp, "2:%d:1:%d:%d:%llu:5:0\n",(event_GetNid(&intrc,erec,0))+1,(event_GetNid(&intrc,erec,0))+1,event_GetTid(&intrc,erec,0)+1,event_GetTi(&intrc,erec,0)-intrc.firsttime);
	      
	      /*Upon exit of a method, write another record indicating
		entry into the previous method on the stack for that
		particular node and thread
	      */
	      if(stkptr[GetNodeId(&intrc,erec)]->tag != -99)
		fprintf(outfp,"2:%d:1:%d:%d:%llu:5:%d\n",(event_GetNid(&intrc,erec,0))+1,(event_GetNid(&intrc,erec,0))+1,event_GetTid(&intrc,erec,0)+1,event_GetTi(&intrc,erec,0)-intrc.firsttime,stkptr[GetNodeId(&intrc,erec)]->tag);
	      if(prvPCF[event_GetEv(&intrc,erec,0)-1] == 0){
		fprintf(pcffp,"%d       %s\n",event_GetEv(&intrc,erec,0),GetEventName(event_GetEv(&intrc,erec,0),&hasParam));
		prvPCF[event_GetEv(&intrc,erec,0)-1] = 1;
	      }
	    }
	  else if (event_GetPar(&intrc,erec,0) == 1)
	    {
	      /* enter new state */
	      stkptr[GetNodeId(&intrc,erec)]++;
	      if ( stkptr[GetNodeId(&intrc,erec)] > (statestk[GetNodeId(&intrc,erec)] + STACKSIZE) )
		{
		  fprintf (stderr, "ERROR: stack overflow on node %d\n", GetNodeId(&intrc,erec));
		  fprintf (stderr, "       event %s at %llu\n", ev->name, event_GetTi(&intrc,erec,0));
		  exit (1);
		}
	      stkptr[GetNodeId(&intrc,erec)]->tag = event_GetEv(&intrc,erec,0);

	      /*Print the record in the form:
		/RecordType(2)for event:NodeID:PTaskID:TaskID:ThreadID:
		:event time:eventType(5) for method entry/exit:event tag(to
		distinguish between events of the same type, 0 for exit and
		event_GetEv(&intrc,erec,0) for entry
	      */

	      fprintf (outfp, "2:%d:1:%d:%d:%llu:5:%d\n",(event_GetNid(&intrc,erec,0))+1,(event_GetNid(&intrc,erec,0))+1,event_GetTid(&intrc,erec,0)+1,event_GetTi(&intrc,erec,0)-intrc.firsttime,event_GetEv(&intrc,erec,0));
	      if(prvPCF[event_GetEv(&intrc,erec,0)-1] == 0){
		fprintf(pcffp,"%d       %s\n",event_GetEv(&intrc,erec,0),GetEventName(event_GetEv(&intrc,erec,0),&hasParam));
		prvPCF[event_GetEv(&intrc,erec,0)-1] = 1;
	      }
	    }
	}
    }

    else if ( outFormat == pv )
      {
	ev = GetEventStruct (event_GetEv(&intrc,erec,0));
	if ((ev!= 0) &&( ( ev->tag != 0 ) || (dynamictrace))) /* dynamic trace doesn't use tag*/
	  {
	    if ( (ev->tag == SEND_EVENT) && pvComm )

	      {
	      /* send message */
		/* In dynamic trace the format for par is
		   31 ..... 24 23 ......16 15..............0
       	           other       type          length
		So, mynode is the sender and its in GetNodeId(&intrc,erec)
		SENDMSG <type> FROM <sender> TO <receiver> LEN <length>
		*/
		/* extract the information from the parameter */

		myid 	= GetNodeId(&intrc,erec) + 1;

		decodeParameter(event_GetPar(&intrc,erec,0), &otherid, &msglen, &msgtag, &comm);
		otherid = otherid+1;


		if (threads)
		  {
		    if (GetMatchingRecv(intrc, msgtag, GetNodeId(&intrc,erec),
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
			otherid	= offset[otherid-1] + event_GetTid(&intrc,erec,0) + 1;
			/* THIS ABOVE IS TRUE ONLY WHEN SAME THREADS COMMUNICATE !! */
#ifdef DEBUG
			printf("ASSUMPTION: SAME THREADIDS ON DIFF NODES COMMUNICATE!!\n");
			printf("SEND: OTHER %d, myid %d len %d tag %d: PAR: %lx\n",
			       otherid, myid, msglen, msgtag, event_GetPar(&intrc,erec,0));
#endif /* DEBUG */
		      }
		  }

#ifdef DEBUG
		printf ("\n\n%llu SENDMSG %d FROM %d TO %d LEN %d\n\n\n",
			event_GetTi(&intrc,erec,0) - intrc.firsttime,
			msgtag, myid , otherid , msglen);
#endif /* DEBUG */

		fprintf (outfp, "%llu SENDMSG %d FROM %d TO %d LEN %d\n",
			 event_GetTi(&intrc,erec,0) - intrc.firsttime,
			 msgtag, myid , otherid , msglen);
	      }
	    else if ( (ev->tag == RECV_EVENT ) && pvComm )
	      {
		/* receive message */
		/* In dynamic trace the format for par is
		   31 ..... 24 23 ......16 15..............0
       	           other       type          length
		   So, mynode is the receiver and its in GetNodeId(&intrc,erec)
		   RECVMSG <type> BY <receiver> FROM <sender> LEN <length>
		*/
		/* extract the information from the parameter */

		myid 	= GetNodeId(&intrc,erec) + 1;

		decodeParameter(event_GetPar(&intrc,erec,0), &otherid, &msglen, &msgtag, &comm);
		otherid = otherid+1;

		if (threads)
		  {
		    if (GetMatchingSend(intrc, msgtag, GetNodeId(&intrc,erec),
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
			otherid	= offset[otherid-1] + event_GetTid(&intrc,erec,0) + 1;
			/* THIS ABOVE IS TRUE ONLY WHEN SAME THREADS COMMUNICATE !! */
#ifdef DEBUG
			printf("ASSUMPTION: SAME THREADIDS ON DIFF NODES COMMUNICATE!!\n");
			printf("RECV: OTHER %d, myid %d len %d tag %d: PAR: %lx\n",
			       otherid, myid, msglen, msgtag, event_GetPar(&intrc,erec,0));
#endif /* DEBUG */
		      }
		  }

#ifdef DEBUG
		printf ("%llu RECVMSG %d BY %d FROM %d LEN %d\n",
			event_GetTi(&intrc,erec,0) - intrc.firsttime,
			msgtag, myid , otherid , msglen);
#endif /* DEBUG */

		fprintf (outfp, "%llu RECVMSG %d BY %d FROM %d LEN %d\n",
			 event_GetTi(&intrc,erec,0) - intrc.firsttime,
			 msgtag, myid , otherid , msglen);
	      }
	    else if (( ev->tag == -9 ) || ( event_GetPar(&intrc,erec,0) == -1))
	      { /* In dynamic tracing, 1/-1 par values are for Entry/Exit resp. */
		/* exit state */
		/* PARVis needs time values relative to the start of the program! */
		stkptr[GetNodeId(&intrc,erec)]--;
		if ( stkptr[GetNodeId(&intrc,erec)] < statestk[GetNodeId(&intrc,erec)] )
		  {
		    fprintf (stderr, "ERROR: stack underflow on node %d\n", GetNodeId(&intrc,erec));
		    fprintf (stderr, "       event %s at %llu\n", ev->name, event_GetTi(&intrc,erec,0));
		    exit (1);
		  }
		if ( pvCompact )
		  fprintf (outfp, "%llu EXCH %d 1 1 %s %d\n",
			   event_GetTi(&intrc,erec,0) - intrc.firsttime, GetNodeId(&intrc,erec)+1,
			   stkptr[GetNodeId(&intrc,erec)]->state, stkptr[GetNodeId(&intrc,erec)]->tag);
		else
		  fprintf (outfp, "%llu EXCHANGE ON CPUID %d UPTO %s %d CLUSTER 1\n",
			   event_GetTi(&intrc,erec,0) - intrc.firsttime, GetNodeId(&intrc,erec)+1,
			   stkptr[GetNodeId(&intrc,erec)]->state, stkptr[GetNodeId(&intrc,erec)]->tag);
	      }
	    else if (event_GetPar(&intrc,erec,0) == 1)
	      {
		/* enter new state */
		stkptr[GetNodeId(&intrc,erec)]++;
		if ( stkptr[GetNodeId(&intrc,erec)] > (statestk[GetNodeId(&intrc,erec)] + STACKSIZE) )
		  {
		    fprintf (stderr, "ERROR: stack overflow on node %d\n", GetNodeId(&intrc,erec));
		    fprintf (stderr, "       event %s at %llu\n", ev->name, event_GetTi(&intrc,erec,0));
		    exit (1);
		  }
		if ( pvCompact )
		  fprintf (outfp, "%llu EXCH %d 1 1 %s %d\n",
			   /*???event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec)+1, ev->state, ev->tag);*/
			   event_GetTi(&intrc,erec,0) - intrc.firsttime, GetNodeId(&intrc,erec)+1, ev->state, ev->no);
		else
		  fprintf (outfp, "%llu EXCHANGE ON CPUID %d DOWNTO %s %d CLUSTER 1\n",
			   /*???event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec)+1, ev->state, ev->tag);*/
			   event_GetTi(&intrc,erec,0) - intrc.firsttime, GetNodeId(&intrc,erec)+1, ev->state, ev->no);
		stkptr[GetNodeId(&intrc,erec)]->state = ev->state;
		/*???stkptr[GetNodeId(&intrc,erec)]->tag = ev->tag;*/
		stkptr[GetNodeId(&intrc,erec)]->tag = ev->no;
	      }
	  }
      }
    else if ( outFormat == dump )
    {
      ptr = GetEventName (event_GetEv(&intrc,erec,0), &hasParam);
      if ( hasParam ) {

	if (IsDoubleParam(event_GetEv(&intrc,erec,0))) { /* user events parameter is a double */
#ifdef TAU_WINDOWS
        fprintf (outfp, "%5ld %30.30s %12I64u %6d %6d %12G\n",
                 intrc.numrec, ptr, event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec), event_GetTid(&intrc,erec,0), event_GetPar(&intrc,erec,0));
#else
        fprintf (outfp, "%5ld %30.30s %12llu %6d %6d %12G\n",
                 intrc.numrec, ptr, event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec), event_GetTid(&intrc,erec,0), event_GetPar(&intrc,erec,0));
#endif
	} else {
#ifdef TAU_WINDOWS
        fprintf (outfp, "%5ld %30.30s %12I64u %6d %6d %12I64d\n",
                 intrc.numrec, ptr, event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec), event_GetTid(&intrc,erec,0), event_GetPar(&intrc,erec,0));
#else
        fprintf (outfp, "%5ld %30.30s %12llu %6d %6d %12lld\n",
                 intrc.numrec, ptr, event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec), event_GetTid(&intrc,erec,0), event_GetPar(&intrc,erec,0));
#endif
	}


      } else {
#ifdef TAU_WINDOWS
        fprintf (outfp, "%5ld %30.30s %12I64u %6d %6d\n",
                 intrc.numrec, ptr, event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec), event_GetTid(&intrc,erec,0));
#else
        fprintf (outfp, "%5ld %30.30s %12llu %6d %6d\n",
                 intrc.numrec, ptr, event_GetTi(&intrc,erec,0), GetNodeId(&intrc,erec), event_GetTid(&intrc,erec,0));
#endif
	/* Changed 12lu to 12llu for unsigned long long time */
      }
    }

    if ( (erec = get_next_rec (&intrc)) == NULL )
      break;
    else
    {
      intrc.numrec++;

      /* -- check clock overflow ---------------------------------------- */
      if ( event_GetTi(&intrc,erec,0) < intrc.lasttime ) intrc.overflows++;
      intrc.lasttime = event_GetTi(&intrc,erec,0);
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
