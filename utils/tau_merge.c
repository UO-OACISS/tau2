/****************************************************************************
 **                      TAU Portable Profiling Package                     **
 **                      http://www.cs.uoregon.edu/research/tau             **
 *****************************************************************************
 **    Copyright 2007                                                       **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 **    Forschungszentrum Juelich                                            **
 ****************************************************************************/

#ifdef __SP1__
# include <Profile/aix.h> /* if its an IBM */
#endif /* __SP1__ */
# include <stdio.h>
# include <stdlib.h>
# include <sys/types.h>
# include <fcntl.h>


#define TAU_MAXPROCS 65536


#ifdef TAU_LARGEFILE
  #define LARGEFILE_OPTION O_LARGEFILE
#else
  #define LARGEFILE_OPTION 0
#endif


#ifdef TAU_NEC
extern "C" {
int getdtablesize(void);
}
#endif /* TAU_NEC */

#ifdef TAU_WINDOWS
  #include <windows.h>	
  #include <io.h>
  #include "getopt.h"
#else
  #define O_BINARY 0
  #include <unistd.h>
#endif

#ifdef FUJITSU
# include <Profile/fujitsu.h>
#endif

#include <string.h>

#define TRACING_ON
#include <Profile/TauTrace.h>

#ifndef TRUE
#   define FALSE  0
#   define TRUE   1
#endif

#define F_EXISTS    0

#define CONTLEN  (sizeof(TAU_EV) - sizeof(long int))

#define STDOUT 1

/* -- buffer sizes ------------------ */
# define INMAX    BUFSIZ   /* records */
# define OUTMAX   BUFSIZ   /* chars   */

int dynamic = TRUE ; /* by default events.<node>.edf files exist */
int dontblock = FALSE; /* by default, block waiting for records, online merge*/
char *mergededffile = NULL; /* default merged EDF file name */

int open_edf_file(char *prefix, int nodeid, int prefix_is_filename);
int parse_edf_file(int node);
int store_merged_edffile(char *filename);
const char *get_event_name(int gid);
int GID(int node, long localEventId); 


struct trcdescr
{
  int     fd;              /* -- input file descriptor                     -- */
  char   *name;            /* -- corresponding file name                   -- */
  int     nid;             /* -- corresponding PTX PID                     -- */
  int     overflows;       /* -- clock overflows in that trace             -- */
  int     contlen;         /* -- length of continuation event buffer       -- */
  long    numrec;          /* -- number of event records already processed -- */
  x_uint64 lasttime;  /* -- timestamp of previous event record        -- */
  x_uint64 offset;    /* -- offset of timestamp                       -- */

/*   TAU_EV  *buffer;    /\* -- input buffer                              -- *\/ */
/*   TAU_EV  *erec;      /\* -- current event record                      -- *\/ */
/*   TAU_EV  *next;      /\* -- next available event record in buffer     -- *\/ */
/*   TAU_EV  *last;      /\* -- last event record in buffer               -- *\/ */

  void  *buffer;    /* -- input buffer                              -- */
  void  *erec;      /* -- current event record                      -- */
  void  *next;      /* -- next available event record in buffer     -- */
  void  *last;      /* -- last event record in buffer               -- */

  int           format;    /* see above */
  int           eventSize; /* sizeof() the corresponding format struct */

} *trcdes;

int outfd; /* output trace file */
static void output_flush(int fd);



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
} TAU_EV32;

/* for 64 bit platforms */
typedef struct {
  x_int64            ev;    /* -- event id        -- */
  x_uint16           nid;   /* -- node id         -- */
  x_uint16           tid;   /* -- thread id       -- */
  x_uint32           padding; /*  space wasted for 8-byte aligning the next item */ 
  x_int64            par;   /* -- event parameter -- */
  x_uint64           ti;    /* -- time [us]?      -- */
} TAU_EV64;


typedef TAU_EV TAU_EV_NATIVE;



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


/* Endian/bitsize stuff */

/* void convertEvent(struct trcdescr *tdes, void *event, int index) { */
/*   TAU_EV32 *event32; */
/*   TAU_EV64 *event64; */

/*   switch (tdes->format) { */
/*   case FORMAT_NATIVE: */
/*   case FORMAT_32: */
/*   case FORMAT_64: */
/*     return; */

/*   case FORMAT_32_SWAP: */
/*     event32 = (TAU_EV32*) event; */
/*     event32[index].ev = swap32(event32[index].ev); */
/*     event32[index].nid = swap16(event32[index].nid); */
/*     event32[index].tid = swap16(event32[index].tid); */
/*     event32[index].par = swap64(event32[index].par); */
/*     event32[index].ti = swap64(event32[index].ti); */
/*     return; */

/*   case FORMAT_64_SWAP: */
/*     event64 = (TAU_EV64*) event; */
/*     event64[index].ev = swap64(event64[index].ev); */
/*     event64[index].nid = swap16(event64[index].nid); */
/*     event64[index].tid = swap16(event64[index].tid); */
/*     event64[index].par = swap64(event64[index].par); */
/*     event64[index].ti = swap64(event64[index].ti); */

/*     //    printf ("event.ti = %llu\n", swap64(event64->ti)); */
/*     return; */
/*   } */

/*   return; */
/* } */

x_int32 event_GetEv(struct trcdescr *tFile, void *event, int index) {
  TAU_EV_NATIVE *nativeEvent;
  TAU_EV32 *event32;
  TAU_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (TAU_EV_NATIVE*)event;
    return nativeEvent[index].ev;
    
  case FORMAT_32:
    event32 = (TAU_EV32*) event;
    return event32[index].ev;
  case FORMAT_32_SWAP:
    event32 = (TAU_EV32*) event;
    return swap32(event32[index].ev);

  case FORMAT_64:
    event64 = (TAU_EV64*) event;
    return event64[index].ev;
  case FORMAT_64_SWAP:
    event64 = (TAU_EV64*) event;
    return swap64(event64[index].ev);
  }
  return 0;
}

void event_SetEv(struct trcdescr *tFile, void *event, int index, x_int32 value) {
  TAU_EV_NATIVE *nativeEvent;
  TAU_EV32 *event32;
  TAU_EV64 *event64;
  x_int64 tmpValue;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (TAU_EV_NATIVE*)event;
    nativeEvent[index].ev = value;
    break;

  case FORMAT_32:
    event32 = (TAU_EV32*) event;
    event32[index].ev = value;
    break;
  case FORMAT_32_SWAP:
    event32 = (TAU_EV32*) event;
    event32[index].ev = swap32(value);
    break;

  case FORMAT_64:
    event64 = (TAU_EV64*) event;
    event64[index].ev = value;
    break;
  case FORMAT_64_SWAP:
    tmpValue = value;
    event64 = (TAU_EV64*) event;
    event64[index].ev = swap64(tmpValue);
    break;
  }
  return;
}

x_uint64 event_GetTi(struct trcdescr *tFile, void *event, int index) {
  TAU_EV_NATIVE *nativeEvent;
  TAU_EV32 *event32;
  TAU_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (TAU_EV_NATIVE*)event;
    return nativeEvent[index].ti;
    
  case FORMAT_32:
    event32 = (TAU_EV32*) event;
    return event32[index].ti;
  case FORMAT_32_SWAP:
    event32 = (TAU_EV32*) event;
    return swap64(event32[index].ti);

  case FORMAT_64:
    event64 = (TAU_EV64*) event;
    return event64[index].ti;
  case FORMAT_64_SWAP:
    event64 = (TAU_EV64*) event;
    return swap64(event64[index].ti);
  }
  return 0;
}

void event_SetTi(struct trcdescr *tFile, void *event, int index, x_uint64 value) {
  TAU_EV_NATIVE *nativeEvent;
  TAU_EV32 *event32;
  TAU_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (TAU_EV_NATIVE*)event;
    nativeEvent[index].ti = value;
    break;
  case FORMAT_32:
    event32 = (TAU_EV32*) event;
    event32[index].ti = value;
    break;
  case FORMAT_32_SWAP:
    event32 = (TAU_EV32*) event;
    event32[index].ti = swap64(value);
    break;
  case FORMAT_64:
    event64 = (TAU_EV64*) event;
    event64[index].ti = value;
    break;
  case FORMAT_64_SWAP:
    event64 = (TAU_EV64*) event;
    event64[index].ti = swap64(value);
    break;
  }
  return;
}


x_uint16 event_GetNid(struct trcdescr *tFile, void *event, int index) {
  TAU_EV_NATIVE *nativeEvent;
  TAU_EV32 *event32;
  TAU_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (TAU_EV_NATIVE*)event;
    return nativeEvent[index].nid;
    
  case FORMAT_32:
    event32 = (TAU_EV32*) event;
    return event32[index].nid;
  case FORMAT_32_SWAP:
    event32 = (TAU_EV32*) event;
    return swap16(event32[index].nid);

  case FORMAT_64:
    event64 = (TAU_EV64*) event;
    return event64[index].nid;
  case FORMAT_64_SWAP:
    event64 = (TAU_EV64*) event;
    return swap16(event64[index].nid);
  }
  return 0;
}

void event_SetNid(struct trcdescr *tFile, void *event, int index, x_uint16 value) {
  TAU_EV_NATIVE *nativeEvent;
  TAU_EV32 *event32;
  TAU_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (TAU_EV_NATIVE*)event;
    nativeEvent[index].nid = value;
    break;

  case FORMAT_32:
    event32 = (TAU_EV32*) event;
    event32[index].nid = value;
    break;
  case FORMAT_32_SWAP:
    event32 = (TAU_EV32*) event;
    event32[index].nid = swap16(value);
    break;

  case FORMAT_64:
    event64 = (TAU_EV64*) event;
    event64[index].nid = value;
    break;
  case FORMAT_64_SWAP:
    event64 = (TAU_EV64*) event;
    event64[index].nid = swap16(value);
    break;
  }
  return;
}


x_uint16 event_GetTid(struct trcdescr *tFile, void *event, int index) {
  TAU_EV_NATIVE *nativeEvent;
  TAU_EV32 *event32;
  TAU_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (TAU_EV_NATIVE*)event;
    return nativeEvent[index].tid;
    
  case FORMAT_32:
    event32 = (TAU_EV32*) event;
    return event32[index].tid;
  case FORMAT_32_SWAP:
    event32 = (TAU_EV32*) event;
    return swap16(event32[index].tid);

  case FORMAT_64:
    event64 = (TAU_EV64*) event;
    return event64[index].tid;
  case FORMAT_64_SWAP:
    event64 = (TAU_EV64*) event;
    return swap16(event64[index].tid);
  }
  return 0;
}


x_uint64 event_GetPar(struct trcdescr *tFile, void *event, int index) {
  TAU_EV_NATIVE *nativeEvent;
  TAU_EV32 *event32;
  TAU_EV64 *event64;

  switch (tFile->format) {
  case FORMAT_NATIVE:
    nativeEvent = (TAU_EV_NATIVE*)event;
    return nativeEvent[index].par;
    
  case FORMAT_32:
    event32 = (TAU_EV32*) event;
    return event32[index].par;
  case FORMAT_32_SWAP:
    event32 = (TAU_EV32*) event;
    return swap64(event32[index].par);

  case FORMAT_64:
    event64 = (TAU_EV64*) event;
    return event64[index].par;
  case FORMAT_64_SWAP:
    event64 = (TAU_EV64*) event;
    return swap64(event64[index].par);
  }
  return 0;
}


void determineFormat(struct trcdescr *tdes) {
  int bytesRead;
  int formatFound;
  TAU_EV32 event32;
  TAU_EV64 event64;

  formatFound = 0;
/*   printf ("determining format!\n"); */
/*   printf ("sizeof(TAU_EV32) = %d\n", sizeof(TAU_EV32)); */
/*   printf ("sizeof(TAU_EV64) = %d\n", sizeof(TAU_EV64)); */


/*   printf ("par32 : %d\n", (long)&event32.par - (long)&event32); */
/*   printf ("par64 : %d\n", (long)&event64.par - (long)&event64); */


/*   lseek(tdes->fd, 0, SEEK_SET); */
  bytesRead = read(tdes->fd, &event32, sizeof(TAU_EV32));
  lseek(tdes->fd, 0, SEEK_SET);
  bytesRead = read(tdes->fd, &event64, sizeof(TAU_EV64));
  lseek(tdes->fd, 0, SEEK_SET);

  /* 32 bit regular */
  if (event32.par == 3) {
    tdes->format = FORMAT_32;
    tdes->eventSize = sizeof(TAU_EV32);
    formatFound = 1;
/*     printf ("32 regular!\n"); */
  }


  /* 32 bit swapped */
  if (swap64(event32.par) == 3) {
    if (formatFound == 1) { /* shouldn't happen, if it does, go to native */
      tdes->format = FORMAT_NATIVE;
      tdes->eventSize = sizeof(TAU_EV_NATIVE);
      return;
    }
    tdes->format = FORMAT_32_SWAP;
    tdes->eventSize = sizeof(TAU_EV32);
    formatFound = 1;
/*     printf ("32 swapped!\n"); */
  }

  /* 64 bit regular*/
  if (event64.par == 3) {
    if (formatFound == 1) { /* shouldn't happen, if it does, go to native*/
      tdes->format = FORMAT_NATIVE;
      tdes->eventSize = sizeof(TAU_EV_NATIVE);
      return;
    }
    tdes->format = FORMAT_64;
    tdes->eventSize = sizeof(TAU_EV64);
    formatFound = 1;
/*     printf ("64 regular!\n"); */
  }


  /* 64 bit swapped*/
  if (swap64(event64.par) == 3) {
    if (formatFound == 1) { /* shouldn't happen, if it does, go to native*/
      tdes->format = FORMAT_NATIVE;
      tdes->eventSize = sizeof(TAU_EV_NATIVE);
      return;
    }
    tdes->format = FORMAT_64_SWAP;
    tdes->eventSize = sizeof(TAU_EV64);
    formatFound = 1;
/*     printf ("64 swapped!\n"); */
  }

  if (formatFound == 0) {
    fprintf (stderr, "Warning: couldn't determine format, using native!\n");
    tdes->format = FORMAT_NATIVE;
    tdes->eventSize = sizeof(TAU_EV_NATIVE);
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



/* -------------------------------------------------------------------------- */
/* -- input buffer handling                                                -- */
/* -------------------------------------------------------------------------- */

static void *get_next_rec(struct trcdescr *tdes)
{
  long no;
  const char *last_event_name;

/*   printf ("get_next_rec called\n"); */
/*   printf ("tdes->eventSize = %d\n", tdes->eventSize); */
/*   printf ("next = 0x%x\n", tdes->next); */
/*   printf ("last = 0x%x\n", tdes->last); */

  if ( (tdes->last == NULL) || (tdes->next > tdes->last) )
  {
    /* -- input buffer empty: read new records from file -------------------- */
    /*if ( (no = read (tdes->fd, tdes->buffer, INMAX * sizeof(TAU_EV))) != (INMAX * sizeof(TAU_EV)) )*/
    if ( (no = read (tdes->fd, tdes->buffer, INMAX * tdes->eventSize)) != (INMAX * tdes->eventSize) ) {
      if ( no == 0 ) {
		
#ifdef DEBUG
	printf("Received EOF on trace \n");	
#endif /* DEBUG */
        /* -- no more event record: ----------------------------------------- */
	if (tdes->last != NULL) {
#ifdef DEBUG
	  printf("Last rec not null\n");	
#endif /* DEBUG */
	  /*last_event_name = get_event_name(tdes->last->ev);*/
	  last_event_name = get_event_name(event_GetEv(tdes,tdes->last,0));
	  if (last_event_name != NULL)
	  { /* the last event in the trace file is WALL_CLOCK. Is it EOF? */
#ifdef DEBUG
	    printf("Last_event_name = %s\n", last_event_name);
#endif /* DEBUG */
            if ((strcmp(last_event_name, "\"WALL_CLOCK\"")==0)|| (dontblock==TRUE))
	    { /* It is the end. Close the trace file */
#ifdef DEBUG
	      printf("last_event_name = %s\n", last_event_name);
#endif /* DEBUG */
	      close(tdes->fd);
	      tdes->fd = -1;
              return ( (TAU_EV *) NULL);
	    }
	    else
	    {
#ifdef DEBUG
	      printf("Blocking...");
#endif /* DEBUG */
	      store_merged_edffile(mergededffile);
	      output_flush(outfd);
	      /* Block waiting for the trace to get some more records in it */
	      while ((no = read (tdes->fd, tdes->buffer, INMAX * tdes->eventSize)) == 0)
	      {
#ifdef DEBUG
		printf("WAITING... no = %d, node filename = %s \n", no, tdes->name);
#endif /* DEBUG */

#ifdef TAU_WINDOWS
		 Sleep(1);
#else
  	 	 sleep(1);
#endif
	      }
	      /* got the trace data! */
#ifdef DEBUG
	      printf("Read %d bytes\n", no);
#endif /* DEBUG */
	      if ((no < 0) || (no % tdes->eventSize != 0)) {
		close(tdes->fd);
		tdes->fd = -1;
		return ((TAU_EV *)NULL);
	      } else {
#ifdef DEBUG
	        printf("Got trace data!\n");
#endif /* DEBUG */
                /* -- we got some event records ------------------------- */
    		tdes->next = tdes->buffer;
    		/*tdes->last = tdes->buffer + (no / tdes->eventSize) - 1;*/
		tdes->last = (char*)tdes->buffer + no - tdes->eventSize;
  		/*return (tdes->erec = tdes->next++);*/

		tdes->erec = tdes->next;
		tdes->next = (void*)(((char*)tdes->next) + tdes->eventSize);
		return tdes->erec;
	      }
	    }
	
	  } 
	} /* valid last event */
#ifdef DEBUG
	printf("Last rec null, closing ...\n");	
#endif /* DEBUG */
        close (tdes->fd);
        tdes->fd = -1;
        return ( NULL);
      } /* possible EOF */
      else if ( (no % tdes->eventSize) != 0 )
      {
        /* -- read error: --------------------------------------------------- */
        fprintf (stderr, "%s: read error\n", tdes->name);
        exit (1);
      }
    }


    /* -- we got some event records ----------------------------------------- */
    tdes->next = tdes->buffer;
    /*tdes->last = tdes->buffer + (no / tdes->eventSize) - 1;*/
    tdes->last = (char*)tdes->buffer + no - tdes->eventSize;

  }

  /*  return (tdes->erec = tdes->next++);*/
  tdes->erec = tdes->next;
  tdes->next = ((char*)tdes->next) + tdes->eventSize;

  return tdes->erec;
}

/* -------------------------------------------------------------------------- */
/* -- Routines Output*: Simple output buffering                            -- */
/* -------------------------------------------------------------------------- */

static char  outbuffer[OUTMAX];
static int   outidx = 0;
static char *outptr = outbuffer;

static void output_flush(int fd)
{
  if ( outidx > 0 )
  {
    if ( write (fd, outbuffer, outidx) != outidx )
    {
      perror ("write output");
      exit (1);
    }
    outidx = 0;
    outptr = outbuffer;
  }
}

static void output(int fd, char *data, size_t l)
{
  if ( outidx + l >= BUFSIZ )
  {
    /* not enough space for data in output buffer: flush buffer */
    output_flush (fd);
  }

  memcpy (outptr, data, l);
  outptr += l;
  outidx += l;
}

/* -------------------------------------------------------------------------- */
/* -- FILE DESCRIPTOR HANDLING ---------------------------------------------- */
/* -------------------------------------------------------------------------- */

#ifndef TAU_WINDOWS
  #include <sys/time.h>
  #include <sys/resource.h>
#endif



int cannot_get_enough_fd(int need)
{
#ifdef TAU_WINDOWS
  return FALSE; /* no getdtablesize() in windows*/
#else	
# if defined(__hpux) || defined(sun)
  /* -- system supports get/setrlimit (RLIMIT_NOFILE) -- */
  struct rlimit rlp;

  getrlimit (RLIMIT_NOFILE, &rlp);
  if ( rlp.rlim_max < need )
    return (TRUE);
  else if ( rlp.rlim_cur < need )
  {
    rlp.rlim_cur = need;
    setrlimit (RLIMIT_NOFILE, &rlp);
  }
  return (FALSE);
# else
#   if defined(_SEQUENT_) || defined(sequent)
      /* -- system provides get/setdtablesize -- */
      int max = getdtablesize();
      return ( (max < need) && (setdtablesize (need) != need) );
#   else
      /* -- system provides only getdtablesize -- */
      int max = getdtablesize();
      return ( max < need );
#   endif
# endif
#endif /* TAU_WINDOWS */
}


/* -------------------------------------------------------------------------- */
/* -- TAU_MERGE MAIN PROGRAM ----------------------------------------------- */
/* -------------------------------------------------------------------------- */

extern char *optarg;
extern int   optind;
char **edfnames; /* if they're specified */
int edfspecified; /* tau_events also needs this */

int main(int argc, char *argv[])
{
  int i, active, numtrc, source, errflag, first;
  int adjust, min_over, reassembly;
  int numedfprocessed;
  int startedfindex, edfcount;
  x_uint64 min_time, first_time;
  long numrec;
  char *trcfile;
  void *erec;
# ifdef __ksr__
  int *sequence;
  int last_pthread, num_pthreads;
# endif
# if defined(__ksr__) || defined(__CM5__)
  x_uint64 last_time;
# endif

  TAU_EV nativeEvent;
  edfcount   = 0;
  numtrc     = 0;
  numrec     = 0;
  errflag    = FALSE;
  dontblock  = FALSE; /* by default, block */
  adjust     = FALSE;
  reassembly = TRUE;
  edfspecified = FALSE; /* by default edf files are not specified on cmdline */
  numedfprocessed = 0; /* only used with -e events.*.edf files are specified */
  first_time = 0L;
  mergededffile = strdup("tau.edf"); /* initialize it */

  while ( (i = getopt (argc, argv, "arne:m:")) != EOF )
  {
    switch ( i )
    {
      case 'a': /* -- adjust first time to zero -- */
                adjust = TRUE;
                break;

      case 'r': /* -- do not reassemble long events -- */
                reassembly = FALSE;
                break;

      case 'e': /* -- EDF files specified on the commandline -- */
                edfspecified = TRUE;
		numedfprocessed = 0;
		edfnames = (char **) malloc (argc * (sizeof(char *))); 
		/* first array */
		for (i = optind-1; i < argc; i++)
		{
		  if(strstr(argv[i], ".edf") != 0)
		  {
#ifdef DEBUG
		    printf("Processing Event file %s for node %d\n", argv[i],
				    numedfprocessed);
#endif /* DEBUG */
	            open_edf_file(argv[i], numedfprocessed, TRUE);
		    numedfprocessed++; 
		    /* store the name of the edf file so that we can re-open 
		     * it later if event files need to be re-read */
		    edfnames[edfcount] = strdup(argv[i]);
		    edfcount++; /* increment the count */
		  }
		  else 
		    break; /* come out of the loop! */
		}
		optind += numedfprocessed - 1; 
#ifdef DEBUG
		printf("numedfprocessed = %d, optind = %d, argc = %d, argv[optind] = %s\n",
				numedfprocessed, optind, argc, argv[optind]);
#endif /* DEBUG */
                break;

      case 'm': /* -- name of the merged edf file (instead of tau.edf) */
		mergededffile = strdup(argv[optind-1]);
#ifdef DEBUG
		printf("merged edf file = %s\n", mergededffile);
#endif /* DEBUG */
		break;

      case 'n': /* -- do not block for records at end of trace -- */
		dontblock = TRUE; 
		break;

      default : /* -- ERROR -- */
                errflag = TRUE;
                break;

    }
  }

#ifdef DEBUG
  printf("optind = %d, argc = %d\n", optind, argc);
#endif /* DEBUG */
  /* -- check whether enough file descriptors available: -------------------- */
  /* -- max-file-descriptors - 4 (stdin, stdout, stderr, output trace) ------ */
  active = argc - optind - 1;
  if ( cannot_get_enough_fd (active + 4) )
  {
      fprintf (stderr, "%s: too many input traces:\n", argv[0]);
      fprintf (stderr, "  1. merge half of the input traces\n");
      fprintf (stderr, "  2. merge other half and output of step 1\n");
      fprintf (stderr, "  Or use, \"tau_treemerge.pl\"\n");
      exit (1);
  }
  trcdes = (struct trcdescr *) malloc (active * sizeof(struct trcdescr));

  for (i=optind; i<argc-1; i++) {
    /*     printf ("opening %s!\n", argv[i]);  */
    /* -- open input trace -------------------------------------------------- */
    if ( (trcdes[numtrc].fd = open (argv[i], O_RDONLY | O_BINARY | LARGEFILE_OPTION )) < 0 ) {
      /*       printf ("failed!\n"); */
      perror (argv[i]);
      errflag = TRUE;
    } else {
      /*       printf ("success!\n"); */
      
      
      /* determine format */
      determineFormat(&trcdes[numtrc]);

      trcdes[numtrc].name      = argv[i];
      trcdes[numtrc].buffer    = (void *) malloc (INMAX * trcdes[numtrc].eventSize);
      trcdes[numtrc].erec      = NULL;
      trcdes[numtrc].next      = NULL;
      trcdes[numtrc].last      = NULL;
      trcdes[numtrc].overflows = 0;


      /* -- read first event record ----------------------------------------- */
      if ( (erec = get_next_rec (trcdes + numtrc)) == NULL ) {
        /* -- no event record: ---------------------------------------------- */
        fprintf (stderr, "%s: warning: trace empty - ignored\n",
                 trcdes[numtrc].name);
        trcdes[numtrc].numrec = 0L;
        active--;
      }

      /* We can't do this check because the event is EV_INIT, but its ev id
       * is different from original creation time after merging */
      /* -- check first event record ---------------------------------------- */
      /* else if ( (erec->ev != TAU_EV_INIT) && (erec->ev != TAU_EV_INITM) )
      {
        fprintf (stderr, "%s: no valid event trace\n", trcdes[numtrc].name);
        exit (1);
      } */
      else
      {

/* 	printf ("first record has Ev = %d\n", event_GetEv(trcdes+numtrc,erec,0)); */

        if ( event_GetNid(trcdes+numtrc, erec, 0) > TAU_MAXPROCS )
          fprintf (stderr,
           "%s: warning: node id %d too big for this machine (max. %d nodes)\n",
           trcdes[numtrc].name, event_GetNid(trcdes, erec, 0), TAU_MAXPROCS);

        trcdes[numtrc].numrec = 1L;
        if ( event_GetEv(trcdes+numtrc, erec, 0) == TAU_EV_INIT ) {
	  if (!dynamic) { /* for dynamic trace, don't change this to INITM */
            /*erec->ev = TAU_EV_INITM;*/
	    event_SetEv(trcdes+numtrc, erec, 0, TAU_EV_INITM);
	  }
          /*trcdes[numtrc].nid = erec->nid;*/
          trcdes[numtrc].nid = event_GetNid(trcdes+numtrc, erec, 0);
        } else {
          trcdes[numtrc].nid = -1;
	}

        /*trcdes[numtrc].lasttime = erec->ti;*/
        trcdes[numtrc].lasttime = event_GetTi(trcdes+numtrc,erec,0);
        trcdes[numtrc].offset   = 0L;


	if(dynamic)
	{
	/* parse edf file for this trace */
	  if (edfspecified == FALSE)
	  { /* use default edf file names  "events.*.edf" */
	    char eventfilename[2048];
	    sprintf(eventfilename, "events.%d.edf", trcdes[numtrc].nid); 
	    open_edf_file(eventfilename, numtrc, TRUE);
	  }
	  if (edfspecified && numtrc >= numedfprocessed) {
	    fprintf (stderr, "Error: When specifying -e, you must specify one .edf file for each trace\n");
	    exit(-1);
	  }
	  parse_edf_file(numtrc);
	}
        numtrc++;
      }
    }
  }
  if (dynamic)
  {
    /* all edf files have been parsed now - store the final edf merged file */
    store_merged_edffile(mergededffile);
  }

  if ( (numtrc < 1) || errflag )
  {
    fprintf (stderr,
             "usage: %s [-a] [-r] [-n] [-e eventedf*] [-m mergededf] inputtraces* (outputtrace|-) \n", argv[0]);
    fprintf(stderr,
    "Note: %s assumes edf files are named events.<nodeid>.edf and \n", argv[0]);
    fprintf(stderr,"      generates a merged edf file tau.edf\n");
    fprintf(stderr,"-a : adjust first time to zero\n");
    fprintf(stderr,"-r : do not reassemble long events\n");
    fprintf(stderr,"-n : do not block waiting for new events. Offline merge\n");
    fprintf(stderr,"-e <files> : provide a list of event definition files corresponding to traces\n");
    fprintf(stderr,"-m <mergededf> : specify the name of the merged event definition file\n");
    fprintf(stderr,"e.g., > %s tautrace.*.trc app.trc\n", argv[0]);
    fprintf(stderr,"e.g., > %s -e events.[0-255].edf -m ev0_255merged.edf tautrace.[0-255].*.trc app.trc\n", argv[0]);



    exit (1);
  }

  /* -- output trace file --------------------------------------------------- */
  if ( strcmp ((argv[argc-1]), "-") == 0 )
  {
    outfd = STDOUT;
  }
  else
  {
    trcfile = (char *) malloc ((strlen(argv[argc-1])+5) * sizeof(char));
    if ( strcmp ((argv[argc-1])+strlen(argv[argc-1])-4, ".trc") == 0 )
      strcpy (trcfile, argv[argc-1]);
    else
      sprintf (trcfile, "%s.trc", argv[argc-1]);

    if ( access (trcfile, F_EXISTS) == 0 && isatty(2) )
    {
      fprintf (stderr, "%s exists; override [y]? ", trcfile);
      if ( getchar() == 'n' ) exit (1);
    }
    if ( (outfd = open (trcfile, O_WRONLY|O_CREAT|O_TRUNC|O_BINARY| LARGEFILE_OPTION, 0644)) < 0 )
    {
      perror (trcfile);
      exit (1);
    }
  }

# if defined(__ksr__) && defined(__TAU__)
  /* ------------------------------------------------------------------------ */
  /* -- determine clock offset for KSR-1 ------------------------------------ */
  /* -- NB: really need a better algorithm here ----------------------------- */
  /* ------------------------------------------------------------------------ */
# define timestamp(i) \
                     (trcdes[sequence[i]].erec->ti + trcdes[sequence[i]].offset)
# define STEP 1

  sequence     = (int *) malloc (numtrc * sizeof(int));
  last_time    = 0L;
  num_pthreads = numtrc;
  last_pthread = numtrc - 1;

  do
  {
    /* -- search for next "in_barrier" event in each trace and -------------- */
    /* -- store sequence number --------------------------------------------- */
    for (i=0; i<numtrc; i++)
    {
      if ( trcdes[i].nid == (TAU_MAXPROCS-1) ) {
	/* master */
        num_pthreads = numtrc - 1;
        last_pthread = numtrc - 2;
      } else {
        do {
          erec = get_next_rec (trcdes + i);
        } while ( (erec != NULL) && (erec->ev != TAU_IN_BARRIER) );
	
        if ( erec == NULL )
          break;
        else
          sequence[erec->par] = i;
      }
    }

    if ( erec != NULL )  {
      /* -- the first at this barrier must at least arrived later than the -- */
      /* -- last at the last barrier ---------------------------------------- */
      if ( timestamp(0) <= last_time ) {
        trcdes[sequence[0]].offset += last_time - timestamp(0) + STEP;
      }

      for (i=1; i<num_pthreads; i++) {
        if ( timestamp(i) <= timestamp(i-1) ) {
          trcdes[sequence[i]].offset += timestamp(i-1) - timestamp(i) + STEP;
        }
      }
      last_time = timestamp(last_pthread);
    }
  } while ( erec != NULL );

  /* -- only on the first worker pthread trace we should run into EndOfTrace  */
  /* -- This is normally trace 0 unless trace 0 is the master pthread trace - */
  if ( i > (0 + (trcdes[0].nid == (TAU_MAXPROCS - 1))) )
  {
    fprintf (stderr, "%s: missing barrier\n", trcdes[i].name);
    exit (1);
  }

  /* -- report offset found and re-open/re-initialize all traces ------------ */
  for (i=0; i<numtrc; i++)
  {
    if ( trcdes[i].offset )
      printf ("%s: offset %ld\n", trcdes[i].name, trcdes[i].offset);

    /* -- if file was closed during the search for barriers, re-open it ----- */
    if ( trcdes[i].fd = -1 )
    {
      if ( (trcdes[i].fd = open (trcdes[i].name, O_RDONLY|O_BINARY| LARGEFILE_OPTION)) < 0 )
      {
        perror (argv[i]);
        exit (1);
      }
    }
    /* -- otherwise reset file offset --------------------------------------- */
    else if ( lseek (trcdes[i].fd, 0, SEEK_SET) == -1 )
    {
      fprintf (stderr, "%s: cannot reset trace\n", trcdes[i].name);
      exit (1);
    }
    trcdes[i].erec = NULL;
    trcdes[i].next = NULL;
    trcdes[i].last = NULL;

    erec = get_next_rec (trcdes + i);
    trcdes[i].numrec = 1L;
    if ( erec->ev == TAU_EV_INIT )
    {
      if (!dynamic) { /* don't change this for dynamic */
        erec->ev = TAU_EV_INITM;
      }
      trcdes[i].nid = erec->nid;
    }
    else
      trcdes[i].nid = -1;

    trcdes[i].lasttime = erec->ti = erec->ti + trcdes[i].offset;
  }
  printf ("\n");
# else
#   if defined(__CM5__) && defined(__TAU__)
    /* ---------------------------------------------------------------------- */
    /* -- determine clock offset for TMC CM-5 ------------------------------- */
    /* ---------------------------------------------------------------------- */

    /* -- search for sync marker in all traces ------------------------------ */
    /* -- and determine highest clock value --------------------------------- */
    last_time = 0L;
    for (i=0; i<numtrc; i++)
    {
      do
      {
        erec = get_next_rec (trcdes + i);
      }
      while ( (erec != NULL) && (erec->ev != TAU_SYNC_MARK) );

      if ( erec == NULL )
      {
        fprintf (stderr, "%s: cannot find sync marker\n", trcdes[i].name);
        exit (1);
      }

      if ( erec->ti > last_time ) last_time = erec->ti;
    }

    /* -- compute and report offset found and re-initialize all traces ------ */
    for (i=0; i<numtrc; i++)
    {
      trcdes[i].offset = last_time - trcdes[i].erec->ti;
      printf ("%s: offset %ld\n", trcdes[i].name, trcdes[i].offset);

      /* -- reset file offset ----------------------------------------------- */
      if ( lseek (trcdes[i].fd, 0, SEEK_SET) == -1 )
      {
        fprintf (stderr, "%s: cannot reset trace\n", trcdes[i].name);
        exit (1);
      }
      trcdes[i].erec = NULL;
      trcdes[i].next = NULL;
      trcdes[i].last = NULL;

      erec = get_next_rec (trcdes + i);
      trcdes[i].numrec = 1L;
      if ( erec->ev == TAU_EV_INIT )
      {
        erec->ev = TAU_EV_INITM;
        trcdes[i].nid = erec->nid;
      }
      else
        trcdes[i].nid = -1;

      trcdes[i].lasttime = erec->ti = erec->ti + trcdes[i].offset;
    }
    printf ("\n");
#   endif
# endif

  /* ------------------------------------------------------------------------ */
  /* -- merge files --------------------------------------------------------- */
  /* ------------------------------------------------------------------------ */
  source = 0;

  do
  {
    /* -- compute minimum of all timestamps and store the index of the ------ */
    /* -- corresponding trace in source ------------------------------------- */
    first = TRUE;
    for (i=0; i<numtrc; i++)
    {
      if ( trcdes[i].fd != -1 )
      {
        if ( first )
        {
          min_time = trcdes[i].lasttime;
          min_over = trcdes[i].overflows;
          source = i;
          first  = FALSE;
        }
        else if ( trcdes[i].overflows < min_over )
        {
          min_time = trcdes[i].lasttime;
          min_over = trcdes[i].overflows;
          source = i;
        }
        else if ( (trcdes[i].overflows == min_over) &&
                  (trcdes[i].lasttime < min_time) )
        {
          min_time = trcdes[i].lasttime;
          source = i;
        }
      }
    }

    if ( adjust ) {
      if ( numrec == 0 ) {
	/*first_time = trcdes[source].erec->ti;*/
	first_time = event_GetTi(trcdes+source,erec,0);
      }
      /*trcdes[source].erec->ti -= first_time;*/
      event_SetTi(trcdes+source,erec,0,event_GetTi(trcdes+source,erec,0)-first_time);
    }
    /* -- correct event id to be global event id ---------------------------- */

    /* OLD : trcdes[source].erec->ev = GID(trcdes[source].nid, trcdes[source].erec->ev);
     */
    /*trcdes[source].erec->ev = GID(source, trcdes[source].erec->ev);*/
    event_SetEv(trcdes+source,trcdes[source].erec,0,GID(source, event_GetEv(trcdes+source,trcdes[source].erec,0)));



    /*    output (outfd, (char *) trcdes[source].erec, sizeof(TAU_EV));*/

    nativeEvent.ev = event_GetEv(trcdes+source,trcdes[source].erec,0);
    nativeEvent.nid = event_GetNid(trcdes+source,trcdes[source].erec,0);
    nativeEvent.tid = event_GetTid(trcdes+source,trcdes[source].erec,0);
    nativeEvent.par = event_GetPar(trcdes+source,trcdes[source].erec,0);
    nativeEvent.ti = event_GetTi(trcdes+source,trcdes[source].erec,0);
    /* printf ("writing out record with ev = %d, nid = %d\n", nativeEvent.ev, nativeEvent.nid);  */
    output (outfd, (char *) &nativeEvent, sizeof(TAU_EV));

    numrec++;

# if defined(__ksr__) && defined(__TAU__)
    erec = trcdes[source].erec;
    if ( erec->ev == TAU_IN_BARRIER )
    {
      if ( ((last_pthread + 1) % num_pthreads) != erec->par )
        fprintf (stderr, "tau_Barrier sequence error at %ld (%d -> %d)\n",
                 erec->ti, last_pthread, erec->par);
      last_pthread = erec->par;
    }
# endif

    /* -- get next event record(s) from same trace -------------------------- */
    do
    {
      if ( (erec = get_next_rec (trcdes + source)) == NULL ) {
        active--;
        break;
      } else  {
        trcdes[source].numrec++;

          /* -- correct nid event field (only the first time) --------------- */
          if ( trcdes[source].nid != -1 ) {
	    /*erec->nid = trcdes[source].nid;*/
	    /* Removed code to explicitly set the node id to the node id of 
	the trace. This hinders merging traces that have remote one-sided events
	where a given trace has events that take place on a remote node as in 
	send or receive in one-sided put operations. */
	    /*event_SetNid(trcdes+source,erec,0,trcdes[source].nid); */
	  }

          /* -- correct clock ----------------------------------------------- */
          /*erec->ti += trcdes[source].offset;*/
	  event_SetTi(trcdes+source,erec,0,event_GetTi(trcdes+source,erec,0)+trcdes[source].offset);

          /* -- check clock overflow ---------------------------------------- */
          /*if ( erec->ti < trcdes[source].lasttime ) {*/
          if ( event_GetTi(trcdes+source,erec,0) < trcdes[source].lasttime ) {
	    trcdes[source].overflows++;
	  }

          /*trcdes[source].lasttime = erec->ti;*/
          trcdes[source].lasttime = event_GetTi(trcdes+source,erec,0);

          /* -- remember continuation event length -------------------------- */
          /*trcdes[source].contlen = erec->par;*/
          trcdes[source].contlen = event_GetPar(trcdes+source,erec,0);

      }
    }
    /*while ( erec->ev == TAU_EV_CONT_EVENT );*/
    while ( event_GetEv(trcdes+source,erec,0) == TAU_EV_CONT_EVENT );
  }
  while ( active > 0 );
  for (i=0; i<numtrc; i++)
  { 
    if (outfd != STDOUT) 
    {
      fprintf (stderr, "%s: %ld records read.\n",
             trcdes[i].name, trcdes[i].numrec);
    }
  }

  output_flush (outfd);
  close (outfd);
  exit (0);
}
