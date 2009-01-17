/*********************************************************************/
/* 	TAU Tracing 						     */
/* 	U.Oregon, ACL, LANL (C) 1997				     */
/*                  pC++/Sage++  Copyright (C) 1994                  */
/*  Indiana University  University of Oregon  University of Rennes   */
/*********************************************************************/

/*
 * tau_events.cpp
 */

# include <stdio.h>
# include <stdlib.h>
# include <sys/types.h>
# include <fcntl.h>
#ifdef TAU_WINDOWS
  #include <io.h>
#else
  #include <unistd.h>
#endif

# include <string.h>

#define TRACING_ON
#include <Profile/TauTrace.h>

# ifndef TRUE
#   define FALSE  0
#   define TRUE   1
# endif

# define F_EXISTS    0

#ifdef TAU_DOT_H_LESS_HEADERS 
#include <vector>
#include <map>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
# include <vector.h>
# include <map.h>
#endif /*  TAU_DOT_H_LESS_HEADERS */
#ifdef KAI
using namespace std;
#endif /* KAI*/ 

# define FILENAME_SIZE 	1024
# define MAX_OPEN_FILES  16*1024
#ifdef TAU_HP_GNU
# define LINEMAX	2*1024
#else
# define LINEMAX	64*1024
#endif /* TAU_HP_GNU */
FILE * edfFiles[MAX_OPEN_FILES]; /* array of descriptors */

int  dynamictrace = FALSE;

extern char *mergededffile; /* name of merged edf file */
extern char **edfnames; /* names of edf files, if specified by the user */
extern int edfspecified; /* whether edf files are specified by the user */

#ifndef TAU_NEC
extern "C" {
#endif /*TAU_NEC compiles tau_merge.c as a C++ file */
  int open_edf_file(char *prefix, int nodeid, int prefix_is_filename);
  int parse_edf_file(int node);
  int store_merged_edffile(char *filename);
  int GID(int node, long localEventId);
#ifndef TAU_NEC
}
#endif /* TAU_NEC */
char header[256]; /* File header like # FunctionId Group Tag "Name Type" Par */

struct ltstr
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) < 0;
  }
};

struct EventDescr {
  int  gid; /* global event id */
  char state[64]; /* state as in TAU_VIZ */
  int  tag; /* -7 for send etc. */
  char param[64]; /* param as in EntryExit */
  public:
    EventDescr(int evGlobalId, char *evState, int evTag, char *evParam)
    {
      gid = evGlobalId;
      tag = evTag;
      strcpy(state, evState);
      strcpy(param, evParam);
    }
    EventDescr()
    {
    } 
    EventDescr(const EventDescr & X) 
    {
      gid = X.gid;
      tag = X.tag;
      strcpy(state, X.state);
      strcpy(param, X.param);
    }
    EventDescr& operator= (const EventDescr& X) 
    {
      gid = X.gid;
      tag = X.tag;
      strcpy(state, X.state);
      strcpy(param, X.param);
      return *this;
    }  
    ~EventDescr() { }
};

struct trcdescr
{
  int     fd;              /* -- input file descriptor                     -- */
  char   *name;            /* -- corresponding file name                   -- */
  int     nid;             /* -- corresponding PTX PID                     -- */
  int     overflows;       /* -- clock overflows in that trace             -- */
  int     contlen;         /* -- length of continuation event buffer       -- */
  long    numrec;          /* -- number of event records already processed -- */
  unsigned long lasttime;  /* -- timestamp of previous event record        -- */
  unsigned long offset;    /* -- offset of timestamp                       -- */

  PCXX_EV  *buffer;    /* -- input buffer                              -- */
  PCXX_EV  *erec;      /* -- current event record                      -- */
  PCXX_EV  *next;      /* -- next available event record in buffer     -- */
  PCXX_EV  *last;      /* -- last event record in buffer               -- */
} ;
extern struct trcdescr *trcdes;
map<const char*, EventDescr *, ltstr> eventNameMap;
/* eventNameMap stores name to global event id mapping */

int globalEventId = 1; /* numbers start from 1 */

map<long, int, less<long> > nodeEventTable[MAX_OPEN_FILES];
/* nodeEventTable stores nodeid, localevent ->globaleventid mapping */
/* e.g., nodeEventTable[0] gives event map for node 0, emap[60000] 
gives global id say 105 of event 60000 */


int get_nodeid(int edf_file_index)
{
  return trcdes[edf_file_index].nid; /* return the node id associated with it*/
}

int open_edf_file(char *prefix, int nodeid, int prefix_is_filename)
{
  char filename[FILENAME_SIZE];

  if (nodeid > MAX_OPEN_FILES)
  {
    printf("nodeid %d exceeds MAX_OPEN_FILES. Recompile with extended limit\n", nodeid);
    exit(1);
  }

  if (prefix_is_filename)
  { /* Use prefix as the file name. Don't add any numbers to it. */
    strcpy(filename, prefix);
  }
  else 
  { /* default mode of operation, use prefix and node id */
    sprintf(filename, "%s.%d.edf", prefix,nodeid);
  }

  if ( (edfFiles[nodeid] = fopen (filename, "r")) == NULL )
  {
    perror (filename);
    exit (1);
  }

  return 1;
}

int parse_edf_file(int node) 
{
  int i,j,k;
  char linebuf[LINEMAX], eventname[LINEMAX], traceflag[32]; 
  char *stlEvName;
  int numevents;
  long localEventId;
  EventDescr inputev;
  map<const char*, EventDescr *, ltstr>::iterator iter;
  
#ifdef DEBUG
  printf("parse edf file: node = %d\n", node);
#endif /* DEBUG */
  fgets (linebuf, LINEMAX, edfFiles[node]);
  sscanf (linebuf, "%d %s", &numevents, traceflag);

  if (strcmp(traceflag, "dynamic_trace_events") == 0) 
  {
    dynamictrace = TRUE;
  }

  for (i=0; i<numevents; i++)
  {
    fgets (linebuf, LINEMAX, edfFiles[node]);
    if ( (linebuf[0] == '\n') || (linebuf[0] == '#') )
    {
      if (linebuf[0] == '#') /* store header for output file */
      {
	strcpy(header, linebuf);
      }
      /* -- skip empty and comment lines -- */
      i--;
      continue;
    }

    localEventId = -1;
    eventname[0]  = '\0';
    inputev.param[0] = '\0';
    if (dynamictrace) /* get eventname in quotes */
    { 
      memset(inputev.state,0,sizeof(inputev.state));
      sscanf (linebuf, "%ld %s %d", &localEventId, inputev.state, &inputev.tag);
#ifdef DEBUG
      printf("Got localEventId %d state %s tag %d\n", localEventId, inputev.state, inputev.tag);
#endif /* DEBUG */
      for(j=0; linebuf[j] !='"'; j++)
	;
      eventname[0] = linebuf[j];
      j++;
      /* skip over till eventname begins */
      for (k=j; linebuf[k] != '"'; k++)
      {
	eventname[k-j+1] = linebuf[k];
      } 
      eventname[k-j+1] = '"';
      eventname[k-j+2] = '\0'; /* terminate eventname */

      strcpy(inputev.param, &linebuf[k+2]);

#ifdef DEBUG 
      printf(" Got eventname=%s param=%s\n", eventname, inputev.param);
#endif /* DEBUG */
      /* now that we have eventname, see if it exists in the map */
      /* Since pointers create a problem with stl we have to do this */
      stlEvName = new char[strlen(eventname)+1];
      strcpy(stlEvName, eventname);
      if((iter = eventNameMap.find((const char *)stlEvName)) != eventNameMap.end()) 
      {
	/* Found eventname in the map */
#ifdef DEBUG
	printf("Found %s in map \n", eventname);
#endif /* DEBUG */
        /* add to nodeEventTable the global id of the event */
	nodeEventTable[node][localEventId] = eventNameMap[(const char *)stlEvName]->gid;
	delete[] stlEvName; /* don't need this if its already there */
#ifdef DEBUG
	printf("node %d local %ld global %d %s \n", node, localEventId,
	  nodeEventTable[node][localEventId], stlEvName);
#endif /* DEBUG */
      } 
      else 
      { /* Not found. Create a new entry in the map! */
#ifdef DEBUG
	printf("Event %s not found in map. Assigning new event id %d\n",
	  eventname, globalEventId);
#endif /* DEBUG */

	EventDescr *e = new  EventDescr(globalEventId,
          inputev.state, inputev.tag, inputev.param);
	eventNameMap[(const char *)stlEvName] = e;
	// Adds a null record and creates the name key in the map
	// Note: don't delete stlEvName - STL needs it.
	nodeEventTable[node][localEventId] = globalEventId;

#ifdef DEBUG
	printf("node %d local %ld global %d %s \n", node, localEventId,
	  nodeEventTable[node][localEventId], stlEvName);
#endif /* DEBUG */

	globalEventId ++;
      }  


    } /* not dynamic trace- what is to be done? */ 
    else 
    {  
      sscanf (linebuf, "%ld %s %d %s %s", &localEventId, 
        inputev.state, &inputev.tag, eventname, inputev.param);
    }

    if ( (localEventId < 0) || !*eventname )
    {
      fprintf (stderr, "events.%d.edf: blurb in line %d\n", node, i+2);
      exit (1);
    }
 
  } /* i < numevents  for loop*/
  return 1;
}  

extern "C" const char *get_event_name(int gid)
{
  map<const char*, EventDescr *, ltstr>::iterator it;
  for (it = eventNameMap.begin(); it != eventNameMap.end(); it++)
  {
#ifdef DEBUG
    printf("get_event_name: Examining %s id %d: Looking for %d\n", 
		    (*it).first, (*it).second->gid, gid);
#endif
    if ((*it).second->gid == gid)
    {
#ifdef DEBUG
      printf("get_event_name: Returning  %s\n", (*it).first);
#endif
      return (*it).first;
    }
  }
  return NULL;
}

int store_merged_edffile(char *filename)
{
  FILE *fp;
  char *errormsg;
  map<const char*, EventDescr *, ltstr>::iterator it;

  if ((fp = fopen (filename, "w+")) == NULL) {
    errormsg = new char[strlen(filename)+32]; 
    sprintf(errormsg,"Error: Could not create %s",filename);
    perror(errormsg);
    return 0;
  }

  fprintf(fp,"%d dynamic_trace_events\n%s",eventNameMap.size(), header);
#ifdef DEBUG
  printf("%d dynamic_trace_events\n%s",eventNameMap.size(), header);
  printf("\nEND OF HEADER\n");
#endif 

  for (it = eventNameMap.begin(); it != eventNameMap.end(); it++) 
  {
    fprintf(fp, "%d %s %d %s %s", (*it).second->gid, (*it).second->state, 
      (*it).second->tag, (*it).first, (*it).second->param);

#ifdef DEBUG
    printf("BEGIN:%d %s %d %s %s:END\n", (*it).second->gid, (*it).second->state, 
      (*it).second->tag, (*it).first, (*it).second->param);
#endif /* DEBUG */
  }  
  
#ifdef DEBUG
  printf("Closing file %s\n", filename);
#endif /* DEBUG */
  fflush(fp);
  fclose(fp);
  return 1;
}

int GID(int node, long local)
{
  map<long, int, less<long> >::iterator it;
  if ((it = nodeEventTable[node].find(local)) != nodeEventTable[node].end())
  { /* found it 
     printf("local %d global %d: ", local, (*it).second);
     */
     return (*it).second; 
  }
  else
  { /* couldn't locate it, must re-read the event table again and try it */
#ifdef DEBUG
    printf("GID: closing event file on node %d and re-reading it.", node);
    printf("Looking for id %d\n",local);
#endif /* DEBUG */
    fclose(edfFiles[node]);

    /* OLD 
    open_edf_file("events", node, FALSE);
    */
    /* use default edf file names  "events.*.edf" */
    if (edfspecified == FALSE)
    {
      char eventfilename[2048];
      sprintf(eventfilename, "events.%d.edf", get_nodeid(node));
      open_edf_file(eventfilename, node, TRUE);
#ifdef DEBUG
      printf("re-opening %s\n", eventfilename);
#endif /* DEBUG */
    }
    else  /* edf file is not specified */
    { /* Hey, we need to know the edf file name! */
      open_edf_file(edfnames[node], node, TRUE); 
#ifdef DEBUG
      printf("re-opening edf file (specified by user): %s\n", edfnames[node]);
#endif /* DEBUG */
    }

    parse_edf_file(node);
    store_merged_edffile(mergededffile); /* update the merged edf file */
    return nodeEventTable[node][local];
  }
  /* OLD 
  int ret =  nodeEventTable[node][local];
  printf("local = %d, global = %d \n",local,  ret);
  return ret;
  */
}
  
#ifdef OLDMAIN
int main(int argc, char **argv)
{
  int numtraces, i;
  /* get default edf File prefix and open edf files */

  strcpy(edfFilePrefix, "events");
  
  numtraces = 4;

  for(i=0; i <numtraces; i++)
  {
    open_edf_file(edfFilePrefix, i, FALSE);
    parse_edf_file(i);
  }

  printf("NODE 3 LOCAL ID 60000 global id = %d\n", GID(3,60000));
  store_merged_edffile("events.edf");

}  
#endif /* OLDMAIN */
