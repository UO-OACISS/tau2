/*********************************************************************
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
# include <unistd.h>

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

#ifdef TAU_DOT_H_LESS_HEADERS 
#include <vector>
#include <map>
#else /* TAU_DOT_H_LESS_HEADERS */
# include <vector.h>
# include <map.h>
#endif /*  TAU_DOT_H_LESS_HEADERS */
#ifdef KAI
using namespace std;
#endif /* KAI*/ 

# define FILENAME_SIZE 	1024
# define MAX_OPEN_FILES  256
# define LINEMAX	64*1024
FILE * edfFiles[MAX_OPEN_FILES]; /* array of descriptors */

int  dynamictrace = FALSE;

extern "C" {
  int open_edf_file(char *prefix, int nodeid);
  int parse_edf_file(int node);
  int store_merged_edffile(char *filename);
  int GID(int node, long localEventId);
}

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

map<const char*, EventDescr, ltstr> eventNameMap;
/* eventNameMap stores name to global event id mapping */

int globalEventId = 1; /* numbers start from 1 */

map<long, int, less<long> > nodeEventTable[MAX_OPEN_FILES];
/* nodeEventTable stores nodeid, localevent ->globaleventid mapping */
/* e.g., nodeEventTable[0] gives event map for node 0, emap[60000] 
gives global id say 105 of event 60000 */

int open_edf_file(char *prefix, int nodeid)
{
  char filename[FILENAME_SIZE];

  if (nodeid > MAX_OPEN_FILES)
  {
    printf("nodeid %d exceeds MAX_OPEN_FILES. Recompile with extended limit\n", nodeid);
    exit(1);
  }

  sprintf(filename, "%s.%d.edf", prefix,nodeid);
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
  map<const char*, EventDescr, ltstr>::iterator iter;
  
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
	nodeEventTable[node][localEventId] = eventNameMap[(const char *)stlEvName].gid;
	delete stlEvName; /* don't need this if its already there */
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

	eventNameMap[(const char *)stlEvName] = EventDescr(globalEventId,
	  inputev.state, inputev.tag, inputev.param);
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

int store_merged_edffile(char *filename)
{
  FILE *fp;
  char *errormsg;
  map<const char*, EventDescr, ltstr>::iterator it;

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
    fprintf(fp, "%d %s %d %s %s", (*it).second.gid, (*it).second.state, 
      (*it).second.tag, (*it).first, (*it).second.param);

#ifdef DEBUG
    printf("BEGIN:%d %s %d %s %s:END\n", (*it).second.gid, (*it).second.state, 
      (*it).second.tag, (*it).first, (*it).second.param);
#endif /* DEBUG */
  }  
  
  return 1;
}

int GID(int node, long local)
{
  return nodeEventTable[node][local];
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
    open_edf_file(edfFilePrefix, i);
    parse_edf_file(i);
  }

  printf("NODE 3 LOCAL ID 60000 global id = %d\n", GID(3,60000));
  store_merged_edffile("events.edf");

}  
#endif /* OLDMAIN */
