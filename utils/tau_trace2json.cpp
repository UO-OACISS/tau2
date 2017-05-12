/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2003  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, Germany                                     **
****************************************************************************/
/***************************************************************************
**	File 		: tau_trace2json.cpp				  **
**	Description 	: TAU trace to json format converter tool         **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 	                  **
***************************************************************************/
#include <TAU_tf.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <map>
using namespace std;
int debugPrint = 0;
int jsonPrint = 1;
#define dprintf if (debugPrint) printf
#define json_printf if (jsonPrint) printf

/* implementation of callback routines */
map< pair<int,int>, int, less< pair<int,int> > > EOF_Trace;
int EndOfTrace = 0;  /* false */
/* implementation of callback routines */
int EnterState(void *userData, double time, 
		unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
  dprintf("Entered state %d time %g nid %d tid %d\n", 
		  stateid, time, nodeid, tid);
  json_printf("  {\n"); 
  json_printf("    \"state\": \"entry\",\n"); 
  json_printf("    \"event-id\": \"%d\",\n", stateid); 
  json_printf("    \"time\": \"%g\",\n", time); 
  json_printf("    \"node-id\": \"%d\",\n", nodeid); 
  json_printf("    \"thread-id\": \"%d\"\n", tid); 
  json_printf("  },\n"); 
  return 0;
}

int LeaveState(void *userData, double time, unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
  dprintf("Leaving state %d time %g nid %d tid %d\n", stateid, time, nodeid, tid);
  json_printf("  {\n"); 
  json_printf("    \"state\": \"exit\",\n"); 
  json_printf("    \"event-id\": \"%d\",\n", stateid); 
  json_printf("    \"time\": \"%g\",\n", time); 
  json_printf("    \"node-id\": \"%d\",\n", nodeid); 
  json_printf("    \"thread-id\": \"%d\"\n", tid); 
  json_printf("  },\n"); 
  return 0;
}


int ClockPeriod( void*  userData, double clkPeriod )
{
  dprintf("Clock period %g\n", clkPeriod);
  json_printf("  {\n"); 
  json_printf("    \"clock-period\": \"%g\"\n", clkPeriod); 
  json_printf("  },\n"); 
  return 0;
}

int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
  dprintf("DefThread nid %d tid %d, thread name %s\n", 
		  nodeToken, threadToken, threadName);
  json_printf("  {\n"); 
  json_printf("    \"thread-name\": \"%s\",\n", threadName); 
  json_printf("    \"node-id\": \"%d\",\n", nodeToken); 
  json_printf("    \"thread-id\": \"%d\"\n", threadToken); 
  json_printf("  },\n"); 
  EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 0; /* initialize it */
  return 0;
}

int EndTrace( void *userData, unsigned int nodeToken, unsigned int threadToken)
{
  dprintf("EndTrace nid %d tid %d\n", nodeToken, threadToken);
  json_printf("  {\n"); 
  json_printf("    \"state-of-trace\": \"end\",\n");
  json_printf("    \"node-id\": \"%d\",\n", nodeToken); 
  json_printf("    \"thread-id\": \"%d\"\n", threadToken); 
  json_printf("  },\n"); 
  EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 1; /* flag it as over */
  /* yes, it is over */
  map < pair<int, int>, int, less< pair<int,int> > >::iterator it;
  EndOfTrace = 1; /* Lets assume that it is over */
  for (it = EOF_Trace.begin(); it != EOF_Trace.end(); it++)
  { /* cycle through all <nid,tid> pairs to see if it really over */
    if ((*it).second == 0)
    {
      EndOfTrace = 0; /* not over! */
      /* If there's any processor that is not over, then the trace is not over */
    }
  }
  return 0;
}

int DefStateGroup( void *userData, unsigned int stateGroupToken, 
		const char *stateGroupName )
{
  dprintf("StateGroup groupid %d, group name %s\n", stateGroupToken, 
		  stateGroupName);
  json_printf("  {\n"); 
  json_printf("    \"group\": \"definition\",\n"); 
  json_printf("    \"group-id\": \"%d\",\n",stateGroupToken);
  json_printf("    \"group-name\": \"%s\"\n", stateGroupName); 
  json_printf("  },\n"); 
  return 0;
}

int DefState( void *userData, unsigned int stateToken, const char *stateName, 
		unsigned int stateGroupToken )
{
  dprintf("DefState stateid %d stateName %s stategroup id %d\n",
		  stateToken, stateName, stateGroupToken);
  json_printf("  {\n"); 
  json_printf("    \"state\": \"definition\",\n"); 
  json_printf("    \"event-id\": \"%d\",\n", stateToken); 
  json_printf("    \"group-id\": \"%d\",\n", stateGroupToken); 
  json_printf("    \"event-name\": %s\n", stateName); 
  json_printf("  },\n"); 
  return 0;
}

int DefUserEvent( void *userData, unsigned int userEventToken,
		const char *userEventName, int monotonicallyIncreasing )
{

  dprintf("DefUserEvent event id %d user event name %s, monotonically increasing = %d\n", userEventToken,
		  userEventName, monotonicallyIncreasing);
  json_printf("  {\n"); 
  json_printf("    \"user-event\": \"definition\",\n"); 
  json_printf("    \"user-event-id\": \"%d\",\n", userEventToken); 
  json_printf("    \"monotonically-increasing\": \"%d\",\n", monotonicallyIncreasing); 
  json_printf("    \"user-event-name\": %s\n", userEventName); 
  json_printf("  },\n"); 
  return 0;
}

int EventTrigger( void *userData, double time, 
		unsigned int nodeToken,
		unsigned int threadToken,
	       	unsigned int userEventToken,
		long long userEventValue)
{
  dprintf("EventTrigger: time %g, nid %d tid %d event id %d triggered value %lld \n", time, nodeToken, threadToken, userEventToken, userEventValue);
  json_printf("  {\n"); 
  json_printf("    \"user-event\": \"trigger\",\n"); 
  json_printf("    \"timestamp\": \"%g\",\n", time); 
  json_printf("    \"user-event-id\": \"%d\",\n", userEventToken); 
  json_printf("    \"node-id\": \"%d\",\n", nodeToken); 
  json_printf("    \"thread-id\": \"%d\",\n", threadToken); 
  json_printf("    \"triggered-value\": \"%lld\"\n", userEventValue); 
  json_printf("  },\n"); 
  return 0;
}

int SendMessage( void *userData, double time, 
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
		unsigned int messageTag,
		unsigned int messageComm )
{
  dprintf("SendMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n", 
		  time, 
		  sourceNodeToken, sourceThreadToken,
		  destinationNodeToken, destinationThreadToken,
		  messageSize, messageTag);
  json_printf("  {\n"); 
  json_printf("    \"message-event\": \"send\",\n"); 
  json_printf("    \"timestamp\": \"%g\",\n", time); 
  json_printf("    \"source-node-id\": \"%d\",\n", sourceNodeToken); 
  json_printf("    \"source-thread-id\": \"%d\",\n", sourceThreadToken); 
  json_printf("    \"destination-node-id\": \"%d\",\n", destinationNodeToken); 
  json_printf("    \"destination-thread-id\": \"%d\",\n", destinationThreadToken); 
  json_printf("    \"message-size\": \"%d\",\n", messageSize); 
  json_printf("    \"message-tag\": \"%lld\"\n", messageTag); 
  json_printf("  },\n"); 
  return 0;
}

int RecvMessage( void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
                unsigned int messageTag,
		unsigned int messageComm )
{
  dprintf("RecvMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n", 
		  time, 
		  sourceNodeToken, sourceThreadToken,
		  destinationNodeToken, destinationThreadToken,
		  messageSize, messageTag);
  json_printf("  {\n"); 
  json_printf("    \"message-event\": \"receive\",\n"); 
  json_printf("    \"timestamp\": \"%g\",\n", time); 
  json_printf("    \"source-node-id\": \"%d\",\n", sourceNodeToken); 
  json_printf("    \"source-thread-id\": \"%d\",\n", sourceThreadToken); 
  json_printf("    \"destination-node-id\": \"%d\",\n", destinationNodeToken); 
  json_printf("    \"destination-thread-id\": \"%d\",\n", destinationThreadToken); 
  json_printf("    \"message-size\": \"%d\",\n", messageSize); 
  json_printf("    \"message-tag\": \"%lld\"\n", messageTag); 
  json_printf("  },\n"); 
  return 0;
}

int main(int argc, char **argv)
{
  Ttf_FileHandleT fh;
  int recs_read, pos;
  char *trace_file;
  char *edf_file;
  int no_state_flag=0, no_message_flag=0;
  /* main program: Usage app <trc> <edf> [-nostate] [-nomessage] */
  if (argc < 3)
  {
    printf("Usage: %s <TAU trace> <edf file> [-nostate] [-nomessage] [-v] [-nojson]\n", 
		    argv[0]);
    return 1;
  }
  
  for (int i = 0; i < argc ; i++)
  {
    switch(i) {
      case 0:
	trace_file = argv[1];
	break;
      case 1:
	edf_file = argv[2];
	break;
      default:
	if (strcmp(argv[i], "-nostate")==0)
	{
	  no_state_flag = 1;
	}
	if (strcmp(argv[i], "-nomessage")==0)
	{
	  no_message_flag = 1;
	}
	if (strcmp(argv[i], "-v")==0)
	{
	  debugPrint = 1;
        }
	if (strcmp(argv[i], "-nojson")==0)
	{
	  jsonPrint = 0;
        }
	break;
    }
  }

  fh = Ttf_OpenFileForInput( argv[1], argv[2]);

  if (!fh)
  {
    printf("ERROR:Ttf_OpenFileForInput fails");
    return 1;
  }

  Ttf_CallbacksT cb;
  /* Fill the callback struct */
  cb.UserData = 0;
  cb.DefClkPeriod = ClockPeriod;
  cb.DefThread = DefThread;
  cb.DefStateGroup = DefStateGroup;
  cb.DefState = DefState;
  cb.DefUserEvent = DefUserEvent;
  cb.EventTrigger = EventTrigger;
  cb.EndTrace = EndTrace;

  /* should state transitions be displayed? */
  if (no_state_flag)
  {
    cb.EnterState = 0;
    cb.LeaveState = 0;
  }
  else
  {
    cb.EnterState = EnterState;
    cb.LeaveState = LeaveState;
  }

  /* should messages be displayed? */
  if (no_message_flag)
  {
    cb.SendMessage = 0;
    cb.RecvMessage = 0;
  }
  else
  {
    cb.SendMessage = SendMessage;
    cb.RecvMessage = RecvMessage;
  }

  json_printf("{\n");
  /* Go through each record until the end of the trace file */
  do {
    recs_read = Ttf_ReadNumEvents(fh,cb, 1024);
    if (recs_read != 0)
      dprintf("Read %d records\n", recs_read);
  }
  while ((recs_read >=0) && (!EndOfTrace));

  /* We need to close the trace so there is no dangling comma "," before 
   * the final "}" record in json - so we create a dummy end-of-output record, 
   * to get our commas in order */

  json_printf("  { \n");
  json_printf("    \"end-of-output\": \"1\" \n");
  json_printf("  } \n");
  json_printf("}\n");
  Ttf_CloseFile(fh);
  return 0;
}


