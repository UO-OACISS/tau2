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
**	File 		: app.cpp 					  **
**	Description 	: TAU trace format reader application             **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 	                  **
***************************************************************************/
#include <TAU_tf.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <map>
using namespace std;
int debugPrint = 1;
#define dprintf if (debugPrint) printf

/* implementation of callback routines */
map< pair<int,int>, int, less< pair<int,int> > > EOF_Trace;
int EndOfTrace = 0;  /* false */
/* implementation of callback routines */
int EnterState(void *userData, double time, 
		unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
  dprintf("Entered state %d time %g nid %d tid %d\n", 
		  stateid, time, nodeid, tid);
  return 0;
}

int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int stateid)
{
  dprintf("Leaving state %d time %g nid %d tid %d\n", stateid, time, nid, tid);
  return 0;
}


int ClockPeriod( void*  userData, double clkPeriod )
{
  dprintf("Clock period %g\n", clkPeriod);
  return 0;
}

int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
  dprintf("DefThread nid %d tid %d, thread name %s\n", 
		  nodeToken, threadToken, threadName);
  EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 0; /* initialize it */
  return 0;
}

int EndTrace( void *userData, unsigned int nodeToken, unsigned int threadToken)
{
  dprintf("EndTrace nid %d tid %d\n", nodeToken, threadToken);
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
  return 0;
}

int DefState( void *userData, unsigned int stateToken, const char *stateName, 
		unsigned int stateGroupToken )
{
  dprintf("DefState stateid %d stateName %s stategroup id %d\n",
		  stateToken, stateName, stateGroupToken);
  return 0;
}

int DefUserEvent( void *userData, unsigned int userEventToken,
		const char *userEventName, int monotonicallyIncreasing )
{

  dprintf("DefUserEvent event id %d user event name %s, monotonically increasing = %d\n", userEventToken,
		  userEventName, monotonicallyIncreasing);
  return 0;
}

int EventTrigger( void *userData, double time, 
		unsigned int nodeToken,
		unsigned int threadToken,
	       	unsigned int userEventToken,
		long long userEventValue)
{
  dprintf("EventTrigger: time %g, nid %d tid %d event id %d triggered value %lld \n", time, nodeToken, threadToken, userEventToken, userEventValue);
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
    printf("Usage: %s <TAU trace> <edf file> [-nostate] [-nomessage]\n", 
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

  /* Go through each record until the end of the trace file */
  do {
    recs_read = Ttf_ReadNumEvents(fh,cb, 1024);
    if (recs_read != 0)
      cout <<"Read "<<recs_read<<" records"<<endl;
  }
  while ((recs_read >=0) && (!EndOfTrace));
  Ttf_CloseFile(fh);
  return 0;
}


