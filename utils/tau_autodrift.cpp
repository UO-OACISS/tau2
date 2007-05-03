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
#include <iostream>
#include <map>
#include <float.h>

using namespace std;
int debugPrint = 1;
#define dprintf if (debugPrint) printf


Ttf_FileHandleT outFile;
double driftrate;


int beginOffsetEvent = -1;
int endOffsetEvent = -1;
double beginOffset = DBL_MIN;
double endOffset = DBL_MIN;
double beginOffsetTimestamp = DBL_MIN;
double endOffsetTimestamp = DBL_MIN;



/* implementation of callback routines */
map< pair<int,int>, int, less< pair<int,int> > > EOF_Trace;
int EndOfTrace = 0;  /* false */
/* implementation of callback routines */
int EnterState(void *userData, double time, 
		unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
//   dprintf("Entered state %d time %g nid %d tid %d\n", 
// 		  stateid, time, nodeid, tid);
  Ttf_EnterState(outFile, time*driftrate, nodeid, tid, stateid);
  return 0;
}

int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int stateid)
{
  Ttf_LeaveState(outFile, time*driftrate, nid, tid, stateid);
//   dprintf("Leaving state %d time %g nid %d tid %d\n", stateid, time, nid, tid);
  return 0;
}


int ClockPeriod( void*  userData, double clkPeriod )
{
//   dprintf("Clock period %g\n", clkPeriod);
  return 0;
}

int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
//   dprintf("DefThread nid %d tid %d, thread name %s\n", 
// 		  nodeToken, threadToken, threadName);
  EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 0; /* initialize it */
  Ttf_DefThread(outFile, nodeToken, threadToken, threadName);

  return 0;
}

int EndTrace( void *userData, unsigned int nodeToken, unsigned int threadToken)
{
//   dprintf("EndTrace nid %d tid %d\n", nodeToken, threadToken);
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
//   dprintf("StateGroup groupid %d, group name %s\n", stateGroupToken, 
// 		  stateGroupName);
  Ttf_DefStateGroup(outFile, stateGroupName, stateGroupToken);
  return 0;
}

int DefState( void *userData, unsigned int stateToken, const char *stateName, 
		unsigned int stateGroupToken )
{
//   dprintf("DefState stateid %d stateName %s stategroup id %d\n",
// 		  stateToken, stateName, stateGroupToken);
  string unquotedName = stateName;
  unquotedName = unquotedName.substr(1,unquotedName.size()-2);
  Ttf_DefState(outFile, stateToken, unquotedName.c_str(), stateGroupToken);
  return 0;
}

int DefUserEvent( void *userData, unsigned int userEventToken,
		const char *userEventName, int monotonicallyIncreasing )
{

//   dprintf("DefUserEvent event id %d user event name %s, monotonically increasing = %d\n", userEventToken,
// 		  userEventName, monotonicallyIncreasing);
  string unquotedName = userEventName;
  unquotedName = unquotedName.substr(1,unquotedName.size()-2);

  Ttf_DefUserEvent(outFile, userEventToken, unquotedName.c_str(), monotonicallyIncreasing);
  return 0;
}

int EventTrigger( void *userData, double time, 
		unsigned int nodeToken,
		unsigned int threadToken,
	       	unsigned int userEventToken,
		long long userEventValue)
{
//   dprintf("EventTrigger: time %g, nid %d tid %d event id %d triggered value %lld \n", time, nodeToken, threadToken, userEventToken, userEventValue);
  Ttf_EventTrigger (outFile, time*driftrate, nodeToken, threadToken, userEventToken, userEventValue);
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
  Ttf_SendMessage( outFile, time*driftrate, sourceNodeToken, sourceThreadToken, destinationNodeToken, destinationThreadToken,
		   messageSize, messageTag, messageComm);
//   dprintf("SendMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n", 
// 		  time, 
// 		  sourceNodeToken, sourceThreadToken,
// 		  destinationNodeToken, destinationThreadToken,
// 		  messageSize, messageTag);
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
  Ttf_RecvMessage( outFile, time*driftrate, sourceNodeToken, sourceThreadToken, destinationNodeToken, destinationThreadToken,
		   messageSize, messageTag, messageComm);
//   dprintf("RecvMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n", 
// 		  time, 
// 		  sourceNodeToken, sourceThreadToken,
// 		  destinationNodeToken, destinationThreadToken,
// 		  messageSize, messageTag);
  return 0;
}


int FindDrift_DefUserEvent( void *userData, unsigned int userEventToken,
		  const char *userEventName, int monotonicallyIncreasing ) {
  if (0 == strcmp(userEventName,"\"TauTraceClockOffsetStart\"")) {
    beginOffsetEvent = userEventToken;
  } else if (0 == strcmp(userEventName,"\"TauTraceClockOffsetEnd\"")) {
    endOffsetEvent = userEventToken;
  }
  return 0;
}

int FindDrift_EventTrigger( void *userData, double time, 
		unsigned int nodeToken,
		unsigned int threadToken,
	       	unsigned int userEventToken,
		long long userEventValue) {
  if (userEventToken == beginOffsetEvent) {
    beginOffset = userEventValue;
    beginOffsetTimestamp = time;
  } else if (userEventToken == endOffsetEvent) {
    endOffset = userEventValue;
    endOffsetTimestamp = time;
  } 
  return 0;
}



void findOffset(char *trace, char *edf) {
  int recs_read, pos;
  Ttf_FileHandleT fh = Ttf_OpenFileForInput(trace, edf);

  if (!fh) {
    printf("ERROR:Ttf_OpenFileForInput fails");
    exit(1);
  }

  Ttf_CallbacksT cb;
  /* Fill the callback struct */
  cb.UserData = 0;
  cb.DefClkPeriod = 0;
  cb.DefThread = 0;
  cb.DefStateGroup = 0;
  cb.DefState = 0;
  cb.DefUserEvent = FindDrift_DefUserEvent;
  cb.EventTrigger = FindDrift_EventTrigger;
  cb.EndTrace = EndTrace;
  cb.EnterState = 0;
  cb.LeaveState = 0;
  cb.SendMessage = 0;
  cb.RecvMessage = 0;

  /* Go through each record until the end of the trace file */
  do {
    recs_read = Ttf_ReadNumEvents(fh,cb, 1024);
  }
  while ((recs_read >=0) && (!EndOfTrace));
  Ttf_CloseFile(fh);
}

int main(int argc, char **argv) {
  Ttf_FileHandleT fh;
  int recs_read, pos;
  char *trace_file;
  char *edf_file;
  int no_state_flag=0, no_message_flag=0;
  /* main program: Usage app <trc> <edf> [-nostate] [-nomessage] */
  if (argc < 3)
  {
    printf("Usage: %s <TAU trace> <edf file> <drift> <drifttimestamp> [-nostate] [-nomessage]\n", 
		    argv[0]);
    exit(1);
  }
  
  double drift;
  double drifttimestamp;

  for (int i = 0; i < argc ; i++) {
    switch(i) {
      case 0:
	trace_file = argv[1];
	break;
      case 1:
	edf_file = argv[2];
	break;
    }
  }

  
  findOffset(argv[1], argv[2]);




  if (beginOffset == DBL_MIN || endOffset == DBL_MIN) {
    printf ("Couldn't find start and end offsets!\n");
    exit(1);
  }
  
  drift = endOffset - beginOffset;
  double duration =  endOffsetTimestamp - beginOffsetTimestamp;
  
  
  if (drift == 0.0) {
    driftrate = 1;
  } else {
    driftrate = 1 + (drift / duration);
  }
  
  printf ("Offset was %G at %G, and %G at %G\n", beginOffset, beginOffsetTimestamp, endOffset, endOffsetTimestamp);
  
  printf ("Using Drift offset of %G, rate = %.16G\n", drift, driftrate);
  
  fh = Ttf_OpenFileForInput( argv[1], argv[2]);
  
  outFile = Ttf_OpenFileForOutput("adjusted.trc","adjusted.edf");

  if (!fh) {
    printf("ERROR:Ttf_OpenFileForInput fails");
    exit(1);
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
  cb.EnterState = EnterState;
  cb.LeaveState = LeaveState;
  cb.SendMessage = SendMessage;
  cb.RecvMessage = RecvMessage;

  // reset this flag, otherwise it will only read 1024 records
  EndOfTrace = 0;

  /* Go through each record until the end of the trace file */
  do {
    recs_read = Ttf_ReadNumEvents(fh,cb, 1024);
  }
  while ((recs_read >=0) && (!EndOfTrace));
  Ttf_CloseFile(fh);

  
  Ttf_CloseOutputFile(outFile);

  return 0;
}


