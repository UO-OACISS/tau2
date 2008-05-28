/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
**			http://www.paratools.com                           **
*****************************************************************************
**    Copyright 2005  						   	   **
**    ParaTools, Inc.                                                      **
****************************************************************************/
/***************************************************************************
**	File 		: tau2otf.cpp 					  **
**	Description 	: TAU to OTF translator                           **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@paratools.com   	                  **
***************************************************************************/
#include <TAU_tf.h>
#include <stdio.h>
#include <iostream>
#include <stddef.h>
#include <otf.h> /* OTF header file */
#include <map>
#include <vector>
#include <stack>
using namespace std;
int debugPrint = 0;
bool multiThreaded = false;
#define dprintf if (debugPrint) printf

/* The choice of the following numbers is arbitrary */
#define TAU_SAMPLE_CLASS_TOKEN   71
#define TAU_DEFAULT_COMMUNICATOR 0 /* they don't belong to a process group */
#define TAU_SCL_NONE 0
#define TAU_OTF_FORMAT 1
#define TAU_MAJOR 2
#define TAU_MINOR 15
#define TAU_SUB 0
#define TAU_NO_PARENT 0
#define TAU_OTF_FILE_MANAGER_LIMIT 250
#define TAU_GLOBAL_STREAM_ID 0

/* Convert each time stamp to 1000 times its value and pass it as uint64_t */
#define TAU_MULT 1000

uint64_t TauGetClockTicksInGHz(double time)
{
  return (uint64_t) (time * TAU_MULT); 
}

/* any unique id */

/* implementation of callback routines */
map< pair<int,int>, int, less< pair<int,int> > > EOF_Trace;
map< int,int, less<int > > numthreads; 
/* numthreads[k] is no. of threads in rank k */

int EndOfTrace = 0;  /* false */

/* Define limits of sample data (user defined events) */
struct {
  unsigned long long umin;
  unsigned long long umax;
} taulongbounds = { 0, (unsigned long long)~(unsigned long long)0 };

struct {
  double fmin;
  double fmax;
} taufloatbounds = {-1.0e+300, +1.0e+300};

/* These structures are used in user defined event data */



/* Global data */
int sampgroupid = 1;
int sampclassid = 2; 
vector<stack <unsigned int> > callstack;
int *offset = 0; 


/* FIX GlobalID so it takes into account numthreads */
/* utilities */
int GlobalId(int localnodeid, int localthreadid)
{

  if (multiThreaded) /* do it for both single and multi-threaded */
  {
    if (offset == (int *) NULL)
    {
      printf("Error: offset vector is NULL in GlobalId()\n");
      return localnodeid+1;
    }
    
    /* for multithreaded programs, modify this routine */
    return offset[localnodeid]+localthreadid+1;  /* for single node program */
  }
  else
  {  /* OTF node nos run from 1 to N, TAU's run from 0 to N-1 */
    return localnodeid+1;
  }

}

/* implementation of callback routines */
/***************************************************************************
 * Description: EnterState is called at routine entry by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EnterState(void *userData, double time, 
		unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
  int cpuid = GlobalId(nodeid, tid);
  dprintf("Entered state %d time %g cpuid %d\n", 
		  stateid, time, cpuid);

  if (cpuid >= (int) callstack.size()+1) 
  {
    fprintf(stderr, "ERROR: tau2otf: EnterState() cpuid %d exceeds callstack size %d\n", cpuid, callstack.size());
    exit(1);
  }
	
  callstack[cpuid].push(stateid);

/* OLD : 
  OTF_Writer_writeDownto((OTF_Writer*)userData, TauGetClockTicksInGHz(time), stateid, cpuid, TAU_SCL_NONE);
*/
  OTF_Writer_writeEnter((OTF_Writer*)userData, TauGetClockTicksInGHz(time), stateid, cpuid, TAU_SCL_NONE);
  return 0;
}

/***************************************************************************
 * Description: EnterState is called at routine exit by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int statetoken)
{
  int cpuid = GlobalId(nid, tid);
  int stateid = callstack[cpuid].top();
  callstack[cpuid].pop();

  dprintf("Leaving state time %g cpuid %d \n", time, cpuid);
  
/* OLD: 
  OTF_Writer_writeUpfrom((OTF_Writer*)userData, TauGetClockTicksInGHz(time), stateid, cpuid, TAU_SCL_NONE);
*/
  /* we can write stateid = 0 if we don't need stack integrity checking */
  OTF_Writer_writeLeave((OTF_Writer*)userData, TauGetClockTicksInGHz(time), stateid, cpuid, TAU_SCL_NONE);
  return 0;
}

/***************************************************************************
 * Description: ClockPeriod (in microseconds) is specified here. 
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int ClockPeriod( void*  userData, double clkPeriod )
{
  
  dprintf("Clock period %g\n", clkPeriod);
  OTF_Writer_writeDefTimerResolution((OTF_Writer*)userData, TAU_GLOBAL_STREAM_ID, TauGetClockTicksInGHz(1/clkPeriod));

  return 0;
}

/***************************************************************************
 * Description: DefThread is called when a new nodeid/threadid is encountered.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
  dprintf("DefThread nid %d tid %d, thread name %s\n", 
		  nodeToken, threadToken, threadName);
  EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 0; /* initialize it */
  numthreads[nodeToken] = numthreads[nodeToken] + 1; 
  if (threadToken > 0) multiThreaded = true; 
  return 0;
}

/***************************************************************************
 * Description: EndTrace is called when an EOF is encountered in a tracefile.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
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

/***************************************************************************
 * Description: DefStateGroup registers a profile group name with its id.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefStateGroup( void *userData, unsigned int stateGroupToken, 
		const char *stateGroupName )
{
  
  dprintf("StateGroup groupid %d, group name %s\n", stateGroupToken, 
		  stateGroupName);

  /* create a default activity (group) */
  OTF_Writer_writeDefFunctionGroup((OTF_Writer*)userData, TAU_GLOBAL_STREAM_ID, stateGroupToken, stateGroupName);
  return 0;
}

/***************************************************************************
 * Description: DefState is called to define a new symbol (event). It uses
 *		the token used to define the group identifier. 
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefState( void *userData, unsigned int stateToken, const char *stateName, 
		unsigned int stateGroupToken )
{
  
  dprintf("DefState stateid %d stateName %s stategroup id %d\n",
		  stateToken, stateName, stateGroupToken);

  /* We need to remove the backslash and quotes from "\"funcname\"" */
  char *name = strdup(stateName);
  int len = strlen(name);
  if ((name[0] == '"' ) && (name[len-1] == '"'))
  {
     name += 1;
     name[len-2] = '\0';
  }

  /* create a state record */
  OTF_Writer_writeDefFunction((OTF_Writer*)userData, TAU_GLOBAL_STREAM_ID, stateToken, (const char *) name, stateGroupToken, TAU_SCL_NONE);

  return 0;
}

/***************************************************************************
 * Description: DefUserEvent is called to register the name and a token of the
 *  		user defined event (or a sample event in Vampir terminology).
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefUserEvent( void *userData, unsigned int userEventToken,
		const char *userEventName , int monotonicallyIncreasing)
{

  dprintf("DefUserEvent event id %d user event name %s\n", userEventToken,
		  userEventName);
  int dodifferentiation;

  /* We need to remove the backslash and quotes from "\"funcname\"" */
  char *name = strdup(userEventName);
  int len = strlen(name);
  if ((name[0] == '"' ) && (name[len-1] == '"'))
  {
     name += 1;
     name[len-2] = '\0';
  }

  /* create a state record */
  if (monotonicallyIncreasing)
  {
    dodifferentiation = 1; /* for hw counter data */
    OTF_Writer_writeDefCounter((OTF_Writer*)userData, TAU_GLOBAL_STREAM_ID, userEventToken,  (const char *) name, OTF_COUNTER_TYPE_ACC+OTF_COUNTER_SCOPE_START, sampclassid, "#/s");
  }
  else
  { /* for non monotonically increasing data */
    dodifferentiation = 0; /* for TAU user defined events */
    OTF_Writer_writeDefCounter((OTF_Writer*)userData, TAU_GLOBAL_STREAM_ID, userEventToken,  (const char *) name, OTF_COUNTER_TYPE_ABS+OTF_COUNTER_SCOPE_POINT, sampclassid, "#");
    /* NOTE: WE DO NOT HAVE THE DO DIFFERENTIATION PARAMETER YET IN OTF */
  } 

  return 0;
}

/***************************************************************************
 * Description: EventTrigger is called when a user defined event is triggered.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EventTrigger( void *userData, double time, 
		unsigned int nodeToken,
		unsigned int threadToken,
	       	unsigned int userEventToken,
		long long userEventValue)
{
  int cpuid = GlobalId (nodeToken, threadToken); /* GID */
  dprintf("EventTrigger: time %g, cpuid %d event id %d triggered value %lld \n", time, cpuid, userEventToken, userEventValue);


  /* write the sample data */
  OTF_Writer_writeCounter((OTF_Writer*)userData, TauGetClockTicksInGHz(time), cpuid, userEventToken, userEventValue); 
  return 0;
}

/***************************************************************************
 * Description: SendMessage is called when a message is sent by a process.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int SendMessage( void *userData, double time, 
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
		unsigned int messageTag, 
		unsigned int messageComm)
{

  int source = GlobalId(sourceNodeToken, sourceThreadToken);
  int dest   = GlobalId(destinationNodeToken, destinationThreadToken);

  dprintf("SendMessage: time %g, source cpuid %d , destination cpuid %d, size %d, tag %d\n", 
		  time, 
		  source, dest, 
		  messageSize, messageTag);

  OTF_Writer_writeSendMsg((OTF_Writer*)userData, TauGetClockTicksInGHz(time), source, dest, TAU_DEFAULT_COMMUNICATOR, messageTag, messageSize, TAU_SCL_NONE);

  return 0;
}

/***************************************************************************
 * Description: RecvMessage is called when a message is received by a process.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int RecvMessage( void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
		unsigned int messageTag,
		unsigned int messageComm)
{

  int source = GlobalId(sourceNodeToken, sourceThreadToken);
  int dest   = GlobalId(destinationNodeToken, destinationThreadToken);

  dprintf("RecvMessage: time %g, source cpuid %d, destination cpuid %d, size %d, tag %d\n", 
		  time, 
		  source, dest, 
		  messageSize, messageTag);

  OTF_Writer_writeRecvMsg((OTF_Writer*)userData, TauGetClockTicksInGHz(time), dest, source, TAU_DEFAULT_COMMUNICATOR, messageTag, messageSize, TAU_SCL_NONE);

  return 0;
}

/***************************************************************************
 * Description: To clean up and reset the end of file marker, we invoke this.
 ***************************************************************************/
int ResetEOFTrace(void)
{
  /* mark all entries of EOF_Trace to be false */
  for (map< pair<int,int>, int, less< pair<int,int> > >:: iterator it = 
		  EOF_Trace.begin(); it != EOF_Trace.end(); it++)
  { /* Explicilty mark end of trace to be not over */ 
    (*it).second = 0;
  }

  return 0;
}

/***************************************************************************
 * Description: The main entrypoint. 
 ***************************************************************************/
int main(int argc, char **argv)
{
  Ttf_FileHandleT fh;
  OTF_FileManager* manager;
  int num_streams = 1;
  int num_nodes = -1;
  int recs_read;
  char *trace_file;
  char *edf_file;
  char *out_file = NULL; 
  int no_message_flag=0;
  int compress_flag = 0; /* by default do not compress traces */
  OTF_FileCompression compression = OTF_FILECOMPRESSION_UNCOMPRESSED; 
  int i; 
  /* main program: Usage app <trc> <edf> [-a] [-nomessage] */
  if (argc < 4)
  {
    printf("Usage: %s <TAU trace> <edf file> <out file> [-n streams] [-nomessage]  [-z] [-v]\n", 
		    argv[0]);
    printf(" -n <streams> : Specifies the number of output streams (default 1)\n");
    printf(" -nomessage : Suppress printing of message information in the trace\n");
    printf(" -z : Enable compression of trace files. By default it is uncompressed.\n");
    printf(" -v         : Verbose\n");
    printf(" Trace format of <out file> is OTF \n");

    printf(" e.g.,\n");
    printf(" %s merged.trc tau.edf app.otf\n", argv[0]);
    exit(1);
  }
  
/***************************************************************************
 ***************************************************************************/
  for (i = 0; i < argc ; i++)
  {
    switch(i) {
      case 0:
	trace_file = argv[1];
	break;
      case 1:
	edf_file = argv[2];
	break;
      case 2: 
	out_file = argv[3]; 
	break; 
      default:
	if (strcmp(argv[i], "-n")==0)
        {
	  num_streams = atoi(argv[i+1]); 
	  i++; 
        }
	if (strcmp(argv[i], "-s")==0)
        {
	  num_nodes = atoi(argv[i+1]); 
	  i++; 
        }
	if (strcmp(argv[i], "-nomessage")==0)
	{
	  no_message_flag = 1;
	}
	if (strcmp(argv[i], "-z")==0)
	{
	  compress_flag = 1;
	}
	if (strcmp(argv[i], "-v") == 0)
        {
	  debugPrint = 1;
	}
	break;
    }
  }
  /* Finished parsing commandline options, now open the trace file */

  fh = Ttf_OpenFileForInput( argv[1], argv[2]);

  if (!fh)
  {
    printf("ERROR:Ttf_OpenFileForInput fails");
    exit(1);
  }


  dprintf("Using %d streams\n", num_streams);

  manager = OTF_FileManager_open(TAU_OTF_FILE_MANAGER_LIMIT);

  /* Define the file control block for output trace file */
  void *fcb = (void *) OTF_Writer_open(out_file, num_streams, manager);


  /* check and verify that it was opened properly */
  if (fcb == 0)
  {
    perror(out_file);
    exit(1);
  }

  /* enble compression if it is specified by the user */
  if (compress_flag)
  {
    compression = OTF_FILECOMPRESSION_COMPRESSED; 
    OTF_Writer_setCompression((OTF_Writer *)fcb, compression);
  }

  /* Write the trace file header */
 
  
  OTF_Writer_writeDefCreator((OTF_Writer *)fcb, TAU_GLOBAL_STREAM_ID, "tau2otf converter version 2.15.x");
  OTF_Writer_writeDefCounterGroup((OTF_Writer *)fcb, TAU_GLOBAL_STREAM_ID, sampclassid, "TAU counter data");


  int totalnidtids;

  if (num_nodes == -1) {
    /* in the first pass, we determine the no. of cpus and other group related
     * information. In the second pass, we look at the function entry/exits */ 
    
    Ttf_CallbacksT firstpass;
    /* In the first pass, we just look for node/thread ids and def records */
    firstpass.UserData = fcb;
    firstpass.DefThread = DefThread;
    firstpass.EndTrace = EndTrace;
    firstpass.DefClkPeriod = ClockPeriod;
    firstpass.DefThread = DefThread;
    firstpass.DefStateGroup = DefStateGroup;
    firstpass.DefState = DefState;
    firstpass.SendMessage = 0; /* Important to declare these as null! */
    firstpass.RecvMessage = 0; /* Important to declare these as null! */
    firstpass.DefUserEvent = 0;
    firstpass.EventTrigger = 0; /* these events are ignored in the first pass */
    firstpass.EnterState = 0;   /* these events are ignored in the first pass */
    firstpass.LeaveState = 0;   /* these events are ignored in the first pass */
    

    /* Go through all trace records */
    do {
      recs_read = Ttf_ReadNumEvents(fh,firstpass, 1024);
#ifdef DEBUG 
      if (recs_read != 0)
	cout <<"Read "<<recs_read<<" records"<<endl;
#endif 
    }
    while ((recs_read >=0) && (!EndOfTrace));
    

    /* reset the position of the trace to the first record */
    for (map< pair<int,int>, int, less< pair<int,int> > >:: iterator it = 
	   EOF_Trace.begin(); it != EOF_Trace.end(); it++)
      { /* Explicilty mark end of trace to be not over */ 
	(*it).second = 0;
      }
    totalnidtids = EOF_Trace.size(); 
  } else {
    totalnidtids = num_nodes;
    for (i=0; i<num_nodes; i++) {
      numthreads[i] = 1; 
    }
  }

  /* This is ok for single threaded programs. For multi-threaded programs
   * we'll need to modify the way we describe the cpus/threads */
/* THERE'S NO NEED TO WRITE THE TOTAL NO. OF CPUS in OTF */
/*
  OTF_WriteDefsyscpunums(fcb, 1, &totalnidtids);
*/

  /* create the thread ids */
  unsigned int groupid = 1;
  int tid = 0; 
  int nodes = numthreads.size(); /* total no. of nodes */ 
  int *threadnumarray = new int[nodes]; 
  offset = new int[nodes+1];
  offset[0] = 0; /* no offset for node 0 */
  for (i=0; i < nodes; i++)
  { /* one for each node */
    threadnumarray[i] = numthreads[i]; 
    offset[i+1] = offset[i] + numthreads[i]; 
  }
  unsigned int *cpuidarray = new unsigned int[totalnidtids]; /* max */
  /* next, we write the cpu name and a group name for node/threads */
  for (i=0; i < nodes; i++)
  { 
    char name[32];
    for (tid = 0; tid < threadnumarray[i]; tid++)
    {
      sprintf(name, "node %d, thread %d", i, tid);
      int cpuid = GlobalId(i,tid);
      cpuidarray[tid] = cpuid;
      dprintf("Calling OTF_Writer_writeDefProcess cpuid %d name %s\n", cpuid, name);
      OTF_Writer_writeDefProcess((OTF_Writer *)fcb, TAU_GLOBAL_STREAM_ID, cpuid, name, TAU_NO_PARENT);
    }
    if (multiThreaded) 
    { /* define a group for these cpus only if it is a multi-threaded trace */
      sprintf(name, "Node %d", i);
      groupid ++; /* let flat group for samples take the first one */
      /* Define a group: threadnumarray[i] represents no. of threads in node */
    
      OTF_Writer_writeDefProcessGroup((OTF_Writer *)fcb, TAU_GLOBAL_STREAM_ID, groupid, name, threadnumarray[i], 
	(uint32_t*) cpuidarray);
    }
  }
  delete[] cpuidarray;


  unsigned int *idarray = new unsigned int[totalnidtids];
  for (i = 0; i < totalnidtids; i++)
  { /* assign i to each entry */
    idarray[i] = i+1;
  }
  
  /* create a callstack on each thread/process id */
  dprintf("totalnidtids  = %d\n", totalnidtids);
  //callstack = new stack<unsigned int> [totalnidtids](); 
  callstack.resize(totalnidtids+1);

  /* Define group ids */
  char name[1024];
  strcpy(name, "TAU default group");
  OTF_Writer_writeDefProcessGroup((OTF_Writer *)fcb, TAU_GLOBAL_STREAM_ID, sampgroupid, name, totalnidtids, idarray);

  EndOfTrace = 0;
  /* now reset the position of the trace to the first record */ 
  Ttf_CloseFile(fh);
  /* Re-open it for input */
  fh = Ttf_OpenFileForInput( argv[1], argv[2]);

  if (!fh)
  {
    printf("ERROR:Ttf_OpenFileForInput fails the second time");
    exit(1);
  }

  dprintf("Re-analyzing the trace file \n");
 

  Ttf_CallbacksT cb;
  /* Fill the callback struct */
  cb.UserData = fcb;
  cb.DefClkPeriod = 0;
  cb.DefThread = 0;
  cb.DefStateGroup = 0;
  cb.DefState = 0;
  cb.DefUserEvent = DefUserEvent;
  cb.EventTrigger = EventTrigger;
  cb.EndTrace = EndTrace;

  /* should state transitions be displayed? */
  /* Of course! */
  cb.EnterState = EnterState;
  cb.LeaveState = LeaveState;
/*
  cb.EnterState = 0;
  cb.LeaveState = 0;
*/

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
#ifdef DEBUG  
    if (recs_read != 0)
      cout <<"Read "<<recs_read<<" records"<<endl;
#endif /* DEBUG */
  }
  while ((recs_read >=0) && (!EndOfTrace));

  /* dummy records */
  Ttf_CloseFile(fh);

  /* close VTF file */
  OTF_Writer_close((OTF_Writer *)fcb);
  return 0;
}

/* EOF tau2otf.cpp */


/***************************************************************************
 * $RCSfile: tau2otf.cpp,v $   $Author: amorris $
 * $Revision: 1.5 $   $Date: 2008/05/28 21:21:27 $
 * VERSION_ID: $Id: tau2otf.cpp,v 1.5 2008/05/28 21:21:27 amorris Exp $
 ***************************************************************************/


