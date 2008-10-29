/*************************************************************************
**			TAU Portable Profiling Package			**
**			http://www.cs.uoregon.edu/research/paracomp/tau	**
**************************************************************************
**	Copyright 2005							**
**	Department of Computer and Information Science, University of Oregon**
**	Advanced Computing Laboratory, Los Alamos National Laboratory	**
**	Research Center Juelich, Germany				**
**************************************************************************
**	File 		: tau2elg.cpp					**
**	Description : TAU to Epilog translator				**
**	Author		: Sameer Shende + Wyatt Spear	   		**
**	Contact		: sameer@cs.uoregon.edu + wspear@cs.uoregon.edu **
*************************************************************************/
#include <TAU_tf.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stddef.h>
#include <elg_rw.h>
#include <map>
#include <vector>
#include <stack>
using namespace std;

int debugPrint = 0;
bool multiThreaded = false;
#define dprintf if (debugPrint) printf

/* The choice of the following numbers is arbitrary */
#define TAU_SAMPLE_CLASS_TOKEN   71
#define TAU_DEFAULT_COMMUNICATOR 42
/* any unique id */

/* implementation of callback routines */
map< pair<int,int>, int, less< pair<int,int> > > EOF_Trace;
map< int,int, less<int > > numthreads; 
map<int,int> matchreg;
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
int sampgroupid = 0;
int sampclassid = 0; 
vector<stack <unsigned int> > callstack;
int *offset = 0; 
double clockp = 0;
unsigned int maxthreads = 0;
unsigned int maxnodes = 0;
int strings=0;
int ecount = 0;

/* FIX GlobalID so it takes into account numthreads */
/* utilities */
int GlobalId(int localnodeid, int localthreadid)
{
	if (multiThreaded)
	{
		if (offset == (int *) NULL)
		{
			printf("Error: offset vector is NULL in GlobalId()\n");
			return localnodeid;
		}
		/* for multithreaded programs, modify this routine */
		return offset[localnodeid]+localthreadid;  /* for single node program */
	}
	else
	{ 
		return localnodeid;
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
	dprintf("Entered state %d time %g nid %d tid %d\n", stateid, time, nodeid, tid);
	unsigned int cpuid = GlobalId(nodeid, tid);
	if (cpuid >= callstack.size()) 
	{
		fprintf(stderr, "ERROR: tau2elg: EnterState() cpuid %d exceeds callstack size %d\n", cpuid, callstack.size());
		exit(1);
	}
	callstack[cpuid].push(stateid);
	ElgOut_write_ENTER((ElgOut*)userData, cpuid, time * 1e-6, matchreg[stateid], 0, NULL);
	return 0;
}

int CountEnter(void *userData, double time, 
		unsigned int nodeid, unsigned int tid, unsigned int stateid){ecount++; return 0;}
int CountLeave(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int stateid) {
  ecount++; return 0;
}

/***************************************************************************
 * Description: LeaveState is called at routine exit by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int stateid)
{
	dprintf("Leaving state time %g nid %d tid %d\n", time, nid, tid);
	int cpuid = GlobalId(nid, tid);
	/*int stateid = callstack[cpuid].top();*/
	callstack[cpuid].pop();
	ElgOut_write_EXIT((ElgOut*)userData, cpuid, time * 1e-6, 0, NULL);	
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
	if(clkPeriod==0)
		return 0;
	//clockp = 1000000/clkPeriod;//1/(clkPeriod/1000000);
	clockp = clkPeriod;//1/(clkPeriod/1000000);
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
	dprintf("DefThread nid %d tid %d, thread name %s\n", nodeToken, threadToken, threadName);
	EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 0; /* initialize it */
	numthreads[nodeToken] = numthreads[nodeToken] + 1;
	if (threadToken > 0) multiThreaded = true;
	if (maxnodes < nodeToken) maxnodes = nodeToken;
	if (maxthreads < threadToken) maxthreads = threadToken;
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
			/* If there's any processor that is not over, 
			 * then the trace is not over */
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
	dprintf("StateGroup groupid %d, group name %s\n", stateGroupToken, stateGroupName);
	/* create a default activity (group) */
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
	dprintf("DefState stateid %d stateName %s stategroup id %d\n", stateToken, stateName, stateGroupToken);
	/* We need to remove the backslash and quotes from "\"funcname\"" */
	char *name = strdup(stateName);
	int len = strlen(name);
	int regs = matchreg.size();
	if ((name[0] == '"' ) && (name[len-1] == '"'))
	{
		name += 1;
		name[len-2] = '\0';
	}
	/* create a state record */
	ElgOut_write_STRING((ElgOut*)userData,regs,0,name);
	ElgOut_write_REGION((ElgOut*)userData, 
		regs,
		strings,
		ELG_NO_ID,
		ELG_NO_LNO,
		ELG_NO_LNO, strings, ELG_FUNCTION);//last ELG_NO_ID for descript
	strings++;
	matchreg[stateToken]=regs;
	return 0;
}

/***************************************************************************
 * Description: DefUserEvent is called to register the name and a token of the
 *  	user defined event (or a sample event in Vampir terminology).
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefUserEvent( void *userData, unsigned int userEventToken,
		const char *userEventName , int monotonicallyIncreasing)
{
	dprintf("DefUserEvent event id %d user event name %s\n", userEventToken, userEventName);
	/* VTF3_WriteDefsampclass(userData, userEventToken, userEventName);*/
	//int iscpugrpsamp = 1;
	//int dodifferentiation;
	/* We need to remove the backslash and quotes from "\"funcname\"" */
	char *name = strdup(userEventName);
	int len = strlen(name);
	if ((name[0] == '"' ) && (name[len-1] == '"'))
	{
		name += 1;
		name[len-2] = '\0';
	}
	ElgOut_write_STRING((ElgOut*)userData,userEventToken,0,userEventName);
	/* create a state record */
	if (monotonicallyIncreasing)
	{
		//dodifferentiation = 1; /* for hw counter data */
		//VTF3_WriteDefsamp(userData, userEventToken, TAU_SAMPLE_CLASS_TOKEN, 
		//iscpugrpsamp, sampgroupid, VTF3_VALUETYPE_UINT, 
		//(void *) &taulongbounds, dodifferentiation, VTF3_DATAREPHINT_BEFORE, 
		//(const char *) name, "#/s");
 
		ElgOut_write_METRIC((ElgOut*)userData, userEventToken,
			userEventToken,
			ELG_NO_ID,
			ELG_FLOAT, 
			ELG_COUNTER,
			ELG_START);
	}
	else
	{	/* for non monotonically increasing data */
		ElgOut_write_METRIC((ElgOut*)userData, userEventToken,
			userEventToken,
			ELG_NO_ID,
			ELG_FLOAT, 
			ELG_SAMPLE,
			ELG_NO_ID);
		//dodifferentiation = 0; /* for TAU user defined events */
		//VTF3_WriteDefsamp(userData, userEventToken, TAU_SAMPLE_CLASS_TOKEN, 
		//iscpugrpsamp, sampgroupid, VTF3_VALUETYPE_UINT, 
		//(void *) &taulongbounds, dodifferentiation, VTF3_DATAREPHINT_BEFORE, 
		//(const char *) name, "#");
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
	dprintf("EventTrigger: time %g, nid %d tid %d event id %d triggered value %lld \n", time, nodeToken, threadToken, userEventToken, userEventValue);
	//int type = VTF3_VALUETYPE_UINT;
	int cpuid = GlobalId (nodeToken, threadToken); /* GID */
	int samplearraydim = 1; 
	/* write the sample data */
	//VTF3_WriteSamp(userData, time, cpuid, samplearraydim, 
	//(const int *) &userEventToken, &type, &userEventValue);
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
		unsigned int messageComm )
{
	dprintf("SendMessage:time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n", time, sourceNodeToken, sourceThreadToken, destinationNodeToken, destinationThreadToken, messageSize, messageTag);
	int source = GlobalId(sourceNodeToken, sourceThreadToken);
	int dest   = GlobalId(destinationNodeToken, destinationThreadToken);
	ElgOut_write_MPI_SEND((ElgOut*)userData, source, time * 1e-6, dest, 0, messageTag,messageSize);//elg_ui4 cid = 0
	//VTF3_WriteSendmsg(userData, time, source, dest, TAU_DEFAULT_COMMUNICATOR, 
	//messageTag, messageSize, VTF3_SCLNONE);
	return 0;
}
int CountSend(void *userData, double time, 
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
	        unsigned int messageTag,
		unsigned int messageComm)
{
  ecount++; return 0;
}
int CountRecv(void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
	        unsigned int messageTag,
		unsigned int messageComm)
{
  ecount++; return 0;
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
		unsigned int messageComm )
{
	dprintf("RecvMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n", time, sourceNodeToken, sourceThreadToken, destinationNodeToken, destinationThreadToken, messageSize, messageTag);
	int source = GlobalId(sourceNodeToken, sourceThreadToken);
	int dest = GlobalId(destinationNodeToken, destinationThreadToken);
	ElgOut_write_MPI_RECV((ElgOut*)userData, dest, time * 1e-6, source, 0, messageTag);//elg_ui4 cid, = 0
	//VTF3_WriteRecvmsg(userData, time, dest, source, TAU_DEFAULT_COMMUNICATOR, 
	//messageTag, messageSize, VTF3_SCLNONE);
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
	{ /*Explicilty mark end of trace to be not over */ 
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
	int recs_read;//int pos;
	char *trace_file;
	char *edf_file;
	char *out_file; 
	//int output_format = VTF3_FILEFORMAT_STD_BINARY; /* Binary by default */  //int no_state_flag=0;
	int no_message_flag=0;
	int i; 
	/* main program: Usage app <trc> <edf> [-a] [-nomessage] */
	if (argc < 3)
	{
		printf(
	"Usage: %s <TAU trace> <edf file> <out file> [-nomessage]  [-v]\n", 
		    argv[0]);
	//printf(" -a         : ASCII VTF3 file format\n");
	//printf(" -fa        : FAST ASCII VTF3 file format\n");
	printf(
	" -nomessage : Suppress printing of message information in the trace\n");
	printf(" -v         : Verbose\n");
	printf(" Default trace format of <out file> is Epilog binary\n");

	printf(" e.g.,\n");
	printf(" %s merged.trc tau.edf app.elg\n", argv[0]);
	exit(1);
	}
	/***************************************************************************
	* -a stands for ASCII, -fa stands for FAST ASCII and -v is for verbose. 
	***************************************************************************/
	for (i = 0; i < argc ; i++)
	{
		switch(i){
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
				if (strcmp(argv[i], "-a")==0)
				{ /* Use ASCII format */}
				if (strcmp(argv[i], "-fa")==0)
				{/* Use FAST ASCII format */}
				if (strcmp(argv[i], "-nomessage")==0)
				{
					no_message_flag = 1;
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
	/* Open elg Trace file for output */
	int endian = 0;
	unsigned char SwapTest[2] = { 1, 0 };
	if( *(short *) SwapTest == 1 )
	{    /* little endian */
		endian = ELG_LITTLE_ENDIAN;
	}
	else
	{    /* big endian */
		endian = ELG_BIG_ENDIAN;
	}
	/* Define the file control block for output trace file */
#ifdef ELG_UNCOMPRESSED
	ElgOut* elgo = ElgOut_open(out_file, endian, ELG_UNCOMPRESSED);
#else  /* ELG_UNCOMPRESSED */
	ElgOut* elgo = ElgOut_open(out_file, endian);
#endif /* ELG_UNCOMPRESSED */
	/* check and verify that it was opened properly */
	if (elgo == NULL)
	{
		perror(out_file);
		exit(1);
	}
	/* in the first (true) pass, we determine the no. of cpus and other group related
	* information. In the second pass, we look at the function entry/exits */ 
	Ttf_CallbacksT firstpass;
	/* In the first pass, we just look for node/thread ids and def records */
	firstpass.UserData = elgo;
	firstpass.DefThread = DefThread;
	firstpass.EndTrace = EndTrace;
	firstpass.DefClkPeriod = ClockPeriod;
	firstpass.DefStateGroup = DefStateGroup;
	firstpass.DefState = DefState;
	firstpass.SendMessage = CountSend; /* Important to declare these as null! */
	firstpass.RecvMessage = CountRecv; /* Important to declare these as null! */
	firstpass.DefUserEvent = 0;
	firstpass.EventTrigger = 0; /* these events are ignored in the first pass */
	firstpass.EnterState = CountEnter;   /* these events are ignored in the first pass */
	firstpass.LeaveState = CountLeave;   /* these events are ignored in the first pass */
	/* Go through all trace records */
	//int ecount = 0;
	do{
		recs_read = Ttf_ReadNumEvents(fh,firstpass, 1024);
		//ecount+=recs_read;
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
	int totalnidtids = EOF_Trace.size(); 
	/* This is ok for single threaded programs. For multi-threaded programs
	* we'll need to modify the way we describe the cpus/threads */
	//VTF3_WriteDefsyscpunums(fcb, 1, &totalnidtids);
	char *machinen="Generic";
	ElgOut_write_STRING(elgo,strings,0,machinen);
	ElgOut_write_MACHINE(elgo, 0, numthreads.size(), strings);//ELG_NO_ID
	strings++;
	#ifdef ELG_MPI_WIN
		ElgOut_write_MPI_COMM(elgo, 0,0, 0, NULL);
	#else
		ElgOut_write_MPI_COMM(elgo, 0, 0, NULL);
	#endif
	/* Then write out the thread names if it is multi-threaded */
	if (multiThreaded)
	{ /* create the thread ids */
		unsigned int groupid = 0x1 << 31; /* Valid vampir group id nos */
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
			char noden[512];
			sprintf(noden,"NODE %d\0",i);
			ElgOut_write_STRING(elgo,strings,0,noden);
			ElgOut_write_NODE((ElgOut*)elgo, i, 0, threadnumarray[i], strings, clockp);
			strings++;
			
			char name[32];
			for (tid = 0; tid < threadnumarray[i]; tid++)
			{
				sprintf(name, "node %d, thread %d", i, tid);
				int cpuid = GlobalId(i,tid);
				cpuidarray[tid] = cpuid;
				ElgOut_write_THREAD((ElgOut*)elgo, cpuid, 0, ELG_NO_ID);
				ElgOut_write_LOCATION((ElgOut*)elgo, cpuid, 0, i, 0, tid);
				//VTF3_WriteDefcpuname(fcb, cpuid, name);
			}
			sprintf(name, "Node %d", i);
			groupid ++; /* let flat group for samples take the first one */
			/* Define a group: threadnumarray[i] represents no. of threads in node */
			//VTF3_WriteDefcpugrp(fcb, groupid, threadnumarray[i], 
			//(const unsigned int *) cpuidarray, name);
		}
		delete[] cpuidarray;
	}
	else
	{
		for(unsigned int i =0; i<numthreads.size();i++)
		{
			char noden[512];
			sprintf(noden,"NODE %d\0",i);
			ElgOut_write_STRING(elgo,strings,0,noden);
			ElgOut_write_NODE((ElgOut*)elgo, i, 0, 1, strings, clockp);
			strings++;
			//ElgOut_write_THREAD((ElgOut*)elgo, i, i, ELG_NO_ID);
			ElgOut_write_LOCATION((ElgOut*)elgo, i, 0, i, i, 0);
		}	
	}
	ElgOut_write_NUM_EVENTS((ElgOut*)elgo,ecount);//2428324
	ElgOut_write_LAST_DEF(elgo);
	unsigned int *idarray = new unsigned int[totalnidtids];
	for (i = 0; i < totalnidtids; i++)
	{ /* assign i to each entry */
		idarray[i] = i;
	}
	/* create a callstack on each thread/process id */
	dprintf("totalnidtids  = %d\n", totalnidtids);
	//callstack = new stack<unsigned int> [totalnidtids](); 
	callstack.resize(totalnidtids);
	/* Define group ids */
	char name[1024];
	strcpy(name, "TAU sample group name");
	//VTF3_WriteDefcpugrp(fcb, sampgroupid, totalnidtids, idarray, name);
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
	cb.UserData = elgo;
	cb.DefClkPeriod = 0;
	cb.DefThread = 0;
	cb.DefStateGroup = 0;
	cb.DefState = 0;
	cb.DefUserEvent = 0;
	cb.EventTrigger = 0;
	cb.EndTrace = EndTrace;
	/* should state transitions be displayed? */
	/* Of course! */
	cb.EnterState = EnterState;
	cb.LeaveState = LeaveState;
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
	//int writtenchars;
	//size_t writtenbytes;
	/* Go through each record until the end of the trace file */
	do{
		recs_read = Ttf_ReadNumEvents(fh,cb, 1024);
		#ifdef DEBUG  
		if (recs_read != 0)
			cout <<"Read "<<recs_read<<" records"<<endl;
		#endif /* DEBUG */
	}
	while ((recs_read >=0) && (!EndOfTrace));
	/* dummy records */
	Ttf_CloseFile(fh);
	/* close ELG file */
	ElgOut_close(elgo);
	return 0;
}
/* EOF tau2elg.cpp */
