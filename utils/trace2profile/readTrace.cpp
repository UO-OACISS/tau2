#include <trace2profile.h>
#include <handlers.h>
#include <TAU_tf.h>
using namespace std;

Ttf_FileHandleT fh;

//map<int,bool> EOF_Trace;//Id to (false: still in/true:exited)
map<pair<int,int>,int> ThreadID;//NID/TID

/* implementation of callback routines */
/***************************************************************************
 * Description: DefThread is called when a new nodeid/threadid is encountered.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
	unsigned int processToken;
	ThreadID[pair<int,int>(nodeToken,threadToken)]= processToken = ThreadID.size();
	ThreadDef(nodeToken,threadToken,processToken,threadName);
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
	char *name = strdup(stateName);
	int len = strlen(name);
	if ((name[0] == '"' ) && (name[len-1] == '"'))
	{
		name += 1;
		name[len-2] = '\0';
	}
	StateDef(stateToken,name,stateGroupToken);
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
	StateGroupDef(stateGroupToken, stateGroupName );
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
	char *name = strdup(userEventName);
	int len = strlen(name);
	if ((name[0] == '"' ) && (name[len-1] == '"'))
	{
		name += 1;
		name[len-2] = '\0';
	}
	UserEventDef(userEventToken,name , monotonicallyIncreasing);
	return 0;
}

/***************************************************************************
 * Description: ClockPeriod (in microseconds) is specified here. 
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int ClockPeriod( void*  userData, double clkPeriod )
{
	double x=1/clkPeriod;
	#ifdef DEBUG 
	cout << "Clock: " << x << endl;
	#endif
	ClockPeriodDef(clkPeriod );
	return 0;
}

/***************************************************************************
 * Description: EndTrace is called when an EOF is encountered in a tracefile.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EndTrace( void *userData, unsigned int nodeToken, unsigned int threadToken)
{
	EndTraceDef(ThreadID[pair<int,int>(nodeToken,threadToken)]);//nodeToken, threadToken, 
	return 0;
}

/***************************************************************************
 * Description: EventTrigger is called when a user defined event is triggered.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EventTrigger( void *userData, double time, 
		unsigned int nid,
		unsigned int tid,
		unsigned int userEventToken,
		long long userEventValue)
{ 
	EventTriggerDef(time, ThreadID[pair<int,int>(nid,tid)], userEventToken,userEventValue);//nid,tid,
	return 0;
}

/***************************************************************************
 * Description: EnterState is called at routine entry by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EnterState(void *userData, double time, 
		unsigned int nid, unsigned int tid, unsigned int stateid)
{
	EnterStateDef(time,ThreadID[pair<int,int>(nid,tid)],stateid);//nid,tid,
	
	return 0;
}



/***************************************************************************
 * Description: LeaveState is called at routine exit by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int stateid)
{
	LeaveStateDef(time,ThreadID[pair<int,int>(nid,tid)], stateid);// nid, tid,

	return 0;
}



/***************************************************************************
 * This routine runs through the relevant callback methods
 ***************************************************************************/
void ReadTraceFile()
{
	/* in the first pass, we determine the no. of cpus and other group related
	* information. In the second pass, we look at the function entry/exits */ 
	int recs_read=0;
	fh = Ttf_OpenFileForInput(Converter::trc, Converter::edf);
	if (!fh)
	{
		fprintf(stderr,"ERROR:Ttf_OpenFileForInput failed\n");
		exit(1);
	}
	
	
	//char prefix [32];
	//InitSnapshot();
	
	Ttf_CallbacksT firstpass;
	/* In the first pass, we just look for node/thread ids and def records */
	firstpass.UserData = 0;
	firstpass.DefThread = DefThread;
	firstpass.EndTrace = EndTrace;
	firstpass.DefClkPeriod = ClockPeriod;
	firstpass.DefStateGroup = DefStateGroup;
	firstpass.DefState = DefState;
	firstpass.SendMessage = 0; /* Important to declare these as null! */
	firstpass.RecvMessage = 0; /* Important to declare these as null! */
	firstpass.DefUserEvent = DefUserEvent;
	firstpass.EventTrigger = 0; /* these events are ignored in the first pass */
	firstpass.EnterState = 0;   /* these events are ignored in the first pass */
	firstpass.LeaveState = 0;   /* these events are ignored in the first pass */

	/* Go through all trace records */
	do
	{
		recs_read = Ttf_ReadNumEvents(fh,firstpass, 1024);
		#ifdef DEBUG 
		if (recs_read != 0)
		cout <<"Read "<<recs_read<<" records"<<endl;
		#endif 
	}
	while ((recs_read >=0) && (!Converter::EndOfTrace));
	
	/* now reset the position of the trace to the first record */
	Ttf_CloseFile(fh);
	/* Re-open it for input */
	fh = Ttf_OpenFileForInput(Converter::trc, Converter::edf);
	
	ProcessDefs();
	
	if (!fh)
	{
		printf("ERROR:Ttf_OpenFileForInput fails the second time");
		//snapshot.close();
		exit(1);
	}
	

	Ttf_CallbacksT cb;
	/* Fill the callback struct */
	cb.UserData = 0;
	cb.DefClkPeriod = 0;
	cb.DefThread = 0;
	cb.DefStateGroup = 0;
	cb.DefState = 0;
	cb.DefUserEvent = 0;
	cb.EventTrigger = EventTrigger;
	cb.EndTrace = EndTrace;
	cb.EnterState = EnterState;
	cb.LeaveState = LeaveState;
	cb.SendMessage = 0;
	cb.RecvMessage = 0;
	/* Go through each record until the end of the trace file */
	do
	{
		recs_read = Ttf_ReadNumEvents(fh,cb, 1024);
		#ifdef DEBUG  
		if (recs_read != 0)
		cout <<"Read "<<recs_read<<" records"<<endl;
		#endif /* DEBUG */
	}
	while ((recs_read >=0) && (!Converter::EndOfTrace));

	/* dummy records */
	Ttf_CloseFile(fh);
	//thisShot=lastTime;
	
}
