/*
 * tau2profile.cpp
 * Author: Wyatt Spear
 * This program reads a specified TAU trace and converts it into an equivalent set of TAU
 * profile files.  An interval may be specified in the command line to have the program 
 * produce trace file 'snapshots' of the trace-state at each interval.
 * 
 * TODO:  Improve command line/help interface.  Add more conditions for snapshot output.
 * e.g.: User Event count, state entry/exit count.
 */

#include <stdlib.h>
#include <iostream>
#include <TAU_tf.h>
#include <fstream>
#include <string>
#include <stack>
#include <map>

using namespace std;

/*
 * This holds a single monotonically increasing event
 */
class MonIncEvent {
	/*Observed values must be held
	 * until all events are encountered
	 * so they can be processed at the same time*/
	public:
	long long holdvalue;
	long long inclusive;
	long long exclusive;
	long long topCount;
	long long fullCount;
	unsigned int eventToken;
	const char * eventName;
};

/*
 * This holds a single state description
 * including any monotonically increasing
 * events defined in the trace
 */
class State {
	public:
	double inclusive;
	double exclusive;
	int calls;
	int subroutines;
	double topTime;/*Time this State was put on/returned to the top of the stack*/
	double fullTime;/*Time this state was put on the stack.*/
	int countRec;/*Counts the depth of recursion*/
	unsigned int stateToken;
	unsigned int stateGroupToken;
	const char * stateName;
	/*These are for use with monotonically increasing events*/
	map<int,MonIncEvent> allmi;
	unsigned int countMIE;/*Counts monotonically increasing events*/
	int lastaction;/*1=enter,-1=leave,0=undefined*/
	double holdvalue;/**/
};

/*
 * This holds a single non-monotonically increasing event
 */
class UserEvent{
	public:
	const char * userEventName;
	int userEventToken;
	int numevents;
	int tricount;
	long long max;
	long long min;
	long long sum;
	long long sumsqr;
};

/*
 * This represents a single thread
 * It holds a representation of each (possible) state
 * and each non-monotonically increasing event
 */
class Thread {
	public:
	map<int,State>allstate;
	map<int,UserEvent>allevents;
	int nodeToken;
	int threadToken;
	int lastState;/*The last state entered in this thread*/
	stack<unsigned int> callstack;/*State IDs*/
	const char * threadName;
	bool finished;
};

/*Each thread in the trace is held here, mapped to the node and thread ids
 * The number indicating if the thread has exited is also mapped here*/
//map<pair<int,int>,pair<int,Thread> > EOF_Trace;//, less<pair<int,int> > 
map<int,Thread> ThreadMap;//global id to Thread object
//map<int,bool> EOF_Trace;//Id to (false: still in/true:exited)
map<pair<int,int>,int> ThreadID;//NID/TID
/*Each state in the trace is held here, mapped to its thread id.  
 * This is copied into each thread once it is initialized*/
map<int,State> allstate;
/*Each monotonically increasing event is held here, mapped to its
 * event id.  This is copied to each state once initialized.*/
map<int,MonIncEvent> allmoninc;
/*Each user event is held here, mapped to its event id.
 * This is copied to each thread after it is initialized.*/
map<int,UserEvent>allevents;
/*This maps group ids to group names*/
map<unsigned int,const char*> groupids;

void ReadFile();
void Usage();
void PrintProfiles(map< int,Thread> &mainmap);//, less< pair<int,int> > 
void PrintSnapshot(double time);
void SnapshotControl(double time, int token);
//Thread LeaveAll(Thread thread, double time);
void StateLeave(double time, Thread &thread, unsigned int stateid);
void StateEnter(double time, Thread &thread, unsigned int stateid);

Ttf_FileHandleT fh;
int EndOfTrace=0;  /* false */
int snapshot=-1;
int printshot=0;
unsigned int miecount=0;
int debugPrint=0;
double segmentInterval=-1;
double nextShot=-1;
char * trc = NULL;
char * edf = NULL;
char * out = NULL;

int trigEvent=-1;
int trigCount=1;
int trigSeen=0;

/* implementation of callback routines */
/***************************************************************************
 * Description: DefThread is called when a new nodeid/threadid is encountered.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
	Thread local;
	local.nodeToken=nodeToken;
	local.threadToken=threadToken;
	local.threadName=threadName;
	local.lastState=-1;
	local.finished=false;
	int curid;
	//EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = pair<int,Thread>(0,local);
	ThreadID[pair<int,int>(nodeToken,threadToken)]= curid =ThreadID.size();
	//EOF_Trace[curid]=false;
	ThreadMap[curid]=local;
	
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
	State local;
	local.calls=0;
	local.subroutines=0;
	local.inclusive=0;
	local.exclusive=0;
	local.topTime=-1;
	local.fullTime=-1;
	local.countRec=0;
	
	/*These are all for monotonically increasing events*/
	local.countMIE=0;
	local.lastaction=0;
	local.holdvalue=0;
	
	local.stateToken=stateToken;
	local.stateGroupToken=stateGroupToken;
	local.stateName=stateName;
	allstate[stateToken]=local;
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
	groupids[stateGroupToken]=stateGroupName;
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
	/* create a state record */
	if (monotonicallyIncreasing)
	{
		/* for hw counter data */
		char *name = strdup(userEventName);
		int len = strlen(name);
		if ((name[0] == '"' ) && (name[len-1] == '"'))
		{
			name += 1;
			name[len-2] = '\0';
		}
		MonIncEvent local;
		local.eventName=name;
		local.eventToken=userEventToken;
		local.exclusive=0;
		local.inclusive=0;
		local.topCount=-1;
		local.fullCount=-1;
		local.holdvalue=-1;
		allmoninc[userEventToken]=local;
		miecount++;
	}
	else
	{ /* for non monotonically increasing data */
		UserEvent local;

		local.userEventName=userEventName;
		local.max=0;
		local.min=0;
		local.numevents=0;
		local.sum=0;
		local.sumsqr=0;
		local.tricount=0;
		local.userEventToken=userEventToken;
		allevents[userEventToken]=local;
	} 
	return 0;
}

/***************************************************************************
 * Description: ClockPeriod (in microseconds) is specified here. 
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int ClockPeriod( void*  userData, double clkPeriod )
{
	return 0;
}

/***************************************************************************
 * Description: EndTrace is called when an EOF is encountered in a tracefile.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EndTrace( void *userData, unsigned int nodeToken, unsigned int threadToken)
{
	//(EOF_Trace[pair<int,int> (nodeToken,threadToken) ]).first = 1; /* flag it as over */
	(ThreadMap[ThreadID[pair<int,int>(nodeToken,threadToken)]]).finished=true;
	/* yes, it is over */
	map < int,Thread >::iterator it;//pair<int, int>, pair<int,Thread>, less< pair<int,int> > 
	EndOfTrace = 1; /* Lets assume that it is over */
	for (it = ThreadMap.begin(); it != ThreadMap.end(); it++)
	{/* cycle through all <nid,tid> pairs to see if it really over */
		if ((*it).second.finished == false)//.second.
		{
			EndOfTrace = 0; /* not over! */
			/* If there's any processor that is not over, then the trace is not over */
		}
	}
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
	SnapshotControl(time,-2);
	int curid=ThreadID[pair<int,int>(nid,tid)];
	/* write the sample data */
	if(allevents.count(userEventToken)>0)
	{	/*not monotonically increasing*/
		/*Each NMI event is between two '0' events which are ignored. (every 2nd of 3 events is used)*/

		//ThreadMap[curid] <- EOF_Trace[pair<int,int>(nid,tid)].second
		if(ThreadMap[curid].allevents[userEventToken].tricount==0)
		{
			ThreadMap[curid].allevents[userEventToken].tricount++;
			return(0);
		}
		if(ThreadMap[curid].allevents[userEventToken].tricount==2)
		{
			ThreadMap[curid].allevents[userEventToken].tricount=0;
			return(0);
		}
		/*Update the event data*/
		ThreadMap[curid].allevents[userEventToken].tricount++;
		ThreadMap[curid].allevents[userEventToken].numevents++;
		ThreadMap[curid].allevents[userEventToken].sum+=userEventValue;
		ThreadMap[curid].allevents[userEventToken].sumsqr+=
		(userEventValue*userEventValue);
		if(userEventValue > ThreadMap[curid].allevents[userEventToken].max)
		{
			ThreadMap[curid].allevents[userEventToken].max=userEventValue;
		}
		if(userEventValue<ThreadMap[curid].allevents[userEventToken].min
		||ThreadMap[curid].allevents[userEventToken].min==0)
		{
			ThreadMap[curid].allevents[userEventToken].min=userEventValue;
		}
	}
	else
	{	/*monotonically increasing*/
		int stateid=ThreadMap[curid].lastState;
		if(stateid==-1)
		{
			cout << "State not set" << endl;
		}
		ThreadMap[curid].allstate[stateid].allmi[userEventToken].holdvalue=userEventValue;
		ThreadMap[curid].allstate[stateid].countMIE++;
		/*If we have encountered every event associated with this state we can process all at once*/
		if(ThreadMap[curid].allstate[stateid].countMIE==(miecount+1))
		{/*lastaction indicates if we should batch process a state exit or a state enter*/
			if(ThreadMap[curid].allstate[stateid].lastaction==1)
			{
				//ThreadMap[curid]=
				StateEnter(time,ThreadMap[curid],stateid);
			}
			else
			if(ThreadMap[curid].allstate[stateid].lastaction==-1)
			{
				//ThreadMap[curid]=
				StateLeave(time,ThreadMap[curid],stateid);
			}
			else
			if(ThreadMap[curid].allstate[stateid].lastaction==0)
			{
				cout << "Action Detection Fault(Event Trigger)" << endl;
			}
			ThreadMap[curid].allstate[stateid].countMIE=0;
			ThreadMap[curid].allstate[stateid].lastaction=0;
			ThreadMap[curid].lastState=-1;
		}
	}
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
	SnapshotControl(time,-2);
	
	/*
	 * If we are recording user defined events, all state control is managed by 'EventTrigger'
	 */
	int curid=ThreadID[pair<int,int>(nid,tid)];
	if(miecount>0)
	{
		ThreadMap[curid].lastState=stateid;
		ThreadMap[curid].allstate[stateid].lastaction=1;
		ThreadMap[curid].allstate[stateid].holdvalue=time;
		if(ThreadMap[curid].allstate[stateid].countMIE>0)
		{
			cout << "User Event Couting Fault (EnterState)" << endl;
		}
		ThreadMap[curid].allstate[stateid].countMIE++;
		return 0;
	}
	
	//ThreadMap[curid]=
	
	StateEnter(time,(ThreadMap[curid]),stateid);
	
	return 0;
}

/***************************************************************************
 * 
 * This routine registers the entry into state 'stateid' from thread 'threadin' at time 'time'.
 * It returns 'threadin' with these modifications.
 * 
 ***************************************************************************/
void StateEnter(double time, Thread &threadin, unsigned int stateid)
{
	/*If there is another state in the callstack then the state we are entering
	* is a subroutine of the previous state.  The previous state's time is no 
	* longer exclusive so we record the exclusive time seen so far and stop.*/
	if(!threadin.callstack.empty())
	{
		int lastcall=threadin.callstack.top();
		threadin.allstate[lastcall].subroutines++;
		
		threadin.allstate[lastcall].exclusive+=time-threadin.allstate[lastcall].topTime;
		threadin.allstate[lastcall].topTime=-1;
		
		if(miecount>0)
		{
			for(map<int,MonIncEvent>:: iterator miit=
			threadin.allstate[stateid].allmi.begin();
			miit!=threadin.allstate[stateid].allmi.end();miit++)
			{
				/*Here we need to add to the previous event's eclusive time the current event's observed time minus the previous's top*/
				int loctoken=(*miit).second.eventToken;
				threadin.allstate[lastcall].allmi[loctoken].exclusive+=(*miit).second.holdvalue-
				threadin.allstate[lastcall].allmi[loctoken].topCount;
				threadin.allstate[lastcall].allmi[loctoken].topCount=-1;
			}
		}
	}
	/*Increase the number of times this state has been entered and mark
	 * the time it spends on the top of the stack.*/
	threadin.allstate[stateid].calls++;
	allstate[stateid].calls++;
	
	threadin.allstate[stateid].topTime=time;
	/*If this state has not exeted after a previous call it is in recursion.  We
	 * only give it a new start time if it is not in recursion.*/
	if(threadin.allstate[stateid].fullTime==-1)
	{
		threadin.allstate[stateid].fullTime=time;
	}
	
	if(miecount>0)
	{
		for(map<int,MonIncEvent>:: iterator miit=
		threadin.allstate[stateid].allmi.begin();
		miit!=threadin.allstate[stateid].allmi.end();miit++)
		{
			(*miit).second.topCount=(*miit).second.holdvalue;
			if((*miit).second.fullCount==-1)
			{
				(*miit).second.fullCount=(*miit).second.holdvalue;
			}
		}
	}	
	
	/*Add one the number of calls to this state currently active and put the
	 * state on the callstack.*/
	threadin.allstate[stateid].countRec++;
	threadin.callstack.push(stateid);
	
	//return threadin;
}

/***************************************************************************
 * Description: LeaveState is called at routine exit by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int stateid)
{
	SnapshotControl(time,stateid);
	
	/*
	 * If we are recording user defined events, all state control is managed by 'EventTrigger'
	 */
	int curid=ThreadID[pair<int,int>(nid,tid)];
	if(miecount>0)
	{
		ThreadMap[curid].lastState=stateid;
		ThreadMap[curid].allstate[stateid].lastaction=-1;
		ThreadMap[curid].allstate[stateid].holdvalue=time;
		if(ThreadMap[curid].allstate[stateid].countMIE>0)
		{
			cout << "User Event Couting Fault (LeaveState)" << endl;
		}
		ThreadMap[curid].allstate[stateid].countMIE++;
		return 0;
	}
	
	//ThreadMap[curid]=
	StateLeave(time,(ThreadMap[curid]),stateid);

	return 0;
}

/***************************************************************************
 * 
 * This routine registers the exit of state 'stateid' from thread 'threadin' at time 'time'.
 * It returns 'threadin' with these modifications.
 * 
 ***************************************************************************/
void StateLeave(double time, Thread &threadin, unsigned int stateid)
{
	/*Add the exclusive time recorded since the routine was last on the top 
	 * of the stack.  It is no longer on the stack so stop recording topTime.
	 * Decrement the number of recursive calls to this routine.*/
	threadin.allstate[stateid].exclusive+=
	time-threadin.allstate[stateid].topTime;
	threadin.allstate[stateid].topTime=-1;
	if(miecount>0&&printshot)
	{
		for(map<int,MonIncEvent>:: iterator miit=
		threadin.allstate[stateid].allmi.begin();
		miit!=threadin.allstate[stateid].allmi.end();miit++)
		{
			(*miit).second.exclusive+=(*miit).second.holdvalue-(*miit).second.topCount;
			(*miit).second.topCount=-1;
		}
	}
	threadin.allstate[stateid].countRec--;
	/*If we are no longer in recursion (all calls to this routine have exited)
	 * then we can record the inclusive time for this routine from its initial
	 * call and stop.*/
	if(threadin.allstate[stateid].countRec==0)
	{
		threadin.allstate[stateid].inclusive+=
		time-threadin.allstate[stateid].fullTime;
		threadin.allstate[stateid].fullTime=-1;
		
		if(miecount>0&&printshot)
		{
			for(map<int,MonIncEvent>:: iterator miit=
			threadin.allstate[stateid].allmi.begin();
			miit!=threadin.allstate[stateid].allmi.end();miit++)
			{
				(*miit).second.inclusive+=((*miit).second.holdvalue)-((*miit).second.fullCount);
				(*miit).second.fullCount=-1;
			}
		}
	}
	/*Pop this routine off of the callstack.  If there is another routine start
	 * recording its exclusive time again.*/
	threadin.callstack.pop();
	if(!threadin.callstack.empty())
	{	
		int lastcall=threadin.callstack.top();
		
		threadin.allstate[lastcall].topTime=time;
		if(miecount>0&&printshot)
		{
			for(map<int,MonIncEvent>:: iterator miit=
			threadin.allstate[stateid].allmi.begin();
			miit!=threadin.allstate[stateid].allmi.end();miit++)
			{
				int loctoken = (*miit).second.eventToken;
				threadin.allstate[lastcall].allmi[loctoken].topCount=(*miit).second.holdvalue;
			}
		}
	}
	//return threadin;
}

/***************************************************************************
 * This routine runs through the relevant callback methods
 ***************************************************************************/
void ReadFile()
{
	/* in the first pass, we determine the no. of cpus and other group related
	* information. In the second pass, we look at the function entry/exits */ 
	int recs_read=0;
	fh = Ttf_OpenFileForInput(trc, edf);
	if (!fh)
	{
		fprintf(stderr,"ERROR:Ttf_OpenFileForInput failed\n");
		exit(1);
	}
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
	while ((recs_read >=0) && (!EndOfTrace));
	
	/*If there are monotoncally increasing events add their map to each state*/
	if(miecount>0)
	{
		for(map<int,State>:: iterator sit=allstate.begin();sit!=allstate.end();sit++)
		{
			(*sit).second.allmi=allmoninc;
		}
	}
	
	/* reset the position of the trace to the first record 
	Initialize global id map*/
	for (map< int, Thread >:: iterator it = //, less< pair<int,int> > 
	ThreadMap.begin(); 
	it != ThreadMap.end(); it++)
	{ /* Explicilty mark end of trace to be not over */ 
		(*it).second.allstate=allstate;
		(*it).second.allevents=allevents;
		(*it).second.finished = 0;
	}
	EndOfTrace = 0;
	/* now reset the position of the trace to the first record */ 
	Ttf_CloseFile(fh);
	/* Re-open it for input */
	fh = Ttf_OpenFileForInput(trc, edf);
	
	if (!fh)
	{
		printf("ERROR:Ttf_OpenFileForInput fails the second time");
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
	while ((recs_read >=0) && (!EndOfTrace));

	/* dummy records */
	Ttf_CloseFile(fh);
	PrintProfiles(ThreadMap);
}

/***************************************************************************
 * Given a map 'mainmap' conforming to the 'whole trace' data structure, this routine will cycle
 * through each thread and state to print out the profile statistics for the whole program
 ***************************************************************************/
void PrintProfiles(map<int,Thread> &mainmap){//, less< pair<int,int> > 
	char prefix [32];
	string s_out="";
	string s_prefix="";
	string cmd="";
	double mean=0;
	double sum=0;
	double num=0;
	if(out!=NULL)
	{
		s_out=out;
		s_out+="/";
	}
	
	if(snapshot>-1)
	{
		snprintf(prefix, sizeof(prefix), "snapshot_%d/",snapshot);
		s_out+=prefix;
		cmd="mkdir ";
		cmd+=s_out;
		system(cmd.c_str());
	}
	
	if(miecount>0)
	{
		s_out+="MULTI__GET_TIME_OF_DAY/";
		cmd="mkdir "+s_out;
		system(cmd.c_str());
	}

	ofstream profile;
	int countFunc=0;
	for(map<int,State>:: iterator stateCount = allstate.begin(); stateCount!=allstate.end(); stateCount++)
	{
		if((*stateCount).second.calls>0)
			countFunc++;
	}
	for (map< int,Thread >:: iterator it = mainmap.begin(); // less< pair<int,int> > 
	it != mainmap.end(); it++)
	{
		char filename [32];
		snprintf(filename, sizeof(filename), "profile.%d.0.%d",((*it).second).nodeToken,((*it).second).threadToken);
		s_prefix=s_out+filename;
		profile.open(s_prefix.c_str());
		profile.precision(16);
		profile << countFunc << " templated_functions";
		if(miecount>0)
		{
			profile << "_MULTI_GET_TIME_OF_DAY";
		}
		profile << endl;
		profile << "# Name Calls Subrs Excl Incl ProfileCalls" << endl;
		for(map<int,State>::iterator st = ((*it).second).allstate.begin(); 
		st !=((*it).second).allstate.end();st++)
		{
			if(allstate[(*st).second.stateToken].calls>0)
			profile << ((*st).second).stateName << " " << ((*st).second).calls 
			<< " " << ((*st).second).subroutines << " " << ((*st).second).exclusive 
			<< " " << ((*st).second).inclusive << " " << "0" << " GROUP=\"" 
			<< groupids[((*st).second).stateGroupToken]<<"\""<< endl;
		}
		profile << "0 aggregates" << endl;
		if(allevents.size()>0)
		{
			profile << allevents.size() << " userevents" << endl 
			<<  "# eventname numevents max min mean sumsqr" << endl;
			for(map<int,UserEvent>::iterator st = ((*it).second).allevents.begin(); 
			st !=((*it).second).allevents.end();st++)
			{
				mean = 0;
				if(((*st).second).numevents>0)
				{
					sum=((*st).second).sum;
					num=((*st).second).numevents;
					mean = sum/num;
				}
				profile << ((*st).second).userEventName << " " 
				<< ((*st).second).numevents << " " << ((*st).second).max 
				<< " " << ((*st).second).min << " " << mean << " " << ((*st).second).sumsqr << endl;
			}
		}
		profile.close();
	}
	
	if(miecount>0)
	{
		unsigned int eventID=0;
		const char * eventname;
		string base = "";
		string s_name="";
		if(out!=NULL)
		{
			base=out;
			base+="/";
		}
		for(map<int,MonIncEvent>:: iterator miit=allmoninc.begin();
		miit!=allmoninc.end();miit++)
		{
			s_name="";
			if(snapshot>-1)
			{
				snprintf(prefix, sizeof(prefix), "snapshot_%d/",snapshot);
				s_name+=prefix;
				//cmd="mkdir ";
				//cmd+=s_out;
				//system(cmd.c_str());
			}
			eventID=(*miit).first;
			eventname=(*miit).second.eventName;
			s_name+="MULTI__";
			s_name+=eventname;
			s_out=base+s_name+"/";
			cmd="mkdir "+s_out;
			system(cmd.c_str());
			
			countFunc=0;
			for(map<int,State>:: iterator stateCount = allstate.begin(); stateCount!=allstate.end(); stateCount++)
			{
				if((*stateCount).second.calls>0)
					countFunc++;
			}
			for (map< int,Thread >:: iterator it = mainmap.begin(); //, less< pair<int,int> > 
			it != mainmap.end(); it++)
			{
				char filename [32];
				snprintf(filename, sizeof(filename), "profile.%d.0.%d",(*it).second.nodeToken,((*it).second).threadToken);
				s_prefix=s_out+filename;
				profile.open(s_prefix.c_str());
				profile.precision(16);
				profile << countFunc << " templated_functions_" << "MULTI_" << eventname << endl;
				profile << "# Name Calls Subrs Excl Incl ProfileCalls" << endl;
				for(map<int,State>::iterator st = ((*it).second).allstate.begin(); 
				st !=((*it).second).allstate.end();st++)
				{
					if(allstate[(*st).second.stateToken].calls>0)
					profile << ((*st).second).stateName << " " << ((*st).second).calls 
					<< " " << ((*st).second).subroutines << " " << ((*st).second).allmi[eventID].exclusive 
					<< " " << ((*st).second).allmi[eventID].inclusive << " " << "0" << " GROUP=\"" 
					<< groupids[((*st).second).stateGroupToken]<<"\""<< endl;
				}
				profile << "0 aggregates" << endl;
				if(allevents.size()>0)
				{
					profile << allevents.size() << " userevents" << endl 
					<<  "# eventname numevents max min mean sumsqr" << endl;
					for(map<int,UserEvent>::iterator st = ((*it).second).allevents.begin(); 
					st !=((*it).second).allevents.end();st++)
					{
						mean = 0;
						if(((*st).second).numevents>0)
						{
							sum=((*st).second).sum;
							num=((*st).second).numevents;
							mean = sum/num;
						}
						profile << ((*st).second).userEventName << " " 
						<< ((*st).second).numevents << " " << ((*st).second).max 
						<< " " << ((*st).second).min << " " << mean << " " << ((*st).second).sumsqr << endl;
					}
				}
				profile.close();
			}
		}
	}	
	return;	
}

/***************************************************************************
 * Given the time stamp for the cut-off time, this routine will iterate through the current trace-state
 * and exit every state on every stack, effectively creating a snapshot profile up to the given time.
 * The profile generated will then be printed.
 ***************************************************************************/
void PrintSnapshot(double time)
{
	map<int,Thread> finalizer = ThreadMap;//, less<pair<int,int> > 
	printshot=1;
	for(map< int, Thread >:: iterator it = finalizer.begin(); //, less< pair<int,int> > 
	it != finalizer.end(); it++)
	{
		while((*it).second.callstack.size()>0)
		{
			//(*it).second=
			StateLeave(time,(*it).second,(*it).second.callstack.top());
		}
	}
	PrintProfiles(finalizer);
	printshot=0;
}

/***************************************************************************
 * If 'time' is greater than or equal to the end of the next specified interval this will
 * cut print out the snapshot as of the specified interval and set the time for the next one.
 ***************************************************************************/
void SnapshotControl(double time, int token)
{
	if(snapshot>-1 && time>=nextShot)
	{
		PrintSnapshot(nextShot);
		nextShot+=segmentInterval;
		snapshot++;
	}
	else
	if(token==trigEvent)
	{
		trigSeen++;
		if(trigSeen==trigCount)
		{
			PrintSnapshot(time);
			snapshot++;
			trigSeen=0;
		}
	}
}

/***************************************************************************
 * Prints usage info
 ***************************************************************************/
void Usage()
{
	cout << "You must specify a valid .trc and .edf file for conversion."<<endl;
	cout << "These may be followed by any of the arguments:\n "<<endl;
	cout << "-d <directory>:  Output profile files to the directory "
		 << "specified rather than the current directory.\n"<< endl;
	cout << "-s <interger n>: Output a profile snapshot of the trace every n "
		 << "time units.\n" << endl;
	cout << "e.g. $tau2profile tau.trc tau.edf -s 25000"  << endl;
}

/***************************************************************************
 * The main function reads user input and starts conversion procedure.
 ***************************************************************************/
int main(int argc, char **argv)
{
	int i; 
	/* main program: Usage app <trc> <edf> [-a] [-nomessage] */
	if (argc < 2)
	{
		Usage();
		exit(1);
	}
	for (i = 0; i < argc ; i++)
	{
		switch(i) 
		{
			case 0:
			break;
			case 1:
			/*trace_file*/ 
			trc = argv[1];
			break;
			case 2:
			/*edf_file*/ 
			edf = argv[2];
			break;			
			default:
			/*if (strcmp(argv[i], "-v") == 0)
			{
				debugPrint = 1;
				break;
			}
			else*/
			if(strcmp(argv[i],"-s")==0)
			{/*Segment interval*/
				if(argc>i+1)
				{
					i++;
					segmentInterval=atof(argv[i]);
					trigEvent=-1;
				}
				break;
			}
			else
			if(strcmp(argv[i],"-d")==0)
			{/*Output Directory*/
				if(argc>i+1)
				{
					i++;
					out=argv[i];
				}
				break;
			}
			else
			if(strcmp(argv[i],"-e")==0)
			{/*Event Trigger*/
				if(argc>i+1)
				{
					i++;
					trigEvent=atoi(argv[i]);
					segmentInterval=0;
				}
				break;
			}
			else
			if(strcmp(argv[i],"-c")==0)
			{/*Event count*/
				if(argc>i+1)
				{
					i++;
					trigCount=atoi(argv[i]);
				}
				break;
			}
			else
			{
				Usage();
				exit(1);
			}
			break;
		}
	}
	/* Finished parsing commandline options, now open the trace file */
	if(segmentInterval>0)
	{
		snapshot=0;
		nextShot=segmentInterval;
	}
	else
	if(trigEvent>-1)
	{
		snapshot=0;
	}
	ReadFile();
}
