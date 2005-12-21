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

struct MonIncEvent {
	long long holdvalue;
	long long inclusive;
	long long exclusive;
	long long topCount;
	long long fullCount;
	unsigned int eventToken;
	const char * eventName;
};

struct State {
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

struct UserEvent{
	const char * userEventName;
	int userEventToken;
	int numevents;
	int tricount;
	long long max;
	long long min;
	long long sum;
	long long sumsqr;
};

struct Thread {
	map<int,State>allstate;
	map<int,UserEvent>allevents;
	int nodeToken;
	int threadToken;
	int lastState;
	stack<unsigned int> callstack;/*State IDs*/
	const char * threadName;
};

map<pair<int,int>,pair<int,Thread>, less<pair<int,int> > > EOF_Trace;
map<int,State> allstate;
map<int,MonIncEvent> allmoninc;
map<int,UserEvent>allevents;
map<unsigned int,const char*> groupids;

void ReadFile();
void Usage();
void PrintProfiles(map< pair<int,int>, pair<int,Thread>, less< pair<int,int> > > mainmap);
void PrintSnapshot(double time);
void SnapshotControl(double time);
Thread LeaveAll(Thread thread, double time);
Thread StateLeave(double time, Thread thread, unsigned int stateid);
Thread StateEnter(double time, Thread thread, unsigned int stateid);

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
	EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = pair<int,Thread>(0,local);
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
	(EOF_Trace[pair<int,int> (nodeToken,threadToken) ]).first = 1; /* flag it as over */
	/* yes, it is over */
	map < pair<int, int>, pair<int,Thread>, less< pair<int,int> > >::iterator it;
	EndOfTrace = 1; /* Lets assume that it is over */
	for (it = EOF_Trace.begin(); it != EOF_Trace.end(); it++)
	{/* cycle through all <nid,tid> pairs to see if it really over */
		if (((*it).second).first == 0)
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
	SnapshotControl(time);
	/* write the sample data */
	if(allevents.count(userEventToken)>0)
	{
		if(EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].tricount==0)
		{
			EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].tricount++;
			return(0);
		}
		if(EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].tricount==2)
		{
			EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].tricount=0;
			return(0);
		}
		EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].tricount++;
		EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].numevents++;
		EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].sum+=userEventValue;
		EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].sumsqr+=
		(userEventValue*userEventValue);
		if(userEventValue > EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].max)
		{
			EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].max=userEventValue;
		}
		
		if(userEventValue<EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].min
		||EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].min==0)
		{
			EOF_Trace[pair<int,int>(nid,tid)].second.allevents[userEventToken].min=userEventValue;
		}
	}
	else
	{	/*monotonically increasing*/
		int stateid=EOF_Trace[pair<int,int>(nid,tid)].second.lastState;
		if(stateid==-1)
		{
			cout << "State not set" << endl;
		}
		EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].allmi[userEventToken].holdvalue=userEventValue;
		EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].countMIE++;
		/*If we have encountered every event associated with this state we can process all at once*/
		if(EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].countMIE==(miecount+1))
		{
			if(EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].lastaction==1)
			{
				EOF_Trace[pair<int,int>(nid,tid)].second=StateEnter(time,EOF_Trace[pair<int,int>(nid,tid)].second,stateid);
			}
			else
			if(EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].lastaction==-1)
			{
				EOF_Trace[pair<int,int>(nid,tid)].second=StateLeave(time,EOF_Trace[pair<int,int>(nid,tid)].second,stateid);
			}
			else
			if(EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].lastaction==0)
			{
				cout << "Action Detection Fault(Event Trigger)" << endl;
			}
			EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].countMIE=0;
			EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].lastaction=0;
			EOF_Trace[pair<int,int>(nid,tid)].second.lastState=-1;
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
	SnapshotControl(time);
	
	/*
	 * If we are recording user defined events, all state control is managed by 'EventTrigger'
	 * TBD: Reduce redundancy between Enter/Exit state and EventTrigger
	 */
	if(miecount>0)
	{
		EOF_Trace[pair<int,int>(nid,tid)].second.lastState=stateid;
		EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].lastaction=1;
		EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].holdvalue=time;
		if(EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].countMIE>0)
		{
			cout << "User Event Couting Fault (EnterState)" << endl;
		}
		EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].countMIE++;
		return 0;
	}
	
	EOF_Trace[pair<int,int>(nid,tid)].second=
	StateEnter(time,(EOF_Trace[pair<int,int>(nid,tid)].second),stateid);
	
	return 0;
}

/***************************************************************************
 * 
 * This routine registers the entry into state 'stateid' from thread 'threadin' at time 'time'.
 * 
 ***************************************************************************/
Thread StateEnter(double time, Thread threadin, unsigned int stateid)
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
	
	return threadin;
}

/***************************************************************************
 * Description: LeaveState is called at routine exit by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int stateid)
{
	SnapshotControl(time);
	
	/*
	 * If we are recording user defined events, all state control is managed by 'EventTrigger'
	 * TBD: Reduce redundancy between Enter/Exit state and EventTrigger
	 */
	if(miecount>0)
	{
		EOF_Trace[pair<int,int>(nid,tid)].second.lastState=stateid;
		EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].lastaction=-1;
		EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].holdvalue=time;
		if(EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].countMIE>0)
		{
			cout << "User Event Couting Fault (LeaveState)" << endl;
		}
		EOF_Trace[pair<int,int>(nid,tid)].second.allstate[stateid].countMIE++;
		return 0;
	}
	
	EOF_Trace[pair<int,int>(nid,tid)].second=
	StateLeave(time,(EOF_Trace[pair<int,int>(nid,tid)].second),stateid);

	return 0;
}

/***************************************************************************
 * 
 * This routine registers the exit of state 'stateid' from thread 'threadin' at time 'time'.
 * 
 ***************************************************************************/
Thread StateLeave(double time, Thread threadin, unsigned int stateid)
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
	return threadin;
}

/*
 * This routine runs through the relevant callback methods
 */
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
	for (map< pair<int,int>, pair<int,Thread>, less< pair<int,int> > >:: iterator it = 
	EOF_Trace.begin(); 
	it != EOF_Trace.end(); it++)
	{ /* Explicilty mark end of trace to be not over */ 
		(*it).second.second.allstate=allstate;
		(*it).second.second.allevents=allevents;
		(*it).second.first = 0;
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
	PrintProfiles(EOF_Trace);
}

/*
 * Given a map conforming to the 'whole trace' data structure, this routine will cycle
 * through each thread and state to print out the profile statistics for the whole program
 */
void PrintProfiles(map< pair<int,int>, pair<int,Thread>, less< pair<int,int> > > mainmap){
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
		sprintf(prefix,"snapshot_%d/",snapshot);
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
	for (map< pair<int,int>, pair<int,Thread>, less< pair<int,int> > >:: iterator it = mainmap.begin(); 
	it != mainmap.end(); it++)
	{
		char filename [32];
		sprintf(filename,"profile.%d.0.%d",((*it).first).first,((*it).first).second);
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
		for(map<int,State>::iterator st = ((*it).second).second.allstate.begin(); 
		st !=((*it).second).second.allstate.end();st++)
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
			for(map<int,UserEvent>::iterator st = ((*it).second).second.allevents.begin(); 
			st !=((*it).second).second.allevents.end();st++)
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
				sprintf(prefix,"snapshot_%d/",snapshot);
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
			for (map< pair<int,int>, pair<int,Thread>, less< pair<int,int> > >:: iterator it = mainmap.begin(); 
			it != mainmap.end(); it++)
			{
				char filename [32];
				sprintf(filename,"profile.%d.0.%d",((*it).first).first,((*it).first).second);
				s_prefix=s_out+filename;
				profile.open(s_prefix.c_str());
				profile.precision(16);
				profile << countFunc << " templated_functions_" << "MULTI_" << eventname << endl;
				profile << "# Name Calls Subrs Excl Incl ProfileCalls" << endl;
				for(map<int,State>::iterator st = ((*it).second).second.allstate.begin(); 
				st !=((*it).second).second.allstate.end();st++)
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
					for(map<int,UserEvent>::iterator st = ((*it).second).second.allevents.begin(); 
					st !=((*it).second).second.allevents.end();st++)
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

/*
 * Given the time stamp for the cut-off time, this routine will iterate through the current trace-state
 * and exit every state on every stack, effectively creating a snapshot profile up to the given time.
 * The profile generated will then be printed.
 */
void PrintSnapshot(double time)
{
	map<pair<int,int>,pair<int,Thread>, less<pair<int,int> > > finalizer = EOF_Trace;
	printshot=1;
	for(map< pair<int,int>, pair<int,Thread>, less< pair<int,int> > >:: iterator it = finalizer.begin(); 
	it != finalizer.end(); it++)
	{
		while((*it).second.second.callstack.size()>0)
		{
			(*it).second.second=StateLeave(time,(*it).second.second,(*it).second.second.callstack.top());
		}
	}
	PrintProfiles(finalizer);
	printshot=0;
}

/*
 * If 'time' is greater than or equal to the end of the next specified interval this will
 * cut print out the snapshot as of the specified interval and set the time for the next one.
 */
void SnapshotControl(double time)
{
	if(snapshot>-1 && time>=nextShot)
	{
		PrintSnapshot(nextShot);
		nextShot+=segmentInterval;
		snapshot++;
	}
}

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

/*
 * The main function reads user input and starts conversion procedure.
 */
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
			{
				if(argc>i+1)
				{
					i++;
					segmentInterval=atof(argv[i]);
				}
				break;
			}
			else
			if(strcmp(argv[i],"-d")==0)
			{
				if(argc>i+1)
				{
					i++;
					out=argv[i];
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
	ReadFile();
}
