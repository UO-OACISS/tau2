#include <handlers.h>
#include <trace2profile.h>

using namespace std;

/* implementation of callback routines */
/***************************************************************************
 * Description: DefThread is called when a new nodeid/threadid is encountered.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int ThreadDef(unsigned int nodeToken, unsigned int threadToken, unsigned int processToken, const char *threadName )
{
	Thread &local = *(new Thread());
	//cout << nodeToken << " Process " << &local << endl;
	local.nodeToken=nodeToken;
	local.threadToken=threadToken;
	Converter::threadnames[processToken] = new string(threadName);//local.threadName=threadName;
	local.lastState=-1;
	local.finished=false;
	
	local.nextShot=-1;
	local.thisShot=0;
	local.lastTime=0;
	
	local.trigSeen=0;
	
	local.snapshot=-1;
	local.processToken=processToken;
	
	
	if(Converter::segmentInterval>0)
	{
		local.snapshot=0;
		local.nextShot=Converter::segmentInterval;
	}
	else
	if(Converter::trigEvent>-1)
	{
		local.snapshot=0;
	}
	
	//int curid;
	//EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = pair<int,Thread>(0,local);
	//Converter::ThreadID[pair<int,int>(nodeToken,threadToken)]= curid =Converter::ThreadID.size();
	//EOF_Trace[curid]=false;
	Converter::ThreadMap[processToken]=&local;
	//cout << "Printing Thread: " << processToken << endl;
	//SnapshotThreadPrint(processToken,nodeToken,threadToken);
	
	return 0;
}


/***************************************************************************
 * Description: DefState is called to define a new symbol (event). It uses
 *		the token used to define the group identifier. 
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int StateDef(unsigned int stateToken, const char *stateName, unsigned int stateGroupToken )
{
	State &local = *(new State());
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
	//local.stateName=stateName;
	Converter::statenames[stateToken]= new string(stateName);
	Converter::allstate[stateToken]=&local;
	return 0;
}


/***************************************************************************
 * Description: DefStateGroup registers a profile group name with its id.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int StateGroupDef(unsigned int stateGroupToken, const char *stateGroupName )
{
	Converter::groupnames[stateGroupToken] = new string(stateGroupName);
	return 0;
}

/***************************************************************************
 * Description: DefUserEvent is called to register the name and a token of the
 *  		user defined event (or a sample event in Vampir terminology).
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int UserEventDef(unsigned int userEventToken,const char *userEventName , int monotonicallyIncreasing)
{	
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
		/* for hw counter data */
		MonIncEvent &local = *(new MonIncEvent());
		Converter::monincnames[userEventToken] = new string(userEventName);//local.eventName=name;
		local.eventToken=userEventToken;
		local.exclusive=0;
		local.inclusive=0;
		local.topCount=-1;
		local.fullCount=-1;
		local.holdvalue=-1;
		Converter::allmoninc.push_back(&local);
		//Converter::miecount++;
		Converter::monincids.insert(userEventToken);
	}
	else
	{ /* for non monotonically increasing data */
		UserEvent &local=*(new UserEvent());

		Converter::usereventnames[userEventToken] = new string(userEventName);//local.userEventName=name;
		local.max=0;
		local.min=0;
		local.numevents=0;
		local.sum=0;
		local.sumsqr=0;
		local.tricount=0;
		local.userEventToken=userEventToken;
		Converter::allevents.push_back(&local);
	} 
	return 0;
}

/***************************************************************************
 * Description: ClockPeriod (in microseconds) is specified here. 
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int ClockPeriodDef(double clkPeriod )
{
	return 0;
}

/***************************************************************************
 * Description: EndTrace is called when an EOF is encountered in a tracefile.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EndTraceDef(unsigned int processToken)//unsigned int nodeToken, unsigned int threadToken, 
{
	//(EOF_Trace[pair<int,int> (nodeToken,threadToken) ]).first = 1; /* flag it as over */
	(Converter::ThreadMap[processToken])->finished=true;
	/* yes, it is over */
	map <unsigned  int,Thread*>::iterator it;//pair<int, int>, pair<int,Thread>, less< pair<int,int> > 
	Converter::EndOfTrace = 1; /* Lets assume that it is over */
	for (it = Converter::ThreadMap.begin(); it != Converter::ThreadMap.end(); it++)
	{/* cycle through all <nid,tid> pairs to see if it really over */
		if ((*it).second->finished == false)//.second.
		{
			Converter::EndOfTrace = 0; /* not over! */
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
int EventTriggerDef(double time,
		unsigned int pid,
		unsigned int userEventToken,
		long long userEventValue)//unsigned int nid,unsigned int tid,
{ 
	if(time>Converter::lastTime)Converter::lastTime=time;
	//unsigned int curid=pid;//Converter::ThreadID[pair<int,int>(nid,tid)];
	
	Thread &threadin=*(Converter::ThreadMap[pid]);
	
	threadin.lastTime=time;//Converter::ThreadMap[curid].lastTime=time;
	//lastTime=time;
	SnapshotControl(time,-2, threadin);
	
	/* write the sample data */
	if(Converter::monincids.count(userEventToken)==0)
	{	/*not monotonically increasing*/
		/*Each NMI event is between two '0' events which are ignored. (every 2nd of 3 events is used)*/

		//ThreadMap[curid] <- EOF_Trace[pair<int,int>(nid,tid)].second
		
		UserEvent &thisUE=*(threadin.allevents[userEventToken]);
		
		if(thisUE.tricount==0)
		{
			thisUE.tricount++;
			return(0);
		}
		if(thisUE.tricount==2)
		{
			thisUE.tricount=0;
			return(0);
		}
		/*Update the event data*/
		thisUE.tricount++;
		thisUE.numevents++;
		thisUE.sum+=userEventValue;
		thisUE.sumsqr+=
		(userEventValue*userEventValue);
		if(userEventValue > thisUE.max)
		{
			thisUE.max=userEventValue;
		}
		if(userEventValue < thisUE.min
		||thisUE.min==0)
		{
			thisUE.min=userEventValue;
		}
	}
	else
	{	/*monotonically increasing*/
		
		//cout << userEventToken << " entered " <<  pid << " with " << userEventValue << endl;
		
		int stateid=threadin.lastState;
		if(stateid==-1)
		{
			cout << "State not set" << endl;
		}
		
		State &thisState=*(threadin.allstate[stateid]);
		
		thisState.allmi[userEventToken]->holdvalue=userEventValue;
		thisState.countMIE++;
		/*If we have encountered every event associated with this state we can process all at once*/
		if(thisState.countMIE==(Converter::monincids.size()+1))
		{/*lastaction indicates if we should batch process a state exit or a state enter*/
			if(thisState.lastaction==1)
			{
				//ThreadMap[curid]=
				StateEnter(time,threadin,thisState);
			}
			else
			if(thisState.lastaction==-1)
			{
				//ThreadMap[curid]=
				StateLeave(time,threadin,thisState);
			}
			else
			if(thisState.lastaction==0)
			{
				cout << "Action Detection Fault(Event Trigger)" << endl;
			}
			thisState.countMIE=0;
			thisState.lastaction=0;
			threadin.lastState=-1;
		}
	}
	return 0;
}

/***************************************************************************
 * Description: EnterState is called at routine entry by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EnterStateDef(double time, unsigned int pid, unsigned int stateid)//unsigned int nid, unsigned int tid, 
{
	if(time>Converter::lastTime)Converter::lastTime=time;
	//unsigned int curid=pid;
	Thread &threadin=*(Converter::ThreadMap[pid]);
	threadin.lastTime=time;
	//lastTime=time;
	SnapshotControl(time,-2, threadin);
	State &thisState=*(threadin.allstate[stateid]);
	/*
	 * If we are recording user defined events, all state control is managed by 'EventTrigger'
	 */
	//int curid=ThreadID[pair<int,int>(nid,tid)];
	if(Converter::monincids.size()>0)
	{
		
		threadin.lastState=stateid;
		thisState.lastaction=1;
		thisState.holdvalue=time;
		if(thisState.countMIE>0)
		{
			cout << "User Event Couting Fault (EnterState)" << endl;
		}
		thisState.countMIE++;
		return 0;
	}
	
	//ThreadMap[curid]=
	
	StateEnter(time,threadin,thisState);
	
	return 0;
}



/***************************************************************************
 * Description: LeaveState is called at routine exit by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int LeaveStateDef(double time,unsigned int pid, unsigned int stateid)// unsigned int nid, unsigned int tid, 
{
	if(time>Converter::lastTime)Converter::lastTime=time;
	//unsigned int curid=pid;//Converter::ThreadID[pair<int,int>(nid,tid)];
	Thread &threadin=*(Converter::ThreadMap[pid]);
	State &thisState=*(threadin.allstate[stateid]);
	threadin.lastTime=time;
	//lastTime=time;
	SnapshotControl(time,stateid, threadin);
	
	/*
	 * If we are recording user defined events, all state control is managed by 'EventTrigger'
	 */
	//int curid=ThreadID[pair<int,int>(nid,tid)];
	if(Converter::monincids.size()>0)
	{
		threadin.lastState=stateid;
		thisState.lastaction=-1;
		thisState.holdvalue=time;
		if(thisState.countMIE>0)
		{
			cout << "User Event Couting Fault (LeaveState)" << endl;
		}
		thisState.countMIE++;
		return 0;
	}
	
	//ThreadMap[curid]=
	StateLeave(time,threadin,thisState);

	return 0;
}
