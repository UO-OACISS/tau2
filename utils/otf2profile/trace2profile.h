#ifndef TRACE2PROFILE_H_
#define TRACE2PROFILE_H_

#include <iostream>
#include <string>
#include <stack>
#include <map>
#include <vector>
#include <set>

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
	//const char * eventName;
	
	long long bakholdvalue;
	long long bakinclusive;
	long long bakexclusive;
	long long baktopCount;
	long long bakfullCount;
	
	
	void BackupMIE(){
		bakholdvalue=holdvalue;
		bakinclusive=inclusive;
		bakexclusive=exclusive;
		baktopCount=topCount;
		bakfullCount=fullCount;
	}
	
	void RestoreMIE(){
		holdvalue=bakholdvalue;
		inclusive=bakinclusive;
		exclusive=bakexclusive;
		topCount=baktopCount;
		fullCount=bakfullCount;
	}
	
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
	//const char * stateName;
	/*These are for use with monotonically increasing events*/
	map<unsigned int,MonIncEvent*> allmi;
	unsigned int countMIE;/*Counts monotonically increasing events*/
	int lastaction;/*1=enter,-1=leave,0=undefined*/
	double holdvalue;/**/
	
	
	double bakinclusive;
	double bakexclusive;
	int bakcalls;
	int baksubroutines;
	double baktoptime;
	double bakfulltime;
	int bakcountrec;
	unsigned int bakcountMIE;
	int baklastaction;
	double bakholdvalue;
	
	void BackupState(){
		
		for(map<unsigned int,MonIncEvent*>:: iterator it = allmi.begin(); it!=allmi.end(); it++)
		{
			(*it).second->BackupMIE();
		}
		
		bakinclusive=inclusive;
		bakexclusive=exclusive;
		bakcalls=calls;
		baksubroutines=subroutines;
		baktoptime=topTime;
		bakfulltime=fullTime;
		bakcountrec=countRec;
		bakcountMIE=countMIE;
		baklastaction=lastaction;
		bakholdvalue=holdvalue;
		

		
	}
	
	void RestoreState(){
		
		for(map<unsigned int,MonIncEvent*>:: iterator it = allmi.begin(); it!=allmi.end(); it++)
		{
			(*it).second->RestoreMIE();
		}
		
		inclusive=bakinclusive;
		exclusive=bakexclusive;
		calls=bakcalls;
		subroutines=baksubroutines;
		topTime=baktoptime;
		fullTime=bakfulltime;
		countRec=bakcountrec;
		countMIE=bakcountMIE;
		lastaction=baklastaction;
		holdvalue=bakholdvalue;
	}
	
	
	/*~State(){
		for(map<unsigned int,MonIncEvent*>:: iterator mit = allmi.begin(); mit!=allmi.end(); mit++)
		{
				delete (*mit).second;
				(*mit).second=NULL;
				
		}
		delete &allmi;
		//&allmi=NULL;
	
	}*/
	
	
};

/*
 * This holds a single non-monotonically increasing event
 */
class UserEvent{
	public:
	//const char * userEventName;
	unsigned int userEventToken;
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
	map<unsigned int,State*>allstate;
	map<unsigned int,UserEvent*>allevents;
	unsigned int nodeToken;
	unsigned int threadToken;
	unsigned int processToken;
	int lastState;/*The last state entered in this thread*/
	int currentState;/*The state being exited, by virtue of being the last state entered*/
	vector<State*> callstack;/*State IDs*/  //unsigned int
	//const char * threadName;
	bool finished;
	
	double nextShot;//=-1;
	double thisShot;//=0;
	double lastTime;//=0;
	
	int trigSeen;//=0;
	
	int snapshot;//=-1;
	
	
	int baklastState;
	double baklastTime;
	vector<State*> bakcallstack;
	
	void BackupThread(){
		for(map<unsigned int,State*>:: iterator sit = allstate.begin(); sit!=allstate.end(); sit++)
		{
			(*sit).second->BackupState();
		}
		baklastState=lastState;
		baklastTime=lastTime;
		bakcallstack.clear();
		for(vector<State*>::iterator cit=callstack.begin(); cit!=callstack.end();cit++)
		{
			bakcallstack.push_back(*cit);
		}
		//callstack.clear();
		
	}
	
	void RestoreThread(){
		for(map<unsigned int,State*>:: iterator sit = allstate.begin(); sit!=allstate.end(); sit++)
		{
			(*sit).second->RestoreState();
		}
		lastState=baklastState;
		lastTime=baklastTime;
		callstack.clear();
		for(vector<State*>::iterator cit=bakcallstack.begin(); cit!=bakcallstack.end();cit++)
		{
			callstack.push_back(*cit);
		}
		bakcallstack.clear();
	
	}
	
	/*
	Thread(const Thread& t){
	
		for(vector<State*>::const_iterator sit=t.callstack.begin();sit!=t.callstack.end();sit++){
			callstack.push_back((new State(**sit)));
		}
	
	}
	
	Thread(){}
	*/
	/*~Thread(){
	
		for(map<unsigned int,State*>:: iterator sit = allstate.begin(); sit!=allstate.end(); sit++)
		{
			delete (*sit).second;
		}
		delete &allstate;
		for(map<unsigned int,UserEvent*>:: iterator uit = allevents.begin(); uit!=allevents.end(); uit++)
		{
			delete (*uit).second;
		}
		delete &allevents;
		delete &callstack;
	}*/
	
};

class Converter{
public:
/*Each thread in the trace is held here, mapped to the process id*/
static map<unsigned int,Thread*> ThreadMap;
/*Each state in the trace is held here, mapped to its event id.  
 * This is copied into each thread once it is initialized*/
static map<unsigned int,State*> allstate;
/*Each monotonically increasing event is held here, mapped to its
 * event id.  This is copied to each state once initialized.*/
static vector<MonIncEvent*> allmoninc;
static set <unsigned int> monincids;
/*Each user event is held here, mapped to its event id.
 * This is copied to each thread after it is initialized.*/
static vector<UserEvent*> allevents;

/*These map names to their ids*/
static map<unsigned int, string*> groupnames;
static map<unsigned int, string*> statenames;
static map<unsigned int, string*> usereventnames;
static map<unsigned int, string*> monincnames;
static map<unsigned int, string*> threadnames;

static int EndOfTrace;  /* false */
//static bool printshot;
//static unsigned int miecount;
//static int debugPrint;
static double segmentInterval;
static double lastTime;
static int trigEvent;
static int trigCount;

static char * trc;
static char * edf;
static char * out;

/*~Converter()
{
	for(map<unsigned int,Thread*>:: iterator it = ThreadMap.begin(); it!=ThreadMap.end(); it++)
	{
		delete (*it).second;
	}
	
}*/

};

void StateLeave(double time, Thread &thread, State &state);
void StateEnter(double time, Thread &thread, State &state);
void SnapshotControl(double time, int stateToken, Thread &thread);
void ReadTraceFile();
void ReadOTFFile();
void ProcessDefs();

#endif /*TRACE2PROFILE_H_*/
