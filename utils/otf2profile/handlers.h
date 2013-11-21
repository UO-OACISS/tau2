#ifndef HANDLERS_H_
#define HANDLERS_H_

int ThreadDef(unsigned int nodeToken, unsigned int threadToken, unsigned int processToken, const char *threadName);
int StateDef(unsigned int stateToken, const char *stateName, unsigned int stateGroupToken );
int StateGroupDef(unsigned int stateGroupToken, const char *stateGroupName );
int UserEventDef(unsigned int userEventToken,const char *userEventName , int monotonicallyIncreasing);
int ClockPeriodDef(double clkPeriod );
int EndTraceDef(unsigned int processToken);//unsigned int nodeToken, unsigned int threadToken, 
int EventTriggerDef(double time, 
		unsigned int pid,
		unsigned int userEventToken,
		long long userEventValue);//unsigned int nid,unsigned int tid,
int EnterStateDef(double time, unsigned int pid, unsigned int stateid);//unsigned int nid, unsigned int tid, 
int LeaveStateDef(double time, unsigned int pid);//, unsigned int stateid unsigned int nid, unsigned int tid, 

#endif /*HANDLERS_H_*/
