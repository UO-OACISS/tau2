
#include "TAU_tf.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>

using namespace std;

#include <float.h>


Ttf_FileHandleT theFile;
map <string,int> eventMap; 
map <string,int> groupMap; 
map <string,int> userEventMap;

int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken, const char *threadName);


int DefStateGroup(void *userData, unsigned int stateGroupToken, const char *stateGroupName);
int DefState(void *userData, unsigned int stateToken, const char *stateName, unsigned int stateGroupToken);
int DefUserEvent(void *userData, unsigned int userEventToken,
		 const char *userEventName, int monotonicallyIncreasing);


int EventTrigger(void *userData, double time,
		 unsigned int nodeToken, unsigned int threadToken, unsigned int userEventToken,
		 //double userEventValue)
		 long long userEventValue);


int SendMessage(void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken, unsigned int messageSize, unsigned int messageTag, unsigned int messageComm);

int RecvMessage(void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken, unsigned int messageSize, unsigned int messageTag, unsigned int messageComm);

int EnterState(void *userData, double time, unsigned int nodeid, unsigned int tid, unsigned int stateid);
int LeaveState(void *userData, double time, unsigned int nodeid, unsigned int tid, unsigned int stateid);
int EndTrace(void *userData, unsigned int nodeToken, unsigned int threadToken);



// an interface for the various record types
class Record {
	public:
		Record(double time) {
			this->time = time;	
		}
		double getTime() {
			return time;	
		}
		virtual void writeRecord(Ttf_FileHandleT tFile) = 0;
		
	protected:
		double time;
};


class EnterStateRecord : public Record {
	public:
		EnterStateRecord(double time, int nid, int tid, int stateToken) : 
					Record(time) {
			this->stateToken = stateToken;
			this->nid = nid;
			this->tid = tid;
		}

		void writeRecord(Ttf_FileHandleT tFile) {
			printf ("writing EnterState record: state=%d, nid=%d, tid=%d\n",stateToken,nid,tid);
			Ttf_EnterState(tFile, time, nid, tid, stateToken);
		}

	private:
		int nid;
		int tid;
		int stateToken;
};

class LeaveStateRecord : public Record {
	public:
		LeaveStateRecord(double time, int nid, int tid, int stateToken) : 
					Record(time) {
			this->stateToken = stateToken;
			this->nid = nid;
			this->tid = tid;
		}

		void writeRecord(Ttf_FileHandleT tFile) {
			Ttf_LeaveState(tFile, time, nid, tid, stateToken);
		}

	private:
		int nid;
		int tid;
		int stateToken;
};


class TriggerRecord : public Record {
	public:
		TriggerRecord(double time, int nid, int tid, int userEventToken, x_uint64 value) : Record(time) {
			this->nid = nid;
			this->tid = tid;
			this->userEventToken = userEventToken;
			this->value = value;
		}
		
		void writeRecord(Ttf_FileHandleT tFile) {
			Ttf_EventTrigger(tFile, time, nid, tid, userEventToken, value )	;
		}
	
	private:
		int nid;
		int tid;
		int userEventToken;
		double value;
};



class MessageRecord : public Record {
	public:
		MessageRecord(double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken, unsigned int messageSize, unsigned int messageTag, int messageComm, bool send) : Record(time) {
			this->sourceNid = sourceNodeToken;
			this->sourceTid = sourceThreadToken;
			this->destNid = destinationNodeToken;
			this->destTid = destinationThreadToken;
			this->size = messageSize;
			this->tag = messageTag;
			this->comm = messageComm;
			this->send = send;
		}

		void writeRecord(Ttf_FileHandleT tFile) {
		if (send) {
				Ttf_SendMessage(tFile, time, sourceNid, sourceTid, destNid, destTid, size, tag, comm);
			} else {
				Ttf_RecvMessage(tFile, time, sourceNid, sourceTid, destNid, destTid, size, tag, comm);
			}
		}
		
	private:
		unsigned int sourceNid, sourceTid, destNid, destTid, size, tag, comm;
		bool send;
	
	
};


class Trace {


  public:
    Trace(Ttf_FileHandleT tFile) : esr(0,0,0,0), lsr(0,0,0,0), tr(0,0,0,0,0), mr(0,0,0,0,0,0,0,0,0)
{
    this->tFile = tFile;
	this->finished = false;
	this->lastRecord = NULL;
    callbacks.DefClkPeriod = NULL;
    callbacks.DefThread = DefThread;
    callbacks.DefStateGroup = DefStateGroup;
    callbacks.DefState = DefState;
    callbacks.DefUserEvent = DefUserEvent;
    callbacks.EventTrigger = EventTrigger;
    callbacks.EndTrace = EndTrace;
    callbacks.EnterState = EnterState;
    callbacks.LeaveState = LeaveState;
    callbacks.SendMessage = SendMessage;
    callbacks.RecvMessage = RecvMessage;

} 

Record* getLastRecord() {
	return lastRecord;	
}

bool isFinished() {
	return finished;	
}

int readNumEvents(int num) {
	printf ("%p: reading data\n", this);
	callbacks.UserData = this; // must do this here, otherwise we might have the wrong pointer
    int recs_read = Ttf_ReadNumEvents(tFile, callbacks, num);
    return recs_read;
}

int defThread(unsigned int nodeToken, unsigned int threadToken, const char *threadName) {
    printf("defThread!\n");
    nodeMap[pair<int,int> (nodeToken,threadToken)] = false;
    Ttf_DefThread(theFile, nodeToken, threadToken, threadName);
    return 0;
}

int defStateGroup(unsigned int stateGroupToken, const char *stateGroupName) {
    printf("defStateGroup : %d : %s\n", stateGroupToken, stateGroupName);

	map<string, int>::iterator it =	groupMap.find(stateGroupName);
	if (it == groupMap.end()) { // not found, add it
		int globalId = groupMap.size();
		groupMap[stateGroupName] = globalId;
		localGroupMap[stateGroupToken] = globalId;
		Ttf_DefStateGroup(theFile, stateGroupName, globalId);
	} else {
		localGroupMap[stateGroupToken] = (*it).second;
	}

    return 0;
}

int defState(unsigned int stateToken, const char *stateName, int stateGroupToken) {
    printf("defState : %d : %s\n", stateToken, stateName);

	map<string, int>::iterator it =	eventMap.find(stateName);
	if (it == eventMap.end()) { // not found, add it
		printf ("didn't find %s in the map\n", stateName);
		int globalId = eventMap.size()+1;
		eventMap[stateName] = globalId;
		localEventMap[stateToken] = globalId;
		
		string crappedName = stateName;
		crappedName = crappedName.substr(1,crappedName.size()-2);
		Ttf_DefState(theFile, globalId, crappedName.c_str(), localGroupMap[stateGroupToken]);
	} else {
		printf ("found %s in the map\n", stateName);
		localEventMap[stateToken] = (*it).second;
	}

    
    return 0;
}


int defUserEvent(unsigned int userEventToken, const char *userEventName, int monotonicallyIncreasing) {
    printf("defUserEvent : %d : %s\n", userEventToken, userEventName);

	map<string, int>::iterator it =	userEventMap.find(userEventName);
	if (it == userEventMap.end()) { // not found, add it
		int globalId = userEventMap.size()+1;
		userEventMap[userEventName] = globalId;
		localUserEventMap[userEventToken] = globalId;
		
		string crappedName = userEventName;
		crappedName = crappedName.substr(1,crappedName.size()-2);
		Ttf_DefUserEvent(theFile, globalId, crappedName.c_str(), monotonicallyIncreasing);
	} else {
		localUserEventMap[userEventToken] = (*it).second;
	}

    return 0;
}

int eventTrigger(double time, unsigned int nodeToken, unsigned int threadToken, unsigned int userEventToken,
		 //double userEventValue)
		 long long userEventValue) {
    printf("eventTrigger!\n");
    tr = TriggerRecord(time, nodeToken, threadToken, userEventToken, userEventValue);
    lastRecord = &tr;
    return 0;
}
int sendMessage(double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken, unsigned int messageSize, unsigned int messageTag, 
		unsigned int messageComm) {
    printf("sendMessage!\n");
    mr = MessageRecord(time, sourceNodeToken, sourceThreadToken, destinationNodeToken, 
				       destinationThreadToken, messageSize, messageTag, messageComm, true);
    lastRecord = &mr;
    return 0;
}

int recvMessage(double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken, unsigned int messageSize, unsigned int messageTag, 
		unsigned int messageComm) {
    printf("recvMessage!\n");
    mr = MessageRecord(time, sourceNodeToken, sourceThreadToken, destinationNodeToken, 
				       destinationThreadToken, messageSize, messageTag, messageComm, false);
    lastRecord = &mr;
    return 0;
}
int enterState(double time, unsigned int nodeid, unsigned int tid, unsigned int stateid) {
	printf("EnterState callback: state=%d, nid=%d, tid=%d\n", stateid,nodeid,tid);
    esr = EnterStateRecord(time,nodeid,tid,localEventMap[stateid]);
    lastRecord = &esr;
    return 0;
}
int leaveState(double time, unsigned int nodeid, unsigned int tid, unsigned int stateid) {
    printf("leaveState!\n");
    lsr = LeaveStateRecord(time,nodeid,tid,localEventMap[stateid]);
	lastRecord = &lsr;
    return 0;
}

int endTrace(unsigned int nodeToken, unsigned int threadToken) {
    printf("%p: endTrace!\n",this);
    nodeMap[pair<int,int> (nodeToken,threadToken)] = true;

	map< pair<int, int> , bool, less< pair<int,int> > >::iterator it;
	printf ("map size = %d\n", nodeMap.size());
	finished = true;
	for (it = nodeMap.begin(); it != nodeMap.end(); it++) {
		if ((*it).second == false) {
			finished = false;	
		}
	}
	return 0;
}


int writeRecord(Ttf_FileHandleT tFile) {
	printf ("%p: writing data\n", this);
	lastRecord->writeRecord(tFile);
	lastRecord = NULL;	
	return 0;
}

private:

	Ttf_FileHandleT tFile;
	Ttf_CallbacksT callbacks;
	Record *lastRecord;
	bool finished;

	EnterStateRecord esr;
	LeaveStateRecord lsr;
	TriggerRecord tr;
	MessageRecord mr;

	map< pair<int, int> , bool, less< pair<int,int> > > nodeMap;

	map< int, int > localEventMap; // map local event id's to global
	map< int, int > localGroupMap; // map local group id's to global
	map< int, int > localUserEventMap; // map local event id's to global

};


int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken, const char *threadName)
{
    return ((Trace *) userData)->defThread(nodeToken, threadToken, threadName);
}
int DefStateGroup(void *userData, unsigned int stateGroupToken, const char *stateGroupName)
{
    return ((Trace *) userData)->defStateGroup(stateGroupToken, stateGroupName);
}

int DefState(void *userData, unsigned int stateToken, const char *stateName, unsigned int stateGroupToken)
{
    return ((Trace *) userData)->defState(stateToken, stateName, stateGroupToken);
}


int DefUserEvent(void *userData, unsigned int userEventToken,
		 const char *userEventName, int monotonicallyIncreasing)
{
    return ((Trace *) userData)->defUserEvent(userEventToken, userEventName, monotonicallyIncreasing);
}
int EventTrigger(void *userData, double time,
		 unsigned int nodeToken, unsigned int threadToken, unsigned int userEventToken,
		 //double userEventValue)
		 long long userEventValue)
{
    return ((Trace *) userData)->eventTrigger(time, nodeToken, threadToken, userEventToken, userEventValue);

}
int SendMessage(void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken, unsigned int messageSize, unsigned int messageTag, unsigned int messageComm)
{
    return ((Trace *) userData)->sendMessage(time, sourceNodeToken, sourceThreadToken, destinationNodeToken,
					     destinationThreadToken, messageSize, messageTag, messageComm);
}
int RecvMessage(void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken, unsigned int messageSize, unsigned int messageTag, unsigned int messageComm)
{
    return ((Trace *) userData)->recvMessage(time, sourceNodeToken, sourceThreadToken, destinationNodeToken,
					     destinationThreadToken, messageSize, messageTag, messageComm);
}

int EnterState(void *userData, double time, unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
    return ((Trace *) userData)->enterState(time, nodeid, tid, stateid);
}
int LeaveState(void *userData, double time, unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
    return ((Trace *) userData)->leaveState(time, nodeid, tid, stateid);
}
int EndTrace(void *userData, unsigned int nodeToken, unsigned int threadToken)
{
    return ((Trace *) userData)->endTrace(nodeToken, threadToken);
}


bool dontBlock = false;
bool adjustToZero = false;


void usage(char *name)
{
    fprintf(stderr,
	    "Usage: %s [-a] [-n] [-e eventedf*] [-m mergededf] inputtraces* (outputtrace|-) \n\n", name);
    fprintf(stderr, "Options:\n\n");
    fprintf(stderr, "-a : adjust first time to zero\n");
    fprintf(stderr, "-n : do not block waiting for new events. Offline merge\n");
    fprintf(stderr, "-e <files> : provide a list of event definition files corresponding to traces\n");
    fprintf(stderr, "-m <mergededf> : specify the name of the merged event definition file\n\n");
    fprintf(stderr, "Note: %s assumes edf files are named events.<nodeid>.edf and \n", name);
    fprintf(stderr, "      generates a merged edf file tau.edf unless -m is specified\n\n");
    fprintf(stderr, "e.g., > %s tautrace.*.trc app.trc\n", name);
    fprintf(stderr, "e.g., > %s -e events.[0-255].edf -m ev0_255merged.edf tautrace.[0-255].*.trc app.trc\n\n",
	    name);
    exit(1);
}


int getNodeId(char *name)
{
    // name is something like tautrace.32.0.4.trc
    // we want to return 32

    char buffer[1024];
    unsigned int i = 0;
    while (i < strlen(name) && name[i] != '.') {
	i++;
    }
    i++;
    int bufpos = 0;
    while (i < strlen(name) && name[i] != '.') {
	buffer[bufpos++] = name[i];
	i++;
    }
    buffer[bufpos] = 0;

    return atoi(buffer);
}

int cannot_get_enough_fd(int need)
{
#ifdef TAU_WINDOWS
    return false;		// no getdtablesize() in windows
#else
# if defined(__hpux) || defined(sun)
    /* -- system supports get/setrlimit (RLIMIT_NOFILE) -- */
    struct rlimit rlp;

    getrlimit(RLIMIT_NOFILE, &rlp);
    if (rlp.rlim_max < need) {
	return (TRUE);
    } else if (rlp.rlim_cur < need) {
	rlp.rlim_cur = need;
	setrlimit(RLIMIT_NOFILE, &rlp);
    }
    return (FALSE);
# else
#   if defined(_SEQUENT_) || defined(sequent)
    /* -- system provides get/setdtablesize -- */
    int max = getdtablesize();
    return ((max < need) && (setdtablesize(need) != need));
#   else
    /* -- system provides only getdtablesize -- */
    int max = getdtablesize();
    return (max < need);
#   endif
# endif
#endif				/* TAU_WINDOWS */
}


extern char *optarg;
extern int optind;
char **edfNames;		/* if they're specified */
int edfCount = 0;
bool edfSpecified = false;

int main(int argc, char *argv[])
{

    char *mergedEdfFile = "tau.edf";	/* initialize it */

    int numEdfProcessed;
    int i;
    while ((i = getopt(argc, argv, "ane:m:")) != EOF) {
	switch (i) {
	case 'a':		/* -- adjust first time to zero -- */
	    adjustToZero = true;
	    break;


	case 'e':		/* -- EDF files specified on the commandline -- */
	    edfSpecified = true;
	    numEdfProcessed = 0;
	    edfNames = (char **) malloc(argc * (sizeof(char *)));
	    for (i = optind - 1; i < argc; i++) {
		if (strstr(argv[i], ".edf") != 0) {
		    //open_edf_file(argv[i], numedfprocessed, TRUE);
		    numEdfProcessed++;
		    /* store the name of the edf file so that we can re-open 
		     * it later if event files need to be re-read */
		    edfNames[edfCount] = strdup(argv[i]);
		    edfCount++;	/* increment the count */
		} else {
		    break;	/* come out of the loop! */
		}
	    }
	    optind += numEdfProcessed - 1;
	    break;

	case 'm':		/* -- name of the merged edf file (instead of tau.edf) */
	    mergedEdfFile = strdup(argv[optind - 1]);
	    break;

	case 'n':		/* -- do not block for records at end of trace -- */
	    dontBlock = true;
	    break;

	default:		/* -- ERROR -- */
	    usage(argv[0]);
	    break;

	}
    }



    int active = argc - optind - 1;
    if (cannot_get_enough_fd(active + 4)) {
	fprintf(stderr, "%s: too many input traces:\n", argv[0]);
	fprintf(stderr, "  1. merge half of the input traces\n");
	fprintf(stderr, "  2. merge other half and output of step 1\n");
	fprintf(stderr, "  Or use, \"tau_treemerge.pl\"\n");
	exit(1);
    }


	printf ("ok...\n");
	theFile = Ttf_OpenFileForOutput("out.trc","out.edf");
	if (theFile == NULL) {
		fprintf (stderr, "Error opening	trace for output\n");
		exit(1);
	}


    vector < Trace* > traces;

    // get all the traces and edfs setup
    for (i = optind; i < argc - 1; i++) {
		char *traceFilename = argv[i];
		char *edfFilename;
		if (edfSpecified) {
		    if (traces.size() >= edfCount) {
				fprintf(stderr, "Error: Not enough EDF files were specified!\n");
				exit(1);
		    }
		    edfFilename = edfNames[traces.size()];
		} else {

	    	edfFilename = (char *) malloc(2048);
		    snprintf(edfFilename, 2048,  "events.%d.edf", getNodeId(traceFilename));
		}

		printf("processing trace: %s, with edf = %s\n", traceFilename, edfFilename);

		Ttf_FileHandleT tFile = Ttf_OpenFileForInput(traceFilename, edfFilename);
		Ttf_SetSubtractFirstTimestamp(tFile, false);

		if (tFile == NULL) {
		    fprintf(stderr, "Error opening trace: %s, with edf = %s\n", traceFilename, edfFilename);
		    exit(1);
		}

		Trace *trace = new Trace(tFile);

		while (trace->getLastRecord() == NULL) {
			trace->readNumEvents(1);
		}
		traces.push_back(trace);

    }

	printf ("Here now!...\n");

    if (traces.size() < 1) {
		usage(argv[0]);
    }


	bool allFinished = false;

	while (!allFinished) {
//		printf ("processing...\n");
		double lowestTime = DBL_MAX;
		int lowestIndex = -1;
		
		for (unsigned int i=0; i < traces.size(); i++) {
			if (!traces[i]->isFinished()) {
				double time = traces[i]->getLastRecord()->getTime();
				if (time < lowestTime) {
					lowestTime = time;
					lowestIndex = i;	
				}
			}
		}
		
		if (lowestIndex == -1) {
			allFinished = true;	
		} else {
	//			printf ("lowest time with %d\n", lowestIndex);
			traces[lowestIndex]->writeRecord(theFile);
			while (traces[lowestIndex]->getLastRecord() == NULL && !traces[lowestIndex]->isFinished()) {
				traces[lowestIndex]->readNumEvents(1);
			}
		}

/*
		allFinished = true;
		for (unsigned int i=0; i < traces.size(); i++) {
			if (!traces[i].isFinished()) {
				allFinished = false;	
			}
		}		
		*/
	}
	
	Ttf_CloseOutputFile(theFile);

	printf ("done!\n");
}
