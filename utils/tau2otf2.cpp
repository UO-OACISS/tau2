/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
**			http://www.paratools.com                           **
*****************************************************************************
**    Copyright 2005  						   	   **
**    ParaTools, Inc.                                                      **
****************************************************************************/
/***************************************************************************
**	File 		: tau2otf2.cpp 					  **
**	Description 	: TAU to OTF translator                           **
**	Author		: Sameer Shende, Wyatt Spear					  **
**	Contact		: sameer@paratools.com   	                  **
***************************************************************************/
#include <TAU_tf.h>
#include <stdio.h>
#include <iostream>
#include <stddef.h>
#include <otf2/otf2.h> /* OTF header file */
#include <map>
#include <unordered_map>
#include <vector>
#include <stack>
#include <inttypes.h>
#include <cstring>
#include <cstdlib>
using namespace std;
int debugPrint = 0;
int remoteThread = -1;
bool multiThreaded = false;
uint32_t string_id = 0;
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
#define TAU_MULT 10000

static inline int
localmax(int a,int b){
    if(a>b)return a;
    return b;
}static inline void
check_pointer
(void* pointer,
const char* description
){
    if ( pointer == NULL )
    {
        printf( "\nERROR: %s\n\n", description );
        exit( EXIT_FAILURE );
    }
}/*
static inline void
otf2_get_parameters
(    int    argc,
    char** argv
);

static inline void
check_pointer
(    void* pointer,
    char* description
);
*/
static inline void
check_status
(OTF2_ErrorCode status,
const char*             description
);

static OTF2_FlushType
pre_flush
(void*         userData,
OTF2_FileType fileType,
uint64_t      locationId,
void*         callerData,
bool          final
);

static OTF2_TimeStamp
post_flush
(void*         userData,
OTF2_FileType fileType,
uint64_t      locationId
);

static OTF2_FlushCallbacks flush_callbacks;
/*=
{    .otf2_pre_flush  = pre_flush,
    .otf2_post_flush = post_flush
};*/

static uint64_t
get_time
(void
);

uint64_t* locations;//[ locations ];

uint64_t TauGetClockTicksInGHz(double time)
{    return (uint64_t) (time * TAU_MULT);
}/* any unique id */

/* implementation of callback routines */
map< pair<int,int>, int, less< pair<int,int> > > EOF_Trace;
map< int,int, less<int > > numthreads;
/* numthreads[k] is no. of threads in rank k */

unordered_map<int,int> regionMap;
int regionIDTracker=0;
unordered_map<int,int> metricMap;
int metricIDTracker=0;
unordered_map<int,int> groupMap;
int groupIDTracker=0;


int EndOfTrace = 0;  /* false */
int location_count = 0;

/* empty string definition */
uint32_t STRING_EMPTY=0;
//enum
//{
//   STRING_EMPTY
//};
enum
{    COUNTS
};

/* definition IDs for MPI comms */
enum
{    MPI_COMM_MPI_COMM_WORLD,
    MPI_COMM_MPI_COMM_SELF
};

/* definition IDs for Groups */
enum
{    GROUP_MPI_LOCATIONS,
    GROUP_MPI_COMM_WORLD,
    GROUP_MPI_COMM_SELF,
    GROUP_ALL_LOCATIONS
};

/* definition IDs for metric classes and instances */
enum
{    METRIC_CLASS_1,
    METRIC_CLASS_2,
    METRIC_CLASS_3,
    METRIC_CLASS_4,
    METRIC_INSTANCE_1,

    NUM_OF_CLASSES
};

/* Define limits of sample data (user defined events) */
/*
struct {
unsigned long long umin;
unsigned long long umax;
} taulongbounds = { 0, (unsigned long long)~(unsigned long long)0 };

struct {
double fmin;
double fmax;
} taufloatbounds = {-1.0e+300, +1.0e+300};
*/
/* These structures are used in user defined event data */

/* Global data */
int sampgroupid = 1;
int sampclassid = 2;
vector<stack <unsigned int> > callstack;
int *offset = 0;

struct group {
    const char* name;
    unsigned int groupid;
    vector < uint64_t > members;
};

//Map of groupid to group object with other invo
map< unsigned int, group > groups;

struct source_info{
    bool sourceFound;
    unsigned int startline;
    unsigned int stopline;
    const char* event_name;
    const char* sourcefile;
    source_info(){
        startline=0;
        stopline=0;
        event_name="";
        sourcefile="";
        sourceFound=false;
    }
};

int maxTauStringId=0;

void tau_trim(char * s) {
    char * p = s;
    int l = strlen(p);

    while((l > 0) && isspace(p[l - 1])) p[--l] = 0;
    while(* p && isspace(* p)) ++p, --l;

    memmove(s, p, l + 1);
}

char* tau_strdup(const char* in_string) {
    // add one more character for the null terminator
    int length = strlen(in_string) + 1;
    char* new_string = (char*)calloc(length, sizeof(char));
    strncpy(new_string,  in_string, length); 
    return new_string;
}

source_info parseSourceInfo(const char * name){
    source_info info = source_info();

    const char* throttled = "[THROTTLED]";
    const char* openmp = "[OpenMP]";
    const char* openmplocation = "[OpenMP location:";
    char* working = tau_strdup(name);
    //char* short_name = "";
    // printf("'%s'\n", working);
    // parse the components out of the timer name
    if (strstr(name, throttled) != NULL) {
        // 'MPI_Irecv() [THROTTLED]'
        // special case, handle it
        int length = strlen(name) - 11;
        working[length] = 0; // new terminator
        return parseSourceInfo(working);
    }  else if (strstr(name, openmp) != NULL) {
        // 'barrier enter/exit [OpenMP]'
        // special case, handle it
        info.event_name = tau_strdup(name);
    } else if (strstr(name, openmplocation) != NULL) {
        // 'paralleldo [OpenMP location: file:/global/u2/k/khuck/src/XGC-1_CPU/pushe2.F95 <72, 150>]'
        // special case, handle it
        char* tmp = strtok(working, " ");
        tau_trim(tmp);
        info.event_name = tau_strdup(tmp);
        tmp = strtok(NULL, ":");
        tmp = strtok(NULL, ":");
        tmp = strtok(NULL, " ");
        info.sourcefile = tau_strdup(tmp);//timer->source_file
        tmp = strtok(NULL, " <,>]");
        info.startline = atoi(tmp);//timer->line_number
        tmp = strtok(NULL, " <,>]");
        info.stopline = atoi(tmp);//timer->line_number_end
    } else if (strstr(name, "[") != NULL && strstr(name, "}") != NULL) {
        // regular case
        // get the function signature
        char* tmp = strtok(working, "[");
        tau_trim(tmp);
        info.event_name = tau_strdup(tmp);
        // get the filename
        tmp = strtok(NULL, "}");
        info.sourcefile = tau_strdup(tmp+1);//timer->source_file
        // get the line and column numbers
        tmp = strtok(NULL, " {},-");
        info.startline = atoi(tmp);//timer->line_number
        tmp = strtok(NULL, " {},-");
        //timer->column_number = atoi(tmp);
        tmp = strtok(NULL, " {},-");
        info.stopline = atoi(tmp);//timer->line_number_end
        tmp = strtok(NULL, " {},-");
        //timer->column_number_end = atoi(tmp);
    } else {
        // simple case.
        info.event_name = tau_strdup(name);
    }
    free(working);

    //info.event_name=short_name;

    return info;
}

/* FIX GlobalID so it takes into account numthreads */
/* utilities */
int GlobalId(int localnodeid, int localthreadid)
{    if (multiThreaded) /* do it for both single and multi-threaded */
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
    {  /* OTF node nos run from 1 to N, TAU's run from 0 to N-1 */
        return localnodeid;
    }
}

bool firstState=true;
double firstRealTime=0;

/* implementation of callback routines */
/***************************************************************************
* Description: EnterState is called at routine entry by trace input library
* 		This is a callback routine which must be registered by the
* 		trace converter.
***************************************************************************/
int EnterState(void *userData, double time,
unsigned int nid, unsigned int tid, unsigned int tauStateid)
{    int cpuid = GlobalId(nid, tid);

    int stateid=regionMap[tauStateid];

    dprintf("Entered state %d time %g cpuid %d\n",
    stateid, time, cpuid);

    if (cpuid >= (int) callstack.size()+1)
    {
        fprintf(stderr, "ERROR: tau2otf: EnterState() cpuid %d exceeds callstack size %d\n", cpuid, (int)callstack.size());
        exit(1);
    }

    callstack[cpuid].push(stateid);

    /* OLD :
OTF_Writer_writeDownto((OTF_Writer*)userData, TauGetClockTicksInGHz(time), stateid, cpuid, TAU_SCL_NONE);
*/
    //OTF2_EvtWriter_Enter((OTF2_EvtWriter*)userData, TauGetClockTicksInGHz(time), stateid, cpuid, TAU_SCL_NONE);
    OTF2_AttributeList* attributes = NULL;

    OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter((OTF2_Archive_struct*)userData, locations[ numthreads[nid] * nid + tid ] );

    OTF2_EvtWriter_Enter(evt_writer, attributes, TauGetClockTicksInGHz(time),stateid);

    if(firstState){
        firstState=false;
        firstRealTime=TauGetClockTicksInGHz(time);
    }

    return 0;
}

double lastt=0;

/***************************************************************************
* Description: EnterState is called at routine exit by trace input library
* 		This is a callback routine which must be registered by the
* 		trace converter.
***************************************************************************/
int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int tauStatetoken)
{    
    int statetoken=regionMap[tauStatetoken];
    int cpuid = GlobalId(nid, tid);
    if(callstack[cpuid].size()<1){
        printf("FAULT: Exit from empty stack on nid/tid/cpu: %d/%d/%d. Statetoken: %d\n", nid,tid,cpuid,statetoken);
        return 0;
    }
    int stateid = callstack[cpuid].top();
    callstack[cpuid].pop();

    dprintf("Leaving state %d time %g cpuid %d \n", stateid, time, cpuid);

    /* OLD:
OTF_Writer_writeUpfrom((OTF_Writer*)userData, TauGetClockTicksInGHz(time), stateid, cpuid, TAU_SCL_NONE);
*/

    OTF2_AttributeList* attributes = NULL;

    OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter((OTF2_Archive_struct*)userData, locations[ numthreads[nid] * nid + tid ] );

    /* we can write stateid = 0 if we don't need stack integrity checking */
    OTF2_EvtWriter_Leave(evt_writer, attributes, TauGetClockTicksInGHz(time), statetoken);
    
    return 0;
}

/***************************************************************************
* Description: ClockPeriod (in microseconds) is specified here.
* 		This is a callback routine which must be registered by the
* 		trace converter.
***************************************************************************/
int ClockPeriod( void*  userData, double clkPeriod )
{    dprintf("Clock period %g\n", clkPeriod);
    //OTF2_EvtWriter_writeDefTimerResolution((OTF2_EvtWriter*)userData, TAU_GLOBAL_STREAM_ID, TauGetClockTicksInGHz(1/clkPeriod));

    return 0;
}

/***************************************************************************
* Description: DefThread is called when a new nodeid/threadid is encountered.
* 		This is a callback routine which must be registered by the
* 		trace converter.
***************************************************************************/

int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{    dprintf("DefThread nid %d tid %d, thread name %s\n",
    nodeToken, threadToken, threadName);
    EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 0; /* initialize it */
    numthreads[nodeToken] = numthreads[nodeToken] + 1;
    if (threadToken > 0) multiThreaded = true;

    location_count++;
    return 0;
}

/***************************************************************************
* Description: EndTrace is called when an EOF is encountered in a tracefile.
* 		This is a callback routine which must be registered by the
* 		trace converter.
***************************************************************************/
int EndTrace( void *userData, unsigned int nodeToken, unsigned int threadToken)
{    dprintf("EndTrace nid %d tid %d\n", nodeToken, threadToken);
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
int DefStateGroup( void *userData, unsigned int tauStateGroupToken,
const char *stateGroupName )
{    
    int stateGroupToken=groupIDTracker;
    groupMap[tauStateGroupToken]=groupIDTracker++;
    dprintf("StateGroup groupid %d, group name %s\n", stateGroupToken,
    stateGroupName);

    groups[stateGroupToken]=group();
    groups[stateGroupToken].groupid=stateGroupToken;
    groups[stateGroupToken].name=stateGroupName;
    return 0;
}

OTF2_ErrorCode status;
static inline void check_status (OTF2_ErrorCode status, const char* description){
    if ( status != OTF2_SUCCESS )
    {
        printf( "\nERROR: %s\n\n", description );
        exit( EXIT_FAILURE );
    }
}

/***************************************************************************
* Description: DefState is called to define a new symbol (event). It uses
*		the token used to define the group identifier.
* 		This is a callback routine which must be registered by the
* 		trace converter.
***************************************************************************/
int DefState( void *userData, unsigned int tauStateToken, const char *stateName,
unsigned int tauStateGroupToken )
{    
    int stateToken=regionIDTracker;
    regionMap[tauStateToken]=regionIDTracker++;
    int stateGroupToken=groupMap[tauStateGroupToken];
    
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

    OTF2_GlobalDefWriter* glob_def_writer = (OTF2_GlobalDefWriter*)userData;
    //glob_def_writer = OTF2_Archive_GetGlobalDefWriter( archive );

    OTF2_GlobalDefWriter_WriteString( glob_def_writer,
    string_id,
    name );
    int rawnameid = string_id;
    string_id++;

    source_info info = parseSourceInfo(name);

    OTF2_GlobalDefWriter_WriteString( glob_def_writer,
    string_id,
    info.event_name );
    int shortnameid = string_id;
    string_id++;

    OTF2_GlobalDefWriter_WriteString( glob_def_writer,
    string_id,
    info.sourcefile );
    int sourceid = string_id;
    string_id++;

    maxTauStringId=localmax(maxTauStringId,stateToken);
    status = OTF2_GlobalDefWriter_WriteRegion( glob_def_writer,
    stateToken,

    rawnameid,
    shortnameid,
    STRING_EMPTY,
    OTF2_REGION_ROLE_UNKNOWN,
    OTF2_PARADIGM_UNKNOWN,
    OTF2_REGION_FLAG_NONE,
    sourceid,  //Source
    info.startline,  //Begin
    info.stopline );  //End
    check_status( status, "Write region definition" );

    map< unsigned int, group >::iterator it = groups.find(stateGroupToken);
    if(it==groups.end()){
        printf("Warning: event %s (id %d) has undefined group id %d\n", name,stateToken,stateGroupToken);
    }
    else{
        groups[stateGroupToken].members.push_back(stateToken);
    }

    //OTF2_EvtWriter_writeDefFunction((OTF2_EvtWriter*)userData, TAU_GLOBAL_STREAM_ID, stateToken, (const char *) name, stateGroupToken, TAU_SCL_NONE);

    return 0;
}

//int NUM_OF_CLASSES = 1;
/***************************************************************************
* Description: DefUserEvent is called to register the name and a token of the
*  		user defined event (or a sample event in Vampir terminology).
* 		This is a callback routine which must be registered by the
* 		trace converter.
***************************************************************************/
int DefUserEvent( void *userData, unsigned int tauUserEventToken,
const char *userEventName , int monotonicallyIncreasing)
{   
    int userEventToken=metricIDTracker;
    metricMap[tauUserEventToken]=metricIDTracker++;
    dprintf("DefUserEvent event id %d user event name %s\n", userEventToken,
    userEventName);
    //int dodifferentiation;

    /* We need to remove the backslash and quotes from "\"funcname\"" */
    char *name = strdup(userEventName);
    int len = strlen(name);
    if ((name[0] == '"' ) && (name[len-1] == '"'))
    {
        name += 1;
        name[len-2] = '\0';
    }

    OTF2_GlobalDefWriter* glob_def_writer = (OTF2_GlobalDefWriter*)userData;
    //glob_def_writer = OTF2_Archive_GetGlobalDefWriter( archive );

    OTF2_GlobalDefWriter_WriteString( glob_def_writer,
    string_id,
    name );
    maxTauStringId=localmax(maxTauStringId,userEventToken);

    /* create a state record */
    if (monotonicallyIncreasing)
    {
        //printf("MONO %s",name);
        //dodifferentiation = 1; /* for hw counter data */
        OTF2_GlobalDefWriter_WriteMetricMember(glob_def_writer,userEventToken, string_id,  string_id,OTF2_METRIC_TYPE_PAPI,OTF2_METRIC_ACCUMULATED_START, OTF2_TYPE_UINT64,OTF2_BASE_DECIMAL,0,COUNTS);
        OTF2_MetricMemberRef* omr = new OTF2_MetricMemberRef[1];
        omr[0]=userEventToken;

        OTF2_GlobalDefWriter_WriteMetricClass (    glob_def_writer, userEventToken, 1, omr,    OTF2_METRIC_SYNCHRONOUS_STRICT, OTF2_RECORDER_KIND_CPU);

    }
    else
    { /* for non monotonically increasing data */
        //dodifferentiation = 0; /* for TAU user defined events */
        //printf("ATO %s",name);
        OTF2_GlobalDefWriter_WriteMetricMember(glob_def_writer,userEventToken, userEventToken,  userEventToken,OTF2_METRIC_TYPE_OTHER,OTF2_METRIC_ABSOLUTE_POINT, OTF2_TYPE_UINT64,OTF2_BASE_DECIMAL,0,COUNTS);
        OTF2_MetricMemberRef* omr = new OTF2_MetricMemberRef[1];
        omr[0]=userEventToken;

        OTF2_GlobalDefWriter_WriteMetricClass (    glob_def_writer, userEventToken, 1, omr,    OTF2_METRIC_SYNCHRONOUS_STRICT, OTF2_RECORDER_KIND_CPU);

        /* NOTE: WE DO NOT HAVE THE DO DIFFERENTIATION PARAMETER YET IN OTF */
    }

    /*
//TODO: Check this out
status = OTF2_GlobalDefWriter_WriteMetricClass( glob_def_writer,
                                                    METRIC_CLASS_1,
                                                    number_of_members_in_class_1,
                                                    metric_members_of_class_1,
                                                    OTF2_METRIC_SYNCHRONOUS_STRICT );
    check_status( status, "Write metric class definition." );

NUM_OF_CLASSES+=1;
*/
    string_id++;
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
unsigned int tauUserEventToken,
long long userEventValue)
{    
    int userEventToken=metricMap[tauUserEventToken];
    int cpuid = GlobalId (nid, tid); /* GID */
    dprintf("EventTrigger: time %g, cpuid %d event id %d triggered value %lld \n", time, cpuid, userEventToken, userEventValue);

    if(userEventToken==7004){
        remoteThread=(int) userEventValue;
    }

    /* write the sample data */
    //OTF2_EvtWriter_writeCounter((OTF2_EvtWriter*)userData, TauGetClockTicksInGHz(time), cpuid, userEventToken, userEventValue);
    OTF2_MetricValue* omv = new OTF2_MetricValue[1];
    omv[0].unsigned_int=userEventValue;

    OTF2_Type* omt = new OTF2_Type[1];
    omt[0]=OTF2_TYPE_UINT64;

    OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter((OTF2_Archive_struct*)userData, locations[ numthreads[nid] * nid + tid ] );
    OTF2_EvtWriter_Metric( evt_writer,
    NULL,
    TauGetClockTicksInGHz(time),
    userEventToken,
    1,
    omt,
    omv );
    
    lastt=TauGetClockTicksInGHz(time);
    
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
{    if(remoteThread>-1)
    {
        destinationThreadToken=remoteThread;
        remoteThread=-1;
    }

    int source = GlobalId(sourceNodeToken, sourceThreadToken);
    int dest   = GlobalId(destinationNodeToken, destinationThreadToken);

    dprintf("SendMessage: time %g, source cpuid %d , destination cpuid %d, size %d, tag %d\n",
    time,
    source, dest,
    messageSize, messageTag);

    //OTF2_EvtWriter_writeSendMsg((OTF2_EvtWriter*)userData, TauGetClockTicksInGHz(time), source, dest, TAU_DEFAULT_COMMUNICATOR, messageTag, messageSize, TAU_SCL_NONE);
    OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter((OTF2_Archive_struct*)userData, locations[ numthreads[sourceNodeToken] * sourceNodeToken + sourceThreadToken ] );
    OTF2_EvtWriter_MpiSend( evt_writer,
    NULL,
    TauGetClockTicksInGHz(time),
    dest,
    TAU_DEFAULT_COMMUNICATOR,
    messageTag,
    messageSize );

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
{    if(remoteThread>-1)
    {
        sourceThreadToken=remoteThread;
        remoteThread=-1;
    }

    int source = GlobalId(sourceNodeToken, sourceThreadToken);
    int dest   = GlobalId(destinationNodeToken, destinationThreadToken);

    dprintf("RecvMessage: time %g, source cpuid %d, destination cpuid %d, size %d, tag %d\n",
    time,
    source, dest,
    messageSize, messageTag);

    //OTF2_EvtWriter_writeRecvMsg((OTF2_EvtWriter*)userData, TauGetClockTicksInGHz(time), dest, source, TAU_DEFAULT_COMMUNICATOR, messageTag, messageSize, TAU_SCL_NONE);
    OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter((OTF2_Archive_struct*)userData, locations[ numthreads[destinationNodeToken] * destinationNodeToken + destinationThreadToken ] );
    OTF2_EvtWriter_MpiRecv( evt_writer,
    NULL,
    TauGetClockTicksInGHz(time),
    source,
    TAU_DEFAULT_COMMUNICATOR,
    messageTag,
    messageSize );

    return 0;
}

/***************************************************************************
* Description: To clean up and reset the end of file marker, we invoke this.
***************************************************************************/
int ResetEOFTrace(void)
{    /* mark all entries of EOF_Trace to be false */
    for (map< pair<int,int>, int, less< pair<int,int> > >:: iterator it =
    EOF_Trace.begin(); it != EOF_Trace.end(); it++)
    { /* Explicilty mark end of trace to be not over */
        (*it).second = 0;
    }

    return 0;
}

static OTF2_FlushType pre_flush(void*         userData,
OTF2_FileType fileType,
uint64_t      locationId,
void*         callerData,
bool          final
){
    return OTF2_FLUSH;
}

static OTF2_TimeStamp post_flush(void*         userData,
OTF2_FileType fileType,
uint64_t      locationId){
    return get_time();
}

static uint64_t get_time
(void){
    static uint64_t sequence;
    return sequence++;
}

/***************************************************************************
* Description: The main entrypoint.
***************************************************************************/
int main(int argc, char **argv)
{    Ttf_FileHandleT fh;
    // OTF_FileManager* manager;
    OTF2_Archive* archive;
    int num_streams = 1;
    int num_nodes = -1;
    int recs_read;
    char *trace_file;
    char *edf_file;
    char *out_file = NULL;
    int no_message_flag=0;
    OTF2_Compression compress_flag = OTF2_COMPRESSION_NONE; /* by default do not compress traces */
    //OTF_FileCompression compression = OTF_FILECOMPRESSION_UNCOMPRESSED;
    int i;
    /* main program: Usage app <trc> <edf> [-a] [-nomessage] */
    if (argc < 4)
    {
        printf("Usage: %s <TAU trace> <edf file> <out file> [-n streams] [-nomessage]  [-z] [-v]\n",
        argv[0]);
        //    printf(" -n <streams> : Specifies the number of output streams (default 1)\n");
        printf(" -nomessage : Suppress printing of message information in the trace\n");
        printf(" -z : Enable compression of trace files. By default it is uncompressed.\n");
        printf(" -v         : Verbose\n");
        printf(" Trace format of <out file> is OTF \n");

        printf(" e.g.,\n");
        printf(" %s merged.trc tau.edf app\n", argv[0]);
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
                compress_flag = OTF2_COMPRESSION_ZLIB;
            }
            if (strcmp(argv[i], "-v") == 0)
            {
                debugPrint = 1;
            }
            break;
        }
    }
    /* Finished parsing commandline options, now open the trace file */

    fh = Ttf_OpenFileForInput( trace_file, edf_file);

    if (!fh)
    {
        printf("ERROR:Ttf_OpenFileForInput fails");
        exit(1);
    }

    dprintf("Using %d streams\n", num_streams);

    /* Create new archive handle. */
    archive = OTF2_Archive_Open( ".",
    out_file,
    OTF2_FILEMODE_WRITE,
    1024 * 1024,
    4 * 1024 * 1024,
    OTF2_SUBSTRATE_POSIX,
    compress_flag );
    check_pointer( archive, "Create archive" );

    //manager = OTF_FileManager_open(TAU_OTF_FILE_MANAGER_LIMIT);

    /* Define the file control block for output trace file */
    //void *fcb = (void *) OTF2_EvtWriter_open(out_file, num_streams, manager);

    check_pointer( archive, "Create archive" );

    OTF2_ErrorCode status;

    flush_callbacks.otf2_pre_flush  = pre_flush;
    flush_callbacks.otf2_post_flush = post_flush;
    /*=
{    .otf2_pre_flush  = pre_flush,
    .otf2_post_flush = post_flush
};*/

    status = OTF2_Archive_SetFlushCallbacks( archive, &flush_callbacks, NULL );
    check_status( status, "Set flush callbacks." );

    OTF2_Archive_SetSerialCollectiveCallbacks( archive );

    status = OTF2_Archive_SetDescription( archive, "Data converted from TAU trace output" );
    check_status( status, "Set description." );
    status = OTF2_Archive_SetCreator( archive, "tau2otf2 converter version 2.21.x" );
    check_status( status, "Set creator." );

    /* check and verify that it was opened properly
if (fcb == 0)
{    perror(out_file);
    exit(1);
}*/

    /* enble compression if it is specified by the user
if (compress_flag)
{    compression = OTF2_FILECOMPRESSION_COMPRESSED;
    OTF2_EvtWriter_setCompression((OTF2_EvtWriter *)fcb, compression);
}*/

    /* Write the trace file header

OTF2_EvtWriter_writeDefCreator((OTF2_EvtWriter *)fcb, TAU_GLOBAL_STREAM_ID, "");
OTF2_EvtWriter_writeDefCounterGroup((OTF2_EvtWriter *)fcb, TAU_GLOBAL_STREAM_ID, sampclassid, "TAU counter data");
*/

    OTF2_GlobalDefWriter* glob_def_writer = NULL;
    glob_def_writer = OTF2_Archive_GetGlobalDefWriter( archive );

    char name_buffer[ 64 ];

    snprintf( name_buffer, sizeof( name_buffer),  "%s" ,"" );
    OTF2_GlobalDefWriter_WriteString( glob_def_writer,string_id, name_buffer );
    STRING_EMPTY=string_id;
    string_id++;

    /*This doesn't seem to be necessary since we write our groups below...
    OTF2_GlobalDefWriter_WriteLocationGroup( glob_def_writer, 0, 1, OTF2_LOCATION_GROUP_TYPE_PROCESS, 0 );
*/

    int totalnidtids;

    if (num_nodes == -1) {
        /* in the first pass, we determine the no. of cpus and other group related
    * information. In the second pass, we look at the function entry/exits */

        Ttf_CallbacksT firstpass;
        /* In the first pass, we just look for node/thread ids and def records */
        firstpass.UserData = glob_def_writer;
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
        do {
            recs_read = Ttf_ReadNumEvents(fh,firstpass, 1024);
#ifdef DEBUG
            if (recs_read != 0)
            cout <<"Read "<<recs_read<<" records"<<endl;
#endif
        }
        while ((recs_read >0) && (!EndOfTrace));

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
    //unsigned int groupid = 1;
    //int tid = 0;
    int nodes = numthreads.size(); /* total no. of nodes */
    int *threadnumarray = new int[nodes];
    offset = new int[nodes+1];
    offset[0] = 0; /* no offset for node 0 */
    for (i=0; i < nodes; i++)
    { /* one for each node */
        threadnumarray[i] = numthreads[i];
        offset[i+1] = offset[i] + numthreads[i];
    }

    /*
unsigned int *cpuidarray = new unsigned int[totalnidtids]; // max
// next, we write the cpu name and a group name for node/threads
for (i=0; i < nodes; i++)
{    char name[32];
    for (tid = 0; tid < threadnumarray[i]; tid++)
    {
    sprintf(name, "node %d, thread %d", i, tid);
    int cpuid = GlobalId(i,tid);
    cpuidarray[tid] = cpuid;
    dprintf("Calling OTF2_EvtWriter_writeDefProcess cpuid %d name %s\n", cpuid, name);
    OTF2_EvtWriter_writeDefProcess((OTF2_EvtWriter *)fcb, TAU_GLOBAL_STREAM_ID, cpuid, name, TAU_NO_PARENT);
    }
    if (multiThreaded)
    { // define a group for these cpus only if it is a multi-threaded trace
    sprintf(name, "Node %d", i);
    groupid ++; // let flat group for samples take the first one
    // Define a group: threadnumarray[i] represents no. of threads in node

    OTF2_EvtWriter_writeDefProcessGroup((OTF2_EvtWriter *)fcb, TAU_GLOBAL_STREAM_ID, groupid, name, threadnumarray[i],
    (uint32_t*) cpuidarray);
    }
}delete[] cpuidarray;
*/

    /* Generate location IDs. Just to have non-consecutive location IDs. */
    //uint64_t locations[ location_count ];
    locations = new uint64_t[location_count];
    uint64_t mpi_ranks[ nodes ];
    //uint64_t master_threads[ nodes ];
    uint64_t master_threads2[ nodes ];

    for ( uint64_t rank = 0; rank < (uint64_t)nodes; rank++ )
    {
        for ( uint64_t thread = 0; thread < (uint64_t)numthreads[rank]; thread++ )
        {
            locations[ numthreads[rank] * rank + thread ] = GlobalId(rank,thread);// ( rank << 16 ) + thread;
        }
        mpi_ranks[ rank ]      = GlobalId(rank,0);//Just to make sure we have valid rank ids. formerly rank
        //master_threads[ rank ] = rank << 16;
    }

    /*We can't overlap string ids, so the first location id must be above the last tau event id*/
    //uint32_t string = maxTauStringId + 1;

    snprintf( name_buffer, sizeof( name_buffer),  "System" );//PRIu64
    status = OTF2_GlobalDefWriter_WriteString( glob_def_writer,
    string_id,
    name_buffer );
    check_status( status, "Write string definition." );

    OTF2_GlobalDefWriter_WriteSystemTreeNode (glob_def_writer, 0,  string_id,string_id, OTF2_UNDEFINED_UINT32);
    string_id++;

    /* Write location group and location definitions. */
    for ( uint64_t rank = 0; rank < (uint64_t)nodes; ++rank )
    {
        char name_buffer[ 64 ];

        snprintf( name_buffer, sizeof( name_buffer),  "Process %d" , (int)rank );//PRIu64
        status = OTF2_GlobalDefWriter_WriteString( glob_def_writer,
        string_id,
        name_buffer );
        check_status( status, "Write string definition." );

        status = OTF2_GlobalDefWriter_WriteLocationGroup( glob_def_writer,
        rank,
        string_id,
        OTF2_LOCATION_GROUP_TYPE_PROCESS,
        0
	#if OTF2_VERSION_MAJOR > 2
	,
        OTF2_UNDEFINED_LOCATION_GROUP
	#endif
	);//( rank / 2 ) + 1 );
        check_status( status, "Write location group definition." );
        string_id++;
        master_threads2[rank]=rank;

        for ( uint64_t thread = 0; thread < (uint64_t)numthreads[rank]; thread++ )
        {
            if(thread==0){
                if(nodes<=1)
                {
                    snprintf( name_buffer, sizeof( name_buffer),  "Master thread");
                }
                else{
                    snprintf( name_buffer, sizeof( name_buffer),  "Rank");
                }
            }
            else{
                snprintf( name_buffer, sizeof( name_buffer),  "Thread %d" ,(int)thread );
            }

            status = OTF2_GlobalDefWriter_WriteString( glob_def_writer,
            string_id,
            name_buffer );
            check_status( status, "Write string definition." );

            //OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter( archive, locations[ numthreads[rank] * rank + thread ] );

            //check_pointer( evt_writer, "Get event writer." );

            uint64_t num_events = -1;
            //status = OTF2_EvtWriter_GetNumberOfEvents( evt_writer, &num_events );
            check_status( status, "Get the numberof written events." );
            //uint64_t loc = locations[ rank * numthreads[rank] + thread ];
            uint64_t loc = GlobalId(rank,thread);
            dprintf("Writing location: %d\n",(int)loc);
            status = OTF2_GlobalDefWriter_WriteLocation( glob_def_writer,
            loc,
            string_id,
            OTF2_LOCATION_TYPE_CPU_THREAD,
            num_events, rank );
            check_status( status, "Write location definition." );
            string_id++;
        }
    }

    /*unsigned int *idarray = new unsigned int[totalnidtids];
    for (i = 0; i < totalnidtids; i++)
    { // assign i to each entry 
        idarray[i] = i+1;
    }*/

    /* create a callstack on each thread/process id */
    dprintf("totalnidtids  = %d\n", totalnidtids);
    //callstack = new stack<unsigned int> [totalnidtids]();
    callstack.resize(totalnidtids+1);

    /**Set up the state groups*/
    typedef std::map<unsigned int, group>::iterator group_type;
    for(group_type iterator = groups.begin(); iterator != groups.end(); iterator++) {

        const char* stateGroupName=iterator->second.name;
        unsigned int stateGroupToken=iterator->first;
        int nom = iterator->second.members.size();
        //vector<unsigned int> mv = iterator->second.members;
        uint64_t* membs=&iterator->second.members[0];

        dprintf("Writing group id: %d, name: %s to otf2\n",stateGroupToken,stateGroupName);

        OTF2_GlobalDefWriter_WriteString( glob_def_writer,string_id ,
        stateGroupName);
        maxTauStringId=localmax(maxTauStringId,stateGroupToken);

        /* create a default activity (group) */
        //OTF2_GlobalDefWriter_WriteGroup(glob_def_writer, stateGroupToken, stateGroupToken,OTF2_PARADIGM_UNKNOWN, OTF2_GROUP_FLAG_NONE,     OTF2_GROUP_TYPE_REGIONS,0,0);
        OTF2_GlobalDefWriter_WriteGroup(glob_def_writer, stateGroupToken, string_id, OTF2_GROUP_TYPE_REGIONS, OTF2_PARADIGM_UNKNOWN, OTF2_GROUP_FLAG_NONE, nom,membs);

        dprintf("Writing group %s, id %d to otf2\n",stateGroupName,stateGroupToken);

        string_id++;
    } //Setting up state groups

    /* Define group ids */
    //char name[1024];
    //strcpy(name, "TAU default group");
    //OTF2_EvtWriter_writeDefProcessGroup((OTF2_EvtWriter *)fcb, TAU_GLOBAL_STREAM_ID, sampgroupid, name, totalnidtids, idarray);

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
    cb.UserData = archive;
    cb.DefClkPeriod = 0;
    cb.DefThread = 0;
    cb.DefStateGroup = 0;
    cb.DefState = 0;
    cb.DefUserEvent = 0;
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
    while ((recs_read >0) && (!EndOfTrace));

    /* dummy records */
    Ttf_CloseFile(fh);

    status = OTF2_GlobalDefWriter_WriteClockProperties( glob_def_writer,
    1000000000,
    firstRealTime,
    lastt
    #if OTF2_VERSION_MAJOR > 2
    ,
    OTF2_UNDEFINED_TIMESTAMP 
    #endif 
    );
    check_status( status, "Write clock properties." );
    
    
    
    
        snprintf( name_buffer, sizeof( name_buffer),  "MPI_COMM_WORLD" );//PRIu64
    status = OTF2_GlobalDefWriter_WriteString( glob_def_writer,
    string_id,
    name_buffer );
    check_status( status, "Write string definition." );
    int COMM_STRING=string_id;

    OTF2_GlobalDefWriter_WriteGroup( glob_def_writer,
    GROUP_MPI_LOCATIONS /* id */,
    string_id /* name */,
    OTF2_GROUP_TYPE_COMM_LOCATIONS,
    OTF2_PARADIGM_MPI,
    OTF2_GROUP_FLAG_NONE,
    nodes,
    mpi_ranks );
    
    OTF2_GlobalDefWriter_WriteGroup( glob_def_writer,
    GROUP_MPI_COMM_WORLD /* id */,
    STRING_EMPTY /* name */,
    OTF2_GROUP_TYPE_COMM_GROUP,
    OTF2_PARADIGM_MPI,
    OTF2_GROUP_FLAG_NONE,
    nodes,
    master_threads2 );//master_threads );

    status =OTF2_GlobalDefWriter_WriteComm (glob_def_writer, TAU_DEFAULT_COMMUNICATOR, COMM_STRING, GROUP_MPI_COMM_WORLD, OTF2_UNDEFINED_COMM
		    #if OTF2_VERSION_MAJOR > 2
		    , OTF2_COMM_FLAG_CREATE_DESTROY_EVENTS
		    #endif
		    );
    check_status( status, "Write communicator." );


    /* write local mappings for MPI communicators and metrics */
    /* write local mappings for metrics, this is an identity map, just to write
    * something out
    */
    OTF2_IdMap* metric_map = OTF2_IdMap_Create( OTF2_ID_MAP_DENSE,
    NUM_OF_CLASSES );
    check_pointer( metric_map, "Create ID map for metrics." );
    for ( uint32_t c = 0; c < NUM_OF_CLASSES; c++ )
    {
        OTF2_IdMap_AddIdPair( metric_map, c, c );
    }

    for ( uint64_t rank = 0; rank < (uint64_t)nodes; rank++ )
    {
        OTF2_IdMap* mpi_comm_map = OTF2_IdMap_Create( OTF2_ID_MAP_SPARSE, 2 );
        check_pointer( mpi_comm_map, "Create ID map for MPI Comms." );

        /* Each location uses its rank as the communicator id which maps to the global 0 */
        OTF2_IdMap_AddIdPair( mpi_comm_map, rank, MPI_COMM_MPI_COMM_WORLD );
        OTF2_IdMap_AddIdPair( mpi_comm_map, rank + nodes, MPI_COMM_MPI_COMM_SELF );

        for ( uint64_t thread = 0; thread < (uint64_t)numthreads[rank]; thread++ )
        {
            /* Open a definition writer, so the file is created. */
            OTF2_DefWriter* def_writer = OTF2_Archive_GetDefWriter(
            archive,
            locations[ numthreads[rank] * rank + thread ] );
            check_pointer( def_writer, "Get definition writer." );

            /* Write metric mappings to local definitions. */
            status = OTF2_DefWriter_WriteMappingTable( def_writer,
            OTF2_MAPPING_METRIC,
            metric_map );
            check_status( status, "Write Metric mappings." );

            //if ( otf2_MPI || otf2_HYBRID )
            //{
            /* Write MPI Comm mappings to local definitions. */
            status = OTF2_DefWriter_WriteMappingTable( def_writer,

            OTF2_MAPPING_COMM,

            mpi_comm_map );
            check_status( status, "Write MPI Comm mapping." );
            //}

            /* Write clock offsets to local definitions. */
            //status = OTF2_DefWriter_WriteClockOffset( def_writer,0, 0, 0.0 );
            //check_status( status, "Write start clock offset." );
            //status = OTF2_DefWriter_WriteClockOffset( def_writer,lastt, 0, 0.0 );
            //check_status( status, "Write end clock offset." );

        }

        OTF2_IdMap_Free( mpi_comm_map );
    }
    OTF2_IdMap_Free( metric_map );

    /* close VTF file */
    status = OTF2_Archive_Close( archive );
    check_status( status, "Close archive." );
    //OTF2_EvtWriter_close((OTF2_EvtWriter *)fcb);
    return 0;
}/* EOF tau2otf.cpp */

/***************************************************************************
* $RCSfile: tau2otf.cpp,v $   $Author: amorris $
* $Revision: 1.5 $   $Date: 2008/05/28 21:21:27 $
* VERSION_ID: $Id: tau2otf.cpp,v 1.5 2008/05/28 21:21:27 amorris Exp $
***************************************************************************/
