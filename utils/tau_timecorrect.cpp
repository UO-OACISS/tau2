/* tau_timecorrect.c */

/**
 * 
 *
 * correct events time --  
 *  timestamp correction with the controlled logical clock.
 * (C) 1995 ZAM/ KFA Juelich (Trace file handling)
 * (C) 1997 Rolf Rabenseifner RUS Stuttgart (Timestamp Correction)
 * (C) 2005 ZAM/ FZ Juelich (Epilog trace file format)
 *  2006 elg_timecorrect modified by Thierry Lopez and Wyatt Spear to correct TAU trace files
 */

/*
Copyright (c) 1998-2005, Forschungszentrum Juelich GmbH, Federal
Republic of Germany

Copyright (c) 2003-2005, University of Tennessee, Knoxville, United
States of America

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the names of Forschungszentrum Juelich GmbH or the University
  of Tennessee, Knoxville, nor the names of their contributors may be
  used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* --- standard includes for tracefile handling : */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>
#include <map>
#include <iostream>

/* --- TAU includes */
#include <TAU_tf.h>

/* --- error handling */
void MemError()
{
    fprintf(stderr, "\ntau_timecorrect: out of memory\n");
    exit(3);
}

/*  Print Usage 
*   Outputs to the user the calling convention for this program.
*   Needs to be looked at and reworked if needed.
*/
void print_usage()
{
    fprintf(stderr,
            "calling convention:\n"
			"   tau_timecorrect [options] <input.trc> <input.edf> <output.trc> <output.edf>\n\n"
            "try 'tau_timecorrect -h' for more information\n\n");
}


/*  Print Help
*   Outputs the various flags to use for this program.
*   Needs to be looked over and reworked if needed.
*/
void print_help()
{
    print_usage();

    fprintf(stderr,
            "options:\n"
            "   -c   <float>   : expected maximal clock difference, value>=0. Default: 1000 [usec]\n"
            "   -d   <float>   : minimal local timestamp difference, value>0. Default: 0.1 [usec]\n"
            "   -e   <float>   : maximal error rate. if the value is > 0 than the amortisation should\n"
            "                    reduce the error to this maximal error rate;\n"
            "                    if value=0 then no amortisation is done. Default: 0.5 %%\n"
            "   -f   <float>   : factor between 0 and 1, default=(0.8), to multiply each\n"
            "                    mmd and immd value read from the options file given with option -r\n"
            "   -i   <from>_<to>_<float> : individual minimal message delay, >0. [usec]\n"
            "                    <from> and <to> are node numbers, 0..#nodes-1\n"
            "   -m   <float>   : minimal message delay, value>0. Default: 5 [usec]\n"
            "   -R   <filename>: read -c, -m, -i, -d options from options file\n"
            "                    before starting the timestamp corrections, they overwrite\n"
            "                    previous command line options.\n"
            "                    Options file kann be generated with -w or -W option\n"
            "   -r               same as -R but uses default options file name 'timecorrect.opt'" 
            "   -v             : output trace statistics\n"
            "   -W   <filename>: write analyzed 'clock difference', 'minimal message delay',\n"
            "                    'individual minimal message delay', 'minimal local timestamp difference'\n"
            "                    to options file after all timestamp corrections.\n"
            "                    Use created options file with -r or -R option\n"
            "   -w               same as -W but uses default options file name 'timecorrect.opt'" 
            );

    fprintf(stderr,
            "Examples:\n"
            "   Correct timestamps using default options:\n"
            "      tau_timecorrect bad_trace.trc bad_trace.edf good_trace.trc good_trace.edf\n"
            "\n"
            "   Create  options file 'timecorrect.opt':\n"
            "     tau_timecorrect -w ping_pong.trc ping_pong.edf dummy.trc dummy.edf\n"
            "   Correct timestamps using options file 'timecorrect.opt':\n"
            "     tau_timecorrect -r bad_trace.trc bad_trace.edf good_trace.trc good_trace.edf\n");
    exit(4);
}


/*--------------------------------------------------------------------*/
/* --- declarations for correct time */


/* This code includes                                         */
/* - the algorithmus for a logical clock with gamma_i^j = 0,  */
/* - the algorithmus for a logical clock with gamma_i^j       */
/*   controled by a controller                                */
/* - the controller                                           */
/* - the amortization                                         */

# define MAX_ANALYSE_POINTS 20
# define PRECISION_OF_DOUBLE 1e-14
# define MAX_DOUBLE 9.9999e30
# define max(A,B) ((A)>(B) ?   (A)  : (B))
# define min(A,B) ((A)<(B) ?   (A)  : (B))
# define Abs(A)   ((A)< 0  ?  (-A)  : (A))
# define sqr(A)   ((A)*(A))
# define pow16(dbl) sqr(sqr(sqr(sqr(dbl))))
/* # define pow16(dbl)  pow(dbl,16.0) */
# define root16(dbl) pow(dbl,0.0625)

#define dprintf if (debugPrint) printf //added in
using namespace std;


#define NO_ID 0xFFFFFFFF  //This is also -1.  This is used in place of void in many cases.

typedef unsigned      char UI1;   /* unsigned integer of size 1 Byte */
typedef unsigned       int UI4;   /* unsigned integer of size 4 Byte */
typedef unsigned long long UI8;   /* unsigned integer of size 8 Byte */
typedef             double D8;    /* real of size 8 Byte (ieee 754) */

//The below type defs are from the orginal ELG_timecorrect file.  They were
//in other files, and I had to move them into this file to make it work.
typedef UI1 Booli;          /* using epilog datatypes */
typedef UI4 procnum;        /* using epilog datatypes */
typedef D8 timestamp;       /* using epilog datatypes */
typedef UI4 msgid1type;     /* using epilog datatypes */
typedef UI4 msgid2type;     /* using epilog datatypes */
# define MAX_TIMESTAMP MAX_DOUBLE       /* for vampir datatypes */

/* implementation of callback routines */
map< pair<int,int>, int, less< pair<int,int> > > EOF_Trace;
int EndOfTrace = 0;  /* false */
int *offset = 0;  //Used to make global ids.
map< int,int, less<int > > numthreads;   //Used in making a global IDs
bool multiThreaded = false;   //This is incase the processors are multi-threaded

/* event type -- unordered set */
typedef enum  //This enum is for figuring out which type of record the algorithm is dealing with.
{ EV_receive, EV_send, EV_internal, EV_user } EVtype;
typedef enum
{ Delta_LCa_C, Delta_LCa_LC } qij_type;
typedef enum
{ p_alg_max, p_alg_avg, p_alg_sqr, p_alg_p16 } p_alg_type;
typedef enum
{ p_crv_spl, p_crv_lin, p_crv_pbl, p_crv_cub } p_crv_type;
typedef enum  //Used by me to determine the difference for entering and leaving states for the correct writing to output.
{ Enter_State, Leave_State } StateType;

/*Tau_rec is based on elg_rec from elg_rw.c line 78
This is used to pass along information and to store into the P array.
This will be a tau_record that is stored with the P array for later writing out.
*/
typedef struct _tau_rec
{
  EVtype rec_type;  //The type of the event.  Send, recieve, or internal(enter,leave state)
  StateType state;  //Tells if it is entering or leaving the state.
  timestamp Cij;  //Time event happened.
  unsigned int nodeid;  //Node ID
  unsigned int tid;  //Thread ID
  unsigned int stateid; //State ID, only used for internal messages (enter and leaving of states.)
  int globalID;  //Global ID used for the algorithm.  Based on nodeid and tid. 

  unsigned int secondNodeID;  //Node ID of the other node, if it exists
  unsigned int secondThreadID;  //Thread ID of the other node, if it exists.
  int secondGlobalID;  //The global id of the second node if it exists.

  unsigned int messageSize; //Only used for send and recieves
  unsigned int messageTag;  //Only used for send and recieves
  unsigned int messageComm; //Only used for send and recieves
  
  long long userEventValue;

}
tau_rec;

/*  Event Entry
*  An event entry is created for each send, recieve, or internal message found.
*  This events are stored in a queue in the P array for processing in the
*  main algorithm.
*/
typedef struct _event_entry
{
    EVtype event_type;
    procnum msg_from;           /* if event_type == EV_receive  */
    timestamp LC_from;          /* if event_type == EV_receive  */
    timestamp LCa_from;         /* if event_type == EV_receive  */
    procnum msg_to;             /* if event_type == EV_send     */
    msgid1type msg_id1;         /* if event_type != EV_internal */
    msgid2type msg_id2;         /* if event_type != EV_internal */
    timestamp Cj;               /* C_i^j (t(e_i^j))   */
    timestamp LCj;              /* LC_i^j  (e_i^j)    */
    timestamp LCaj;             /* LC'_i^j (e_i^j)    */
    struct _event_entry *next;  /* next newer in the input queue */
    /* next newer in the amortisation list */
    /* next (unsorted) in output groups */
    Booli not_the_first_event;
    timestamp discrete_advance; /* if event_type == EV_receive */
    struct _event_entry *corresponding_send;    /*if recv.event processable */
    timestamp corresponding_recv_LCa;   /* if send event, after corresp.
                                           receive event is processed */
    Booli corresponding_recv_LCa_known;
    struct _event_entry *next_newer_recv;       /* in the amortisation list */
    struct _event_entry *next_newer_send;       /* in the amortisation list */
    double amortisation;        /* amortisation at this ... */
    struct _event_entry *next_AM_point; /* next ...amortisation point */
    tau_rec *record;    //This was changed from elg to tau
    timestamp C_from;
    procnum i;  //Process number
}
event_entry;

/*  Csel Entry
*  Filled with send events awaiting to find the corresponding recieve event.
*  
*/
typedef struct _csel_entry
{                               /* corresponding send event list entry */
    timestamp LC;
    timestamp LCa;
    procnum msg_from;
    msgid1type msg_id1;
    msgid2type msg_id2;
    struct _csel_entry *next;
    event_entry *send_event;
    /* this pointer stays valid until the corresponding receive
       was computed, because only send events with known
       corresponding receive's LC' can move from AM_postponed to
       AM_allowed which is the precondition to go to the output
       which will free this send_event pointer */
    timestamp C;
}
csel_entry;

/*  Merge Node
*   Not sure what this does.  I believe it has something to do with
*   taking the records and making sure they are outputed in the correct order.
*/
typedef struct _merge_node
{
    struct _event_entry *event;
    struct _merge_node *left;
    struct _merge_node *right;
    Booli final;
}
merge_node;


/* Process Data
*  This is the heart of data holding section of the algorithm.
*  The P array is made of n, the number or processes, of these structures.
*  Each data holds the current information for the process.
*  It holds the events that it currently is buffering to be processed, reprocessed and outputed.
*  Times of events, Cj, the logical clock times, LCj1, and the adjusted logical clock times, LCa.
*  The merge node where the P array rests.
*  Which event is next to be processed.
*/
typedef struct
{
    event_entry *In_oldest;     /* input queue -- oldest pointer */
    event_entry *In_newest;     /* input queue -- newest pointer */
    csel_entry *csel_first;     /* sorted corresponding send ... */
    csel_entry *csel_last;      /*                ... event list */
    procnum input_processable_next;     /* input scheduler : */
    /* list of processes with possibly
       executable events in In_oldest */
    Booli input_processable;
    Booli initialized;          /* if first event processed? */
    timestamp Cj1;              /* C_i (t(e_i^j-1)) */
    timestamp LCj1;             /* LC_i  (e_i^j-1)  */
    timestamp LCaj1;            /* LC'_i (e_i^j-1)  */
    float gammaj;               /* gamma_i^j */
    timestamp delta;            /* <minimal local timestamp difference */
    timestamp LCaj_Cj;          /* LC'_i(e_i^j) - C_i(t(e_i^j)) */
    timestamp LCaj1_Cj1;        /* LC'_i(e_i^j-1) - C_i(t(e_i^j-1)) */
    double pow16_LCaj1_Cj1;     /* (LC'-C)**16  */
    event_entry *AM_oldest;
    event_entry *AM_oldest_send;
    event_entry *AM_postponed_send;
    timestamp AM_LCa_limit;
    timestamp AM_LCa_next_before_oldest;
    event_entry *AM_todo_recv;
    event_entry *AM_newest_recv;
    event_entry *AM_newest_send;
    event_entry *AM_newest;
    procnum AM_processable_next;        /* amortisation list scheduler */
    Booli AM_processable;
    struct _merge_node *output_merge_node;
    struct _event_entry *output_merge_newest_event;
    /* additional data for other analysis of proc. i */
    timestamp Out_Cj1;          /* measured at output from event_entry */
    timestamp Out_LCaj1;
}
process_data;

/*  immd_entry
*   Currently no idea what this does.  Used in the main algorithm.
*/
typedef struct _immd_entry
{                               /* individual minimal message delay defaults list entry */
    struct _immd_entry *next;
    procnum from, to;
    timestamp immd;
}
immd_entry;

/*--------------------------------------------------------------------*/

/* global data for command line options */
static double clock_period;
static timestamp delta_default, mmd_default, *mmd;
static int mmd_individual;      /* 0=same mmd for all k,i; 1=individual mmds */
static immd_entry *immd_first, *immd_last;
static float mmd_factor;        /* default is 0.8 */
static char *to_optfilename, *from_optfilename;
static UI1 verbose = 0;  //False
static float qmin, qmax, pmin, pmax, D_slow_factor, gamma1_init;
static timestamp min_LCa_C;
static double sum16_LCa_C;
static double max_sum16_LCa_C;
static timestamp AM_cldiff_default;
static double AM_maxerr;


/* support for single location */
static UI4 single_lid;


//Forward declare to get around compiler issues.
void tau_Write_Record(Ttf_FileHandleT OutFile,tau_rec* OutRec);
void correct_time_exec(procnum input_processable_next, Ttf_FileHandleT outputFile, bool cleanUp);

//Frees records, because the records are malloced.
void tau_free(tau_rec* rec)
{
	free(rec);
}

/*  Default_opt
*   The above global variables are set to default options if the user
*   had not used any flags or additional arguements.
*/
void default_opt()
{
    clock_period = 1.0;         /* = 1.0 sec, no option, only internal */
    delta_default = 0.1e-6 / clock_period;
    /* = 0.1 us, minimal difference between
       two events in the same process */
    mmd_default = 5e-6 / clock_period;
    /* = 5 us, minimal message delay */
    mmd_individual = 0;
    immd_first = NULL;
    immd_last = NULL;
    mmd = &mmd_default;
    mmd_factor = 0.8;           /* due to clock drifts the given
                                   mmd's from the last run on the
                                   from_optfile may be larger than the
                                   real mmd's and must be reduced
                                   therefore by an mmd_factor < 1 */
    to_optfilename = NULL;
    from_optfilename = NULL;
    qmin = 1.2;                 /* when LC'-C > qmin * max(LC-C)
                                   the controller starts its work, i.e.
                                   it reduces gamma.
                                   allowed: qmin >= 1
                                   advice:  1.2 (1.0 .. 2.0) */
    qmax = 3.0;                 /* when LC'-C >= qmax * max(LC-C)
                                   the controller reduces gamma to zero,
                                   i.e. the controlled logical clock
                                   runs like the simple logical clock.
                                   allowed: qmax > qmin
                                   advice:  2*(qmin+0.5) */
    pmin = 0.0;                 /* when min(LC'-C) > pmin * max(LC'-C)
                                   the controller starts its work, i.e.
                                   it reduces gamma.
                                   allowed: pmin >= 0
                                   advice:  0.0  */
    pmax = 1.0;                 /* when min(LC'-C) >= pmax * max(LC'-C)
                                   the controller reduces gamma to zero,
                                   i.e. the controlled logical clock
                                   runs like the simple logical clock.
                                   allowed: pmax > pmin
                                   advice:  1.0 (0.5..1.0) */
    D_slow_factor = 0.5;        /* computing max(LC-C) the controller
                                   forgets with D_slow_factor*gamma1_init
                                   the past. This prohibits that some time
                                   after a pertubation the controller
                                   starts to reduce gamma due to the
                                   rule based on qmin.
                                   0 means that a pertubation won't be
                                   forgotten in computing max(LC-C).
                                   allowed: 0 <= D_slow_factor < 1
                                   advice:  0.5 (0.5 .. 0.8) */
    gamma1_init = 2e-5;         /* a pertubation of delta will be for-
                                   gotten in delta/gamma1_init, e.g. with
                                   gamma1_init=1e-3, a clock error correc-
                                   tion of 1ms (i.e. LC-C = LC'-C = 1ms)
                                   will influence LC' for 1ms/1e-3 = 1sec;
                                   gamma1_init=1  implies LC' := LC.
                                   allowed: 0 <= gamma1_init <= 1
                                   advice:  2e-5 (0.05 .. 1e-4)   */
    AM_cldiff_default = 1000e-6 / clock_period;
    /* expected maximal clock diff. in  us */
    AM_maxerr = 0.005;          /* wished maximal error */
}



/*--------------------------------------------------------------------*/
/* Korrigieren der Zeitstempel in einer Datei: */
/* global between Init, several invocations of Exec, and Finish */
static procnum n;  //The number of processes
static process_data *P;  //The P array used to hold each of the processes informatiion.
static merge_node *merge_root, *node, *nextnode;  //Merge array used to for outputing I believe.
static event_entry final_event_struct;
static event_entry *final_event = &final_event_struct;
static timestamp Dmaxj, LC_Dmaxj;
static procnum index_min_LCa_C;

static float gamma_max;
static long In_length;  //Keeps track of the number of events left to be processed. Should be 0 when done.

static timestamp AM_interval;   /* amortisation interval in cl.periods */
static timestamp AM_cldiff;     /* exp. max. clock diff. in cl.periods */
static procnum AM_out_filled;   /* number of fille output lists
                                   to be merged */

static double min_delta = MAX_DOUBLE;
static double *ki_mmd;
static long count_err_gt_maxerr = 0;
static double max_err_gt_maxerr = 0;
static double sum_err_gt_maxerr = 0;
static long count_err_le_maxerr = 0;
static double max_err_le_maxerr = 0;
static double sum_err_le_maxerr = 0;
static long count_err_eq_zero = 0;
static double min_err = MAX_DOUBLE;

static int _TRUE_ = 1;
static int _FALSE_ = 0;

/* GlobalID
*  This is used to convert nodes with threads to a single id.
*  The main algorithm uses a single id to use in the P array and this
*  funtion allows the records node id and thread id to be compressed into one id.
*/
int GlobalId(int localNodeId, int localThreadId)
{
	if (multiThreaded)
	{
		if (offset == (int *) NULL)
		{
			printf("Error: offset vector is NULL in GlobalId()\n");
			return localNodeId;
		}
		/* for multithreaded programs, modify this routine */
		return offset[localNodeId]+localThreadId;  /* for single node program */
	}
	else
	{ 
		return localNodeId;
	}

}


/* This section begins the functions needed to be able to read in and write out
*  tau trace files.
*/



/* DefThread
*  This is used for both multi and single threaded programs.  It reads
*  in the process, writes the process out, increments the global number of processes, 
*  then checks to see if the program is multi-threaded.
*/
int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
  EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 0; /* initialize it */
  Ttf_DefThread(userData, nodeToken, threadToken, threadName);  //Write it out
  
  n++;  //Increase the number of processes

  numthreads[nodeToken] = numthreads[nodeToken] + 1;
  if (threadToken > 0) multiThreaded = true;  //Is it multi-threaded.

  return 0;
}


/* DefStateGroup  
*  This function reads in the state group and writes it back out.
*  The algorithm does not need to know what the states are called.
*/
int DefStateGroup( void *userData, unsigned int stateGroupToken, 
		const char *stateGroupName )
{
  Ttf_DefStateGroup(userData, stateGroupName, stateGroupToken);
  return 0;
}

/* DefState
*  The function reads a defined state from the trace file and writes the information
*  to the output trace file.
*  The main algorithm does not need this information.
*/
int DefState( void *userData, unsigned int stateToken, const char *stateName, 
		unsigned int stateGroupToken )
{
	if(stateToken<60000)  
	{
		string name = stateName;
		name=name.substr(1,name.length()-2);
		Ttf_DefState(userData, stateToken, name.c_str(), stateGroupToken);
	}
  return 0;
}

/* End Trace
*  This function is called when the reader has come to the end of the trace.
*  It checks to see if all the information had been pulled out of the trace file.
*  This function is not needed for the main algorithm.
*/
int EndTrace( void *userData, unsigned int nodeToken, unsigned int threadToken)
{
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


int DefUserEvent( void *userData, unsigned int userEventToken,
		const char *userEventName, int monotonicallyIncreasing )
{
	string name = userEventName;
	name = name.substr(1,name.length()-2);
	Ttf_DefUserEvent(userData,userEventToken, name.c_str(), monotonicallyIncreasing);
   //dprintf("DefUserEvent event id %d user event name %s, monotonically increasing = %d\n", userEventToken, userEventName, monotonicallyIncreasing);

  return 0;
}


/*  SetEventEntry
*   Used to set an event to default values before it is
*	passed back, the send/recieve/enter/leave function
*	calls where it will be more processed some more.
*/
event_entry* setEventEntry(procnum i)
{
	event_entry* input_entry;

    // store entry into the input queue 
	if ((input_entry = (event_entry*) malloc(sizeof(event_entry))) == NULL)  {
		printf("memory allocation error");
		exit(EXIT_FAILURE);
	}

    input_entry->next = NULL;
    if (P[i].In_oldest == NULL)  //Is there anything in queue right now?
    {                           /* thus machinably */
        P[i].input_processable = _TRUE_;
        P[i].In_oldest = input_entry;
    }
    else
    {                           /* nicht bearbeitbar, da aeltere Events
                                   noch nicht bearbeitet */
        P[i].In_newest->next = input_entry;
    }
    P[i].In_newest = input_entry;

	input_entry->corresponding_recv_LCa_known = _FALSE_;
    input_entry->next_newer_send = NULL;
    input_entry->next_newer_recv = NULL;
	In_length++; 
	input_entry->i = i;

	return input_entry;

}

/* EventTrigger
*  Event trigger is called when the tau reader see an event trigger.
*  This function sets up the structures and then passes it to algorithm
*  to process.
*/

int EventTrigger( void *userData, double time, 
		unsigned int nodeid,
		unsigned int tid,
	       	unsigned int userEventToken,
		long long userEventValue)
{
//	Ttf_EventTrigger(userData,time,nodeToken,threadToken,userEventToken, userEventValue);
//	dprintf("EventTrigger: time %g, nid %d tid %d event id %d triggered value %lld \n", time, nodeToken, threadToken, userEventToken, userEventValue);
	procnum i;
	tau_rec* currentRecord;
	procnum input_processable_next;
	event_entry *input_entry;

	//printf("Entering EnterState \n");

	//Enough memory to create a tau_record.
	if ((currentRecord = (tau_rec*)malloc(sizeof(tau_rec))) == NULL)  {
		printf("memory allocation error");
		exit(EXIT_FAILURE);
	}

	i = single_lid = GlobalId(nodeid, tid);  //Get the global idea for the record

	//Does this global id exist and does it fall into the range of existing processors
	if (i == NO_ID || i >= n) {
		printf("invalid location id %d", i);
		exit(EXIT_FAILURE);
	}

	input_entry = setEventEntry(i);
	
	//Fill out the currentRecord to be stored.
	currentRecord->rec_type = EV_user;
	currentRecord->nodeid = nodeid;
	currentRecord->tid = tid;
	currentRecord->stateid = userEventToken;
	currentRecord->Cij = time;
	currentRecord->globalID = i;
	currentRecord->state = Enter_State;
	currentRecord->userEventValue = userEventValue;

	input_entry->record = currentRecord;
	input_entry->Cj = time;

	//printf("I'm in internal message.  \n");
    input_entry->event_type = EV_user;

	//input_entry has been setup.  Now process the event

	input_processable_next = i;  //Which process has the next event to be processed.
	correct_time_exec(input_processable_next, userData,false);  //Call the algorithm and point to the process to be looked at.
	
	return 0;
}



/* Enter State
*  This function is called when the trace reader has seen that a state has been entered.
*  It creates a tau_record, and fills it out with the nesscary information for the
*  algorithm to be able to process it and then for the progam to write it out later.
*  Finally it calls the algorithm passing in which process has a new record to be looked at.
*/
int EnterState(void *userData, double time, 
		unsigned int nodeid, unsigned int tid, unsigned int stateid)
{	
	procnum i;
	tau_rec* currentRecord;
	procnum input_processable_next;
	event_entry *input_entry;

	//printf("Entering EnterState \n");

	//Enough memory to create a tau_record.
	if ((currentRecord = (tau_rec*)malloc(sizeof(tau_rec))) == NULL)  {
		printf("memory allocation error");
		exit(EXIT_FAILURE);
	}

	i = single_lid = GlobalId(nodeid, tid);  //Get the global idea for the record

	//Does this global id exist and does it fall into the range of existing processors
	if (i == NO_ID || i >= n) {
		printf("invalid location id %d", i);
		exit(EXIT_FAILURE);
	}

	input_entry = setEventEntry(i);
	
	//Fill out the currentRecord to be stored.
	currentRecord->rec_type = EV_internal;
	currentRecord->nodeid = nodeid;
	currentRecord->tid = tid;
	currentRecord->stateid = stateid;
	currentRecord->Cij = time;
	currentRecord->globalID = i;
	currentRecord->state = Enter_State;

	input_entry->record = currentRecord;
	input_entry->Cj = time;

	//printf("I'm in internal message.  \n");
    input_entry->event_type = EV_internal;

	//input_entry has been setup.  Now process the event

	input_processable_next = i;  //Which process has the next event to be processed.
	correct_time_exec(input_processable_next, userData,false);  //Call the algorithm and point to the process to be looked at.
	
	//printf("Finishing the enter state\n");
	return 0;
}




/* Leave State
*  This function is called when the trace reader has seen that a state has been left.
*  It creates a tau_record, and fills it out with the nesscary information for the
*  algorithm to be able to process it and then for the progam to write it out later.
*  Finally it calls the algorithm passing in which process has a new record to be looked at.
*/
int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid, unsigned int stateid)
{

	procnum i;
	tau_rec* currentRecord;
	procnum input_processable_next;
	event_entry *input_entry;

	//printf("Entering Leave State\n");
	//Is there enough memory to create the tau_record.
	if ((currentRecord = (tau_rec*)malloc(sizeof(tau_rec))) == NULL)  {
		printf("memory allocation error");
		exit(EXIT_FAILURE);
	}

	i = single_lid = GlobalId(nid, tid);  //Figure out the global id.

	//Does this global id exist and does it fall into the range of existing processors
	if (i == NO_ID || i >= n) {
		printf("invalid location id %d", i);
		exit(EXIT_FAILURE);
	}

	input_entry = setEventEntry(i);
   	
	//Need to create the record to be stored.
	currentRecord->rec_type = EV_internal;
	currentRecord->nodeid = nid;
	currentRecord->tid = tid;
	currentRecord->stateid = stateid;
	currentRecord->Cij = time;
	currentRecord->globalID = i;
	currentRecord->state = Leave_State;

	input_entry->record = currentRecord;
	input_entry->Cj = time;

	//printf("I'm in internal message.  \n");
    input_entry->event_type = EV_internal;
    
	//input_entry has been setup.  Now process the event

	input_processable_next = i;  //Which process the algorithm should look at next.
	correct_time_exec(input_processable_next, userData,false); //Call the main algorithm to look at process i.

	return 0;
}

/* Send Message
*  This function is called when the trace reader has seen a send message.
*  It creates a tau_record, and fills it out with the nesscary information for the
*  algorithm to be able to process it and then for the progam to write it out later.
*  Finally it calls the algorithm passing in which process has a new record to be looked at.
*/
int SendMessage( void *userData, double time, 
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
		unsigned int messageTag,
		unsigned int messageComm )
{

	procnum i,destinationID;
	tau_rec* currentRecord;
	procnum input_processable_next;
	event_entry *input_entry;

    

//	printf("Entering SendMessage \n");
	//Is there enough memory to create the tau_record.
	if ((currentRecord = (tau_rec*)malloc(sizeof(tau_rec))) == NULL)  {
		printf("memory allocation error");
		exit(EXIT_FAILURE);
	}
	
	i = single_lid = GlobalId(sourceNodeToken, sourceThreadToken);  //Global id of the source node.
	destinationID = GlobalId(destinationNodeToken,destinationThreadToken);  //Global id of the destination node.

	//Does this global id, i, exist and does it fall into the range of existing processors
	if (i == NO_ID || i >= n) {
		printf("invalid location id %d", i);
		exit(EXIT_FAILURE);
	}

	//Does this global id, destination id, exist and does it fall into the range of existing processors
	if (destinationID == NO_ID || destinationID>= n) {
		printf("invalid location id %d", destinationID);
		exit(EXIT_FAILURE);
	}

	input_entry = setEventEntry(i);

	//Create the record to be stored and fill it.
	currentRecord->rec_type = EV_send;

	currentRecord->nodeid = sourceNodeToken;
	currentRecord->tid = sourceThreadToken;
	currentRecord->globalID = i;

	currentRecord->secondNodeID = destinationNodeToken;  
    currentRecord->secondThreadID = destinationThreadToken; 
    currentRecord->secondGlobalID = destinationID;  

	currentRecord->messageSize = messageSize;
	currentRecord->messageTag = messageTag;
	currentRecord->messageComm = messageComm;
	currentRecord->Cij = time;


    input_entry->record = currentRecord;  
    input_entry->Cj = time;
	
//	printf("I'm a send \n");
	input_entry->event_type = EV_send;
    input_entry->msg_to = destinationID;
    input_entry->msg_id1 =  messageComm;
    input_entry->msg_id2 = messageTag;
	
	//input_entry has been setup.  Now process the event

	input_processable_next = i;  //Which process the main algorithm should look at.
	correct_time_exec(input_processable_next, userData,false);  //Call the main algorithm and with which process to look at.

  return 0;
}

/* Recv Message
*  This function is called when the trace reader has seen a recieve message.
*  It creates a tau_record, and fills it out with the nesscary information for the
*  algorithm to be able to process it and then for the progam to write it out later.
*  Finally it calls the algorithm passing in which process has a new record to be looked at.
*/
int RecvMessage( void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
                unsigned int messageTag,
		unsigned int messageComm )
{
	procnum i,sourceID;
	tau_rec* currentRecord;
	procnum input_processable_next;
	event_entry *input_entry;

	//assign a value from the files. 

	//printf("Entering Recvmessage \n");
	
	//Is there enough memory to create the tau_record.
	if ((currentRecord = (tau_rec*)malloc(sizeof(tau_rec))) == NULL)  {
		printf("memory allocation error");
		exit(EXIT_FAILURE);
	}

	//Get the global ids for the source and destination nodes.
    i = single_lid = GlobalId(destinationNodeToken,destinationThreadToken);
	sourceID = GlobalId(sourceNodeToken, sourceThreadToken);

	//Does this global id, i id, exist and does it fall into the range of existing processors
	if (i == NO_ID || i >= n) {
		printf("invalid location id %d", i);
		exit(EXIT_FAILURE);
	}

	//Does this global id, sourceID id, exist and does it fall into the range of existing processors
	if (sourceID == NO_ID || sourceID>= n) {
		printf("invalid location id %d", sourceID);
		exit(EXIT_FAILURE);
	}

	input_entry = setEventEntry(i);

	//Create the record to be stored.
	currentRecord->rec_type = EV_receive;

	currentRecord->nodeid = destinationNodeToken;
	currentRecord->tid = destinationThreadToken;
	currentRecord->globalID = i;

	currentRecord->secondNodeID = sourceNodeToken;  
    currentRecord->secondThreadID = sourceThreadToken; 
    currentRecord->secondGlobalID = sourceID;  

	currentRecord->messageSize = messageSize;
	currentRecord->messageTag = messageTag;
	currentRecord->messageComm = messageComm;
	currentRecord->Cij = time;

    input_entry->record = currentRecord; 
    input_entry->Cj = time;

	//printf("I'm a Recieve! \n");
    input_entry->event_type = EV_receive;
    input_entry->msg_from = sourceID;
    input_entry->msg_id1 = messageComm;
    input_entry->msg_id2 = messageTag;
	
	//input_entry has been setup.  Now process the event

	input_processable_next = i;  //Which process is next to be looked at by the main algorithm.
	correct_time_exec(input_processable_next, userData,false);  //Calls the main algorithm to look at process i.

  return 0;
}

/* tau_Write_Record 
*  This function is used to write all tau_records to the output files.
*  It checks to see what the record type is and then uses the correct output method.
*/
void tau_Write_Record(Ttf_FileHandleT OutFile,tau_rec* OutRec)
{
	switch (OutRec->rec_type)
    {
    case EV_send:
        //printf("I'm a writing a send. \n");
		
		/*printf("SendMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d, messageComm %d\n", 
		  OutRec->Cij, 
		  OutRec->nodeid, OutRec->tid,
		  OutRec->secondNodeID, OutRec->secondThreadID,
		  OutRec->messageSize, OutRec->messageTag,OutRec->messageComm); */

		//The nodeid is the source node since the global key is based off the sends.
		Ttf_SendMessage(OutFile,  OutRec->Cij, 
		  OutRec->nodeid, OutRec->tid, 
		  OutRec->secondNodeID, OutRec->secondThreadID, 
		  OutRec->messageSize,  // length
		  OutRec->messageTag,   // tag
		  OutRec->messageComm);   // communicator

        break;
    case EV_receive:
		//printf("I'm a writing a recieve. \n");
		/*printf("RecvMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d ,messageComm %d\n", 
		  OutRec->Cij, 
		  OutRec->secondNodeID, OutRec->secondThreadID,
		  OutRec->nodeid, OutRec->tid,
		  OutRec->messageSize, OutRec->messageTag,OutRec->messageComm);*/

		//SecondNode holds the source files, since the global key is based off the nodeid which is the dest.
		Ttf_RecvMessage(OutFile, OutRec->Cij, 
		  OutRec->secondNodeID, OutRec->secondThreadID,
		  OutRec->nodeid, OutRec->tid, 
		  OutRec->messageSize,  // length
		  OutRec->messageTag,   // tag
		  OutRec->messageComm);   // communicator

	break;
	case EV_internal:
	//	printf("I'm a internal message. \n");
		if (OutRec->state == Enter_State)
		{
//		printf("I'm a Enter_state message. \n");
		/*printf("Entered state %d time %g nid %d tid %d\n", 
		  OutRec->stateid, OutRec->Cij, OutRec->nodeid, OutRec->tid); */
		Ttf_EnterState(OutFile, (long int)OutRec->Cij, OutRec->nodeid, OutRec->tid, OutRec->stateid);
		}
		else {  if (OutRec->state == Leave_State)
			{
			//printf("I'm in a Leave_state message. \n");
				//printf("Leaving state %d time %g nid %d tid %d\n", OutRec->stateid, OutRec->Cij, OutRec->nodeid, OutRec->tid);
				Ttf_LeaveState(OutFile, (long int) OutRec->Cij, OutRec->nodeid, OutRec->tid, OutRec->stateid);
			}
		}

	break;
	case EV_user:
		Ttf_EventTrigger(OutFile,OutRec->Cij,OutRec->nodeid,OutRec->tid,OutRec->stateid, OutRec->userEventValue);
	break;

    default:
		printf("This is an unknown message type.  \n"); 
		exit(EXIT_FAILURE);
    }

}

/* Correct_time_init  
*  This is used by the program to figure out the base values that the main algorithm will use.
*  This funtion is called early on.
*/
void correct_time_init()
{
    n = 0;  //The number of processes is 0
    P = NULL;  //The P array is empty.
    In_length = 0;  //The number of events left to be processed is 0.
    gamma_max = 1.0 - gamma1_init;      /* fixed definition */
    Dmaxj = 0;
    index_min_LCa_C = 0;
    min_LCa_C = 0.0;
    sum16_LCa_C = 0;
    max_sum16_LCa_C = 0;
    AM_cldiff = 0;  //I belive AM is amortized clock difference is 0.
    AM_interval = (AM_maxerr > 0 ? AM_cldiff_default / AM_maxerr : 0);
    final_event->LCaj = MAX_TIMESTAMP;
    final_event->next = final_event;
}




/* Handle_def_record
*  Main purpose is to read the input file, count the number of processes and write out all the state and thread information.
*  Takes in the input and output file.
*  Gets the type record being read.
*  If its a defthread, increase the number of processes
*  Any other type, write it out to the ouput files.
*  When the input file is done being read, create a P list based on the number of processes.
*  Also create this merge_node array, which I'm not sure excatly what it does.  Used in the output to file function.
*
*  Return if there is a last defintion or not.
*/
UI4 handle_def_record(Ttf_FileHandleT inputFile, Ttf_FileHandleT outputFile)
{
    UI1 type;
    procnum i, k;
    merge_node *merge_node_array, *upper_lev_merge_node_array;
    int merge_node_array_lng, upper_lev_merge_node_array_lng;
    UI4 lastdef = _FALSE_;
	int recs_read, pos;

	//The callback is used for the tau reader to know what functions to call when it sees a certain record.
	//0 means call nothing.
	Ttf_CallbacksT cb;
  /* Fill the callback struct */
  cb.UserData = outputFile;
  cb.DefClkPeriod = 0;  //could be a clock period
  cb.DefThread = DefThread;
  cb.DefStateGroup = DefStateGroup;
  cb.DefState = DefState;
  cb.DefUserEvent = DefUserEvent;
  cb.EventTrigger = 0; 
  cb.EndTrace = EndTrace;
  cb.EnterState = 0;
  cb.LeaveState = 0;
  cb.SendMessage = 0;
  cb.RecvMessage = 0;

  /* Go through each record until the end of the trace file */
  do {
    recs_read = Ttf_ReadNumEvents(inputFile,cb, 1024);
    }
  while ((recs_read >=0) && (!EndOfTrace));
  EndOfTrace=0;
	if (multiThreaded)
	{ /* create the thread ids */
		int tid = 0; 
		int nodes = numthreads.size(); /* total no. of nodes */ 
		int *threadnumarray = new int[nodes]; 
		offset = new int[nodes+1];
		offset[0] = 0; /* no offset for node 0 */
		for (int q=0; q < nodes; q++)
		{ /* one for each node */
			threadnumarray[q] = numthreads[q]; 
			offset[q+1] = offset[q] + numthreads[q]; 
		}
	}

	//Done looking through the trace file.
	//printf("Found the last def. \n");
	AM_out_filled = 0;
  
        /*
         *    Convert minimal delay of messages from k to i, given  in parameter 
         *    list to adj. matrix. 
         *    Destroy input list.
         *    /param n           [in]         - number of procs
         *    /param immd_first  [in]         - pointer to first list element
         *    /param mmd         [out]        - n x n matrix of delays
         */
        if (immd_first != NULL)
        {
            immd_entry *immd;
            procnum i, k;
            mmd_individual = 1;
            if ((mmd = (timestamp*) malloc(n * n * sizeof(timestamp))) == NULL)
                MemError();
            for (k = 0; k < n; k++)
                for (i = 0; i < n; i++)
                {
                    mmd[k * n + i] = mmd_default;
                }
		
            /* =minimal delay of messages from k to i */
            while (immd_first != NULL)
            {
                immd = immd_first;
                mmd[immd->from * n + immd->to] = immd->immd;
                immd_first = immd->next;
                free(immd);
            }
        }

        /*
         *    Create and initialize \a P                 - array of  Process Data 
         *    Create and initialize \a merge_node_array  - array of numproc size
         *    Each process points to one merge_node.
         */
        if ((P = (process_data*)malloc(n * sizeof(process_data))) == NULL)
            MemError();
        merge_node_array_lng = n;
        if ((merge_node_array = (merge_node*) malloc(merge_node_array_lng *
                                       sizeof(merge_node))) == NULL)
            MemError();
        for (i = 0; i < n; i++)
        {                       /* init... */
            P[i].In_oldest = NULL;
            P[i].csel_first = NULL;
            P[i].csel_last = NULL;
            P[i].input_processable_next = NO_ID;  //Which process to look at after this one.
            P[i].input_processable = _FALSE_;  //Has any events to be processed
            P[i].initialized = _FALSE_;
            P[i].gammaj = gamma_max;
            P[i].LCaj_Cj = 0;  //The events adjust clock time.
            P[i].Cj1 = 0;       /* for initial step of p-algorithm */
            P[i].LCaj1 = 0;     /* for initial step of p-algorithm */
            P[i].LCaj1_Cj1 = 0; /* for initial step of p-algorithm */
            P[i].AM_oldest = NULL;  //Everything with an AM, stands for amortized.
            P[i].AM_oldest_send = NULL;
            P[i].AM_postponed_send = NULL;
            P[i].AM_LCa_limit = 0;
            P[i].AM_todo_recv = NULL;
            P[i].AM_newest_recv = NULL;
            P[i].AM_newest_send = NULL;
            P[i].AM_newest = NULL;
            P[i].AM_processable_next = NO_ID;
            P[i].AM_processable = _FALSE_;
            P[i].output_merge_node = &(merge_node_array[i]);
            P[i].output_merge_node->final = _TRUE_;
            P[i].output_merge_node->event = NULL;
            P[i].output_merge_node->left = NULL;        /* unused because .. */  //Written in previous versions.
            P[i].output_merge_node->right = NULL;       /*  .... final node */
            P[i].output_merge_newest_event = NULL;
        }

        /*
         *   Create tree of \a merge nodes
         */
	
        while (merge_node_array_lng > 1)
        {
            upper_lev_merge_node_array = merge_node_array;
            upper_lev_merge_node_array_lng = merge_node_array_lng;
            merge_node_array_lng = (upper_lev_merge_node_array_lng + 1) / 2;
            if ((merge_node_array = (merge_node*) malloc(merge_node_array_lng *
                                           sizeof(merge_node))) == NULL)
                MemError();
			
			for (i = 0; i < merge_node_array_lng; i++)
            {
                merge_node_array[i].final = _FALSE_;
                merge_node_array[i].event = NULL;
                merge_node_array[i].left =
		  &(upper_lev_merge_node_array[2 * i]);
                merge_node_array[i].right =
                    (2 * i + 1 <
                     upper_lev_merge_node_array_lng ?
                     &(upper_lev_merge_node_array[2 * i + 1]) : NULL);
                /* delete intermediate upper-level-nodes that have only
                   one link */
                if ((merge_node_array[i].left != NULL) &&
                    (!merge_node_array[i].left->final) &&
                    (merge_node_array[i].left->right == NULL))
                    merge_node_array[i].left = merge_node_array[i].left->left;
                if ((merge_node_array[i].right != NULL)
                    && (!merge_node_array[i].right->final)
                    && (merge_node_array[i].right->right == NULL))
                    /* !!!right->right  changed to right->left by Andrej Kuehnal */
                    merge_node_array[i].right =
                        merge_node_array[i].right->left;
            }
        }

        /*
         *   Init  ki_mmd matrix ( n x n ) with MAX_DOUBLE
         */
        merge_root = &(merge_node_array[0]);
        if ((ki_mmd = (double*) malloc(n * n * sizeof(double))) == NULL)
            MemError();
        for (k = 0; k < n; k++) /* sender */
            for (i = 0; i < n; i++)     /* receiver */
                ki_mmd[k * n + i] = +MAX_DOUBLE;
        lastdef = _TRUE_;
       
    return lastdef;
}


/* Handle Event Record
*  This function reads in the input trace file and looks for send and recieve messages, entry and exit of states.
*  When it finds them, it calls the correction functions and has them processed by the main algorithm.
*/
void handle_event_record(Ttf_FileHandleT inputFile,Ttf_FileHandleT outputFile)
{
	int recs_read;
	//Are there any processes to look at?
	if (n == 0)
	{ 
		printf("There don't seem to be any processors found"); 
		exit(EXIT_FAILURE);
	}

	Ttf_CallbacksT cb;
    /* Fill the callback struct */
    cb.UserData = outputFile;
    cb.DefClkPeriod = 0;
    cb.DefThread = 0;
	cb.DefStateGroup = 0;
	cb.DefState = 0;
	cb.DefUserEvent = 0;
	cb.EventTrigger = EventTrigger; 
	cb.EndTrace = EndTrace;
    cb.EnterState = EnterState;
    cb.LeaveState = LeaveState;
    cb.SendMessage = SendMessage;
    cb.RecvMessage = RecvMessage;
 
	  /* Go through each record until the end of the trace file */
	do {
		recs_read = Ttf_ReadNumEvents(inputFile,cb, 1024);
	}
	while ((recs_read >=0) && (!EndOfTrace));
}

/* Correct Time Exec
*  This is the brain of the program.  This is the algorithm that processes
*  records, and adjust their times, from what I can gather, a logical clock approach.
*  Events are ordered based on the sending and receiving of messages.
*  Input_processable_next is used to determine which process is next to be looked at.
*  If clean up is true, it runs through the last events to be processed and writes them out to the output files.
*/
void correct_time_exec(procnum input_processable_next, Ttf_FileHandleT outputFile, bool cleanUp)
{
	tau_rec* OutRec;
    event_entry *output_entry;
    procnum i, k; 
	event_entry *input_entry;
    Booli input_processable;
    procnum AM_processable_next;
    csel_entry *one_csel, *prev_csel;
    timestamp Cij, LCij, LCaij, diff_LCa_C;
    float pij, qij, p_gammaij, q_gammaij;
    double divisor;
    double err, abs_err;

	AM_processable_next = NO_ID;

	/* process message queue */
    do     /* while (input_processable_next != NO_ID) */
    {      /* processing all processable events           */
//      printf("in the do loop %d\n", EV_receive);
        if (input_processable_next != NO_ID)
        {
            i = input_processable_next;
            input_entry = P[i].In_oldest;
            if (input_entry->event_type == EV_receive)
            {   /* whether corresponding sent event already processed */
                /* search from the _beginning_ of the list */
                one_csel = P[i].csel_first;
                input_processable = _FALSE_;
                prev_csel = NULL;
                while (one_csel != NULL)
                {
                    if ((one_csel->msg_from ==
                         input_entry->msg_from)
                        && (one_csel->msg_id1 ==
                            input_entry->msg_id1)
                        && (one_csel->msg_id2 == input_entry->msg_id2))
                    {
                        input_processable = _TRUE_;
                        input_entry->C_from = one_csel->C;
                        input_entry->LC_from = one_csel->LC;
                        input_entry->LCa_from = one_csel->LCa;
						input_entry->corresponding_send =
                            one_csel->send_event;
                        /* remove one_csel from csel list */
                        if (prev_csel == NULL)
                        {
                            P[i].csel_first = one_csel->next;
                        }
                        else
                        {
                            prev_csel->next = one_csel->next;
                        }
                        if (one_csel == P[i].csel_last)
                            P[i].csel_last = prev_csel;
                        free((void *) one_csel);
                        one_csel = NULL;
                    }
                    else
                    {
                        prev_csel = one_csel;
                        one_csel = one_csel->next;
                    }
                }               /* while (one_csel != NULL) */
            }
            else
            {                   /* EV_send or EV_internal */
                input_processable = _TRUE_;
            }

            /* if EV_send, EV_internal or EV_recv with coresponded EV_send */
            if (input_processable)
            {                   /* input_processable */
                /* removing from input queue */
                P[i].In_oldest = input_entry->next;
                if (P[i].In_oldest == NULL)
                {   /* remove this input queue from input_processable_next list */
                    input_processable_next = P[i].input_processable_next;
                    P[i].input_processable = _FALSE_;
                    P[i].input_processable_next = NO_ID;
                }
                In_length--;

                /* processing the controlled logical clock */
                input_entry->not_the_first_event = P[i].initialized;
                Cij = input_entry->Cj;
                LCij = Cij;
                LCaij = Cij;
             

              //  printf("this is i %d \n", i);
                if (!P[i].initialized)
                {
                    //printf("messy!!! \n");
                    P[i].delta = delta_default;
                }
                else            /* P[i].initialized */
                {
                  
                    if ((Cij - P[i].Cj1) < min_delta)
                        min_delta = Cij - P[i].Cj1;
                    LCij = max(LCij, P[i].LCj1 + P[i].delta);
                    LCaij = max(LCaij, P[i].LCaj1 + P[i].delta);
                    LCaij = max(LCaij, P[i].LCaj1 +
                            (timestamp) (P[i].gammaj * (Cij - P[i].Cj1)));
                    /*  ^- type cast to prohibit rounding errors
                       if timestamp != double */
                     
                }
                if (input_entry->event_type == EV_receive)
                {
                   
                    timestamp LCaij_shifted;
                    k = input_entry->msg_from;
                    if ((Cij - input_entry->C_from) < ki_mmd[k * n + i])
                        ki_mmd[k * n + i] = Cij - input_entry->C_from;
                    LCij = max(LCij, input_entry->LC_from +
                               mmd[mmd_individual * (k * n + i)]);
                    LCaij_shifted = LCaij;
                    LCaij = max(LCaij, input_entry->LCa_from +
                                mmd[mmd_individual * (k * n + i)]);
                    input_entry->discrete_advance = LCaij - LCaij_shifted;
                    if (((LCaij - Cij) > AM_cldiff) && (AM_maxerr > 0))
                    {
                        AM_cldiff = LCaij - Cij;
                        AM_interval =
                            max(AM_cldiff, AM_cldiff_default) / AM_maxerr;
                    }
                }
                input_entry->LCj = LCij;
                input_entry->LCaj = LCaij;
                if (input_entry->event_type == EV_receive)
                {
                    input_entry->corresponding_send->corresponding_recv_LCa =
                        LCaij;
                    input_entry->corresponding_send->
                        corresponding_recv_LCa_known = _TRUE_;
                    /* add k to AM_processable_next if the
                       input_entry->corresponding_send is the beginning
                       of the AM_postponed_send list, and if it is not
                       already in the AM_processable list */
                    if ((input_entry->corresponding_send
                         == P[k].AM_postponed_send) && (!P[k].AM_processable))
                    {
                        P[k].AM_processable_next = AM_processable_next;
                        P[k].AM_processable = _TRUE_;
                        AM_processable_next = k;
                    }
                }
                /* storing sent events into csel and putting corresponding
                   non-empty queues into input_processable_next */
                if (input_entry->event_type == EV_send)
                {
                    /* create new csel element */
                    if ((one_csel = (csel_entry*) malloc(sizeof(csel_entry))) == NULL)
                        MemError();
                    one_csel->C = Cij;
                    one_csel->LC = LCij;
                    one_csel->LCa = LCaij;
                    one_csel->msg_from = i;
                    one_csel->msg_id1 = input_entry->msg_id1;
                    one_csel->msg_id2 = input_entry->msg_id2;
                    one_csel->send_event = input_entry;
                    /* add one_csel at the _end_ of the csel list of P[k] */
                    k = input_entry->msg_to;
                    one_csel->next = NULL;
                    if (P[k].csel_last == NULL)
                        P[k].csel_first = one_csel;
                    else
                        P[k].csel_last->next = one_csel;
                    P[k].csel_last = one_csel;
                    /* add k to input_processable_next list if it has an entry in the
                       input queue and if it is not already in the list */
                    if ((P[k].In_oldest != NULL) && (!P[k].input_processable))
                    {
                 
                        P[k].input_processable_next = input_processable_next;
                        P[k].input_processable = _TRUE_;
                        input_processable_next = k;
                    }
                }
                /* logical clock processed */
                P[i].LCaj_Cj = LCaij - Cij;

                /* recalculating gamma -- p-algorithm */
                /* pij = divident / divisor */

                /* computing divident = min(LC'-C) */
                diff_LCa_C = LCaij - Cij;
                if (i == index_min_LCa_C)
                {
                    if (diff_LCa_C > min_LCa_C)
                    {           /*look for new minimum */
                        index_min_LCa_C = 0;
                        min_LCa_C = P[0].LCaj_Cj;
                        for (k = 1; k < n; k++)
                        {
                            if (P[k].LCaj_Cj < min_LCa_C)
                            {
                                index_min_LCa_C = k;
                                min_LCa_C = P[k].LCaj_Cj;
                            }
                        }
                    }
                    else
                    {
                        min_LCa_C = diff_LCa_C;
                    }
                }
                else
                {
                    if (diff_LCa_C < min_LCa_C)
                    {
                        index_min_LCa_C = i;
                        min_LCa_C = diff_LCa_C;
                    }
                }

                /* computing divisor  */

                if (min_LCa_C == 0.0)
                {
                    sum16_LCa_C = 0.0;    /* forces start condition when min_LCa_C > 0 */
                }
                else
                {
                    if (sum16_LCa_C == 0.0)
                    {
                        /* new computation due to - start condition (sum16_LCa_C==0) */
                        sum16_LCa_C = 0;
                        for (k = 0; k < n; k++)
                        {
                            register double diff, pow16_LCa_C;
                            diff = P[k].LCaj_Cj;
                            pow16_LCa_C = pow16(diff);
                            sum16_LCa_C = sum16_LCa_C + pow16_LCa_C;
                            P[k].pow16_LCaj1_Cj1 = pow16_LCa_C;
                        }
                        max_sum16_LCa_C = sum16_LCa_C;
                    }
                    else
                    {
                        /* incremental change */
                        register double pow16_LCa_C;
                        pow16_LCa_C = pow16(diff_LCa_C);
                        sum16_LCa_C =
                            sum16_LCa_C + pow16_LCa_C - P[i].pow16_LCaj1_Cj1;
                        P[i].pow16_LCaj1_Cj1 = pow16_LCa_C;
                        max_sum16_LCa_C = max(max_sum16_LCa_C, sum16_LCa_C);
                        if (sum16_LCa_C <=
                            (PRECISION_OF_DOUBLE * 1e4) * max_sum16_LCa_C)
                        {
                            /* new computation due to risc of incorrectness */
                            sum16_LCa_C = 0;
                            for (k = 0; k < n; k++)
                            {
                                register double diff, pow16_LCa_C;
                                diff = P[k].LCaj_Cj;
                                pow16_LCa_C = pow16(diff);
                                sum16_LCa_C = sum16_LCa_C + pow16_LCa_C;
                                P[k].pow16_LCaj1_Cj1 = pow16_LCa_C;
                            }
                            max_sum16_LCa_C = sum16_LCa_C;
                        }
                    }
                }
                divisor = root16(sum16_LCa_C / sqrt((double) n));
                /* computing divisor and pij */
                pij = (divisor == 0.0 ? 0.0 : (min_LCa_C / divisor));

                if (pij <= pmin)
                {
                    p_gammaij = gamma_max;
                }
                else if (pij >= pmax)
                {
                    p_gammaij = 0.0;
                }
                else
                {
                    float x;
                    x = (pij - pmin) / (pmax - pmin);   /* 0<x<1 */
                    p_gammaij = gamma_max * (1 - x * x);
                }

                /* recalculating gamma -- q-algorithm */
                if (P[i].initialized)
                {
                    Dmaxj = Dmaxj - D_slow_factor * gamma1_init *
                        max(0, (LCij - LC_Dmaxj));
                    Dmaxj = max(0, Dmaxj);
                    LC_Dmaxj = max(LC_Dmaxj, LCij);
                }
                if (LCij - Cij > Dmaxj)
                {
                    Dmaxj = LCij - Cij;
                    LC_Dmaxj = LCij;
                }

                qij = (diff_LCa_C > 0 ? diff_LCa_C / Dmaxj : 1.0);
                if (qij <= qmin)
                {
                    q_gammaij = gamma_max;
                }
                else if (qij >= qmax)
                {
                    q_gammaij = 0.0;
                }
                else
                {
                    float x;
                    x = (qij - qmin + qij - qmax) / (qmax - qmin);      /* -1 < x < 1 */
                    q_gammaij = gamma_max * 0.25 * ((x * x - 3) * x + 2.0);
                }
                P[i].gammaj = min(p_gammaij, q_gammaij);

                /* controller processed */
	
                /* logical clock processed */
                P[i].initialized = _TRUE_;
                P[i].Cj1 = Cij;
                P[i].LCj1 = LCij;
                P[i].LCaj1 = LCaij;

                /* storing the event into the amortisation list of P[i] */
                input_entry->next = NULL;
                if (P[i].AM_newest != NULL)
                    P[i].AM_newest->next = input_entry;
                P[i].AM_newest = input_entry;
                switch (input_entry->event_type)
                {
                case EV_send:
                    if (P[i].AM_newest_send != NULL)
                        P[i].AM_newest_send->next_newer_send = input_entry;
                    P[i].AM_newest_send = input_entry;
                    if (P[i].AM_postponed_send == NULL)
                    {
                        P[i].AM_postponed_send = input_entry;
                        P[i].AM_LCa_limit = input_entry->LCaj;
                    }
                    if (P[i].AM_oldest_send == NULL)
                        P[i].AM_oldest_send = input_entry;
                    break;
                case EV_receive:
                    if (P[i].AM_newest_recv != NULL)
                        P[i].AM_newest_recv->next_newer_recv = input_entry;
                    P[i].AM_newest_recv = input_entry;
                    if (P[i].AM_todo_recv == NULL)
                        P[i].AM_todo_recv = input_entry;
                    break;
                default:
                    ;
                }
                if (P[i].AM_oldest == NULL)
                {
                    P[i].AM_oldest = input_entry;
                }
                if (P[i].AM_postponed_send == NULL)
                    P[i].AM_LCa_limit = input_entry->LCaj;
                /* add i to AM_processable_next if it is not
                   already in the AM_processable list */
                if (!P[i].AM_processable)
                {
                    P[i].AM_processable_next = AM_processable_next;
                    P[i].AM_processable = _TRUE_;
                    AM_processable_next = i;
                }
            }
            else
            {                   /* else -- not input_processable */
                /* remove this input queue from input_processable_next list */
                input_processable_next = P[i].input_processable_next;
                P[i].input_processable = _FALSE_;
                P[i].input_processable_next = NO_ID;
            }                   /* else -- not input_processable */
        }                       /* if (input_processable_next != NO_ID) */
        /* processing all amortisation lists */
        /* in general this are at maximum 2 lists: the list of the
           input event, and the list of a corresponding send event */
        while (AM_processable_next != NO_ID)
        {
            i = AM_processable_next;
            /* process all postponed send events */
            while ((P[i].AM_postponed_send !=
                    NULL)
                   && P[i].AM_postponed_send->corresponding_recv_LCa_known)
            {
                P[i].AM_postponed_send =
                    P[i].AM_postponed_send->next_newer_send;
                P[i].AM_LCa_limit =
                    (P[i].AM_postponed_send ==
                     NULL ? P[i].AM_newest->LCaj : P[i].AM_postponed_send->
                     LCaj);
            }
            /* process the amortization of all receive events
               with LCaj <= AM_LCa_limit */
            while ((P[i].AM_todo_recv != NULL)
                   && (P[i].AM_todo_recv->LCaj <= P[i].AM_LCa_limit))
            {
                if ((P[i].AM_todo_recv->
                     discrete_advance > 0)
                    && (P[i].AM_oldest != P[i].AM_todo_recv)
                    && (AM_maxerr > 0))
                {   /* ---- AMORTISATION ---- */
                    event_entry *first_AM_point;
                    event_entry *current_event;
                    event_entry *found;
                    event_entry *last_found_AM_point;
                    event_entry *left_AM_point;
                    event_entry *right_AM_point;
                    timestamp LCa_lower, LCa_left, LCa_current;
                    timestamp LCa_upper, LCa_right;
                    double AMORTI_lower, AMORTI_left;
                    double AMORTI_upper, AMORTI_right;
                    double allowed_current;
                    double min_gradient, gradient;

                    /* initialisation */
                    AMORTI_upper = P[i].AM_todo_recv->discrete_advance;
                    LCa_upper = P[i].AM_todo_recv->LCaj - AMORTI_upper;

                    /* if first event in the amortisation list is equal to
                       the overall first event on this process then
                       AMORTI_lower can be choosen as minimum of all
                       allowed amortisations at intermediate send events
                       and the AMORTI_upper */

                    if (P[i].AM_oldest->not_the_first_event)
                    {
                        LCa_lower =
                            min(P[i].AM_oldest->LCaj,
                                max(P[i].AM_LCa_next_before_oldest,
                                    P[i].AM_LCa_limit - AM_interval));
                        AMORTI_lower = 0;
                    }
                    else
                    {
                        LCa_lower = P[i].AM_oldest->LCaj;
                        AMORTI_lower = AMORTI_upper;
                        current_event = P[i].AM_oldest_send;
                        while ((current_event != NULL)
                               && (current_event->LCaj < LCa_upper))
                        {
                            procnum k;
                            k = current_event->msg_to;
                            LCa_current = current_event->LCaj;
                            allowed_current =
                                current_event->
                                corresponding_recv_LCa
                                - mmd[mmd_individual * (i * n + k)] -
                                current_event->LCaj;
                            AMORTI_lower = min(AMORTI_lower, allowed_current);
                            current_event = current_event->next_newer_send;
                        }       /*while (...&& (current_event->LCaj < LCa_upper)) */
                    }
                    /* look for the amortisation points, i.e. the lower
                       convex hull from the begin of the amortisation interval
                       over all send events until the receive event with
                       the discrete_advance > 0 */
                    if (P[i].AM_oldest_send == NULL)
                    {
                        /* there is no intermediate send event
                           between lower_C and upper_C */
                        first_AM_point = NULL;
                    }
                    else
                    {           /*(P[i].AM_oldest_send != NULL) */
                        /* computing the convex hull under all
                           (LCa_max_allowed - LCa) values at each send event
                           and AMORTI_lower at LCa_lower
                           P[i].AM_todo_recv->discrete_advance at LCa_upper,
                           with
                           LCa_max_allowed = corresponding_recv_LCa - mmd */
                        first_AM_point = NULL;
                        last_found_AM_point = NULL;
                        do      /* while (found != NULL) */
                        {
                            found = NULL;
                            if (last_found_AM_point == NULL)
                            {
                                LCa_left = LCa_lower;
                                AMORTI_left = AMORTI_lower;
                                current_event = P[i].AM_oldest_send;
                                if (current_event->LCaj == LCa_lower)
                                    current_event =
                                        current_event->next_newer_send;
                            }
                            else
                            {
                                LCa_left = last_found_AM_point->LCaj;
                                AMORTI_left =
                                    last_found_AM_point->amortisation;
                                current_event =
                                    last_found_AM_point->next_newer_send;
                            }
                            min_gradient =
                                (AMORTI_upper - AMORTI_left) / (LCa_upper -
                                                                LCa_left);
                            while ((current_event != NULL)
                                   && (current_event->LCaj < LCa_upper))
                            {
                                procnum k;
                                k = current_event->msg_to;
                                LCa_current = current_event->LCaj;
                                allowed_current
                                    =
                                    current_event->
                                    corresponding_recv_LCa
                                    - mmd[mmd_individual * (i * n + k)] -
                                    current_event->LCaj;
                                gradient =
                                    (allowed_current -
                                     AMORTI_left) / (LCa_current - LCa_left);
                                if (gradient <= min_gradient)
                                {
                                    found = current_event;
                                    min_gradient = gradient;
                                }
                                current_event =
                                    current_event->next_newer_send;
                            }   /*while (current_event->LCaj < LCa_upper) */
                            if (found != NULL)
                            {
                                if (first_AM_point == NULL)
                                {
                                    first_AM_point = found;
                                }
                                else
                                {
                                    last_found_AM_point->next_AM_point =
                                        found;
                                }
                                last_found_AM_point = found;
                                last_found_AM_point->
                                    amortisation
                                    =
                                    AMORTI_left
                                    +
                                    min_gradient *
                                    (last_found_AM_point->LCaj - LCa_left);
                            }   /*if (found != NULL) */
                        }
                        while (found != NULL);
                        if (last_found_AM_point != NULL)
                            last_found_AM_point->next_AM_point =
                                P[i].AM_todo_recv;
                    }           /*(P[i].AM_oldest_send != NULL) */
                    P[i].AM_todo_recv->amortisation = AMORTI_upper;
                    P[i].AM_todo_recv->next_AM_point = NULL;

                    /* now the amortisation can be done */
                    left_AM_point = P[i].AM_oldest;
                    right_AM_point = first_AM_point;
                    if (right_AM_point == NULL)
                        right_AM_point = P[i].AM_todo_recv;
                    left_AM_point->
                        amortisation =
                        AMORTI_lower +
                        (right_AM_point->
                         amortisation -
                         AMORTI_lower) *
                        (left_AM_point->LCaj -
                         LCa_lower) / (right_AM_point->LCaj - LCa_lower);
                    while (right_AM_point != NULL)
                    {
                        LCa_left = left_AM_point->LCaj;
                        LCa_right = right_AM_point->LCaj;
                        AMORTI_left = left_AM_point->amortisation;
                        AMORTI_right = right_AM_point->amortisation;
                        if (right_AM_point == P[i].AM_todo_recv)
                            LCa_right = LCa_right - AMORTI_right;
                        if (AMORTI_right > 0)
                        {
                            current_event = left_AM_point;
                            gradient =
                                (AMORTI_right - AMORTI_left) / (LCa_right -
                                                                LCa_left);
                            while (current_event != right_AM_point)
                            {
                                current_event->
                                    LCaj =
                                    current_event->
                                    LCaj +
                                    AMORTI_left + (current_event->LCaj -
                                                   LCa_left) * gradient;
                                /* */
                                current_event = current_event->next;
                            }
                        }       /*if (AMORTI_right > 0) */
                        left_AM_point = right_AM_point;
                        right_AM_point = right_AM_point->next_AM_point;
                    }           /* while (right_AM_point != NULL) */
                }               /* ---- AMORTISATION ---- */
                P[i].AM_todo_recv = P[i].AM_todo_recv->next_newer_recv;
            }
            /* move all events before (AM_LCa_limit - AM_interval)
               to the output queue */

			
            while ((P[i].AM_oldest != NULL)
                   && (P[i].AM_oldest->LCaj <
                       P[i].AM_LCa_limit - AM_interval))
            {
                input_entry = P[i].AM_oldest;
                LCaij = input_entry->LCaj;
                /* delete input_entry on amortisation list */
                if (P[i].AM_oldest_send == P[i].AM_oldest)
                    P[i].AM_oldest_send =
                        P[i].AM_oldest_send->next_newer_send;
                if (P[i].AM_newest_recv == P[i].AM_oldest)
                    P[i].AM_newest_recv = NULL;
                if (P[i].AM_newest_send == P[i].AM_oldest)
                    P[i].AM_newest_send = NULL;
                if (P[i].AM_newest == P[i].AM_oldest)
                    P[i].AM_newest = NULL;
                P[i].AM_LCa_next_before_oldest = P[i].AM_oldest->LCaj;
                P[i].AM_oldest = P[i].AM_oldest->next;
                /* move input_entry to the output buffer */
                input_entry->next = NULL;
                if (P[i].output_merge_node->event == NULL)
                {
                    AM_out_filled++;
                    P[i].output_merge_node->event = input_entry;
                    P[i].output_merge_newest_event = input_entry;
                }
                else
                {
                    P[i].output_merge_newest_event->next = input_entry;
                    P[i].output_merge_newest_event = input_entry;
                }
            }                   /* while ((... < P[i].AM_LCa_limit - AM_interval)) */
            /* remove this AM list from AM_processable_next list */
            AM_processable_next = P[i].AM_processable_next;
            P[i].AM_processable = _FALSE_;
            P[i].AM_processable_next = NO_ID;
        }                       /* while (AM_processable_next != NO_ID) */

        /* output of all allowed elements from the output queue */
        /* begin of merge-sort algorithm */
       
		if (cleanUp == true)
		{
			
            for (i = 0; i < n; i++)
            {
                if (P[i].output_merge_node->event == NULL)
                {
                    AM_out_filled++;
                    if (P[i].AM_oldest == NULL)
                        P[i].output_merge_node->event = final_event;
                    else
                    {
                        P[i].output_merge_node->event = P[i].AM_oldest;
                        P[i].AM_newest->next = final_event;
                    }
                }
                else
                {
                    if (P[i].AM_oldest == NULL)
                        P[i].output_merge_newest_event->next = final_event;
                    else
                    {
                        P[i].output_merge_newest_event->next = P[i].AM_oldest;
                        P[i].AM_newest->next = final_event;
                    }
                }
                P[i].output_merge_newest_event = NULL;  /*unused in the future */
            }
		}



        while ((AM_out_filled == n) && (merge_root->event != final_event))
        {
            if (merge_root->event != NULL)
            {  
                output_entry = merge_root->event;
                merge_root->event = NULL;
                /* write output_entry to output file and free it */
                OutRec = output_entry->record;
				OutRec->Cij = output_entry->LCaj; //added by me
                tau_Write_Record(outputFile, OutRec);
                i = output_entry->i;
                Cij = output_entry->Cj;
                LCij = output_entry->LCj;
                LCaij = output_entry->LCaj;
                if (output_entry->not_the_first_event)
                {
                    err =
                        (Cij > P[i].Out_Cj1
                         ?
                         ((LCaij - P[i].Out_LCaj1) -
                          (Cij - P[i].Out_Cj1)) / (Cij -
                                                   P[i].Out_Cj1) : 99.999);
                    abs_err = Abs(err);
                    if (abs_err == 0.0)
                    {
                        count_err_eq_zero++;
                    }
                    else if (abs_err <= AM_maxerr)
                    {
                        count_err_le_maxerr++;
                        sum_err_le_maxerr = sum_err_le_maxerr + abs_err;
                        if (abs_err > max_err_le_maxerr)
                            max_err_le_maxerr = abs_err;
                    }
                    else        /* (abs_err > AM_maxerr) */
                    {
                        count_err_gt_maxerr++;
                        sum_err_gt_maxerr = sum_err_gt_maxerr + abs_err;
                        if (abs_err > max_err_gt_maxerr)
                            max_err_gt_maxerr = abs_err;
                    }
                    if (abs_err < min_err)
                        min_err = abs_err;
                }
                P[i].Out_Cj1 = Cij;     /* for next iteration */
                P[i].Out_LCaj1 = LCaij;
                tau_free(OutRec); 
                free(output_entry);
            }
            /* else  the tree is not yet fully filled,
               i.e. we are in the initalization phase */
            /* now get the next event */
            node = merge_root;
            while (!node->final)
            {
                if (node->left->event == NULL)
                    /* goto left to init. parts of the left tree  */
                {
                    node = node->left;
                }
                else if (node->right->event == NULL)
                    /* goto right to init. parts of the right tree */
                {
                    node = node->right;
                }
                else
                {               /* compare, fetch smaller one and goto its node */
                    /* comparison */
                    if (node->left->event->LCaj <= node->right->event->LCaj)
                        nextnode = node->left;
                    else
                        nextnode = node->right;
                    /* fetch */
                    node->event = nextnode->event;
                    if (nextnode->final)
                    {
                        nextnode->event = nextnode->event->next;
                        if (nextnode->event == NULL)
                            AM_out_filled--;
                    }
                    else
                    {
                        nextnode->event = NULL;
                    }
                    /* goto nextnode */
                    node = nextnode;
                }
            }                   /* while (!node.final) */
        }                       /* while (AM_out_filled == n) */
        /* end of merge-sort algorithm */
	}
    while (input_processable_next != NO_ID);

}



/*  Corret Time Finish
*  This is an option to output various information about what program has done.
*/
void correct_time_finish()
{
    FILE *to_optfile = NULL;
    procnum i, k;
    double max_LCa_C, multiplier;
    char *format, *unit;
    double min_ki_mmd, max_ki_mmd, sum_ki_mmd, avg_ki_mmd;
    int cnt_ki_mmd;

    if (to_optfilename != NULL)
        to_optfile = fopen(to_optfilename, "w");

    printf("number of nodes = %1d\n", n);
    if (verbose)
    {
        max_LCa_C = 0;
        for (i = 0; i < n; i++)
            max_LCa_C = max(max_LCa_C, P[i].LCaj1 - P[i].Cj1);
        max_LCa_C = max_LCa_C * clock_period;
        if (max_LCa_C >= 1e+0)
        {
            multiplier = 1e+0;
            unit = "sec";
        }
        else if (max_LCa_C >= 1e-3)
        {
            multiplier = 1e+3;
            unit = "ms";
        }
        else if (max_LCa_C >= 1e-6)
        {
            multiplier = 1e+6;
            unit = "us";
        }
        else if (max_LCa_C >= 1e-9)
        {
            multiplier = 1e+9;
            unit = "ns";
        }
        else
        {
            multiplier = 1e+12;
            unit = "ps";
        }
        if (max_LCa_C * multiplier >= 100)
            format = " %6.2lf";
        else if (max_LCa_C * multiplier >= 10)
            format = " %6.3lf";
        else
            format = " %6.4lf";
        printf
            ("\nDifferences new timestamps - old timestamps at the last event in [%s]\n",
             unit);
        printf("Node  i=");
        for (i = 0; i < min(10, n); i++)
            printf(" %6d", i);
        for (i = 0; i < n; i++)
        {
            if ((i % 10) == 0)
                printf("\n%5d+i ", i);
            printf(format,
                   (P[i].LCaj1 - P[i].Cj1) * clock_period * multiplier);
        }
        printf("\n");
    }                           /*if (verbose) */
    if (AM_maxerr > 0)
    {
        AM_cldiff = AM_cldiff * clock_period * 1e6;
        if (verbose)
        {
            printf
                ("\nMaximal clock differences = %4.2lf us,  used start-value = %4.2lf us\n",
                 AM_cldiff, AM_cldiff_default * clock_period * 1e6);
            if (AM_cldiff >
                1.10 /* +10% */  * AM_cldiff_default * clock_period * 1e6)
                printf
                    ("\nADVICE -- use for correcting similar tracefiles:  -c %1.0lf\n",
                     AM_cldiff);
        }                       /*if (verbose) */
    }
    min_delta = min_delta * clock_period * 1e6  /*[us] */
        ;
    if (verbose)
    {
        printf
            ("\nMinimal difference between the original timestamps of two\n"
             "events in the same process = %5.3lf us\n", min_delta);

        if ((min_delta > 0)
            && (min_delta < delta_default * clock_period * 1e6))
        {
            printf
                ("\nADVICE -- use for correcting similar tracefiles:  -d %5.3lf\n",
                 min_delta);
        }
    }                           /*if (verbose) */
    max_ki_mmd = -MAX_DOUBLE;
    min_ki_mmd = MAX_DOUBLE;
    cnt_ki_mmd = 0;
    sum_ki_mmd = 0;
    for (k = 0; k < n; k++)     /* sender */
        for (i = 0; i < k; i++) /* receiver */
            if ((ki_mmd[k * n + i] < MAX_DOUBLE)
                && (ki_mmd[i * n + k] < MAX_DOUBLE))
            {
                double delay;
                delay =
                    (ki_mmd[k * n + i] +
                     ki_mmd[i * n + k]) / 2 * clock_period * 1e6
                    /*[us] */
                    ;
                ki_mmd[k * n + i] = delay;
                ki_mmd[i * n + k] = delay;
                if (delay < min_ki_mmd)
                    min_ki_mmd = delay;
                if (delay > max_ki_mmd)
                    max_ki_mmd = delay;
                cnt_ki_mmd++;
                sum_ki_mmd = sum_ki_mmd + delay;
            }
            else
            {
                ki_mmd[k * n + i] = MAX_DOUBLE;
                ki_mmd[i * n + k] = MAX_DOUBLE;
            }
    avg_ki_mmd = (cnt_ki_mmd > 0 ? sum_ki_mmd / cnt_ki_mmd : 0);
    if (verbose)
    {
        if (cnt_ki_mmd > 0)
        {
            printf("\nMinimal message delay between 2 processes =\n"
                   "   (min=%5.3lfus,  avg=%5.3lfus,  max=%5.3lfus)\n",
                   min_ki_mmd, avg_ki_mmd, max_ki_mmd );
            printf ("with min/avg/max over all paires of processes with messages in both directions\n");
            printf ("These values have include an error <= 2*(max. clock drifts over the whole time)\n");
            if ((min_ki_mmd > 0) &&
                (Abs (mmd_default * clock_period * 1e6 - mmd_factor * min_ki_mmd) >
                (0.05 * mmd_default * clock_period * 1e6)))
            {
                /* the 80% are necessary because the minimum is computed
                   as average between 2 messages in the opposite directions
                   and we assume that one has the minimal message delay (mmd),
                   but the other has 1.50*mmd, and therefore the average has
                   1.25*mmd, and therefore mmd is 80% of 1.25*mmd. */
                printf
                    ("\nADVICE -- use for correcting similar tracefiles:  -m %5.3lf\n",
                     0.80 * min_ki_mmd);
                printf
                    ("but this value can be meaningless if the clock drifts are to large.\n");
            }
            if ((min_ki_mmd > 0) && (max_ki_mmd > 1.5 * min_ki_mmd) &&
                 ! mmd_individual)
            {
                printf("\n"
                       "ADVICE -- for correcting this tracefile one may get better results if one\n"
                       "uses individual minimal message delays for each pair of processes, i.e.\n");
                if (to_optfilename == NULL)
                {
                    printf ("start two passes, the first with -w timecorrect.opt, "
                            "and the second with -r timecorrect.opt \n");
                }
                else
                {
                    printf("start a second pass with the same input file and "
                           "the option -r %s\n", to_optfilename);
                    if (to_optfile == NULL)
                        printf ("and make that in this first pass the optionfile '%s' is writable!\n",
                                to_optfilename);
                }
            }
        }                       /* cnt_ki_mmd > 0 */
        else
        {                       /* cnt_ki_mmd == 0 */
            printf("\n"
                   "The minimal message delay cannot be computed because there is \n"
                   "not a pair of processes that have exchanged messages in both "
                   "directions.\n");
        }                       /* cnt_ki_mmd == 0 */
    }                           /*if (verbose) */
    if (verbose)
    {
        printf ("\nThe correction of the timestamps causes additional errors on the length of\n"
                "time intervals between two successive events in the same process:\n");
        printf(" %7ld intervals with            error  =      0%%\n",
               count_err_eq_zero);
        printf(" %7ld intervals with      0%% <  error <= %6.3f%%",
               count_err_le_maxerr, 100 * AM_maxerr);
        if (count_err_le_maxerr > 0)
            printf (" : avg.=%6.3f%%, max=%6.3f%%",
                     100 * sum_err_le_maxerr / count_err_le_maxerr,
                     100 * max_err_le_maxerr);
        printf("\n");
        printf (" %7ld intervals with %6.3f%% <  error           ",
                count_err_gt_maxerr, 100 * AM_maxerr);
        if (count_err_gt_maxerr > 0)
            printf (" : avg.=%6.3f%%, max=%6.3f%%",
                    100 * sum_err_gt_maxerr / count_err_gt_maxerr,
                    100 * max_err_gt_maxerr);
        printf("\n");
        printf("Error summary:\n");
        printf
            (" %7ld intervals with %6.3f%% <= error <= %6.3f%%",
             count_err_eq_zero
             +
             count_err_le_maxerr
             + count_err_gt_maxerr, 100 * min_err,
             100 * max(max_err_le_maxerr, max_err_gt_maxerr));
        if ((count_err_eq_zero + count_err_le_maxerr + count_err_gt_maxerr) > 0)
            printf(" : avg.=%6.3f%%\n",
                   100 * (sum_err_le_maxerr +
                          sum_err_gt_maxerr) / (count_err_eq_zero +
                                                count_err_le_maxerr +
                                                count_err_gt_maxerr));
        if (mmd_individual)
        {
            printf("\n"
                   "The local error rates may be greater due to the use of individual minimal\n"
                   "message delays, but the error for time intervals between events on different\n"
                   "nodes may be significantly smaller, but this can not be measured.\n");
        }
    }                           /*if (verbose) */
    if (to_optfile != NULL)
    {
        if (AM_maxerr > 0)
            fprintf(to_optfile, "-c %4.2lf\n", AM_cldiff);
        if (min_delta > 0)
            fprintf(to_optfile, "-d %5.3lf\n", min_delta);
        if ((cnt_ki_mmd > 0) && (min_ki_mmd > 0))
        {
            fprintf(to_optfile, "-m %1.0lf\n", min_ki_mmd);
            for (k = 0; k < n; k++)     /* sender */
                for (i = 0; i < n; i++) /* receiver */
                    if (ki_mmd[k * n + i] != MAX_DOUBLE)
                        fprintf(to_optfile, "-i %1d_%1d_%1.0lf\n", k, i,
                                ki_mmd[k * n + i]);
        }
        fclose(to_optfile);
        printf("write options to file '%s'\n", to_optfilename);
    }
    if (In_length > 0)
    {
/*        elg_error_msg("%1d records not processed\n"
                      "e.g. due to the lack of a corresponding SENDMSG record\n",
                      In_length);
*/
		printf("%1d records not processed \n"
					"e.g due to the lack of a a corresponding SENDMSG record\n",
					In_length);
		exit(EXIT_FAILURE);
    }
    if (P != NULL)
        free(P);
    if (mmd_individual)
        free(mmd);
}

/*--------------------------------------------------------------------*/


/* Correct Time
*  This is the main loop of the program.  Below is the loop information for it.
*      -main message loop 
*   
*   - open input and output trace files 
*   - initialize record handler
*   - read input file record by record
*   - call record handlig function
*   - finalize record handler
*   - print statistics
*   - close input and output files
*/
void correct_time(const char* file1trc, const char* file1edf, const char* file2trc, const char* file2edf)
{
    //void *in, *out;
    //UI4 numrec = 0, done = 0, temp = 0;
    //struct stat statbuf;
	static UI4 lastdef = 0;

	Ttf_FileHandleT infile;
	Ttf_FileHandleT outfile;  

	//Need some error checking
	//Open the input file.
	infile  = Ttf_OpenFileForInput (file1trc, file1edf);
	if (infile == NULL)
	{ 
	printf("Cannot open files %s %s for reading", file1trc, file1edf); 
	exit(EXIT_FAILURE);
	} 
	
	//Open the output file.
	outfile = Ttf_OpenFileForOutput(file2trc, file2edf);  
	if (outfile == NULL)
	{ 
	printf("Cannot open files %s %s for reading", file2trc, file2edf); 
	exit(EXIT_FAILURE);
	} 

	//Setup the global variables for the time correction algorithm.
    correct_time_init();

	//First pass through the file to get the state names and node/thread declarations.  Count the number of processes,
	//create the P array, and output the state names and node/thread declarations to the output file.
	lastdef = handle_def_record(infile, outfile);    
 
	if (lastdef == _FALSE_) {
			printf("I haven't finished counting the number of processors \n");
			exit(EXIT_FAILURE);
	}

	//At the end of the file.  Close it and reopen it.
	Ttf_CloseFile(infile);
	infile = Ttf_OpenFileForInput(file1trc, file1edf);


	//Begin readjusting all the times.
	//Read in all the records dealing with sends, recieves, entering and leaving states to readjust them.
	if (infile != NULL) 
    {
        handle_event_record(infile,outfile);
    }

	//Have it start writing information to the output file.
	correct_time_exec(NO_ID, outfile,true);

	/* print statistics */
    correct_time_finish();

    /* close files */
	Ttf_CloseFile(infile);
	Ttf_CloseOutputFile(outfile);
}

/* Parse Option
*  This reads in the additional options that the user has inputed into the program.
*  I currently don't know what this is doing and should be looked at and adjusted if need be.
*/
void parse_option(const char ch, const char* val)
{
    procnum immd_from, immd_to;
    double immd_us;
    immd_entry *immd;
    switch (ch)
    {
    case 'c':
        AM_cldiff_default = atof(val);  /*[usec] */
		if (AM_cldiff_default < 0) 
		{
            printf("'-c' option must be positive number");
			exit(EXIT_FAILURE);
		}
        AM_cldiff_default *= 1e-6 / clock_period;
        break;
    case 'd':
        delta_default = atof(val);
        if (delta_default <= 0)
		{
            printf("'-d' option must be positive number");
			exit(EXIT_FAILURE);
		}
        delta_default *= 1e-6 / clock_period;
        break;
    case 'e':
        AM_maxerr = atof(val) * 0.01 /*[1%] */ ;
        if (AM_maxerr < 0)
		{
            printf("'-e' option must be not negative number");
			exit(EXIT_FAILURE);
		}
        break;
    case 'f':
        mmd_factor = atof(val);
        if (mmd_factor <= 0)
		{
            printf("'-f' option must be positive number");
			exit(EXIT_FAILURE);
		}
        break;
    case 'i':
        if (3 != sscanf(val, "%d_%d_%lf", &immd_from, &immd_to, &immd_us))
		{
            printf("'-i' option: invalid format");
			exit(EXIT_FAILURE);
		}
        if (immd_us <= 0)
		{
            printf("'immd_us' must be positive");
			exit(EXIT_FAILURE);
		}
        if ((immd = (immd_entry*) malloc(sizeof(immd_entry))) == NULL)
            MemError();
        if (immd_first == NULL)
            immd_first = immd;
        else
            immd_last->next = immd;

        immd_last = immd;
        immd->next = NULL;
        immd->from = immd_from;
        immd->to = immd_to;
        immd->immd = immd_us * 1e-6 / clock_period;
        break;
    case 'm':
        mmd_default = atof(val);        /*[usec] */
        if (mmd_default <= 0)
		{
            printf("'-m' option  must be positive number");
			exit(EXIT_FAILURE);
		}
        mmd_default *= 1e-6 / clock_period;
        break;
    case 'r':
        from_optfilename = "timecorrect.opt";
        break;
    case 'R':
        from_optfilename = strdup(val);
        break;
    case 'v':
        verbose = _TRUE_;
        break;
    case 'w':
        to_optfilename = "timecorrect.opt";
        break;
    case 'W':
        to_optfilename = strdup(val);
        break;
    case '?':
    case 'h':
        print_help();
        exit(0);
        break;
    default:
        print_usage();
        printf("invalid option: '%c'", ch);
		exit(EXIT_FAILURE);
    }
}

/* Read Options
*  This function was used previously to open a file, check to see if it opened
*  and read the parsing info from it.
*/
void read_options(const char* filename)
{
    FILE *optfile;
    char optname[100];
    char optvalue[100];
    int ret;

    optfile = fopen(filename, "r");
    if (optfile == NULL)
	{ 
	printf("Cannot open file %s for reading", filename); 
	exit(EXIT_FAILURE);
	} 

    while ((ret = fscanf(optfile, "-%s %s ", optname, optvalue)) != EOF)
    {
        if (ret != 2)
		{ 
		printf("invalid options file '%s' optname='%s', optvalue='%s'",
		    filename, optname, optvalue); 
		exit(EXIT_FAILURE);
		}
        parse_option(optname[0], optvalue);
    }
    fclose(optfile);
    printf("read options from file '%s'\n", filename);
}

/* Parse Options
*  Reads in the command line options and adjusts the program to those options.
*  I don't really know what this function does, and have not really modified it.
*  Should be looked over and changed if need be.
*/ 
int parse_options(int argc, char* argv[])
{
    int ch;
    immd_entry *immd;

    /* read command line options */
    while ((ch = getopt(argc, argv, "c:d:e:f:i:m:R:W:rwvh?")) != EOF)
        parse_option(ch, optarg);
    
    /* read option file if any */
    if (from_optfilename)
        read_options(from_optfilename);

    /* aplay minimal message delay factor to all related items */
    mmd_default *= mmd_factor;
    immd = immd_first;
    while (immd != NULL)
    {
        immd->immd *= mmd_factor;
        immd = immd->next;
    }
    return optind;
}

/*--------------------------------------------------------------------*/

/* Main
*  Sets the default options, reads in the input and output trace files, then calls
*  the main algorithm to deal with the files.
*/
int main(int argc, char **argv)
{
    printf ("Processing the files.\n");

    //char *infile = NULL;  //Might not need these
    //char *outfile = NULL; //Might not need these
    int opt_ind;

	char *trace_file;
    char *edf_file;
    char *trace_file2;
    char *edf_file2;
	
    /* set default options */
    default_opt();

	//Might have to take care of the below one later.
    /* parse command line, read options from file */
    opt_ind = parse_options(argc, argv); 

    /* check if input and output file names are present */
    if ((argc - opt_ind) != 4)//  NOTE: need to put this back in
    {
        print_usage();
        return 1;
    }

	trace_file = argv[1];
	edf_file = argv[2];
	trace_file2 = argv[3];
	edf_file2 = argv[4];
	

    /* check if input and output file names differs */
    /*infile = argv[optind];
    outfile = argv[optind + 1];
    if (strcmp(infile, outfile) == 0)
        elg_error_msg("<infile> has same name as <outfile> has");
    */

	/* correct timestamps for event records */    
	correct_time(trace_file, edf_file, trace_file2, edf_file2);

    return 0;
}
