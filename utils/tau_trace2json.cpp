/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2003  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, Germany                                     **
****************************************************************************/
/***************************************************************************
**	File 		: tau_trace2json.cpp				  **
**	Description 	: TAU trace to json format converter tool         **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 	                  **
***************************************************************************/
#include <TAU_tf.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;
int debugPrint = 0;
int jsonPrint = 1;
int chromeFormat = 0;
int ignoreAtomic = 0;
int printstdout = 0;
const char* filename="events.json";
#define dprintf if (debugPrint) printf

ofstream json_event_out;
ofstream json_index_out;
//#define json_index_out cout

/* map of metadata */
map<string,string> my_metadata;

/* map of user events */
struct my_user_event {
  int _id;
  string _name;
  bool _mi;
  my_user_event(int id, string name, bool mi) : _id(id), _name(name), _mi(mi) {};
};
map<int,my_user_event*> my_user_event_set;

/* map of groups */
struct my_group {
  int _id;
  string _name;
  my_group(int id, string name) : _id(id), _name(name) {};
};
map<int,my_group*> my_group_set;

/* map of states */
struct my_state {
  int _event_id;
  int _group_id;
  string _name;
  my_state(int event_id, int group_id, string name) : _event_id(event_id),
        _group_id(group_id), _name(name) {};
};
map<int,my_state*> my_state_set;

/* map of threads */
struct my_thread {
  int _node_id;
  int _thread_id;
  string _name;
  my_thread(int node_id, int thread_id, string name) : _node_id(node_id),
        _thread_id(thread_id), _name(name) {};
};
set<my_thread*> my_thread_set;

/* implementation of callback routines */
map< pair<int,int>, int, less< pair<int,int> > > EOF_Trace;
int EndOfTrace = 0;  /* false */
/* implementation of callback routines */
int EnterState(void *userData, double time,
		unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
  dprintf("Entered state %d time %g nid %d tid %d\n",
		  stateid, time, nodeid, tid);

  if(chromeFormat){
		json_event_out << "{";
	  //json_event_out << "\t\t\t\"event-id\": \"" << stateid << "\",\n";
	  json_event_out << "\"name\": " << my_state_set[stateid]->_name << ", ";
		json_event_out << "\"cat\": \"TAU\", ";
		//Event type. B=Begin.
		json_event_out << "\"ph\": \"B\", ";
	  json_event_out << "\"ts\": \"" << time << "\", ";
	  json_event_out << "\"pid\": \"" << nodeid << "\", ";
	  json_event_out << "\"tid\": \"" << tid << "\" ";
	  json_event_out << "},\n";
  }
	else{
  json_event_out << "{";
  json_event_out << "\"event-type\": \"entry\", ";
  //json_event_out << "\t\t\t\"event-id\": \"" << stateid << "\",\n";
  json_event_out << "\"name\": " << my_state_set[stateid]->_name << ", ";
  json_event_out << "\"time\": \"" << time << "\", ";
  json_event_out << "\"node-id\": \"" << nodeid << "\", ";
  json_event_out << "\"thread-id\": \"" << tid << "\" ";
  json_event_out << "},\n";
  }
  return 0;
}

int LeaveState(void *userData, double time, unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
  dprintf("Leaving state %d time %g nid %d tid %d\n", stateid, time, nodeid, tid);
	dprintf("chromeFormat: %d\n", chromeFormat);
	if(chromeFormat){
		dprintf("In chrome leave\n");
		json_event_out << "{";
	  //json_event_out << "\t\t\t\"event-id\": \"" << stateid << "\",\n";
	  json_event_out << "\"name\": " << my_state_set[stateid]->_name << ", ";
		json_event_out << "\"cat\": \"TAU\", ";
		//Event type. E=End.
		json_event_out << "\"ph\": \"E\", ";
	  json_event_out << "\"ts\": \"" << time << "\", ";
	  json_event_out << "\"pid\": \"" << nodeid << "\", ";
	  json_event_out << "\"tid\": \"" << tid << "\" ";
	  json_event_out << "},\n";
  }
	else{
  json_event_out << "{ ";
  json_event_out << "\"event-type\": \"exit\", ";
  //json_event_out << "\"event-id\": \"" << stateid << "\", ";
  json_event_out << "\"name\": " << my_state_set[stateid]->_name << ", ";
  json_event_out << "\"time\": \"" << time << "\", ";
  json_event_out << "\"node-id\": \"" << nodeid << "\", ";
  json_event_out << "\"thread-id\": \"" << tid << "\" ";
  json_event_out << "},\n";
  }
  return 0;
}


int ClockPeriod( void*  userData, double clkPeriod )
{
  dprintf("Clock period %g\n", clkPeriod);
  char buf[1024];
  snprintf(buf, sizeof(buf),  "%g", clkPeriod);
  //my_metadata["clock-period"] = std::to_string(clkPeriod);
  my_metadata["clock-period"] = string(buf);
  my_metadata["clock-units"] = "seconds";
  return 0;
}

std::string escape_json(const char *unescaped) {
	  std::string s(unescaped);
    std::ostringstream o;
    for (char const &c:s){ //(auto c = s.cbegin(); c != s.cend(); c++) {
        switch (c) {
        //case '"': o << "\\\""; break; //This catches the leading and trailing quote and internal quotes should already be escaped. Revisit if needed.
        case '\\': o << "\\\\"; break;
        case '\b': o << "\\b"; break;
        case '\f': o << "\\f"; break;
        case '\n': o << "\\n"; break;
        case '\r': o << "\\r"; break;
        case '\t': o << "\\t"; break;
        default:
            if ('\x00' <= c && c <= '\x1f') {
                o << "\\u"
                  << std::hex << std::setw(4) << std::setfill('0') << (int)c;
            } else {
                o << c;
            }
        }
    }
    return o.str();
}

int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
  dprintf("DefThread nid %d tid %d, thread name %s\n",
		  nodeToken, threadToken, threadName);
  EOF_Trace[pair<int,int> (nodeToken,threadToken) ] = 0; /* initialize it */
  my_thread *g = new my_thread(nodeToken, threadToken, threadName);
  my_thread_set.insert(g);
  return 0;
}

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
      /* If there's any processor that is not over, then the trace is not over */
    }
  }
  return 0;
}

int DefStateGroup( void *userData, unsigned int stateGroupToken,
		const char *stateGroupName )
{
  dprintf("StateGroup groupid %d, group name %s\n", stateGroupToken,
		  stateGroupName);
  my_group *g = new my_group(stateGroupToken, stateGroupName);
  my_group_set[stateGroupToken] = g;
  return 0;
}

int DefState( void *userData, unsigned int stateToken, const char *stateName,
		unsigned int stateGroupToken )
{
  dprintf("DefState stateid %d stateName %s stategroup id %d\n",
		  stateToken, stateName, stateGroupToken);
  my_state *s = new my_state(stateToken, stateGroupToken, stateName);
  my_state_set[stateToken] = s;
  return 0;
}

int DefUserEvent( void *userData, unsigned int userEventToken,
		const char *userEventName, int monotonicallyIncreasing )
{
  const char *escapedName = escape_json(userEventName).c_str();

  dprintf("DefUserEvent event id %d user event name %s, monotonically increasing = %d\n", userEventToken,
		  userEventName, monotonicallyIncreasing);

			//printf("Orig name: %s, escaped name: %s\n",userEventName,escapedName);

  my_user_event *ue = new my_user_event(userEventToken, escape_json(userEventName).c_str(), monotonicallyIncreasing);
  my_user_event_set[userEventToken] = ue;
  return 0;
}

int EventTrigger( void *userData, double time,
		unsigned int nodeToken,
		unsigned int threadToken,
	       	unsigned int userEventToken,
		long long userEventValue)
{
  dprintf("EventTrigger: time %g, nid %d tid %d event id %d triggered value %lld \n", time, nodeToken, threadToken, userEventToken, userEventValue);
	if(ignoreAtomic){
		return 0;
	}
	if(chromeFormat){
		json_event_out << "{";
	  //json_event_out << "\t\t\t\"event-id\": \"" << stateid << "\",\n";
	  json_event_out << "\"name\": " << my_user_event_set[userEventToken]->_name << ", ";
		json_event_out << "\"cat\": \"AtomicEvent\", ";
		//Event type. C=Counter.
		json_event_out << "\"ph\": \"C\", ";
	  json_event_out << "\"ts\": \"" << time << "\", ";
	  json_event_out << "\"pid\": \"" << nodeToken << "\", ";
	  json_event_out << "\"tid\": \"" << threadToken << "\", ";
		json_event_out << "\"args\": {\"counts\": " << userEventValue << "} ";
	  json_event_out << "},\n";
  }
	else{
  json_event_out << "{ ";
  json_event_out << "\"event-type\": \"counter\", ";
  json_event_out << "\"time\": \"" << time << "\", ";
  //json_event_out << "\"event-id\": \"" << userEventToken << "\", ";
  json_event_out << "\"name\": " << my_user_event_set[userEventToken]->_name << ", ";
  json_event_out << "\"node-id\": \"" << nodeToken << "\", ";
  json_event_out << "\"thread-id\": \"" << threadToken << "\", ";
  json_event_out << "\"value\": \"" << userEventValue << "\" ";
  json_event_out << "},\n";
  }
  return 0;
}

int SendMessage( void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
		unsigned int messageTag,
		unsigned int messageComm )
{
  dprintf("SendMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n",
		  time,
		  sourceNodeToken, sourceThreadToken,
		  destinationNodeToken, destinationThreadToken,
		  messageSize, messageTag);
			if(chromeFormat){
				json_event_out << "{ ";
				json_event_out << "\"name\": \"MPI\", ";
			  json_event_out << "\"cat\": \"Message\", ";
			  json_event_out << "\"ts\": \"" << time << "\", ";
			  json_event_out << "\"pid\": \"" << sourceNodeToken << "\", ";
			  json_event_out << "\"tid\": \"" << sourceThreadToken << "\", ";
				json_event_out << "\"ph\": \"s\", ";
				json_event_out << "\"bp\": \"e\", ";
			  json_event_out << "\"id\": \"" << destinationNodeToken << "\", ";
				json_event_out << "\"scope\": \"" << messageTag << "\", ";
				json_event_out << "\"args\": {\"destNode\": " << destinationNodeToken << ",\"destThread\": " << destinationThreadToken << ",\"message-size\": " << messageSize << ",\"message-tag\": " << messageTag << "} ";
			  json_event_out << "},\n";
			}
			else{
  json_event_out << "{ ";
  json_event_out << "\"event-type\": \"send\", ";
  json_event_out << "\"timestamp\": \"" << time << "\", ";
  json_event_out << "\"source-node-id\": \"" << sourceNodeToken << "\", ";
  json_event_out << "\"source-thread-id\": \"" << sourceThreadToken << "\", ";
  json_event_out << "\"destination-node-id\": \"" << destinationNodeToken << "\", ";
  json_event_out << "\"destination-thread-id\": \"" << destinationThreadToken << "\", ";
  json_event_out << "\"message-size\": \"" << messageSize << "\", ";
  json_event_out << "\"message-tag\": \"" << messageTag << "\" ";
  json_event_out << "},\n";
  }
  return 0;
}

int RecvMessage( void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken,
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
                unsigned int messageTag,
		unsigned int messageComm )
{
  dprintf("RecvMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n",
		  time,
		  sourceNodeToken, sourceThreadToken,
		  destinationNodeToken, destinationThreadToken,
		  messageSize, messageTag);
			if(chromeFormat){
				json_event_out << "{ ";
				json_event_out << "\"name\": \"MPI\", ";
			  json_event_out << "\"cat\": \"Message\", ";
			  json_event_out << "\"ts\": \"" << time << "\", ";
			  json_event_out << "\"pid\": \"" << destinationNodeToken << "\", ";
			  json_event_out << "\"tid\": \"" << destinationThreadToken << "\", ";
				json_event_out << "\"ph\": \"f\", ";
				json_event_out << "\"bp\": \"e\", ";
			  json_event_out << "\"id\": \"" << destinationNodeToken << "\", ";
				json_event_out << "\"scope\": \"" << messageTag << "\", ";
				json_event_out << "\"args\": {\"destNode\": " << destinationNodeToken << ",\"destThread\": " << destinationThreadToken << ",\"message-size\": " << messageSize << ",\"message-tag\": " << messageTag << "} ";
			  json_event_out << "},\n";
			}
			else{
  json_event_out << "{ ";
  json_event_out << "\"event-type\": \"receive\", ";
  json_event_out << "\"timestamp\": \"" << time << "\", ";
  json_event_out << "\"source-node-id\": \"" << sourceNodeToken << "\", ";
  json_event_out << "\"source-thread-id\": \"" << sourceThreadToken << "\", ";
  json_event_out << "\"destination-node-id\": \"" << destinationNodeToken << "\", ";
  json_event_out << "\"destination-thread-id\": \"" << destinationThreadToken << "\", ";
  json_event_out << "\"message-size\": \"" << messageSize << "\", ";
  json_event_out << "\"message-tag\": \"" << messageTag << "\" ";
  json_event_out << "},\n";
  }
  return 0;
}

void write_definitions(void) {
  /* user events */
  json_index_out << "\t\"counters\": [";
  map<int,my_user_event*>::iterator ue_it;
  bool first = true;
  for (ue_it = my_user_event_set.begin(); ue_it != my_user_event_set.end(); ue_it++) {
    if (first) { first = false; } else { json_index_out << ","; }
    json_index_out << "\n\t\t{\n";
    json_index_out << "\t\t\t\"id\": \"" << (*ue_it).second->_id << "\",\n";
    json_index_out << "\t\t\t\"monotonically-increasing\": ";
    if ((*ue_it).second->_mi == 0) {
        json_index_out << "false,\n";
    } else {
        json_index_out << "true,\n";
    }
    json_index_out << "\t\t\t\"name\": " << (*ue_it).second->_name;
    json_index_out << "\n\t\t}";
  }
  json_index_out << "\n\t],\n";
  /* groups */
  json_index_out << "\t\"groups\": [";
  map<int,my_group*>::iterator gr_it;
  first = true;
  for (gr_it = my_group_set.begin(); gr_it != my_group_set.end(); gr_it++) {
    if (first) { first = false; } else { json_index_out << ","; }
    json_index_out << "\n\t\t{\n";
    json_index_out << "\t\t\t\"id\": \"" << (*gr_it).second->_id << "\",\n";
    json_index_out << "\t\t\t\"name\": \"" << (*gr_it).second->_name << "\"";
    json_index_out << "\n\t\t}";
  }
  json_index_out << "\n\t],\n";
  /* states */
  json_index_out << "\t\"states\": [";
  map<int,my_state*>::iterator st_it;
  first = true;
  for (st_it = my_state_set.begin(); st_it != my_state_set.end(); st_it++) {
    if (first) { first = false; } else { json_index_out << ","; }
    json_index_out << "\n\t\t{\n";
    json_index_out << "\t\t\t\"event-id\": \"" << (*st_it).second->_event_id << "\",\n";
    json_index_out << "\t\t\t\"group-id\": \"" << (*st_it).second->_group_id << "\",\n";
    json_index_out << "\t\t\t\"name\": " << (*st_it).second->_name;
    json_index_out << "\n\t\t}";
  }
  json_index_out << "\n\t],\n";
  /* threads */
  json_index_out << "\t\"threads\": [";
  set<my_thread*>::iterator thr_it;
  first = true;
  for (thr_it = my_thread_set.begin(); thr_it != my_thread_set.end(); thr_it++) {
    if (first) { first = false; } else { json_index_out << ","; }
    json_index_out << "\n\t\t{\n";
    json_index_out << "\t\t\t\"node-id\": \"" << (*thr_it)->_node_id << "\",\n";
    json_index_out << "\t\t\t\"thread-id\": \"" << (*thr_it)->_thread_id << "\",\n";
    json_index_out << "\t\t\t\"name\": \"" << (*thr_it)->_name << "\"";
    json_index_out << "\n\t\t}";
  }
  json_index_out << "\n\t],\n";
}

void write_metadata(void) {
	if(!chromeFormat){
  json_index_out << "{\n";
  json_index_out << "\t\"metadata\": {";
  /* user events */
  map<string,string>::iterator it;
  bool first = true;
  for (it = my_metadata.begin(); it != my_metadata.end(); it++) {
    if (first) { first = false; } else { json_index_out << ","; }
    json_index_out << "\n\t\t\"" << (*it).first << "\": \"" << (*it).second << "\"";
  }
  json_index_out << "\n\t},\n";
  write_definitions();
  }
  ifstream json_trace_in;
  json_trace_in.open(filename);
  std::string line;
	if(printstdout)
  {
		while(std::getline(json_trace_in, line)) json_index_out << line << '\n' ;
	}
  json_trace_in.close();
	if(printstdout)
 {
	 remove(filename);
 }
	if(!chromeFormat){
  json_index_out << "}\n";
  }
}

int main(int argc, char **argv)
{
  Ttf_FileHandleT fh;
  int recs_read, pos;
  char *trace_file;
  char *edf_file;
  int no_state_flag=0, no_message_flag=0;
  /* main program: Usage app <trc> <edf> [-nostate] [-nomessage] */
  if (argc < 3)
  {
    printf("Usage: %s <TAU trace> <edf file> [-nostate] [-nomessage] [-v] [-nojson] [-chrome] [-ignoreatomic] [-o filename.json] [-print]\n",
		    argv[0]);
    return 1;
  }

  for (int i = 0; i < argc ; i++)
  {
    switch(i) {
      case 0:
	trace_file = argv[1];
	break;
      case 1:
	edf_file = argv[2];
	break;
      default:
	if (strcmp(argv[i], "-nostate")==0)
	{
	  no_state_flag = 1;
	}
	if (strcmp(argv[i], "-nomessage")==0)
	{
	  no_message_flag = 1;
	}
	if (strcmp(argv[i], "-v")==0)
	{
	  debugPrint = 1;
        }
	if (strcmp(argv[i], "-nojson")==0)
	{
	  jsonPrint = 0;
        }
	if (strcmp(argv[i], "-chrome")==0)
	{
		 chromeFormat = 1;
			     }
	if (strcmp(argv[i], "-ignoreatomic")==0)
	{
			ignoreAtomic = 1;
	}
	if (strcmp(argv[i], "-o")==0)
	{
		  i++;
			filename = argv[i];
	}

	if (strcmp(argv[i], "-print")==0)
	{
		  printstdout=1;
	}

	break;
    }
  }

  fh = Ttf_OpenFileForInput(trace_file, edf_file);

  if (!fh)
  {
    printf("ERROR:Ttf_OpenFileForInput fails");
    return 1;
  }

  /* open the output files */
  if (jsonPrint) {
    json_event_out.open(filename, ios::out | ios::trunc);
		//If we truncate the timestamps we get invalid trace renderings.
		json_event_out.setf(ios_base::fixed);
    //json_index_out.open("trace.json", ios::out | ios::trunc);
		json_index_out.open(filename, ios::out | ios::trunc);
		json_index_out.setf(ios_base::fixed);
		if(chromeFormat){
			json_event_out << "[\n";
		}
		else{
    json_event_out << "\t\"trace events\": [\n";
	  }
  }

  Ttf_CallbacksT cb;
  /* Fill the callback struct */
  cb.UserData = 0;
  cb.DefClkPeriod = ClockPeriod;
  cb.DefThread = DefThread;
  cb.DefStateGroup = DefStateGroup;
  cb.DefState = DefState;
  cb.DefUserEvent = DefUserEvent;
  cb.EventTrigger = EventTrigger;
  cb.EndTrace = EndTrace;

  /* should state transitions be displayed? */
  if (no_state_flag)
  {
    cb.EnterState = 0;
    cb.LeaveState = 0;
  }
  else
  {
    cb.EnterState = EnterState;
    cb.LeaveState = LeaveState;
  }

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
    if (recs_read != 0)
      dprintf("Read %d records\n", recs_read);
  }
  while ((recs_read >=0) && (!EndOfTrace));

  Ttf_CloseFile(fh);

  if(chromeFormat){
		//The chrome trace reader accepts unterminated lists.
		//json_event_out << "]\n";
		json_event_out.close();
		write_metadata();
	}
  else if (jsonPrint) {
    json_event_out << "{ ";
    json_event_out << "\"event-type\": \"trace end\" ";
    json_event_out << "}\n";
    json_event_out << "]\n";
    json_event_out.close();
    write_metadata();
    //json_index_out.close();
  }
  return 0;
}
