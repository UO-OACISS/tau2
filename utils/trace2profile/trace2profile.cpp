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

#include <fstream>
#include <iomanip>
#include <trace2profile.h>
#include <sstream>

#include <stdlib.h>
#include <string.h>

ofstream snapshot;

char * Converter::trc = NULL;
char * Converter::edf = NULL;
char * Converter::out = NULL;

int Converter::trigEvent=-1;

int Converter::EndOfTrace=0;  /* false */

//bool Converter::printshot=false;
//unsigned int Converter::miecount=0;
//int Converter::debugPrint=0;
double Converter::segmentInterval=-1;
double Converter::lastTime=0;
int Converter::trigCount=1;

map<unsigned int,Thread*> Converter::ThreadMap;
//map<pair<int,int>,int> Converter::ThreadID;//NID/TID
map<unsigned int,State*> Converter::allstate;
vector<MonIncEvent*> Converter::allmoninc;
set<unsigned int> Converter::monincids;
vector<UserEvent*> Converter::allevents;

map<unsigned int, string*> Converter::groupnames;
map<unsigned int, string*> Converter::statenames;
map<unsigned int, string*> Converter::usereventnames;
map<unsigned int, string*> Converter::monincnames;
map<unsigned int, string*> Converter::threadnames;

bool useSnapshot;


void Usage();
void PrintSnapshot(double time, int threadToken, bool printProfile);

/***************************************************************************
 * 
 * This routine registers the entry into state 'stateid' from thread 'threadin' at time 'time'.
 * It returns 'threadin' with these modifications.
 * 
 ***************************************************************************/
void StateEnter(double time, Thread &threadin, State &statein)
{
	/*If there is another state in the callstack then the state we are entering
	* is a subroutine of the previous state.  The previous state's time is no 
	* longer exclusive so we record the exclusive time seen so far and stop.*/
	
	//State thisstate=*(threadin.allstate[stateid]);
	State *lastcall=NULL;
	if(!threadin.callstack.empty())
	{		
		lastcall=threadin.callstack.back();//)  .top();
		
		//if(threadin.processToken==1 && lastcall->stateToken==1)
			//cout << "enter sub " << endl;
		
		lastcall->subroutines++;
		lastcall->exclusive+=time-lastcall->topTime;
		lastcall->topTime=-1;
	}
	/*Increase the number of times this state has been entered and mark
	 * the time it spends on the top of the stack.*/
	statein.calls++;
	Converter::allstate[statein.stateToken]->calls++;
	
	statein.topTime=time;
	/*If this state has not exited after a previous call it is in recursion.  We
	 * only give it a new start time if it is not in recursion.*/
	if(statein.fullTime==-1)
	{
		statein.fullTime=time;
	}
	
	if(Converter::monincids.size()>0)
	{
		for(map<unsigned int,MonIncEvent*>:: iterator miit=statein.allmi.begin();miit!=statein.allmi.end();miit++)
		{
			
			if(!threadin.callstack.empty())
			{/*Here we need to add to the previous event's exclusive time the current event's observed time minus the previous's top*/
				MonIncEvent &thisMI=*(lastcall->allmi[(*miit).first]);
				
				//if(threadin.processToken==1 && lastcall->stateToken==1)
					//cout << "in enter adding to ex, which is: " << thisMI.exclusive << " " << (*miit).second->holdvalue << " - " << thisMI.topCount << " = " << ((*miit).second->holdvalue-thisMI.topCount) << endl;
			
				thisMI.exclusive+=((*miit).second->holdvalue-thisMI.topCount);
				//if(threadin.processToken==1 && lastcall->stateToken==1)
					//cout << thisMI.exclusive << endl;
				thisMI.topCount=-1;
			}
			
			(*miit).second->topCount=(*miit).second->holdvalue;
			if((*miit).second->fullCount==-1)
			{
				(*miit).second->fullCount=(*miit).second->holdvalue;
			}
		}
	}	
	
	/*Add one the number of calls to this state currently active and put the
	 * state on the callstack.*/
	statein.countRec++;
	threadin.callstack.push_back(&statein);
	
	//return threadin;
}


/***************************************************************************
 * 
 * This routine registers the exit of state 'stateid' from thread 'threadin' at time 'time'.
 * It returns 'threadin' with these modifications.
 * 
 ***************************************************************************/
void StateLeave(double time, Thread &threadin, State &statein)
{
	/*Add the exclusive time recorded since the routine was last on the top 
	 * of the stack.  It is no longer on the stack so stop recording topTime.
	 * Decrement the number of recursive calls to this routine.*/
	statein.exclusive+=time-statein.topTime;
	if(statein.exclusive <0)
		cout << "Exclusive stamp: "<< setprecision(8) <<  time << " topTime: " << statein.topTime << endl;
	
	statein.topTime=-1;
	statein.countRec--;
	/*If we are no longer in recursion (all calls to this routine have exited)
	 * then we can record the inclusive time for this routine from its initial
	 * call and stop.*/
	if(statein.countRec==0)
	{
		statein.inclusive+=time-statein.fullTime;
		statein.fullTime=-1;
		
		if(statein.inclusive <0)
			cout << "Inclusive stamp: "<<  time << " topTime: " << statein.fullTime << endl;

	}
	/*Pop this routine off of the callstack.  If there is another routine start
	 * recording its exclusive time again.*/
	threadin.callstack.pop_back();
	State *lastcall=NULL;
	if(!threadin.callstack.empty())
	{	
		lastcall=threadin.callstack.back();//top();
		lastcall->topTime=time;
	}
	
	if(Converter::monincids.size()>0)//&&Converter::printshot
	{
		for(map<unsigned int,MonIncEvent*>:: iterator miit=statein.allmi.begin();miit!=statein.allmi.end();miit++)
		{
			if((*miit).second->topCount<0)
				cout << "First mie process not initialized (topcount)." << endl;
			//if(threadin.processToken==1 && statein.stateToken==1)	
			//cout << "in leave adding to ex, which is: " << (*miit).second->exclusive << " " << (*miit).second->holdvalue << " - " << (*miit).second->topCount << " = " << (*miit).second->holdvalue-(*miit).second->topCount << endl;
				
			(*miit).second->exclusive+=((*miit).second->holdvalue-(*miit).second->topCount);
			(*miit).second->topCount=-1;
			
			if(statein.countRec==0)
			{
				(*miit).second->inclusive+=((*miit).second->holdvalue)-((*miit).second->fullCount);
				(*miit).second->fullCount=-1;
			}
			if(!threadin.callstack.empty())
			{
				//unsigned int loctoken = (*miit).second->eventToken;
				lastcall->allmi[(*miit).first]->topCount=(*miit).second->holdvalue;
			}
		}
	}
}



void InitSnapshot(){

	string s_out="";
	string s_prefix="";
	string cmd="";
	if(Converter::out!=NULL)
	{
		s_out=Converter::out;
		s_out+="/";
	}
	char filename [32];
	sprintf(filename,"profile.xml");//%d.0.%d",((*it).second).nodeToken,((*it).second).threadToken);
	s_prefix=s_out+filename;
	snapshot.open(s_prefix.c_str());//, ofstream::app|ofstream::out
	snapshot.precision(16);
	
	snapshot << "<profile_xml>" << endl;

}


void writeStringXML(string *stringIn) {
	
  const char* s = stringIn->c_str();
  if (!s) return;
  
  bool useCdata = false;
  
  if (strchr(s, '<') || strchr(s, '&')) {
    useCdata = true;
  }
  
  if (strstr(s, "]]>")) {
    useCdata = false;
  }
  
  if (useCdata) {
    //fprintf (profile,"<![CDATA[%s]]>",s);
    snapshot << "<![CDATA[" << s << "]]>";
    return;
  }

  // could grow up to 5 times in length
  char *str = (char *) malloc (6*strlen(s));
  char *d = str;
  while (*s) {
    if ((*s == '<') || (*s == '>') || (*s == '&')) {
      // escape these characters
      if (*s == '<') {
	strcpy (d,"&lt;");
	d+=4;
      }
      
      if (*s == '>') {
	strcpy (d,"&gt;");
	d+=4;
      }
      
      if (*s == '&') {
	strcpy (d,"&amp;");
	d+=5;
      }
    } else {
      *d = *s;
      d++; 
    }
    
    s++;
  }
  *d = 0;
  
  //fprintf (profile,"%s",str);
  snapshot << str;
  free (str);
}


void SnapshotDefs(){
	snapshot << "<definitions thread=\""; 
	
	/*bool first=true;
	for(map<int,Thread>:: iterator threadCount = ThreadMap.begin(); 
	threadCount!=ThreadMap.end(); threadCount++)
	{
		if(!first)
		{
			profile << ",";
		}
		first=false;
		profile << (*threadCount).first;
	}*/
	snapshot << "*";
	
	snapshot << "\">"  << endl
	<< "<metric id=\""<<0<<"\" name=\"time\" units=\"microseconds\"/>" << endl;
	for(vector<MonIncEvent*>:: iterator mieCount = Converter::allmoninc.begin(); 
	mieCount!=Converter::allmoninc.end(); mieCount++)
	{
		snapshot << "<metric id=\""<< (*mieCount)->eventToken << "\" name=\""; 
		writeStringXML(Converter::monincnames[(*mieCount)->eventToken]); 
		snapshot << "\" units=\"Probably Not Microseconds\"/>" << endl;
	}
	for(map<unsigned int,State*>:: iterator stateCount = Converter::allstate.begin(); 
	stateCount!=Converter::allstate.end(); stateCount++)
	{
		//cout << "With shots: " << Converter::statenames[(*stateCount).second.stateToken] << " Index: " << (*stateCount).second.stateToken << endl; 
		snapshot << "<event id=\""<< (*stateCount).second->stateToken << "\" name=\""; 
		 writeStringXML(Converter::statenames[(*stateCount).second->stateToken]); snapshot << "\" group=\"";
		 writeStringXML(Converter::groupnames[(*stateCount).second->stateGroupToken]);
		 snapshot << "\"/>" << endl;
		
		//cout << "With shots2: " << Converter::statenames[(*stateCount).second.stateToken] << " Index: " << (*stateCount).second.stateToken << endl;
		
	}
	for(vector<UserEvent*>:: iterator eventCount = Converter::allevents.begin(); 
	eventCount!=Converter::allevents.end(); eventCount++)
	{
		snapshot << "<atomic_event id=\""<< (*eventCount)->userEventToken << "\" name=\""; 
		writeStringXML(Converter::usereventnames[(*eventCount)->userEventToken]); 
		snapshot << "\"/>" << endl;
	}
	snapshot << "</definitions>" << endl;
}

void SnapshotClose()
{
	snapshot << "</profile_xml>" << endl;
	snapshot.close();	
}

void SnapshotThreadPrint(unsigned int curid,unsigned int nodeToken,unsigned int threadToken){
	snapshot << "<thread id=\"" << curid << "\" node=\"" << nodeToken 
	<< "\" context=\"0\" thread=\"" << threadToken << "\"/>" << endl;	
}

void ProcessDefs(){
	/*For every thread in the trace*/
	for (map<unsigned int, Thread*>:: iterator it = Converter::ThreadMap.begin(); it != Converter::ThreadMap.end(); it++)
	{ 
		//(*it).second->allstate=Converter::allstate;
		/*Create a new state pointer map using pointers to copies of the global states*/
		(*it).second->allstate=*(new map<unsigned int, State*>());
		for(map<unsigned int, State*>:: iterator sit = Converter::allstate.begin(); sit != Converter::allstate.end(); sit++)
		{
			(*it).second->allstate[(*sit).first]= new State(*((*sit).second));
			
			
			/*If there are monotoncally increasing events add their map to each state*/
			if(Converter::monincids.size()>0)
			{
				(*it).second->allstate[(*sit).first]->allmi = *(new map<unsigned int, MonIncEvent*>());
				for(vector<MonIncEvent*>:: iterator mit=Converter::allmoninc.begin(); mit!=Converter::allmoninc.end();mit++)
				{
					(*it).second->allstate[(*sit).first]->allmi[(*mit)->eventToken]=new MonIncEvent(*(*mit));
					/*TODO*/
					//(*sit).second.allmi=*(new map<unsigned int, MonIncEvent*>());  //Converter::allmoninc;
				}
			}
		}
		
		/*Create a new user event map using pointers to copies of the global user events*/
		(*it).second->allevents=*(new map<unsigned int, UserEvent*>());//Converter::allevents;
		for(vector<UserEvent*>:: iterator eit = Converter::allevents.begin(); eit != Converter::allevents.end(); eit++)
		{
			(*it).second->allevents[(*eit)->userEventToken]= new UserEvent(*(*eit));
		}
		
		//cout << "allstateref " << &((*it).second->allstate) << endl;
		
		/* reset the position of the trace to the first record Initialize global id map*/
		(*it).second->finished = 0;
		/*Print the snapshot thread definitions*/
		SnapshotThreadPrint((*it).second->processToken,(*it).second->nodeToken,(*it).second->threadToken);
	}
	Converter::EndOfTrace = 0;
	 
	SnapshotDefs();
}



/***************************************************************************
 * Given a time and a threadToken, this routine will copy the thread, exit every
 * remaining event at the given time and then print the profile data
 ***************************************************************************/
void PrintSnapshot(double time, Thread &finalizer, bool printProfile){//map<int,Thread>, less< pair<int,int> > 

	//Thread* finalizer = Converter::ThreadMap[threadToken];
	//Converter::printshot=true;
	//for(map< int, Thread >:: iterator it = finalizer.begin(); it != finalizer.end(); it++)//, less< pair<int,int> > 
	//{
	
	//Thread finalizer=printThread;
	if(useSnapshot)
	{
	if(finalizer.callstack.size()>0)
	{
		finalizer.BackupThread();
		while(finalizer.callstack.size()>0)//(*it) = finalizer
		{
			//(*it).second=
			//cout << Converter::miecount << " " << Converter::printshot << endl;
			StateLeave(time,finalizer,*(finalizer.callstack.back()));//top()
		
		}
	}

	double mean=0;
	double sum=0;
	double num=0;
	unsigned int eventID=0;
	//MonIncEvent *MIEvent=NULL;
	//int countFunc=0;
	/*for(map<int,State>:: iterator stateCount = allstate.begin(); stateCount!=allstate.end(); stateCount++)
	{
		if((*stateCount).second.calls>0)
			countFunc++;
	}*/
	//for (map< int,Thread >:: iterator it = mainmap.begin(); it != mainmap.end(); it++){
		snapshot << "<profile thread=\"" << finalizer.processToken <<"\">" << endl;//(*it).first
		snapshot << "<snapshotname>Time step " << time << "</snapshotname>" << endl;
		//<!--#event_id num_metrics (metric_id exclusive inclusive)[1..num_metrics] numcalls numsubr -->
		snapshot << "<interval_data metrics=\"0";
		if(Converter::monincids.size()>0)
		{
			for(vector<MonIncEvent*>:: iterator miit=Converter::allmoninc.begin();miit!=Converter::allmoninc.end();miit++)
			{
				eventID=(*miit)->eventToken;
				snapshot << " " << eventID;
			}
		}
		snapshot << "\">" << endl;
		for(map<unsigned int,State*>::iterator st = finalizer.allstate.begin(); 
		st !=finalizer.allstate.end();st++)
		{
			
			//cout << "With SnapB: " <<Converter::statenames[((*st).second)->stateToken] << " Index: " << ((*st).second)->stateToken << endl;
			
			if(((*st).second)->calls>0)//allstate[(*st).second.stateToken].calls
			{
				snapshot << ((*st).second)->stateToken << " "
				<< ((*st).second)->calls << " " //Was .stateName
				<< ((*st).second)->subroutines << " " //<< " "  << "0"   " GROUP=\"" << groupids[((*st).second).stateGroupToken]<<"\""<<
				<< ((*st).second)->exclusive << " " 
				<< ((*st).second)->inclusive << " ";
				if(Converter::monincids.size()>0)
				{
					for(map<unsigned int,MonIncEvent*>:: iterator miit=((*st).second)->allmi.begin();
					miit!=((*st).second)->allmi.end();miit++)
					{
						//MIEvent = ((*st).second)->allmi[(*miit).first];
						snapshot <<((*miit).second)->exclusive << " " << ((*miit).second)->inclusive << " ";
					}
				}
				snapshot << endl;
			}
		}
		snapshot << "</interval_data>" << endl;
		//profile << "0 aggregates" << endl;
		if(Converter::allevents.size()>0)
		{
			//profile << allevents.size() << " userevents" << endl 
			//<<  "# eventname numevents max min mean sumsqr" << endl;
			snapshot << "<atomic_data>" << endl;
			for(map<unsigned int,UserEvent*>::iterator st = finalizer.allevents.begin(); 
			st !=finalizer.allevents.end();st++)
			{
				mean = 0;
				if(((*st).second)->numevents>0)
				{
					sum=((*st).second)->sum;
					num=((*st).second)->numevents;
					mean = sum/num;
				}
				snapshot << ((*st).second)->userEventToken << " " //was userEventName
				<< ((*st).second)->numevents << " " 
				<< ((*st).second)->max << " " 
				<< ((*st).second)->min << " " 
				<< mean << " " << ((*st).second)->sumsqr << endl;
			}
			snapshot << "</atomic_data>" << endl;
		}
		snapshot << "</profile>" << endl;
		
		//Converter::printshot=false;
	}
		
		if(printProfile)
		{
/***************************************************************************
 * Given a map 'mainmap' conforming to the 'whole trace' data structure, this routine will cycle
 * through each thread and state to print out the profile statistics for the whole program
 ***************************************************************************/
//void PrintSnapshots(map<int,Thread> &mainmap)//, less< pair<int,int> > 
	//char prefix [32];
	string s_out="";
	string s_prefix="";
	string cmd="";
	double mean=0;
	double sum=0;
	double num=0;
	if(Converter::out!=NULL)
	{
		s_out=Converter::out;
		s_out+="/";
	}
	/*
	if(snapshot>-1)
	{
		sprintf(prefix,"snapshot_%d/",snapshot);
		s_out+=prefix;
		cmd="mkdir ";
		cmd+=s_out;
		system(cmd.c_str());
	}*/
	
	if(Converter::monincids.size()>0)
	{
		s_out+="MULTI__GET_TIME_OF_DAY/";
		cmd="mkdir "+s_out;
		system(cmd.c_str());
	}

	string profileUDEs="";

	ofstream profile;
	int countFunc=0;
	for(map<unsigned int,State*>:: iterator stateCount = Converter::allstate.begin(); stateCount!=Converter::allstate.end(); stateCount++)
	{
		if((*stateCount).second->calls>0)
			countFunc++;
	}
	//for (map< int,Thread >:: iterator it = mainmap.begin(); it != mainmap.end(); it++){
		char filename [32];
		sprintf(filename,"profile.%d.0.%d",finalizer.nodeToken,finalizer.threadToken);//((*it).second)
		s_prefix=s_out+filename;
		profile.open(s_prefix.c_str());
		profile.precision(16);
		profile << countFunc << " templated_functions";
		if(Converter::monincids.size()>0)
		{
			profile << "_MULTI_GET_TIME_OF_DAY";
		}
		profile << endl;
		profile << "# Name Calls Subrs Excl Incl ProfileCalls" << endl;
		for(map<unsigned int,State*>::iterator st = finalizer.allstate.begin(); st !=finalizer.allstate.end();st++)
		{
			//cout << "With Prof: " <<*Converter::statenames[((*st).second)->stateToken] << " Index: " << ((*st).second)->stateToken << endl; 
			if(Converter::allstate[(*st).second->stateToken]->calls>0)
			profile << "\"" << *Converter::statenames[((*st).second)->stateToken] << "\" " << ((*st).second)->calls 
			<< " " << ((*st).second)->subroutines << " " << ((*st).second)->exclusive 
			<< " " << ((*st).second)->inclusive << " " << "0" << " GROUP=\"" 
			<< *Converter::groupnames[((*st).second)->stateGroupToken]<<"\""<< endl;
		}
		profile << "0 aggregates" << endl;
		if(Converter::allevents.size()>0)
		{
			ostringstream UDEstream;
			//profileUDEs+=Converter::allevents.size()+" userevents\n";
			//profileUDEs+="# eventname numevents max min mean sumsqr\n";
			
			
			UDEstream << Converter::allevents.size() << " userevents" << endl 
			<<  "# eventname numevents max min mean sumsqr" << endl;
			
			for(map<unsigned int,UserEvent*>::iterator st = finalizer.allevents.begin(); st !=finalizer.allevents.end();st++)
			{
				mean = 0;
				if(((*st).second)->numevents>0)
				{
					sum=((*st).second)->sum;
					num=((*st).second)->numevents;
					mean = sum/num;
				}
				
				UDEstream << "\"" << *Converter::usereventnames[((*st).second)->userEventToken] << "\" " 
				<< ((*st).second)->numevents << " " << ((*st).second)->max 
				<< " " << ((*st).second)->min << " " << mean << " " << ((*st).second)->sumsqr << endl;
			}
			profileUDEs=UDEstream.str();
			profile << profileUDEs;
		}
		profile.close();
	//}
	
	if(Converter::monincids.size()>0)
	{
		unsigned int eventID=0;
		string eventname;
		string base = "";
		string s_name="";
		if(Converter::out!=NULL)
		{
			base=Converter::out;
			base+="/";
		}
		for(map<unsigned int,string*>:: iterator miit=Converter::monincnames.begin();
		miit!=Converter::monincnames.end();miit++)
		{
			s_name="";
			/*if(snapshot>-1)
			{
				sprintf(prefix,"snapshot_%d/",snapshot);
				s_name+=prefix;
				//cmd="mkdir ";
				//cmd+=s_out;
				//system(cmd.c_str());
			}*/
			eventID=(*miit).first;
			eventname=*((*miit).second);//*Converter::monincnames[(*miit).second.eventToken];
			s_name+="MULTI__";
			s_name+=eventname;
			s_out=base+s_name+"/";
			cmd="mkdir "+s_out;
			system(cmd.c_str());
			
			countFunc=0;
			for(map<unsigned int,State*>:: iterator stateCount = Converter::allstate.begin(); stateCount!=Converter::allstate.end(); stateCount++)
			{
				if((*stateCount).second->calls>0)
					countFunc++;
			}
			//for (map< int,Thread >:: iterator it = mainmap.begin(); it != mainmap.end(); it++)
				char filename [32];
				sprintf(filename,"profile.%d.0.%d",finalizer.nodeToken,finalizer.threadToken);
				s_prefix=s_out+filename;
				profile.open(s_prefix.c_str());
				profile.precision(16);
				profile << countFunc << " templated_functions_" << "MULTI_" << eventname << endl;
				profile << "# Name Calls Subrs Excl Incl ProfileCalls" << endl;
				for(map<unsigned int,State*>::iterator st = finalizer.allstate.begin(); st !=finalizer.allstate.end();st++)
				{
					if(Converter::allstate[(*st).second->stateToken]->calls>0)
					profile << "\"" << *Converter::statenames[((*st).second)->stateToken] << "\" " << ((*st).second)->calls 
					<< " " << ((*st).second)->subroutines << " " << ((*st).second)->allmi[eventID]->exclusive 
					<< " " << ((*st).second)->allmi[eventID]->inclusive << " " << "0" << " GROUP=\"" 
					<< *Converter::groupnames[((*st).second)->stateGroupToken]<<"\""<< endl;
				}
				profile << "0 aggregates" << endl;
				if(Converter::allevents.size()>0)
				{
					/*profile << Converter::allevents.size() << " userevents" << endl 
					<<  "# eventname numevents max min mean sumsqr" << endl;
					for(map<unsigned int,UserEvent*>::iterator st = finalizer.allevents.begin(); st !=finalizer.allevents.end();st++)
					{
						mean = 0;
						if(((*st).second)->numevents>0)
						{
							sum=((*st).second)->sum;
							num=((*st).second)->numevents;
							mean = sum/num;
						}
						profile << "\"" << *Converter::usereventnames[((*st).second)->userEventToken] << "\" " 
						<< ((*st).second)->numevents << " " << ((*st).second)->max 
						<< " " << ((*st).second)->min << " " << mean << " " << ((*st).second)->sumsqr << endl;
					}*/
					profile << profileUDEs;
				}
				profile.close();
			}
		}	
	}
	
	if(finalizer.bakcallstack.size()>0)
	{
		finalizer.RestoreThread();
	}
	
	return;	
}

/***************************************************************************
 * Given the time stamp for the cut-off time, this routine will iterate through the current trace-state
 * and exit every state on every stack, effectively creating a snapshot profile up to the given time.
 * The profile generated will then be printed.
 ***************************************************************************/
void PrintProfiles()
{
	//map<int,Thread> finalizer = ThreadMap;//, less<pair<int,int> > 
	//Thread finalizer = ThreadMap[threadToken];
	//printshot=1;
	//Converter::ThreadMap.
	for(map<unsigned int,Thread*>:: iterator it = Converter::ThreadMap.begin(); it != Converter::ThreadMap.end(); it++)//, (map<pair<int,int>,int>:: iterator it = Converter::ThreadID.begin(); it != Converter::ThreadID.end(); it++)
	{
		//cout << "Final Pointer: "<< (*it).second << ": " <<  (*it).second->processToken <<endl;
		
		PrintSnapshot(Converter::lastTime,*((*it).second),true);
		
		/*while(finalizer.callstack.size()>0)//(*it) = finalizer
		{
			//(*it).second=
			StateLeave(time,finalizer,finalizer.callstack.top());
		}*/
	}
	//thisShot=time;
	//PrintProfiles(finalizer, time);
	//printshot=0;	
}


void FinishSnapshots()
{
	PrintProfiles();
	if(useSnapshot)
		SnapshotClose();	
}

/***************************************************************************
 * If 'time' is greater than or equal to the end of the next specified interval this will
 * cut print out the snapshot as of the specified interval and set the time for the next one.
 ***************************************************************************/
void SnapshotControl(double time, int stateToken, Thread& threadin)
{
	//cout << time << endl;
	double nextShot=threadin.nextShot;
	int snapshot=threadin.snapshot;
	while(snapshot>-1 && time>=nextShot)
	{
		//cout << "IN " << setprecision(8) << time << endl;
		PrintSnapshot(nextShot, threadin,false);
		nextShot+=Converter::segmentInterval;
		snapshot++;
	}
	threadin.nextShot=nextShot;
	
	
	if(stateToken==Converter::trigEvent)
	{
		int trigSeen=threadin.trigSeen;
		trigSeen++;
		if(trigSeen==Converter::trigCount)
		{
			PrintSnapshot(time, threadin,false);
			snapshot++;
			trigSeen=0;
		}
		threadin.trigSeen=trigSeen;
	}
	threadin.snapshot=snapshot;
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
	//cout << "-s <interger n>: Output a profile snapshot of the trace every n "
		// << "time units.\n" << endl;
	cout << "e.g. $tau2profile tau.trc tau.edf" << endl; // -s 25000"  << endl;
}

/***************************************************************************
 * The main function reads user input and starts conversion procedure.
 ***************************************************************************/
int main(int argc, char **argv)
{
	int i; 
	bool seenedf=false;
	useSnapshot=false;
	
	/* main program: Usage app <trc> <edf> [-a] [-nomessage] */
	if (argc < 1)
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
			Converter::trc = argv[1];
			break;
			//case 2:
			/*edf_file*/ 
			//Converter::edf = argv[2];
			//break;			
			default:
			/*if (strcmp(argv[i], "-v") == 0)
			{
				debugPrint = 1;
				break;
			}
			else*/
			
			//cout << argv[i] << endl;
			/*TODO: Disabled segmentation until code update*/
			/*
			if(strcmp(argv[i],"-s")==0)
			{//Segment interval
				if(argc>i+1)
				{
					useSnapshot=true;
					i++;
					Converter::segmentInterval=atof(argv[i]);
					Converter::trigEvent=-1;
				}
				break;
			}
			else*/
			if(strcmp(argv[i],"-d")==0)
			{/*Output Directory*/
				if(argc>i+1)
				{
					i++;
					Converter::out=argv[i];
				}
				break;
			}
			else
			/*TODO: disabled event-triggerd segmentation
			 * if(strcmp(argv[i],"-e")==0)
			{//Event Trigger
				if(argc>i+1)
				{
					useSnapshot=true;
					i++;
					Converter::trigEvent=atoi(argv[i]);
					Converter::segmentInterval=0;
				}
				break;
			}
			else*/
			/*TODO: Disabled segmentation until code update*/
			/*
			if(strcmp(argv[i],"-c")==0)
			{//Event count
				if(argc>i+1)
				{
					useSnapshot=true;
					i++;
					Converter::trigCount=atoi(argv[i]);
				}
				break;
			}
			else*/
			{
				//cout << seenedf << " " << i << endl;
				if((!seenedf)&&(i==2))
				{
					//i++;
					Converter::edf = argv[2];
					seenedf=true;
				}
				else
				{
					Usage();
					exit(1);
				}
			}
			break;
		}
	}
	/* Finished parsing commandline options, now open the trace file */
	
	if(useSnapshot)
		InitSnapshot();
	
	if(seenedf)
		ReadTraceFile();
	else
	{
		//TODO: Re-enable OTF with library detection

//		ReadOTFFile();		

		Usage();
		exit(1);
	}

		
	FinishSnapshots();
	//ReadTraceFile();
}
