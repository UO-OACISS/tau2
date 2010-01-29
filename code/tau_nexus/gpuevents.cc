#include "gpuevents.h"
#include<string>
#include<sstream>
#include<unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<iostream>
#include<TAU.h>
#include<stdlib.h>
#include "TAU_tf_headers.h"
//#ifdef TAU_MPITRACE
#include <mpi.h>
//#endif
using namespace std;

extern TAU64 AlignedTime(int device,TAU64 raw_time);
extern "C" int TAUDECL tau_totalnodes(int ,int);
extern "C" void TauTraceEventOnly(long int ev, x_int64 par, int tid);
extern x_uint64 TauTraceGetTimeStamp(int tid);
extern void * message_size, * rcv_message;
#define PATH_SIZE 200

static int last=-1;

    unsigned long
    hash(unsigned char *str)
    {
        unsigned long hash = 5381;
        int c;

        while (c = *str++)
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

        return hash;
    }

//For a given process, process is the unique MPI rank
//Node n is the nth node in the allocation
//Core m is the mth core on node n
int TauGetCpuSite(int *node, int *core, int *rank){
//#ifdef TAU_MPITRACE
	char host_name[MPI_MAX_PROCESSOR_NAME];
	char (*host_names)[MPI_MAX_PROCESSOR_NAME];
	MPI_Comm internode;
	MPI_Comm intranode;

	int nprocs, namelen,n,bytes;

	MPI_Comm_rank(MPI_COMM_WORLD, (int*)rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Get_processor_name(host_name,&namelen);
	bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);

	host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
	
	strcpy(host_names[*rank], host_name);
	for (n=0; n<nprocs; n++)
	{
		MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD); 
	}
	
	unsigned int color;
  color = 0;

  for (n=1; n<nprocs; n++)
  {
	  if(strcmp(host_names[n-1], host_names[n])) color++;
	  if(strcmp(host_name, host_names[n]) == 0) break;
  }
	
	MPI_Comm_split(MPI_COMM_WORLD, color, *rank, &internode);
	MPI_Comm_rank(internode,(int*)core);
	
	MPI_Comm_split(MPI_COMM_WORLD, *core, *rank, &intranode);
	
	MPI_Comm_rank(intranode,(int*)node);
	
	//if(last==*node){
	//	*node++;
	//}
	//last=*node;
//#else
//	*node=0;
//	*core=0;
//	*rank=0;
//#endif
	return 0;
}


EventManager::EventManager()
{

	char dpath[PATH_SIZE];
	trace_enabled=false;
	base_dir=getcwd(dpath,PATH_SIZE);
	trace_enabled=IsTraceSet();
	start_time_stamp=TauTraceGetTimeStamp(0);
	//node_id=-1;
	locrank=-1;
}

EventManager::EventManager(string _base_dir)
{
	TauGetCpuSite(&locnode,&loccore,&locrank);
	fprintf(stdout,"Con Base Dir Node %d, Core %d, Rank %d\n",locnode,loccore,locrank);
	trace_enabled=false;
	base_dir=_base_dir;
	trace_enabled=IsTraceSet();
	start_time_stamp=TauTraceGetTimeStamp(0);
	//node_id=-1;
	locrank=-1;
}

EventManager::EventManager(string _base_dir,bool trace)
{
	TauGetCpuSite(&locnode,&loccore,&locrank);
	fprintf(stdout,"Con w Bool Node %d, Core %d, Rank %d\n",locnode,loccore,locrank);
	trace_enabled=trace;
	base_dir=_base_dir;
	trace_enabled=IsTraceSet();
	start_time_stamp=TauTraceGetTimeStamp(0);
}

EventManager::~EventManager()
{
	list<event_base *>::iterator it;
	for(it=trace_events.begin();it!=trace_events.end();it++)
	{
		event_base * my_ev=*it;
		delete my_ev;
	}
	trace_events.clear();
	map<string,event_base *>::iterator pit;
	for(pit=profiles.begin();pit!=profiles.end();pit++)
	{
		event_base * my_ev=pit->second;
		delete my_ev;
	}
	profiles.clear();
	CloseTraces();
}

void EventManager::SetThread(unsigned int tid)
{
	thread_id=tid;
}

/*
	This routine grabs the TAU_TRACE environment variable 
	to inspect if it's set to 1. 
*/
bool EventManager::IsTraceSet()
{
	 char *my_env=getenv("TAU_TRACE");
         if(!my_env || my_env[0]!='1')
                return false;
	 return true;			
}

// it's older code 
void EventManager::PopulateProfile(event_base * event)
{
	stringstream my_prof;
	if(event->type==DATA)
	{
		mem_event * mevent=(mem_event *) event;
		my_prof<<mevent->tau_context<<"#"<<mevent->device_id<<"#"<<mevent->stream_id<<"#"<<mevent->mem_address;
		string prof_str=my_prof.str();
		map<string,event_base *>::iterator it=profiles.find(prof_str);
		if(it!=profiles.end())
		{
			mem_event * old_mevent=(mem_event *)it->second;
			old_mevent->mem_size+=mevent->mem_size;
			old_mevent->gpu_elapsed_time+=mevent->gpu_elapsed_time;
			old_mevent->calls++;
			delete mevent;
		} 
		else
		{
			mevent->calls=0;
			profiles.insert(profiles.end(),make_pair(prof_str,event));
		}
	}
	else
	{
		kernel_event * kevent=(kernel_event *) event;
		my_prof<<kevent->tau_context<<"#"<<kevent->device_id<<"#"<<kevent->stream_id<<"#"<<kevent->kernel_name;
		string prof_str=my_prof.str();
		map<string,event_base *>::iterator it=profiles.find(prof_str);
		if(it!=profiles.end())
		{
			kernel_event * old_kevent = (kernel_event *) it->second;
			old_kevent->gpu_elapsed_time+=kevent->gpu_elapsed_time;
			old_kevent->calls++;
			delete kevent;
				
		}
		else
		{
			kevent->calls=0;
			profiles.insert(profiles.end(),make_pair(prof_str,event));
			
		}
		
	}	
}

/*
	This routine extracts the number of events with a given 
	device ID , stream and event type. Event types could be 
	memory transfer , kernel launch etc. 
*/
int EventManager::GetEventCount(int dev_id, int stream,event_type type)
{
	int count=0;
	map<string,event_base *>::iterator it;
	for(it=profiles.begin();it!=profiles.end();it++)
	{
		event_base *my_event=it->second;
		if(my_event->device_id==dev_id && my_event->stream_id==stream)
		{
			if(type==ALL)
				count++;
			else if(type==my_event->type)
				count++;
		}	
	}
	
	return count;
}

/*
	This is a helper routine to extract the total time spent in the GPU. 
	It's used to generate top level event for GPU profiles. 
*/

double  EventManager::GetTotalTime(int dev, int stream)
{
	double total_time=0;
        map<string,event_base *>::iterator it;
        for(it=profiles.begin();it!=profiles.end();it++)
        {
                event_base *my_event=it->second;
                if(my_event->device_id==dev && my_event->stream_id==stream)
                {
			total_time+=(double)my_event->gpu_elapsed_time/1000;
                }
        }

        return total_time;	
}

/*
	This routines return the profile output stream for a given node, stream, device and metric. 
*/
fstream * EventManager::GetProfileOut(unsigned int node,unsigned int stream,unsigned int device, string metric_name)
{
	stringstream outfile;
	char hostname[64];
        gethostname(hostname,64);
	outfile<<base_dir<<"/"<<metric_name<<"/taucuda_profile."<<hostname<<"."<<stream<<"."<<device; //get host id was node_id
	//outfile<<base_dir<<"/"<<metric_name<<"/taucuda_profile."<<node<<"."<<stream<<"."<<device;
	//outfile<<base_dir<<"/"<<metric_name<<"/taucuda_profile."<<node<<"."<<stream<<"."<<thread_id;
	string my_path=outfile.str();
	map<string, fstream *>::iterator it=out_files.find(my_path);
	if(it!=out_files.end())
	{
		/*
			If the output stream is already open just return it.
		*/
		dfprintf(stdout,"Found the file for output\n");
		return it->second;
	}
	else
	{
		/*
			The output stream is opened fro the first time. 
			Profile headers are populated including the number of events. 
		*/
		fstream *my_fstr=new fstream(my_path.c_str(),fstream::app|fstream::out);
		out_files.insert(out_files.end(),make_pair(my_path,my_fstr));
		int prof_count;
		if(metric_name=="gpu_elapsed_time")
			prof_count=GetEventCount(device,stream,ALL);
		else if(metric_name=="gpu_memory_transfer")
		{
			prof_count=GetEventCount(device,stream,DATA2D);
			prof_count+=GetEventCount(device,stream,DATAFD);
			prof_count+=GetEventCount(device,stream,DATA);
		}
		else
			prof_count=GetEventCount(device,stream,KERNEL);		
		if(metric_name=="gpu_elapsed_time")
		{
			(*my_fstr)<<prof_count+1<<" templated_functions_MULTI_"<<metric_name<<endl;
			(*my_fstr)<<"# Name Calls Subrs Excl Incl ProfileCalls # <metadata></metadata>"<<endl;
			double total_time=GetTotalTime(device, stream);
			(*my_fstr)<<"\"GPU-Total-Elapsed-Time\" 1"<<" 0 0 "<<total_time<<" 0 GROUP=\"TAU_DEFAULT\""<<endl;
		}
		else
		{
			(*my_fstr)<<prof_count<<" templated_functions_MULTI_"<<metric_name<<endl;
			(*my_fstr)<<"# Name Calls Subrs Excl Incl ProfileCalls # <metadata></metadata>"<<endl;
		}
		return my_fstr;	
	}	
}

/*
	Close a profile file stream after writing out the last meta information.
	Clear the entry from the map of profile files.  
*/

void EventManager::CloseProfileOut(fstream *out_fstr)
{
	map<string,fstream *>::iterator it;
	for(it=out_files.begin();it!=out_files.end();it++)
	{
		if(it->second==out_fstr)
		{
			out_files.erase(it);
			(*out_fstr)<<"0 aggregates"<<endl;
			out_fstr->close();
			delete out_fstr;
		}
	}	
}

/*
	Close all the profile files for a particular metric. 
	Profile end metadata is written out and entries are cleared from the map. 
*/
void EventManager::CloseProfile(string metric_name)
{
	string base_path=base_dir;
	base_path.append("/");
	base_path.append(metric_name);
	base_path.append("/");
	map<string, fstream *>::iterator it;
	for(it=out_files.begin();it!=out_files.end();)
	{
		string out_path=it->first;
		map<string, fstream *>::iterator erase_it=it++;
		dfprintf(stdout,"$$$$$$ %s##%s\n",base_path.c_str(),out_path.c_str());
		if(base_path.compare(0,base_path.length(),out_path,0,base_path.length())==0)
		{
			fstream *out_fstr=erase_it->second;
			(*out_fstr)<<"0 aggregates"<<endl;
			out_files.erase(erase_it);
			out_fstr->close();
			delete out_fstr;				
		}
			
	}			
}

/*
	Returns the trace writer handle for a given node, stream and device. 
*/

Ttf_FileHandleT EventManager::GetTraceOut(unsigned int node,unsigned int stream,unsigned int device)
{
	unsigned int process,core;
	stringstream outfile, edf_str;
	outfile<<base_dir<<"/taucudatrace."<<node<<"."<<stream<<"."<<device<<".trc";
	edf_str<<base_dir<<"/taucudaevents."<<node<<"."<<stream<<"."<<device<<".edf";
	//edf_str<<base_dir<<"/taucudaevents."<<node<<".edf";
	string trc_file=outfile.str();
	map<string, Ttf_FileHandleT>::iterator it=trace_files.find(trc_file);
	if(it!=trace_files.end())
	{
		dfprintf(stdout,"Found the trace file for output\n");
		return it->second;
	}
	else
	{
		/*
			The trace writer is opened for the first time and the trace events are initialized here.  
		*/
		string edf_file=edf_str.str();
		Ttf_FileHandleT my_fstr=Ttf_OpenFileForOutput(trc_file.c_str(),edf_file.c_str());
		stringstream node_info;
		node_info<<"node "<<node;
		Ttf_DefThread(my_fstr, node, device, node_info.str().c_str());
  		Ttf_DefThread(my_fstr, node, device, node_info.str().c_str());
  		Ttf_DefStateGroup(my_fstr, "TAU_DEFAULT", 1);
  		Ttf_DefStateGroup(my_fstr, "TAU_EVENT", 2);
		trace_files.insert(trace_files.end(),make_pair(trc_file,my_fstr));
		file_info my_info={node,stream,device};
		trace_files_info.insert(trace_files_info.end(),make_pair(trc_file,my_info));
		//take care of the top timer
 		Ttf_DefState(my_fstr, 0 ,"GPU_TOP_EVENT", 1);
		/*
			The user events are used for tracking memory transfer.
		*/
 		Ttf_DefUserEvent(my_fstr, TAUCUDA_BLOCK_SIZE ,"Number of Threads in the Block", 2);
 		Ttf_DefUserEvent(my_fstr, TAUCUDA_GRID_SIZE ,"Number of Blocks in the Grid", 2);
 		Ttf_DefUserEvent(my_fstr, TAUCUDA_SSHARED_MEM ," Static Shared Memory Used", 2);
 		Ttf_DefUserEvent(my_fstr, TAUCUDA_DSHARED_MEM ," Dynamic Shared Memory Used", 2);
 		Ttf_DefUserEvent(my_fstr, TAUCUDA_REGISTERS ," Registers Used per Thread", 2);
 		Ttf_DefUserEvent(my_fstr, TAUCUDA_MEM_SEND ,"TAUCUDA_MEM_SEND", 2);
 		Ttf_DefUserEvent(my_fstr, TAUCUDA_MEM_RCV ,"TAUCUDA_MEM_RCV", 2);
 		Ttf_DefUserEvent(my_fstr, TAUCUDA_COPY_MEM_SIZE ,"TAUCUDA_COPY_MEM_SIZE", 2);
         	Ttf_EnterState(my_fstr, start_time_stamp, node, device,0);
		return my_fstr;	
	}	
}

/*
	Trace writers are closed here and cleared from the map. 
*/

void EventManager::CloseTraces()
{
	end_time_stamp=TauTraceGetTimeStamp(0);
	map<string, Ttf_FileHandleT>::iterator it;
	for(it=trace_files.begin();it!=trace_files.end();)
	{
		map<string, Ttf_FileHandleT>::iterator erase_it=it++;
		Ttf_FileHandleT out_fstr=erase_it->second;
		map<string, file_info>::iterator iit=trace_files_info.find(erase_it->first);
	 	Ttf_LeaveState(out_fstr,end_time_stamp, iit->second.node_id,iit->second.device_id, 0);
		Ttf_CloseOutputFile(out_fstr);
		trace_files.erase(erase_it);
	}			
}

/*
	This is a helper function to write out user defined events to the trace file. 
	Trace writer APIs can not be directly used here as we need to use the entire 
	bytes of the parameter. 
*/

void  EventManager::SpitEvent(Ttf_fileT *tFile, unsigned int event_id, unsigned int nid, unsigned int tid, TAU64 time_stamp, TAU64 param)
{
	 if (tFile->tracePosition >= TAU_MAX_RECORDS) {
      		Ttf_FlushTrace(tFile);
    	 }
    	int pos = tFile->tracePosition;		
    	tFile->traceBuffer[pos].ev = event_id;
    	tFile->traceBuffer[pos].nid = nid;
    	tFile->traceBuffer[pos].tid = RtsLayer::getTid();
    	tFile->traceBuffer[pos].ti = (x_uint64)time_stamp;
    	tFile->traceBuffer[pos].par = param;
    	tFile->tracePosition++;
    	tFile->lastTimestamp = time_stamp;
}

/*
	This routine is for tracing kernel launch events. 
*/

void EventManager::TraceEvent(string event_name, TAU64 start_time, TAU64 end_time, TAU32 event_id, 
			unsigned int node, unsigned int stream, unsigned int device, 
			TAU32 g_size, TAU32 b_size, TAU32 s_sm, TAU32 d_sm, TAU32 registers, float occupancy )
{
	 Ttf_FileHandleT file=GetTraceOut(node,stream,device);
	 Ttf_fileT *tFile = (Ttf_fileT*)file;	
	 Ttf_DefState(file, event_id, event_name.c_str(), 1);
	 TAU64 time_stamp = AlignedTime(device, start_time);
	 Ttf_EnterState(file, time_stamp, node, device, event_id);

	//timestamp gets corrected here with the help of the closkc table. 
	 time_stamp=AlignedTime(device, end_time);

	/*
		All the GPU counters are populated as user defined events in the following. 
	*/
	 SpitEvent(tFile,TAUCUDA_GRID_SIZE, node, RtsLayer::getTid(), time_stamp-8,g_size);
	
	 SpitEvent(tFile,TAUCUDA_BLOCK_SIZE, node, RtsLayer::getTid(), time_stamp-6,b_size);
	
	 SpitEvent(tFile,TAUCUDA_SSHARED_MEM, node, RtsLayer::getTid(), time_stamp-4,s_sm);

	 SpitEvent(tFile,TAUCUDA_DSHARED_MEM, node, RtsLayer::getTid(), time_stamp-2,d_sm);

	 SpitEvent(tFile,TAUCUDA_REGISTERS, node, RtsLayer::getTid(), time_stamp-1,registers);

	 Ttf_LeaveState(file, time_stamp, node, device, event_id);	 
}

/*
	This routine is called to generate trace for memory transfer events. 
*/

void EventManager::TraceMemEvent(string event_name, TAU64 start_time, TAU64 end_time, TAU32 event_id, unsigned int node, 
				unsigned int stream, unsigned int device, TAU64 mem_size, TAU64 ctx, TAU64 call_id, event_type type)
{

	 if(type==DATAFD)
	 {
		/*
			If the memory copy is from device to the CPU, marker events as receiver of the memory 
			is populated in the TAU trace. This is done here as this routine gets invoked during 
			context synchronization.
		*/
		TauUserEvent *my_ev = (TauUserEvent*)rcv_message;
                TauUserEvent *my_size = (TauUserEvent*)message_size;
		//60008
		TauTraceEventOnly(my_ev->GetEventId(),ctx ,RtsLayer::getTid());
                TauTraceEventOnly(my_ev->GetEventId(),call_id ,RtsLayer::getTid());
                TauTraceEventOnly(my_size->GetEventId(),mem_size ,RtsLayer::getTid());
		
		//Tau_userevent(rcv_message,(double)ctx);
                //Tau_userevent(rcv_message,(double)call_id);
                //Tau_userevent(message_size,(double)mem_size);
	
	 }

	/*
		Actual GPU event trace is generated in the following after correcting the timestamps. 
	*/

	 Ttf_FileHandleT file=GetTraceOut(node,stream,device);
	 Ttf_fileT *tFile = (Ttf_fileT*)file;	
	 Ttf_DefState(file, event_id, event_name.c_str(), 1);
	 TAU64 time_stamp = AlignedTime(device, start_time);
	 Ttf_EnterState(file, time_stamp, node, device, event_id);
	 time_stamp=AlignedTime(device, end_time);
	 Ttf_LeaveState(file, time_stamp, node, device, event_id);
	 if(type==DATA) return;
 
	
	 if (tFile->tracePosition >= TAU_MAX_RECORDS) {
      		Ttf_FlushTrace(tFile);
    	 }

    	//int pos = tFile->tracePosition;
		
	if(type==DATA2D)
	{
		/*
			For memory copy to the device , receive marker is generated in the TAUCuda trace. 
		*/
		//60008
		SpitEvent(tFile, TAUCUDA_MEM_RCV, node, RtsLayer::getTid(), time_stamp, ctx);
		SpitEvent(tFile, TAUCUDA_MEM_RCV, node, RtsLayer::getTid(), time_stamp, call_id);
		SpitEvent(tFile, TAUCUDA_COPY_MEM_SIZE, node, RtsLayer::getTid(), time_stamp, mem_size);
    		//tFile->traceBuffer[pos].ev = 60008;
	}
	else
	{
		/*
			For memory copy from the device , send marker is generated in the TAUCuda trace. 
		*/
		
		//60007
		SpitEvent(tFile, TAUCUDA_MEM_SEND, node, RtsLayer::getTid(), time_stamp, ctx);
		SpitEvent(tFile, TAUCUDA_MEM_SEND, node, RtsLayer::getTid(), time_stamp, call_id);
		SpitEvent(tFile, TAUCUDA_COPY_MEM_SIZE, node, RtsLayer::getTid(), time_stamp, mem_size);
    		//tFile->traceBuffer[pos].ev = 60007;
	}

	/*	
	 Ttf_RecvMessage(file, time_stamp,                  
	          node_id, RtsLayer::getTid(), // from 1,0
                  node, device, // to 0,0
                  mem_size,  // length
                  //event_id,   // tag
                  42,   // tag
                  0);   // communicator
	 */
}

void SpitEvent(string event_name, string type, TAU32 device_id, TAU32 stream_id, TAU64 time)
{
}

void SpitMessage(TAU32 device_id, TAU32 stream_id, TAU64 size)
{

}


bool EventManager::IsTraceEnabled()
{
	return trace_enabled; 
}

/*
	Helper routine to Create the metric folders.
*/ 

void EventManager::CreateMetricFolder(string metric_name)
{
	//string path_create="mkdir ";
	string path_create;
	path_create.append(base_dir);
	path_create.append("/");
	path_create.append(metric_name);
	//system(path_create.c_str());
	mkdir(path_create.c_str(),0755);
}

/*
	This is older routine 
*/

void EventManager::WriteProfileToFile()
{
	map<string, event_base *>::iterator it;
	for(it=profiles.begin();it!=profiles.end();it++)
	{
		event_base *my_event=it->second;
		fstream *fstr=GetProfileOut(0,my_event->stream_id, my_event->device_id, "gpu_elapsed_time");
		if(my_event->type==DATA)
		{
			mem_event *mevent=(mem_event *) my_event;
			(*fstr)<<"M-"<<mevent->mem_address<<"#C-"<<mevent->tau_context<<"#"<<mevent->mem_size<<" "<<\
				mevent->gpu_elapsed_time<<" "<<mevent->gpu_elapsed_time<<" GROUP=\"TAU_DEFAULT\""<<endl;
			delete mevent;	
		}
		else
		{
			kernel_event *kevent=(kernel_event *) my_event;
			(*fstr)<<"M-"<<kevent->kernel_name<<"#C-"<<kevent->tau_context<<"#"<<" "<<\
				kevent->gpu_elapsed_time<<" "<<kevent->gpu_elapsed_time<<" GROUP=\"TAU_DEFAULT\""<<endl;
			delete kevent;		
		}
	}	
	profiles.clear();		
}


/*
	Writes out elapsed time metric profile 
*/

void EventManager::WriteElaspedMetric()
{
	CreateMetricFolder("gpu_elapsed_time");
	map<string, event_base *>::iterator it;
	for(it=profiles.begin();it!=profiles.end();it++)
	{
		event_base *my_event=it->second;
		fstream *fstr=GetProfileOut(0,my_event->stream_id, my_event->device_id, "gpu_elapsed_time");
		if(my_event->type==DATA2D || my_event->type==DATAFD || my_event->type==DATA)
		{
			/*
				For memory transfer profile use the API name as part of the event name. 
			*/
			mem_event *mevent=(mem_event *) my_event;
			(*fstr)<<"\""<<mevent->mem_api<<"#"<<mevent->tau_context<<"\" "<<mevent->calls <<" 0 "<<\
				(double)mevent->gpu_elapsed_time/1000<<" "<<(double)mevent->gpu_elapsed_time/1000<<" 0 GROUP=\"TAU_DEFAULT\""<<endl;
			//cout<<mevent->mem_api<<"#"<<mevent->tau_context<<"      "<<mevent->calls <<"     "<<mevent->gpu_elapsed_time<<"     "<<mevent->gpu_elapsed_time<<" GROUP=\"TAU_DEFAULT\""<<endl;
		}
		else
		{
			/*
				For kernel launch profile use the kernel name as part of the event name. 
			*/
			kernel_event *kevent=(kernel_event *) my_event;
			(*fstr)<<"\""<<kevent->kernel_name<<"#"<<kevent->tau_context<<"\" "<<kevent->calls <<" 0 "<<\
				(double)kevent->gpu_elapsed_time/1000<<" "<<(double)kevent->gpu_elapsed_time/1000<<" 0 GROUP=\"TAU_DEFAULT\""<<endl;
			//cout<<kevent->kernel_name<<"#"<<kevent->tau_context<<"     "<<kevent->calls <<"     "<<kevent->gpu_elapsed_time<<"     "<<kevent->gpu_elapsed_time<<" GROUP=\"TAU_DEFAULT\""<<endl;
		}
	}
	CloseProfile("gpu_elapsed_time");
}

/*
	Write out memory transfer profile in the respective folder. 
*/

void EventManager::WriteMemTransferMetric()
{
	CreateMetricFolder("gpu_memory_transfer");
	map<string, event_base *>::iterator it;
	for(it=profiles.begin();it!=profiles.end();it++)
	{
		event_base *my_event=it->second;
		if(my_event->type==DATA2D || my_event->type==DATAFD || my_event->type==DATA)
		{
			/*
				It accounts only data size in memory transfer events 
			*/
			fstream *fstr=GetProfileOut(0,my_event->stream_id, my_event->device_id, "gpu_memory_transfer");
			mem_event *mevent=(mem_event *) my_event;
			(*fstr)<<"\""<<mevent->mem_api<<"#"<<mevent->tau_context<<"\" "<<mevent->calls <<" 0 "<<\
				mevent->mem_size<<" "<<mevent->mem_size<<" 0 GROUP=\"TAU_DEFAULT\""<<endl;
			//cout<<mevent->mem_api<<"#"<<mevent->tau_context<<"      "<<mevent->calls <<"     "<<mevent->gpu_elapsed_time<<"     "<<mevent->gpu_elapsed_time<<" GROUP=\"TAU_DEFAULT\""<<endl;
		}
	}
	CloseProfile("gpu_memory_transfer");
}

/*
	Writes out static memory usage profile.
*/

void EventManager::WriteStaicMemoryMetric()
{
	CreateMetricFolder("gpu_static_shared_memory");
	map<string, event_base *>::iterator it;
	for(it=profiles.begin();it!=profiles.end();it++)
	{
		event_base *my_event=it->second;
		if(my_event->type==KERNEL)
		{
			fstream *fstr=GetProfileOut(0,my_event->stream_id, my_event->device_id, "gpu_static_shared_memory");
			kernel_event *kevent=(kernel_event *) my_event;
			(*fstr)<<"\""<<kevent->kernel_name<<"#"<<kevent->tau_context<<"\" "<<kevent->calls <<" 0 "<<\
				kevent->static_shared_mem<<" "<<kevent->static_shared_mem<<" 0 GROUP=\"TAU_DEFAULT\""<<endl;
		}
	}
	CloseProfile("gpu_static_shared_memory");

}

/*
	Writes out dynamic memory usage profile.
*/

void EventManager::WriteDynamicMemoryMetric()
{
	CreateMetricFolder("gpu_dynamic_shared_memory");
	map<string, event_base *>::iterator it;
	for(it=profiles.begin();it!=profiles.end();it++)
	{
		event_base *my_event=it->second;
		if(my_event->type==KERNEL)
		{
			fstream *fstr=GetProfileOut(0,my_event->stream_id, my_event->device_id, "gpu_dynamic_shared_memory");
			kernel_event *kevent=(kernel_event *) my_event;
			(*fstr)<<"\""<<kevent->kernel_name<<"#"<<kevent->tau_context<<"\" "<<kevent->calls <<" 0 "<<\
				kevent->dynamic_shared_mem<<" "<<kevent->dynamic_shared_mem<<" 0 GROUP=\"TAU_DEFAULT\""<<endl;
		}
	}
	CloseProfile("gpu_dynamic_shared_memory");
}

/*
	Writes out core occupancy profiles of the kernel launch.  
*/

void EventManager::WriteOccupancyMetric()
{
	CreateMetricFolder("gpu_core_occupancy");
	map<string, event_base *>::iterator it;
	for(it=profiles.begin();it!=profiles.end();it++)
	{
		event_base *my_event=it->second;
		if(my_event->type==KERNEL)
		{
			fstream *fstr=GetProfileOut(0,my_event->stream_id, my_event->device_id, "gpu_core_occupancy");
			kernel_event *kevent=(kernel_event *) my_event;
			(*fstr)<<"\""<<kevent->kernel_name<<"#"<<kevent->tau_context<<"\" "<<kevent->calls <<" 0 "<<\
				kevent->occupancy<<" "<<kevent->occupancy<<" 0 GROUP=\"TAU_DEFAULT\""<<endl;
		}
	}
	CloseProfile("gpu_core_occupancy");
}

/*
	Writes out profile for register usage per thread. 
*/

void EventManager::WriteThreadRegisterMetric()
{
	CreateMetricFolder("gpu_thread_registers");
	map<string, event_base *>::iterator it;
	for(it=profiles.begin();it!=profiles.end();it++)
	{
		event_base *my_event=it->second;
		if(my_event->type==KERNEL)
		{
			fstream *fstr=GetProfileOut(0,my_event->stream_id, my_event->device_id, "gpu_thread_registers");
			kernel_event *kevent=(kernel_event *) my_event;
			(*fstr)<<"\""<<kevent->kernel_name<<"#"<<kevent->tau_context<<"\" "<<kevent->calls <<" 0 "<<\
				kevent->registers<<" "<<kevent->registers<<" 0 GROUP=\"TAU_DEFAULT\""<<endl;
		}
	}
	CloseProfile("gpu_thread_registers");
}

/*
 Helper routine to extarct the short routine name ignoring the signature. 
*/

string EventManager::ExtractRtnName(string rtn_sign)
{
	int start_indx=rtn_sign.find(" cuda");
	if(start_indx==string::npos)
		return rtn_sign;
	int end_indx=rtn_sign.find('(');
	if(end_indx==string::npos)
		return rtn_sign;
	start_indx++;
	return rtn_sign.substr(start_indx,end_indx-start_indx);
}

#define MAX_DEPTH 3
/*
 for now it retrieves from the current TAU context 
 it will change later depending on the situation.
*/
string EventManager::GetProgramContext()
{
	string tau_context;
	const char *str;
	TAU_QUERY_DECLARE_EVENT(event);
	int depth = TauEnv_get_callpath_depth();
	TAU_QUERY_GET_CURRENT_EVENT(event);
   	TAU_QUERY_GET_EVENT_NAME(event, str);
	if(str)
	{
		tau_context=ExtractRtnName(str);
	}
   	if (depth < 1) {
     		depth = 1;
   	}
	int counter=1;
	/*
		walk through the TAU event stack and concat the names 
		to get the TAU context. 
	*/
	while (str && depth > 0) {
     		tau::Profiler *p = (tau::Profiler *)event;
     		TAU_QUERY_GET_PARENT_EVENT(event);
     		TAU_QUERY_GET_EVENT_NAME(event, str);
     		if (str) {
			tau_context.append("#");
			//tau_context.append(str);
			tau_context.append(ExtractRtnName(str));
			counter++;
			if(counter>MAX_DEPTH)
				break;
     		}
     		depth--;
   	}
	
	return tau_context;
}

/*
	Older routine
*/
int  EventManager::MyGPUNode(int device_id)
{

	/*int gpu_node;//tau_totalnodes(0,0);
	//PMPI_Comm_size( MPI_COMM_WORLD, &gpu_node );
	if(node_id<0)
		node_id=RtsLayer::myNode();
	gpu_node+=node_id;
	
	return gpu_node;
	int gpu_node=device_id+1;
	if(node_id<0)
		node_id=RtsLayer::myNode();
	gpu_node=(gpu_node<<24)|node_id;
	
	return gpu_node;*/

	if(locrank<0){
		TauGetCpuSite(&locnode,&loccore,&locrank);
		fprintf(stdout,"MyGPUNode Node %d, Core %d, Rank %d\n",locnode,loccore,locrank);
	}
		//node_id=locrank;//=RtsLayer::myNode();
	return locrank;//node_id;	
}


/*
this is called when CUDA driver api call gets intercepted.
It sets up the api_event structure which is temporarily stored until 
the corresponding profile callback received.  
*/

void  EventManager::APIEntry(TAU64 ctxt, TAU64 stream, TAU64 eventid, event_type type, char *api_name, TAU32 device_id)
{
	
	api_event my_event;
	my_event.context_id=ctxt;
	my_event.stream_id=stream;
	my_event.event_id=eventid;
	my_event.type=type;
	my_event.api_name=api_name;
	//my_event.tau_context="taucuda_context";
	my_event.tau_context=GetProgramContext();
	my_event.device_id=device_id;
	api_events.insert(api_events.end(),my_event);
	dfprintf(stdout, "####DEVICE: %d \n", device_id);
}
         
/*
	Memory profile events are managed by this routine which is called by the callback handler.  
*/
 
void  EventManager::MemProfileEvent(TAU64 ctxt,TAU64 stream_id,TAU64 eventid, TAU64 start_time, TAU64 end_time, TAU64 mem_size)
{
	api_event my_event;
	stringstream mystr;
	bool flag =false; 
	list<api_event>::iterator it;
	/*
		Try to grap the API event data structure for thie profile callabck. 
	*/
	for(it=api_events.begin();it!=api_events.end();it++)
	{
		my_event=*it;
		if(my_event.context_id==ctxt && my_event.event_id==eventid)
		{
			flag=true;
			api_events.erase(it);
			break;
		}	
	}	
	// just some extra precaution 
	if(!flag)
	{
		dfprintf(stdout,"Didnt find the API event \n");
		current_profile=NULL;
		return;
	}

	if(trace_enabled)
	{
		/*
			If trace is enabled triger the trace event manager after composing the event name. 
		*/
		string event_name=my_event.api_name;
		event_name.append("#");
		event_name.append(my_event.tau_context);
		TraceMemEvent(event_name, start_time, end_time, (TAU32) eventid,MyGPUNode(my_event.device_id),
				stream_id, my_event.device_id, mem_size, ctxt, eventid, my_event.type);  
		return;
	}	

	mystr<<my_event.tau_context<<"#"<<my_event.api_name<<"#"<<stream_id;
	map<string, event_base *>::iterator it1=profiles.find(mystr.str());
	if(it1==profiles.end())
	{
		/*
			If the event record is not found earlier it's a new eventa and it 
			gets initialized in the following. 
		*/
		mem_event *mevent=new mem_event;
		mevent->type=my_event.type;
		mevent->context_id=ctxt;
		mevent->stream_id=stream_id;
		mevent->device_id=my_event.device_id;
		mevent->gpu_elapsed_time=end_time-start_time;
		mevent->calls=1;
		mevent->tau_context=my_event.tau_context;
		mevent->mem_api=my_event.api_name;
		mevent->mem_size=mem_size;
		mevent->kern_context=NULL;
		profiles.insert(profiles.end(), make_pair(mystr.str(),mevent));	
		current_profile=(event_base *)mevent;	
	}
	else
	{
		/*
			Event record found and we need to add up the metric and increment the call 
			count. 
		*/
		mem_event *mevent=(mem_event*) it1->second;
		mevent->gpu_elapsed_time+=end_time-start_time;
		mevent->calls++;
		mevent->mem_size+=mem_size;
		current_profile=(event_base *)mevent;	
	}
}

/*
	This routine handles tracing of kernel profile events 
*/

void  EventManager::KernelTraceEvent(TAU64 ctxt,TAU64 stream_id ,TAU64 eventid, char * callname, TAU64 start_time,TAU64 end_time,
					TAU32 g_size, TAU32 b_size, TAU32 s_sm, TAU32 d_sm, TAU32 registers, float occupancy )
{
	if(!trace_enabled)
		return;
	api_event my_event;
	bool flag =false; 
	list<api_event>::iterator it;
	/*
		Grab the corresponding API event for this profile callback. 
	*/
	for(it=api_events.begin();it!=api_events.end();it++)
	{
		my_event=*it;
		if(my_event.context_id==ctxt && my_event.event_id==eventid)
		{
			flag=true;
			api_events.erase(it);
			break;
		}	
	}	

	if(!flag)
	{
		dfprintf(stdout, "Didnt find the API event\n");
		current_profile=NULL;
		return;
	}
	
	/*
		Use the calname for the event name and trigger event tracer. 
	*/

		string event_name(callname);
		event_name.append("#");
		event_name.append(my_event.tau_context);
		TraceEvent(event_name, start_time, end_time, (TAU32) eventid,
				MyGPUNode(my_event.device_id),stream_id, my_event.device_id, 
				g_size, b_size, s_sm, d_sm, registers, occupancy);  
}

/*       
	This routine manages other parameters associated to Kernel profile events. 
*/
void  EventManager::KernelProfileEvent(TAU64 ctxt,TAU64 stream_id ,TAU64 eventid, char * callname, TAU64 start_time,TAU64 end_time)
{
	if(trace_enabled)
		return;
	api_event my_event;
	bool flag =false; 
	list<api_event>::iterator it;
	for(it=api_events.begin();it!=api_events.end();it++)
	{
		my_event=*it;
		if(my_event.context_id==ctxt && my_event.event_id==eventid)
		{
			flag=true;
			api_events.erase(it);
			break;
		}	
	}	

	if(!flag)
	{
		dfprintf(stdout, "Didnt find the API event\n");
		current_profile=NULL;
		return;
	}
	
	
	stringstream mystr;
	mystr<<my_event.tau_context<<"#"<<callname<<"#"<<stream_id;
	map<string, event_base *>::iterator it1=profiles.find(mystr.str());
	if(it1==profiles.end())
	{
		/*
			If the kernel profile event was not found , a new one is created and initialized. 
			Initialization for kernel event take into account of differnt counters. 
		*/
		kernel_event *kevent=new kernel_event;
		kevent->type=my_event.type;
		kevent->context_id=ctxt;
		kevent->stream_id=stream_id;
		kevent->device_id=my_event.device_id;
		kevent->gpu_elapsed_time=end_time-start_time;
		kevent->calls=1;
		kevent->tau_context=my_event.tau_context;
		kevent->kernel_name=callname;
		kevent->block_dim=0;
		kevent->grid_dim=0;
		kevent->static_shared_mem=0;
		kevent->dynamic_shared_mem=0;
		kevent->registers=0;
		kevent->occupancy=0;
		
		profiles.insert(profiles.end(), make_pair(mystr.str(),kevent));	
		current_profile=(event_base *)kevent;	
	}
	else
	{
		kernel_event *kevent=(kernel_event *) it1->second;
		kevent->gpu_elapsed_time+=end_time-start_time;
		kevent->calls++;
		current_profile=(event_base *)kevent;	
	}
				
}

/*
	This routine was originally designed to enable incrementally update kernel event information. However, this could have been put together 
	with the above routine.  
*/       
void  EventManager::UpdateKernelProfile(TAU32 g_size, TAU32 b_size, TAU32 s_sm, TAU32 d_sm, TAU32 registers, float occupancy)
{
	if(trace_enabled)
		return;
	kernel_event *kevent=(kernel_event *)current_profile;
	if(!kevent)
	{
		dfprintf(stdout,"Current Profile Pointer NULL\n");
		return;
	}
	/*
		Add up the various counters. 
	*/
	kevent->grid_dim+=g_size;
	kevent->block_dim+=b_size;
	kevent->static_shared_mem+=s_sm;
	kevent->dynamic_shared_mem+=d_sm;
	kevent->registers+=registers;
	kevent->occupancy+=occupancy;
}


void EventManager::SynChronize(int stream)
{
	
}

/*
	This is an interface required to be called when the application exits. 
	As of now this gets invoked when shared library gets unloaded or 
*/

void EventManager::ThreadExit()
{

	if(trace_enabled)
		return;
	//node_id=RtsLayer::myNode();
	TauGetCpuSite(&locnode,&loccore,&locrank);
	fprintf(stdout,"Thread Exit Node %d, Core %d, Rank %d\n",locnode,loccore,locrank);
	//locnode=node;
	//loccore=core;
	//locrank=process;
	/*
		Write out all different counter profiles. 
		Special care needs to be taken here when we can support various GPU counters. 
	*/
	WriteElaspedMetric();
	WriteMemTransferMetric();
	WriteOccupancyMetric();
	WriteStaicMemoryMetric();		
	WriteDynamicMemoryMetric();
	WriteThreadRegisterMetric();		
}
