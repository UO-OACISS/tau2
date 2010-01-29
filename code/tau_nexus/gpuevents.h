
/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2009                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory                                        **
****************************************************************************/
/***************************************************************************
**      File            : gpuevents.h                                     **
**      Description     : TAU trace format reader library header files    **
**      Author          : Shangkar Mayanglambam                           **
**      Contact         : smeitei@cs.uoregon.edu                          **
***************************************************************************/



#pragma once
#include<list>
#include<map>
#include<string>
#include<fstream>
#include <TAU_tf.h>
#include "TAU_tf_headers.h"

using namespace std;
#define NAME_SIZE 100
#define dfprintf //
#define TAUCUDA_MEM_SIZE 900001
#define TAUCUDA_MSG_SEND 900002
#define TAUCUDA_MSG_RCV  900003
#define TAUCUDA_BLOCK_SIZE 900004
#define TAUCUDA_GRID_SIZE 900005
#define TAUCUDA_SSHARED_MEM 900006
#define TAUCUDA_DSHARED_MEM 900007
#define TAUCUDA_REGISTERS 900008
#define TAUCUDA_OCCUPANCY 900009

#define TAUCUDA_MEM_SEND 900010
#define TAUCUDA_MEM_RCV 900011
#define TAUCUDA_COPY_MEM_SIZE 900012


enum event_type{DATA,DATA2D,DATAFD,KERNEL,ALL,OTHERS};
typedef unsigned long long TAU64;
typedef unsigned int TAU32;
typedef unsigned short TAU16;
typedef unsigned char TAU8;

/*
	keeps track of the API entry events. 
*/
struct api_event{
	event_type type;
	TAU64 context_id;
	TAU64 stream_id;
	//corresponds to callid
	TAU64 event_id;
	string api_name;
	string tau_context;
	TAU32 device_id;	
};

/*
  A common data structure for event no matter it could be memory transfer or kernel launch. 
*/
struct event_base{
	event_type type;
	TAU64 context_id;
	//TAU64 event_id;
	TAU32 device_id;
	TAU32 stream_id;
	TAU32 cpu_thread_id;
	//TAU64 start_time;
	//TAU64 end_time;
	TAU64 gpu_elapsed_time;
	TAU64 cpu_elapsed_time;
	//char tau_context[NAME_SIZE];
	string tau_context;
	TAU32 calls;
};


enum data_transfer {TOGPU, FROMGPU};
enum transfer_mode {SYNC, ASYNC};

/*
	kernel event inherits the basic event data structure. 
*/
struct kernel_event:event_base
{
	//char kernel_name[NAME_SIZE];
	string kernel_name;
	TAU32 block_dim;
	TAU32 grid_dim;
	//shared memory is per block
	TAU32 static_shared_mem;
	TAU32 dynamic_shared_mem;
	//registers per thread
	TAU32 registers;
	float occupancy;
/*	TAU32 warp_serialized;
	TAU32 gst_total;
	TAU32 gld_total;
	TAU32 gst_unalign;
	TAU32 gld_unalign;
	//this for constant memory
	TAU32 cache_miss;
	TAU32 branches;
	TAU32 divergent_branches;
	*/
};

/*
	memory event inherits the basic event data structure. 
*/
struct mem_event:event_base
{
	string mem_api;
	data_transfer tofrom;
	transfer_mode mode;
	TAU64 mem_size;
	unsigned long mem_address;
	kernel_event * kern_context;	   
};

/*
	data structure to keep track of the files.  
*/

struct file_info{
	unsigned int node_id;
	unsigned int stream_id;
	unsigned int device_id;
};

/*
	This is heart of the event management which manages the events per thread.
	EventManager object gets created for every different thread and the pointer 
	is stored in the thread local variable.     
*/

class EventManager{
	bool trace_enabled;
	list<event_base *> trace_events; 
	list<api_event> api_events;                     // api event tracker list
	//maintains the count as well
	map<string,event_base *> profiles;		// profile event record map 
	map<string,fstream *> out_files;		// output file tracker
	map<string, Ttf_FileHandleT> trace_files;	// trace file tracker
 	map<string, file_info> trace_files_info;	// trace file information 
	void SpitEvent(string event_name, string type, TAU32 device_id, TAU32 stream_id, TAU64 time); 
	void SpitMessage(TAU32 device_id, TAU32 stream_id, TAU64 size);
 
	void PopulateProfile(event_base *event);
	void WriteProfileToFile();
	void CreateMetricFolder(string metric_name);
	fstream * GetProfileOut(unsigned int node, unsigned int stream,unsigned int dev_id, string metric_name);
	Ttf_FileHandleT GetTraceOut(unsigned int node, unsigned int stream,unsigned int dev_id);
	void CloseProfileOut(fstream *out_fstr);
	void CloseProfile(string metric_name);

	void CloseTraces();
	int GetEventCount(int dev_id, int stream,event_type type);
	double GetTotalTime(int dev, int stream);
	//retrieve the current program context
	string GetProgramContext();
	string ExtractRtnName(string rtn_sign);
	bool IsTraceSet();
	void SpitEvent(Ttf_fileT *tFile,unsigned int event_id, unsigned int nid, unsigned int tid, TAU64 timestamp, TAU64 param); 
	//keeps the currently populated profile record
	event_base * current_profile;                    
	string base_dir;                                // base directory for profile generation.   
	unsigned int thread_id;				// thread ID
//	int node_id;					// node ID in MPI applications
	int locnode;
	int locrank;
	int loccore;
	TAU64 start_time_stamp;				// start timestamp used to generate dummy top level event
	TAU64 end_time_stamp;				// end timestamp for dummy top level event. 
public:
	EventManager();
	EventManager(string base_dir);
	EventManager(string base_dir, bool trace);
	~EventManager();
	// this is called when the APi gets invoked 
	void APIEntry(TAU64 ctxt, TAU64 stream, TAU64 eventid, event_type type, char *api_name, TAU32 device_id); 
	//This populates memory profiles 
	void MemProfileEvent(TAU64 ctxt,TAU64 stream_id, TAU64 eventid, TAU64 start_time, TAU64 end_time, TAU64 mem_size);
	//This populates Kernel launch profile event
	void KernelProfileEvent(TAU64 ctxt,TAU64 stream_id,TAU64 eventid, char * callname, TAU64 start_time,TAU64 end_time);
	//This updates Kernel profile 
	void UpdateKernelProfile(TAU32 g_size, TAU32 b_size, TAU32 s_sm, TAU32 d_sm, TAU32 registers, float occupancy);
	
	void KernelTraceEvent(TAU64 ctxt,TAU64 stream_id,TAU64 eventid, char * callname, TAU64 start_time,TAU64 end_time,
				TAU32 g_size, TAU32 b_size, TAU32 s_sm, TAU32 d_sm, TAU32 registers, float occupancy);
	//This handles tracing in general
	void TraceEvent(string event_name, TAU64 start_time, TAU64 end_time, TAU32 event_id, unsigned int node_id, unsigned int stream, unsigned int device,
			TAU32 g_size, TAU32 b_size, TAU32 s_sm, TAU32 d_sm, TAU32 registers, float occupancy); 
	// tracing memory events taken care by this routine 
	void TraceMemEvent(string event_name, TAU64 start_time, TAU64 end_time, TAU32 event_id, unsigned int node_id, unsigned int stream, 
								unsigned int device, TAU64 mem_size, TAU64 ctxt, TAU64 event_id, event_type type);  

	void SetThread(unsigned int tid);

	/*
		Helper routines for computing and wtiting out profiles for different metrics
	*/

	void WriteElaspedMetric();
	void WriteMemTransferMetric();
	void WriteStaicMemoryMetric();
	void WriteDynamicMemoryMetric(); 
	void WriteOccupancyMetric();
	void WriteThreadRegisterMetric();

	int MyGPUNode(int device);
 	
	bool IsTraceEnabled();
	void SynChronize(int stream);
	void ThreadExit();	
}; 
