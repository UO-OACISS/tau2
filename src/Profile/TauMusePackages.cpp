/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.acl.lanl.gov/tau                        **
 * *****************************************************************************
 * **    Copyright 2003                                                       **
 * **    Department of Computer and Information Science, University of Oregon **
 * **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauMusePackage.cpp                                    **
 * **      Description     : TAU MUSE/MAGNET Interface                      **
 * **      Author          : Suravee Suthikulpanit                          **
 * **      Contact         : Suravee@cs.uoregon.edu                         **
 * **      Flags           : Compile with                                   **
 * **                        -DTAU_MUSE                                     **
 * ****************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

#include <Profile/TauMuse.h>

#ifdef TAU_MUSE_EVENT
/*
// FOR TESTING
TAU_REGISTER_EVENT(bandwidth_recv,"bandwidth_recv");
TAU_REGISTER_EVENT(bandwidth_send,"bandwidth_send");
TAU_REGISTER_EVENT(tcpbandwidth_recv,"tcpbandwidth_recv");
TAU_REGISTER_EVENT(tcpbandwidth_send,"tcpbandwidth_send");
TAU_REGISTER_EVENT(accumulator_sock_send,"accumulator_sock_send");
TAU_REGISTER_EVENT(accumulator_sock_recv,"accumulator_sock_recv");
*/
#endif //TAU_MUSE_EVENT


// This function choose the appropriate create_encoder for the handler.
int create_encode_selector(char *handler_name,char *ascii_command,int size, char *binary_command){
	if(!strncmp("TAU_count",handler_name,9)){
		return CreateTauCountEncode(ascii_command,size,binary_command);
	}else if(!strncmp("process_scheduling",handler_name,18)){
		return CreateProcessSchedulingEncode(ascii_command,size,binary_command);
	}else if(!strncmp("accumulator",handler_name,11)){
		return CreateAccumulatorEncode(ascii_command,size,binary_command);
	}else if(!strncmp("bandwidth",handler_name,9)){
		return CreateBandwidthEncode(ascii_command,size,binary_command);
	}else if(!strncmp("tcpbandwidth",handler_name,12)){
		return CreateTcpBandwidthEncode(ascii_command,size,binary_command);
	}else{
		return -1 ;
	}
}

// This function choose the appropriate query_decoder for the handler.
double query_decode_selector(char *handler_name,
			const char *binary_command, const char *binary_reply,
			int size, char *ascii_reply, double data[]){
	if(!strncmp("TAU_count",handler_name,9)){
		return QueryTauCountDecode(binary_command,binary_reply,size,ascii_reply,data);
	}else if(!strncmp("process_scheduling",handler_name,18)){
		return QueryProcessSchedulingDecode(binary_command,binary_reply,size,ascii_reply,data);
	}else if(!strncmp("accumulator",handler_name,11)){
		return QueryAccumulatorDecode(binary_command,binary_reply,size,ascii_reply,data);
	}else if(!strncmp("bandwidth",handler_name,9)){
		return QueryBandwidthDecode(binary_command,binary_reply,size,ascii_reply,data);
	}else if(!strncmp("tcpbandwidth",handler_name,12)){
		return QueryTcpBandwidthDecode(binary_command,binary_reply,size,ascii_reply,data);
	}else{
		return -1 ;
	}
}

// This function choose the appropriate addfilter_encoder.
int addfilter_encode_selector(char *filter_name,char *ascii_command,int size, char *binary_command){
	if(!strcmp("process_filter",filter_name)){
		return AddFilterProcessFilterEncode(ascii_command,size,binary_command);
	}else if(!strcmp("socket_filter",filter_name)){
		return AddFilterSocketFilterEncode(ascii_command,size,binary_command);
	}else{
		return -1 ;
	}
}

// FOR TESTING
#ifdef TAU_MUSE_EVENT
/*
int report_user_defined_events(double data[]){

	char *metrics[10];
	int sizeofMetrics,i;
	for(i=0;i<10;i++)
		metrics[i]=(char*)malloc(100);
	sizeofMetrics=TauMuseGetMetricsNonMono(metrics,10);	

	printf("sizeofMetrics = %d\n",sizeofMetrics);
	
	for(i=0;i<sizeofMetrics;i++){	
		if(!strncmp("bandwidth_recv",metrics[i],14)){
			TAU_EVENT(bandwidth_recv,data[i]);
		}else if(!strncmp("bandwidth_send",metrics[i],14)){
			TAU_EVENT(bandwidth_send,data[i]);
		}else if(!strncmp("tcpbandwidth_recv",metrics[i],17)){
			TAU_EVENT(tcpbandwidth_recv,data[i]);
		}else if(!strncmp("tcpbandwidth_send",metrics[i],17)){
			TAU_EVENT(tcpbandwidth_send,data[i]);
		}else if(!strncmp("accumulator_sock_send",metrics[i],21)){
			TAU_EVENT(accumulator_sock_send,data[i]);
		}else if(!strncmp("accumulator_sock_recv",metrics[i],21)){
			TAU_EVENT(accumulator_sock_recv,data[i]);
		}
	}
	for(i=0;i<10;i++)
		free(metrics[i]);

	return 0;	
}
*/
#endif //TAU_MUSE_EVENT

/////////////////////////////////////////////////////////////////////////////////////////
//=================================================================================
// TAU-MUSE PRE-DEFINED PACKAGE
// 
// Description:
// This function select the appropriate package 
// configuration according to the specified 
// package name. 
//=================================================================================
int monotonic_package_selector(struct package_info *pkg){

//=============================================
// THIS IS FOR MONOTONICALLY INCREASING VALUE
//=============================================

	//*****************************************
	// Package : count	
	//*****************************************	
	if(!strcmp("count",pkg->package_name)){
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=1;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"count");
		
		sprintf(pkg->handlers[0].handler_name,"TAU_count");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
#endif
		return TauMuseCreate(pkg);
	} 
	//*****************************************
	// Package : busy_time
	//*****************************************	
	else if(!strcmp("busy_time",pkg->package_name)){
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=2;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"busy_time");
		
		sprintf(pkg->handlers[0].handler_name,"process_scheduling");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(pkg->handlers[0].filters[0].args[1],0,100);
		sprintf(pkg->handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
		printf("args: %s\n",pkg->handlers[0].filters[0].args[1]);
#endif
		return TauMuseCreate(pkg);
	} 
	//*****************************************
	// Package : idle_time
	//*****************************************	
	else if(!strcmp("idle_time",pkg->package_name)){
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=2;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"idle_time");
		
		sprintf(pkg->handlers[0].handler_name,"process_scheduling");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		// This is to instrument MAGNET_TASK_CTX_IN for pid=0
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter pid=0 event=258");
		memset(pkg->handlers[0].filters[0].args[1],0,100);
		// This is to instrument MAGNET_TASK_CTX_OUT for pid=0
		sprintf(pkg->handlers[0].filters[0].args[1],"process_filter pid=0 event=259");
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
		printf("args: %s\n",pkg->handlers[0].filters[0].args[1]);
#endif
		return TauMuseCreate(pkg);
	} 
	//*****************************************
	// Package : total_time
	//*****************************************	
	else if(!strcmp("total_time",pkg->package_name)){
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=2;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"total_time");
		
		sprintf(pkg->handlers[0].handler_name,"process_scheduling");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(pkg->handlers[0].filters[0].args[1],0,100);
		sprintf(pkg->handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
		printf("args: %s\n",pkg->handlers[0].filters[0].args[1]);
#endif
		return TauMuseCreate(pkg);
	}
	//*****************************************
	// Package : total_time_debug
	// NOTE: This is a hack version for Jeremy
	//*****************************************	
	else if(!strcmp("total_time_debug",pkg->package_name)){
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=2;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"total_time_debug");
		
		sprintf(pkg->handlers[0].handler_name,"process_scheduling pid=%d",getpid());
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter event=258");
		memset(pkg->handlers[0].filters[0].args[1],0,100);
		sprintf(pkg->handlers[0].filters[0].args[1],"process_filter event=259");
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
		printf("args: %s\n",pkg->handlers[0].filters[0].args[1]);
#endif
		return TauMuseCreate(pkg);
	}
	//*****************************************
	// Package : context_switch
	//*****************************************	
	else if(!strcmp("context_switch",pkg->package_name)){
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=2;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"context_switch");
		
		sprintf(pkg->handlers[0].handler_name,"process_scheduling");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(pkg->handlers[0].filters[0].args[1],0,100);
		sprintf(pkg->handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
		printf("args: %s\n",pkg->handlers[0].filters[0].args[1]);
#endif
		return TauMuseCreate(pkg);
	}
	//*****************************************
	// Package : accumulator_sock_send
	//*****************************************	
	else if(!strcmp("accumulator_sock_send",pkg->package_name)){
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		// handler[0]
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=1;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"accumulator_sock_send");
		
		sprintf(pkg->handlers[0].handler_name,"accumulator");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"socket_filter pid=%d event=3",getpid());
		
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
#endif
		return TauMuseCreate(pkg);
	}

	//*****************************************
	// Package : accumulator_sock_recv
	//*****************************************	
	else if(!strcmp("accumulator_sock_recv",pkg->package_name)){
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		// handler[0]
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=1;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"accumulator_sock_recv");
		
		sprintf(pkg->handlers[0].handler_name,"accumulator");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"socket_filter pid=%d event=4",getpid());
		
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
#endif
		return TauMuseCreate(pkg);
	}
	//*****************************************
	// Package : Default
	//*****************************************	
	else{
		pkg->numofhandlers=1;
		pkg->totalcounters=1;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=1;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"count");
		
		sprintf(pkg->handlers[0].handler_name,"TAU_count");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
#endif
		return TauMuseCreate(pkg);
	}
        return 0;
}

//=================================================
// THIS IS FOR NON-MONOTONICALLY INCREASING VALUE
//=================================================
int nonmonotonic_package_selector(struct package_info *pkg){
	//*****************************************
	// Package : bandwidth  
	//*****************************************	
	if(!strcmp("bandwidth",pkg->package_name)){
		
		pkg->numofhandlers=1;
		pkg->totalcounters=2;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=2;
		pkg->handlers[0].filters[0].filter_argc=1;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"bandwidth_recv");
		strcpy((char*)&pkg->handlers[0].metrics[1].info,"bandwidth_send");

		// NOTE:
		// "bandwidth 1" means creating bandwidth 
		// handler with delta time = 1 
		sprintf(pkg->handlers[0].handler_name,"bandwidth 1");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"socket_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
#endif
		return TauMuseCreate(pkg);
	}
	//*****************************************
	// Package : tcpbandwidth  
	//*****************************************	
	else if(!strcmp("tcpbandwidth",pkg->package_name)){

		pkg->numofhandlers=1;
		pkg->totalcounters=2;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=2;
		pkg->handlers[0].filters[0].filter_argc=1;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"tcpbandwidth_recv");
		strcpy((char*)&pkg->handlers[0].metrics[1].info,"tcpbandwidth_send");

		// NOTE:
		// "bandwidth 1" means creating bandwidth 
		// handler with delta time = 1 
		sprintf(pkg->handlers[0].handler_name,"tcpbandwidth 1");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"socket_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
#endif
		return TauMuseCreate(pkg);
	}
	//*****************************************
	// Package : accumulator_sock_send_recv
	// Note This is a hack version for testing.
	//*****************************************	
	else if(!strcmp("accumulator_sock_send_recv",pkg->package_name)){
		pkg->numofhandlers=2;
		pkg->totalcounters=2;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"accumulator_sock_send");
		strcpy((char*)&pkg->handlers[1].metrics[0].info,"accumulator_sock_recv");
		
		// handler[0]
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=1;
		sprintf(pkg->handlers[0].handler_name,"accumulator");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter event=3");
		
		// handler[1]
		pkg->handlers[1].numoffilters=1;
		pkg->handlers[1].numofcounters=1;
		pkg->handlers[1].filters[0].filter_argc=1;
		sprintf(pkg->handlers[1].handler_name,"accumulator");
		memset(pkg->handlers[1].filters[0].args[0],0,100);
		sprintf(pkg->handlers[1].filters[0].args[0],"process_filter event=4");
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
		printf("args: %s\n",pkg->handlers[1].filters[0].args[0]);
#endif
		return TauMuseCreate(pkg);
	}
	//*****************************************
	// Package : Default
	//*****************************************	
	else{
		pkg->numofhandlers=1;
		pkg->totalcounters=2;
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=2;
		pkg->handlers[0].filters[0].filter_argc=1;
		strcpy((char*)&pkg->handlers[0].metrics[0].info,"bandwidth_recv");
		strcpy((char*)&pkg->handlers[0].metrics[1].info,"bandwidth_send");

		// NOTE:
		// "bandwidth 1" means creating bandwidth 
		// handler with delta time = 1 
		sprintf(pkg->handlers[0].handler_name,"bandwidth 1");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"socket_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
#endif
		return TauMuseCreate(pkg);
	}
}

#if 0
//======================================================
// THIS IS FOR MULTIPLE MONOTONICALLY INCREASING VALUES
//======================================================
#ifdef TAU_MUSE_MULTIPLE
	//*****************************************
	// Package : accumulator_sock_send_recv
	//*****************************************	
	if(!strcmp("accumulator_sock_send_recv",pkg->package_name)){
		pkg->numofhandlers=2;
		pkg->totalcounters=2;
		// handler[0]
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=1;
		sprintf(pkg->handlers[0].handler_name,"accumulator");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter event=3");
		
		// handler[1]
		pkg->handlers[1].numoffilters=1;
		pkg->handlers[1].numofcounters=1;
		pkg->handlers[1].filters[0].filter_argc=1;
		sprintf(pkg->handlers[1].handler_name,"accumulator");
		memset(pkg->handlers[1].filters[0].args[0],0,100);
		sprintf(pkg->handlers[1].filters[0].args[0],"process_filter event=4");
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
		printf("args: %s\n",pkg->handlers[1].filters[0].args[0]);
#endif
		return TauMuseCreate();
	}
	//*****************************************
	// Package : Default
	//*****************************************	
	else{
		pkg->numofhandlers=2;
		pkg->totalcounters=2;
		// handler[0]
		pkg->handlers[0].numoffilters=1;
		pkg->handlers[0].numofcounters=1;
		pkg->handlers[0].filters[0].filter_argc=1;
		sprintf(pkg->handlers[0].handler_name,"accumulator");
		memset(pkg->handlers[0].filters[0].args[0],0,100);
		sprintf(pkg->handlers[0].filters[0].args[0],"process_filter event=3");
		
		// handler[1]
		pkg->handlers[1].numoffilters=1;
		pkg->handlers[1].numofcounters=1;
		pkg->handlers[1].filters[0].filter_argc=1;
		sprintf(pkg->handlers[1].handler_name,"accumulator");
		memset(pkg->handlers[1].filters[0].args[0],0,100);
		sprintf(pkg->handlers[1].filters[0].args[0],"process_filter event=4");
#ifdef DEBUG_PROF
		printf("args: %s\n",pkg->handlers[0].filters[0].args[0]);
		printf("args: %s\n",pkg->handlers[1].filters[0].args[0]);
#endif
		return TauMuseCreate();
	}
#endif //TAU_MUSE_MULTIPLE
}
#endif // 0
