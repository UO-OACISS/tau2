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

/* TheMuseSockId() is a global variable now */
int& TheMuseSockId(void)
{
  static int sockid = 0; 
  return sockid;
}

/* TheMusePackage() is a global variable now */
struct package_info& TheMusePackage(void){
        static struct package_info pkg;
        return pkg;
}

/* Get the name of TAU_MUSE predefine package */
char * get_muse_package(void)
{
        char *package = getenv("TAU_MUSE_PACKAGE");
        if (package == (char *) NULL)
        {  /* the user has not specified any handler name */
          return "TAU_count";
        }
        else
          return package;
}

// This function choose the appropriate create_encoder for the handler.
int create_encode_selector(char *handler_name,char *ascii_command,int size, char *binary_command){
	if(!strcmp("TAU_count",handler_name)){
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
	if(!strcmp("TAU_count",handler_name)){
		return QueryTauCountDecode(binary_command,binary_reply,size,ascii_reply,data);
	}else if(!strcmp("process_scheduling",handler_name)){
		return QueryProcessSchedulingDecode(binary_command,binary_reply,size,ascii_reply,data);
	}else if(!strcmp("accumulator",handler_name)){
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

/////////////////////////////////////////////////////////////////////////////////////////
//=================================================================================
// TAU-MUSE PRE-DEFINED PACKAGE
//=================================================================================
int package_selector(int *data){
	char *package=get_muse_package();

	TauMuseInit();
//=============================================
// THIS IS FOR MONOTONICALLY INCREASING VALUE
//=============================================
#ifdef TAU_MUSE	
	//*****************************************
	// Package : count	
	//*****************************************	
	if(!strcmp("count",package)){
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=1;
		
		sprintf(TheMusePackage().handlers[0].handler_name,"TAU_count");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
#endif
		TauMuseCreate();
	} 
	//*****************************************
	// Package : busy_time
	//*****************************************	
	else if(!strcmp("busy_time",package)){
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=2;
		
		sprintf(TheMusePackage().handlers[0].handler_name,"process_scheduling");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(TheMusePackage().handlers[0].filters[0].args[1],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[1]);
#endif
		TauMuseCreate();
	} 
	//*****************************************
	// Package : idle_time
	//*****************************************	
	else if(!strcmp("idle_time",package)){
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=2;
		
		sprintf(TheMusePackage().handlers[0].handler_name,"process_scheduling");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		// This is to instrument MAGNET_TASK_CTX_IN for pid=0
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=0 event=258");
		memset(TheMusePackage().handlers[0].filters[0].args[1],0,100);
		// This is to instrument MAGNET_TASK_CTX_OUT for pid=0
		sprintf(TheMusePackage().handlers[0].filters[0].args[1],"process_filter pid=0 event=259");
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[1]);
#endif
		TauMuseCreate();
	} 
	//*****************************************
	// Package : total_time
	//*****************************************	
	else if(!strcmp("total_time",package)){
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=2;
		
		sprintf(TheMusePackage().handlers[0].handler_name,"process_scheduling");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(TheMusePackage().handlers[0].filters[0].args[1],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[1]);
#endif
		TauMuseCreate();
	}
	//*****************************************
	// Package : context_switch
	//*****************************************	
	else if(!strcmp("context_switch",package)){
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=2;
		
		sprintf(TheMusePackage().handlers[0].handler_name,"process_scheduling");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(TheMusePackage().handlers[0].filters[0].args[1],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[1]);
#endif
		TauMuseCreate();
	}
	//*****************************************
	// Package : accumulator_sock_send
	//*****************************************	
	else if(!strcmp("accumulator_sock_send",package)){
		TheMusePackage().numofhandlers=1;
		// handler[0]
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=1;
		sprintf(TheMusePackage().handlers[0].handler_name,"accumulator");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"socket_filter pid=%d event=3",getpid());
		
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
#endif
		TauMuseCreate();
	}

	//*****************************************
	// Package : accumulator_sock_recv
	//*****************************************	
	else if(!strcmp("accumulator_sock_recv",package)){
		TheMusePackage().numofhandlers=1;
		// handler[0]
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=1;
		sprintf(TheMusePackage().handlers[0].handler_name,"accumulator");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"socket_filter pid=%d event=4",getpid());
		
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
#endif
		TauMuseCreate();
	}

	//*****************************************
	// Package : Default
	//*****************************************	
	else{
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=1;
		
		sprintf(TheMusePackage().handlers[0].handler_name,"TAU_count");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
#endif
		TauMuseCreate();
	}
        return 0;
#endif //TAU_MUSE

//=================================================
// THIS IS FOR NON-MONOTONICALLY INCREASING VALUE
//=================================================
#ifdef TAU_MUSE_EVENT
	//*****************************************
	// Package : bandwidth  
	//*****************************************	
	if(!strcmp("bandwidth",package)){
		if(data == NULL){
			printf("ERROR: Unspecified socket.\n");
			return 0;
		}
		
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=1;

		// NOTE:
		// "bandwidth 1" means creating bandwidth 
		// handler with delta time = 1 
		sprintf(TheMusePackage().handlers[0].handler_name,"bandwidth 1");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"socket_filter pid=%d sid=%d",getpid(),*data);
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
#endif
		TauMuseCreate();
	}
	//*****************************************
	// Package : tcpbandwidth  
	//*****************************************	
	else if(!strcmp("tcpbandwidth",package)){
		if(data == NULL){
			printf("ERROR: Unspecified socket.\n");
			return 0;
		}

		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=1;

		// NOTE:
		// "bandwidth 1" means creating bandwidth 
		// handler with delta time = 1 
		sprintf(TheMusePackage().handlers[0].handler_name,"tcpbandwidth 1");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"socket_filter pid=%d sid=%d",getpid(),*data);
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
#endif
		TauMuseCreate();
	}
	//*****************************************
	// Package : Default
	//*****************************************	
	else{
		if(data == NULL){
			printf("ERROR: Unspecified socket.\n");
			return 0;
		}
		
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=1;

		// NOTE:
		// "bandwidth 1" means creating bandwidth 
		// handler with delta time = 1 
		sprintf(TheMusePackage().handlers[0].handler_name,"bandwidth 1");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"socket_filter pid=%d sid=%d",getpid(),*data);
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
#endif
		TauMuseCreate();
	}
#endif //TAU_MUSE_EVENT

//======================================================
// THIS IS FOR MULTIPLE MONOTONICALLY INCREASING VALUES
//======================================================
#ifdef TAU_MUSE_MULTIPLE
	//*****************************************
	// Package : process_stat
	//*****************************************	
	if(!strcmp("process_stat",package)){
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=3;
		TheMusePackage().handlers[0].filters[0].filter_argc=2;
		
		sprintf(TheMusePackage().handlers[0].handler_name,"process_scheduling");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(TheMusePackage().handlers[0].filters[0].args[1],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[1]);
#endif
		TauMuseCreate();
	}
	//*****************************************
	// Package : accumulator_sock_send_recv
	//*****************************************	
	else if(!strcmp("accumulator_sock_send_recv",package)){
		TheMusePackage().numofhandlers=2;
		// handler[0]
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=1;
		TheMusePackage().handlers[0].filters[0].filter_argc=1;
		sprintf(TheMusePackage().handlers[0].handler_name,"accumulator");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter event=3");
		
		// handler[1]
		TheMusePackage().handlers[1].numoffilters=1;
		TheMusePackage().handlers[1].numofcounters=1;
		TheMusePackage().handlers[1].filters[0].filter_argc=1;
		sprintf(TheMusePackage().handlers[1].handler_name,"accumulator");
		memset(TheMusePackage().handlers[1].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[1].filters[0].args[0],"process_filter event=4");
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
		printf("args: %s\n",TheMusePackage().handlers[1].filters[0].args[0]);
#endif
		TauMuseCreate();
	}
	//*****************************************
	// Package : all
	//*****************************************	
	else if(!strcmp("all",package)){
		TheMusePackage().numofhandlers=2;
		
		//handlers[0] 	
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=3;
		TheMusePackage().handlers[0].filters[0].filter_argc=2;
		sprintf(TheMusePackage().handlers[0].handler_name,"process_scheduling");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(TheMusePackage().handlers[0].filters[0].args[1],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
		//handlers[1] 	
		TheMusePackage().handlers[1].numoffilters=1;
		TheMusePackage().handlers[1].numofcounters=1;
		TheMusePackage().handlers[1].filters[0].filter_argc=1;
		sprintf(TheMusePackage().handlers[1].handler_name,"TAU_count");
		memset(TheMusePackage().handlers[1].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[1].filters[0].args[0],"process_filter pid=%d",getpid());
#ifdef DEBUG_PROF

		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[1]);
		printf("args: %s\n",TheMusePackage().handlers[1].filters[0].args[0]);
#endif
		TauMuseCreate();
	}
	//*****************************************
	// Package : Default
	//*****************************************	
	else{
		TheMusePackage().numofhandlers=1;
		TheMusePackage().handlers[0].numoffilters=1;
		TheMusePackage().handlers[0].numofcounters=3;
		TheMusePackage().handlers[0].filters[0].filter_argc=2;
		
		sprintf(TheMusePackage().handlers[0].handler_name,"process_scheduling");
		memset(TheMusePackage().handlers[0].filters[0].args[0],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[0],"process_filter pid=%d event=258",getpid());
		memset(TheMusePackage().handlers[0].filters[0].args[1],0,100);
		sprintf(TheMusePackage().handlers[0].filters[0].args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[0]);
		printf("args: %s\n",TheMusePackage().handlers[0].filters[0].args[1]);
#endif
		TauMuseCreate();
	}
#endif //TAU_MUSE_MULTIPLE
}
