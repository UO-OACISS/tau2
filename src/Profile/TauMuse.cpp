/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.acl.lanl.gov/tau                        **
 * *****************************************************************************
 * **    Copyright 2003                                                       **
 * **    Department of Computer and Information Science, University of Oregon **
 * **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauMuse.cpp                                    **
 * **      Description     : TAU MUSE/MAGNET Interface                      **
 * **      Author          : Suravee Suthikulpanit                          **
 * **      Contact         : Suravee@cs.uoregon.edu                         **
 * **      Flags           : Compile with                                   **
 * **                        -DTAU_MUSE                                     **
 * ****************************************************************************/
// NOTE: This is implented for using with "count" handler at this point.
// 	 Encoder and Decoder are needed for different handler.
//
/* This file has routines for connecting to the MAGNETD server and sending 
 * commands. */

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/poll.h>

#include <Profile/TauMuse.h>

#define HOST_PORT 		9997			// MUSE server port
#define HOST_IP 		"127.0.0.1"		
#define BUFFERSIZE 		8192
#define MAX_ARGLEN 		255
#define MAX_REPLY_LENGTH 	1024 		 
//#define DEBUG_PROF 		 

/**************************************
* Description	: PACKET PROTOCOL for communicating with
* 		  the MUSE server.
* From		: translator.c
**************************************
The sizes for all parameters are:
<command value> = 1 byte
<handler id> = 1 bytes
<size> = 4 bytes
<ASCII string> = <string length> = 1 byte, <string> <= 254 bytes, <NULL> = 1 byte
<options> = # of bytes specified by <size>
The commands are defined as follows:
HELP:                   <command=0> <ASCII string> <size=0> <NULL>
CREATE:                 <command=1> <ASCII string> <size> <options>
QUERY_MAPPER:   	<command=2> <ASCII string> <size> <options>
QUERY_HANDLER:  	<command=3> <handler id> <size=0>
DESTROY:                <command=4> <handler id> <size=0>
START:                  <command=5> <handler id> <size=0>
STOP:                   <command=6> <handler id> <size=0>
RESETFILTERS:   	<command=7> <handler id> <size=0>
ADDFILTER:              <command=8> <handler id> <ASCII string> <size> <options>
GET:                    <command=9> <size> <options (1st byte matters)>
QUIT:                   <command=10> <size=0>
*****************************************/

/*********************
 * Description	: Send binary command to MUSE server and verify
 * From		: mdsh/mdsh.c <modified>
 *********************/
int send_and_check(int sockfd,int command_length,char *send_buffer,char *recv_buffer){
        struct pollfd poll_fd;
        unsigned int b;
        int network_command_length;

        network_command_length = htonl(command_length);
        // send command_length
        if(send(sockfd, &network_command_length,sizeof(network_command_length),0) == -1){
                perror("send");
                printf("TauMuse.cpp: Unable to send command_length\n");
                close(sockfd);
                return(1);
        }
        // send command
        if(send(sockfd, send_buffer,command_length,0) == -1){
                perror("send");
                printf("TauMuse.cpp: Unable to send command to MUSE\n");
                close(sockfd);
                return(1);
        }
        // receive confirmation
        memset(recv_buffer,0,BUFFERSIZE);
        poll_fd.fd = sockfd;
        poll_fd.events = POLLIN;

        if (poll(&poll_fd, 1, -1) < 0) {
              printf("TauMuse.cpp: poll() failed in server thread: %s", strerror(errno));
        }
        // Figure out what happened 

        if (poll_fd.revents & POLLERR) {
                printf("TauMuse.cpp: Error: poll() returned POLLERR\n");
                kill(0,SIGTERM);
        } else if (poll_fd.revents & POLLHUP) {
                printf("\nTauMuse.cpp: Hang up signal received from server.  Terminating...\n");
                kill(0,SIGTERM);
        } else if (poll_fd.revents & POLLNVAL) {
                printf("\nTauMuse.cpp: Error: poll() returned POLLNVAL\n");
                kill(0,SIGTERM);
        } else if (poll_fd.revents & POLLIN) {
                if ((b = recv(sockfd, recv_buffer, BUFFERSIZE, 0)) == -1) {
                        printf("TauMuse.cpp: recv() failed: %s", strerror(errno));
                        kill(0,SIGTERM);
                }
                if (b == 0) {
                          printf("\nTauMuse.cpp: Hang up signal received from server. Terminating...\n");
                          kill(0,SIGTERM);
                }
                while (b < sizeof(int)) {
                        if ((b += recv(sockfd, &recv_buffer[b], BUFFERSIZE-b, 0)) == -1) {
                                printf("TauMuse.cpp: recv() failed: %s", strerror(errno));
                                kill(0,SIGTERM);
                        }
                }

                b -= sizeof(int);

                while (b < ntohl(((int *)recv_buffer)[0])) {
                        if ((b += recv(sockfd, &recv_buffer[b+sizeof(int)],
                                  BUFFERSIZE-b-sizeof(int), 0)) == -1) {
                                printf("TauMuse.cpp: recv() failed: %s", strerror(errno));
                                kill(0,SIGTERM);
                        }
                }
        }
        return(0);

}

/* Get the name of the MUSE handler */
char * get_muse_handler_name(void)
{
	char *handler = getenv("TAU_MUSE_HANDLER");
	if (handler == (char *) NULL)
	{  /* the user has not specified any handler name */
	  return "TAU_count";
	}
	else
	  return handler;
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

/* TheMuseSockId() is a global variable now */
int& TheMuseSockId(void)
{
  static int sockid = 0; 
  return sockid;
}

/* TheMuseHandlerId() is a global variable now */
int& TheMuseHandlerId(void)
{
  static int handlerid = 0; 
  return handlerid;
}

// This function choose the appropriate create_encoder for the handler.
int create_encode_selector(char *handler_name,char *ascii_command,int size, char *binary_command){
	if(!strcmp("TAU_count",handler_name)){
		return CreateTauCountEncode(ascii_command,size,binary_command);
	}else if(!strcmp("process_scheduling",handler_name)){
		return CreateProcessSchedulingEncode(ascii_command,size,binary_command);
	}else{
		return -1 ;
	}
}

// This function choose the appropriate query_decoder for the handler.
double query_decode_selector(char *handler_name,
			const char *binary_command, const char *binary_reply,
			int size, char *ascii_reply){
	if(!strcmp("TAU_count",handler_name)){
		return QueryTauCountDecode(binary_command,binary_reply,size,ascii_reply);
	}else if(!strcmp("process_scheduling",handler_name)){
		return QueryProcessSchedulingDecode(binary_command,binary_reply,size,ascii_reply);
	}else{
		return -1 ;
	}
}

// This function choose the appropriate addfilter_encoder.
int addfilter_encode_selector(char *filter_name,char *ascii_command,int size, char *binary_command){
	if(!strcmp("process_filter",filter_name)){
		return AddFilterProcessFilterEncode(ascii_command,size,binary_command);
	}else{
		return -1 ;
	}
}

// This is the predefined TAU_MUSE package.
int package_selector(void){
	char *args[MAX_FILTER_ARGS];
	char *package=get_muse_package();
	
	if(!strcmp("count",package)){
		args[0]=(char*)malloc(100);
		memset(args[0],0,100);
		sprintf(args[0],"process_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",args[0]);
#endif
		setenv("TAU_MUSE_HANDLER","TAU_count",1);
		return TauMuseInit("TAU_count",1,args);
	} else if(!strcmp("busy_time",package)){
		args[0]=(char*)malloc(100);
		memset(args[0],0,100);
		sprintf(args[0],"process_filter pid=%d event=258",getpid());
		args[1]=(char*)malloc(100);
		memset(args[1],0,100);
		sprintf(args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",args[0]);
		printf("args: %s\n",args[1]);
#endif
		setenv("TAU_MUSE_HANDLER","process_scheduling",1);
		return TauMuseInit("process_scheduling",2,args);
	} else if(!strcmp("idle_time",package)){
		args[0]=(char*)malloc(100);
		memset(args[0],0,100);
		sprintf(args[0],"process_filter pid=0 event=258");
		args[1]=(char*)malloc(100);
		memset(args[1],0,100);
		sprintf(args[1],"process_filter pid=0 event=259");
#ifdef DEBUG_PROF
		printf("args: %s\n",args[0]);
		printf("args: %s\n",args[1]);
#endif
		setenv("TAU_MUSE_HANDLER","process_scheduling",1);
		return TauMuseInit("process_scheduling",2,args);
	} else if(!strcmp("total_time",package)){
		args[0]=(char*)malloc(100);
		memset(args[0],0,100);
		sprintf(args[0],"process_filter pid=%d event=258",getpid());
		args[1]=(char*)malloc(100);
		memset(args[1],0,100);
		sprintf(args[1],"process_filter pid=%d event=259",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",args[0]);
		printf("args: %s\n",args[1]);
#endif
		setenv("TAU_MUSE_HANDLER","process_scheduling",1);
		return TauMuseInit("process_scheduling",2,args);
	}else{
		//BY Default
		args[0]=(char*)malloc(100);
		memset(args[0],0,100);
		sprintf(args[0],"process_filter pid=%d",getpid());
#ifdef DEBUG_PROF
		printf("args: %s\n",args[0]);
#endif
		setenv("TAU_MUSE_HANDLER","TAU_count",1);
		return TauMuseInit("TAU_count",1,args);
	}
	return 0;
}

//=================================================================================
// TAU-MUSE API IMPLEMENTATION
// -TauMuseInit		:Send command "create","addfilter","start" to magnetd
// -TauMuseQuery	:Send command "query"
// -TauMuseDestroy 	:Send command "stop","destroy","quit"
//================================================================================

/*********************
 * Description	: Initialize socket connecting to MUSE sever
 * 		  - connect
 * 		  - send command create <handler_hane> <args>
 * 		  - send command start <handlerID>
 * NOTE		: This function is called by TauMuseQuery	
 *********************/
int TauMuseInit(char *handler_name, int filter_argc,char *args[]){
        struct sockaddr_in host_addr;
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        char cmdstr[MAX_ARGLEN];
        int command_length;
        unsigned int size_reply;
        int *sizeptr;
         int handlerID;
        unsigned char *byteptr;
	int i;
	int sockfd ;
	
        // ===================================
        // Establish socket and connection
        // ===================================
        // host information
        memset(&host_addr,0,sizeof(host_addr));
        host_addr.sin_family = AF_INET;
        host_addr.sin_addr.s_addr = inet_addr(HOST_IP);
        host_addr.sin_port = htons(HOST_PORT);

        // create socket        
        if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
                perror("socket");
                printf("TauMuse.cpp: Unable to create socket: %s\n",
                                strerror(errno));
                return(1);
        }

	/* Assign sockfd to the global variable TheMuseSockId() */
	TheMuseSockId() = sockfd; 

#ifdef DEBUG_PROF
        printf("TauMuse.cpp: Connecting to magnetd using AF_INET.\n");
#endif /* DEBUG_PROF */
        // connect to magnetd
        if(connect(sockfd,(struct sockaddr *) &host_addr,
                        sizeof(host_addr)) == -1){
                perror("connect");
                printf("TauMuse.cpp: Unable to connect to server: %s\n",
                                strerror(errno));
                return(1);
        }
        // verify connection
        if(recv(sockfd, recv_buffer, BUFFERSIZE,0) == -1){
                perror("recv");
                printf("TauMuse.cpp: Unable to establish connection: %s\n",
                                strerror(errno));
                return(1);
        }
        if(recv_buffer[0] == 1){
                printf("TauMuse.cpp: Connection Refused from server.\n");
                close(sockfd);
                return(1);
        }else if(recv_buffer[0] == 0){
#ifdef DEBUG_PROF
                printf("------------Connection Established-----------\n");
#endif /* DEBUG_PROF */
        }else{
                printf("TauMuse.cpp: Unknown handshake reply.\n");
                close(sockfd);
                return(1);
        }

        // ====================================
        // command "create <handler_name>"
        // ====================================
        sprintf(cmdstr,"create %s",handler_name);
#ifdef DEBUG_PROF
        printf("cmdstr = %s\n",cmdstr);
#endif /* DEBUG_PROF */
        command_length = create_encode_selector(handler_name,cmdstr,BUFFERSIZE,send_buffer);
        send_and_check(sockfd,command_length,send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!handler is created\n");
#endif /* DEBUG_PROF */

	//-------------------------------------------
	// Need to extract information from recv_from	
        // Check HandlerID ... 
        // HACKY!!!... Endian stuff
        byteptr = (unsigned char *)recv_buffer+5;
        handlerID = (int) *byteptr;
#ifdef DEBUG_PROF
        printf("handlerID is %d\n",handlerID);
#endif /* DEBUG_PROF */

	/* Assign it to the global variable */
        TheMuseHandlerId() = (int) *byteptr;
	//-------------------------------------------
	
	// =====================================
	// command "addfilter <handlerID> <filter_name> <args>"
	// =====================================
	// Loop to add multiple filters.	
	for(i=0;i<filter_argc;i++) {	
		sprintf(cmdstr,"addfilter %d %s", handlerID,args[i]);
#ifdef DEBUG_PROF
		printf("cmdstr = %s\n",cmdstr);
#endif /* DEBUG_PROF */
		command_length = addfilter_encode_selector(strtok(args[i]," ")
				,cmdstr,BUFFERSIZE,send_buffer);
		send_and_check(sockfd,command_length,send_buffer,recv_buffer);
#ifdef DEBUG_PROF
		printf("!!!!!!!!!filter is added.\n");
#endif /* DEBUG_PROF */
	}

	// =====================================
        // command "start <handlerID>"
        // =====================================
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        // Command for start
        send_buffer[0] = 5;
	byteptr = (unsigned char *)&send_buffer[sizeof(unsigned char)];	
	*byteptr = (unsigned char)handlerID;

        send_and_check(sockfd,2+sizeof(int),send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!handlerID %d is started\n",handlerID);
#endif /* DEBUG_PROF */

        return(handlerID);
}

/*********************
 * Description	: Destroy socket connecting to MUSE sever
 * 		  - connect
 * 		  - send command stop <handlerID>
 * 		  - send command destroy <handlerID>
 * 		  - send command quit 
 *********************/
void TauMuseDestroy(void){
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        unsigned char *byteptr;
	int handlerID = TheMuseHandlerId(); /* read global */
	int sockfd = TheMuseSockId(); /* read */
        // ====================================
        // command "stop <handlerID>"
        // ====================================
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 6;
        byteptr = (unsigned char *)&handlerID;
        send_buffer[1] = (char)*byteptr;
        send_and_check(sockfd,2+sizeof(int),send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!handlerID %d is stopped\n",handlerID);
#endif /* DEBUG_PROF */
        // ====================================
        // command "destroy <handlerID>"
        // ====================================
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 4;
        send_buffer[1] = (char)*byteptr;
        send_and_check(sockfd,2+sizeof(int),send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!handlerID %d is destroyed\n",handlerID);
#endif /* DEBUG_PROF */
        // ====================================
        // command "quit"
        // ====================================
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 10;
        send_and_check(sockfd,1+sizeof(int),send_buffer,recv_buffer);
}

/*********************
 * Description	: Query_handler from MUSE sever
 * 		  - send command query_handler
 * NOTE		: Return double for now.
 *********************/
double TauMuseQuery(void){
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        char result_buffer[MAX_REPLY_LENGTH];
        unsigned char *byteptr;
	double result;
	
	// This will get the value from environment variable
	// to initilize the appropriate handler and filter arguments.
	static int handlerID = package_selector(); 
	int sockfd = TheMuseSockId(); /* read from the global */

        // ====================================
        // command "query_handler <handlerID>"
        // ====================================
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 3;
        byteptr = (unsigned char *)&handlerID;
        send_buffer[1] = (char)*byteptr;
        send_and_check(sockfd,2+sizeof(int),send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!handlerID %d is queried\n",handlerID);
#endif /* DEBUG_PROF */
        result = (double)query_decode_selector(get_muse_handler_name(),send_buffer,recv_buffer,
                        MAX_REPLY_LENGTH,result_buffer);
#ifdef DEBUG_PROF
	printf("TauMuseQuery---: get_muse_handler_name()=%s\n",get_muse_handler_name());
        printf("TauMuseQuery---: result buffer:\n%s\n",result_buffer);
        printf("TauMuseQuery---: result value passing to TAU: %f\n",result);
#endif /* DEBUG_PROF */
        return result;

}

/* EOF */
