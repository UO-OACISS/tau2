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
#define DIRNAMELENGTH		2048
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

/************************************************************
 * Description	: Send binary command to MUSE server and verify
 * From		: mdsh/mdsh.c <modified>
 ***********************************************************/
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

/////////////////////////////////////////////////////////////////////////////////////////
//=================================================================================
// TAU-MUSE API IMPLEMENTATION
// -TauMuseInit		:Connect to magnetd
// -TauMuseCreate	:Send command "create","addfilter","start" to magnetd
// -TauMuseQuery	:Send command "query"
// -TauMuseDestroy 	:Send command "stop","destroy","quit"
//================================================================================

/*************************************************************
 * Description	: Initialize socket connecting to MUSE sever
 * 		  - connect
 * 		  - send command create <handler_name> <args>
 * 		  - send command start <handlerID>
 * NOTE		: This function is called by TauMuseQuery	
 *************************************************************/
int TauMuseInit(void){
#ifdef AF_UNIX_MODE
	struct sockaddr unix_addr;
	char current_directory[DIRNAMELENGTH];
#else
        struct sockaddr_in host_addr;
#endif //AF_UNIX_MODE
	int sockfd ;
        char recv_buffer[BUFFERSIZE];
	
        // ===================================
        // Establish socket and connection
        // ===================================

#ifdef AF_UNIX_MODE
	// USING AF_UNIX 
	/* fill in the socket structure with host information */
	unix_addr.sa_family = AF_UNIX;
	strcpy(unix_addr.sa_data,"magnetd");

	// GO find the lock file, which is also the socket descriptor 
	getcwd(current_directory,DIRNAMELENGTH);
	if (chdir("/var/lock") < 0) {
	printf("unable to change to lockfile directory: %s", strerror(errno));
	return(1);
	}

	/* grab an Internet domain socket */
	if ((sockfd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
	printf("unable to create socket: %s", strerror(errno));
	return(1);
	}
	/* Assign sockfd to the global variable TheMuseSockId() */
	TheMuseSockId() = sockfd; 

#ifdef DEBUG_PROF
	/* connect to PORT on HOST */
	printf("%s: %d\n", unix_addr.sa_data, unix_addr.sa_family);
	printf("Connecting to magnetd using AF_UNIX...\n");
#endif //DEBUG_PROF

	if (connect(sockfd, &unix_addr,  /* Choose right socket type */
						sizeof(unix_addr)) == -1) {
	printf("unable to connect with AF_UNIX socket: %s", strerror(errno));
	return(1);
	}
	chdir(current_directory);
#else 
        // USING AF_INET	
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
#endif //AF_UNIX_MODE


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
	return(0);
}
/*************************************************************
 * Description	: Create handler according to the handler_name
 * 		  - send command create <handler_name> <args>
 * 		  - send command addfilter <handler_name> <filter_name><args>
 * 		  - send command start <handlerID>
 * NOTE		: This function is called by TauMuseQuery	
 *************************************************************/

int TauMuseCreate(void){
        char cmdstr[MAX_ARGLEN];
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        int command_length;
	int sockfd = TheMuseSockId();
        unsigned char *byteptr;
        int handlerID;
	int i,j,b,k;


	// Loop to create multiple handlers.
	for(j=0;j<TheMusePackage().numofhandlers;j++){
        
	// ====================================
        // command "create <handler_name>"
        // ====================================
        sprintf(cmdstr,"create %s",TheMusePackage().handlers[j].handler_name);
#ifdef DEBUG_PROF
        printf("cmdstr = %s\n",cmdstr);
#endif /* DEBUG_PROF */
	memset(send_buffer,0,BUFFERSIZE);
        command_length = create_encode_selector(TheMusePackage().handlers[j].handler_name,cmdstr,BUFFERSIZE,send_buffer);
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
        printf("!!!!!handlerID is %d\n",handlerID);
#endif /* DEBUG_PROF */

	//-------------------------------------------
	
	// =====================================
	// command "addfilter <handlerID> <filter_name> <args>"
	// =====================================
	// Loop to add multiple filters.	
	for(k=0;k<TheMusePackage().handlers[j].numoffilters;k++) {	
		// Loop to add multiple arguements for each filter.	
		for(i=0;i<TheMusePackage().handlers[j].filters[k].filter_argc;i++) {	
			sprintf(cmdstr,"addfilter %d %s", handlerID,TheMusePackage().handlers[j].filters[k].args[i]);
#ifdef DEBUG_PROF
			printf("cmdstr = %s\n",cmdstr);
#endif /* DEBUG_PROF */
			command_length = addfilter_encode_selector(
					strtok(TheMusePackage().handlers[j].filters[k].args[i]," ")
					,cmdstr,BUFFERSIZE,send_buffer);
#ifdef DEBUG_PROF
			//DEBUG JEREMY
			printf("cl: %u\n", command_length);
			for (b=0;b<command_length;b++)
				printf("%u ", (unsigned char)send_buffer[b]);
			printf("\n");
#endif //DEBUG_PROF
		
			send_and_check(sockfd,command_length,send_buffer,recv_buffer);
#ifdef DEBUG_PROF
			printf("!!!!!!!!!filter is added.\n");
#endif /* DEBUG_PROF */
		} // End for loop for each filter's arguement.
	} // End for loop for each filter.

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

        TheMusePackage().handlers[j].handlerID = handlerID;
	} // End for loop for each handler

	return 0;
}

/************************************************************
 * Description	: Destroy socket connecting to MUSE sever
 * 		  - connect
 * 		  - send command stop <handlerID>
 * 		  - send command destroy <handlerID>
 * 		  - send command quit 
 ***********************************************************/
void TauMuseDestroy(void){
	/*
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        unsigned char *byteptr;
	int handlerID = TheMuseHandlerId(); // read global
	int sockfd = TheMuseSockId(); // read 
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
#endif // DEBUG_PROF 
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
#endif // DEBUG_PROF 
        // ====================================
        // command "quit"
        // ====================================
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 10;
        send_and_check(sockfd,1+sizeof(int),send_buffer,recv_buffer);
	*/
}

#ifdef TAU_MUSE
/************************************************************
 * Description	: Query_handler from MUSE sever for
 * 		  monotonically incresing value.
 * 		  - send command query_handler
 ***********************************************************/
double TauMuseQuery(void){
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        char result_buffer[MAX_REPLY_LENGTH];
        unsigned char *byteptr;
	double result=0.0;
	double data_tmp[1];
	
	// This will get the value from environment variable
	// to initilize the appropriate handler and filter arguments.
	static int initialized=package_selector(NULL); 
	int sockfd = TheMuseSockId(); /* read from the global */
	int handlerID=TheMusePackage().handlers[0].handlerID;

#ifdef DEBUG_PROF
	printf("TauMuseQuery--- : pid=%d\n",getpid());
#endif //DEBUG_PROF
	
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
        printf("TauMuseQuery---: handlerID %d is queried\n",handlerID);
#endif /* DEBUG_PROF */
        result = (double)query_decode_selector(TheMusePackage().handlers[0].handler_name,
			send_buffer,recv_buffer, MAX_REPLY_LENGTH,result_buffer,data_tmp);
#ifdef DEBUG_PROF
	printf("TauMuseQuery---: TheMusePackage().handlers[0].handler_name=%s\n",
			TheMusePackage().handlers[0].handler_name);
        printf("TauMuseQuery---: result value passing to TAU: %f\n",result);
        printf("TauMuseQuery---: result buffer:\n%s\n",result_buffer);
#endif /* DEBUG_PROF */
       
	return result;

}

#endif //TAU_MUSE
#ifdef TAU_MUSE_EVENT

/************************************************************
 * Description	: Query_handler from MUSE sever for 
 * 		  non-monotonically increasing value.
 * 		  - send command query_handler
 ***********************************************************/
double TauMuseEventQuery(int data){
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        char result_buffer[MAX_REPLY_LENGTH];
        unsigned char *byteptr;
	double result;
	double data_tmp[1];
	
	// This will get the value from environment variable
	// to initilize the appropriate handler and filter arguments.
	static int initialized=package_selector(&data); 
	int sockfd = TheMuseSockId(); /* read from the global */
	int handlerID=TheMusePackage().handlers[0].handlerID;

#ifdef DEBUG_PROF
	printf("DEBUG:TauMuseEventQuery : pid=%d,data=%d\n",getpid(),data);
#endif //DEBUG_PROF
	
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
        result = (double)query_decode_selector(TheMusePackage().handlers[0].handler_name,
			send_buffer,recv_buffer, MAX_REPLY_LENGTH,result_buffer,data_tmp);
#ifdef DEBUG_PROF
	printf("TauMuseEventQuery---: TheMusePackage().handlers[0].handler_name=%s\n",
			TheMusePackage().handlers[0].handler_name);
        printf("TauMuseEventQuery---: result buffer:\n%s\n",result_buffer);
        printf("TauMuseEventQuery---: result value passing to TAU: %f\n",result);
#endif /* DEBUG_PROF */
        return result;

}
#endif //TAU_MUSE_EVENT

#ifdef TAU_MUSE_MULTIPLE
/************************************************************
 * Description	: Query_handler from MUSE sever for
 * 		  monotonically incresing value.
 * 		  - send command query_handler
 ***********************************************************/
int TauMuseMultipleQuery(double data[], int array_size){
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        char result_buffer[MAX_REPLY_LENGTH];
        unsigned char *byteptr;
	int i,j;
	int handlerID;
	double data_tmp[MAXNUMOF_COUNTERS];
	double *data_ptr=data;
	int data_counter=0;
	
	// This will get the value from environment variable
	// to initilize the appropriate handler and filter arguments.
	//static int handlerID = package_selector(NULL); 
	static int initialized=package_selector(NULL); 
	int sockfd = TheMuseSockId(); /* read from the global */

	if(array_size<=MAXNUMOF_COUNTERS){
	for(i=0;i<TheMusePackage().numofhandlers;i++){	
		handlerID=TheMusePackage().handlers[i].handlerID;

//#ifdef DEBUG_PROF
		printf("DEBUG:TauMuseMultipleQuery : pid=%d\n",getpid());
//#endif //DEBUG_PROF
	
		// ====================================
		// command "query_handler <handlerID>"
		// ====================================
		// create command in binary
		memset(send_buffer,0,BUFFERSIZE);
		send_buffer[0] = 3;
		byteptr = (unsigned char *)&handlerID;
		send_buffer[1] = (char)*byteptr;
		send_and_check(sockfd,2+sizeof(int),send_buffer,recv_buffer);
//#ifdef DEBUG_PROF
		//printf("!!!!!!!!!handlerID %d is queried\n",handlerID);
		printf("TauMuseMultipleQuery---: TheMusePackage().handlers[%d].handler_name=%s\n"
				,i,TheMusePackage().handlers[i].handler_name);
		//printf("TauMuseMultipleQuery---: result buffer:\n%s\n",result_buffer);
//#endif /* DEBUG_PROF */
		query_decode_selector(TheMusePackage().handlers[i].handler_name,
			send_buffer,recv_buffer, MAX_REPLY_LENGTH,result_buffer,data_tmp);

		//Copy data_tmp over to data
		for(j=0;j<TheMusePackage().handlers[i].numofcounters;j++,data_ptr++){
			// Check if the size of array provided is too small.
			if(data_counter<array_size){
				*data_ptr=data_tmp[j];
				data_counter++;
#ifdef DEBUG_PROF
				printf("handlerID=%d : data_ptr = %f\n",handlerID,*data_ptr);
				printf("data_counter=%d\n",data_counter);
#endif /* DEBUG_PROF */
			}else{
				printf("TauMuseMultipleQuery---: Number of data exceed array_size. array_size=%d, data_counter=%d\n",array_size,data_counter);
				return(-1);
			}
		}
	}
	}else{
		printf("TauMuseMultipleQuery---: array_size is too big\n");
		return(-1);
	}
	return data_counter;
}
#endif //TAU_MUSE_MULTIPLE
/* EOF */
