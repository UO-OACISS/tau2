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
#include <poll.h>

#define HOST_PORT 9997			// MUSE server port
#define HOST_IP "127.0.0.1"		
#define BUFFERSIZE 8192
#define MAX_ARGLEN 255
#define MAX_REPLY_LENGTH 1024 

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
QUERY_MAPPER:   <command=2> <ASCII string> <size> <options>
QUERY_HANDLER:  <command=3> <handler id> <size=0>
DESTROY:                <command=4> <handler id> <size=0>
START:                  <command=5> <handler id> <size=0>
STOP:                   <command=6> <handler id> <size=0>
RESETFILTERS:   <command=7> <handler id> <size=0>
ADDFILTER:              <command=8> <handler id> <ASCII string> <size> <options>
GET:                    <command=9> <size> <options (1st byte matters)>
QUIT:                   <command=10> <size=0>
*****************************************/
/*********************
 * Description	: Struct for input data for count.
 * From		: handler/count/count.h
 *********************/
struct count_handler_input_data
{
        int fsize;
        int fname_size;
};

/*********************
 * Description	: Struct for return data for count.
 * From		: handler/count/count.h
 *********************/
struct count_handler_return_data
{
        int count;
};

/*********************
 * Description	: Encode binary code for command create 
 * 		  which wil be sent to Muse server
 * From		: translator.c <modified>
 *********************/
int encode_create_command_count(char *ascii_command, int size, char *binary_command)
{
        char *arg, temp[MAX_ARGLEN];
        struct count_handler_input_data *chid;
        int *args_size_ptr;

        strncpy(temp, ascii_command, MAX_ARGLEN);
        arg = strtok(temp, " ");
        if (arg)
        {

                if (strcasecmp(arg, "create")==0)
                {

                        binary_command[0] = 1;
                        arg = strtok(NULL, " ");
                        if (strcasecmp(arg, "count")!=0)
                        {
                                printf("TauMuse.cpp: Internal error - string count is not in argument from TAU.");
                                return 0;
                        }
                        else
                        {

                                binary_command[1] = strlen(arg);
                                strncpy(&binary_command[2], arg, binary_command[1]+1);
                                args_size_ptr = (int *) &binary_command[binary_command[1]+3];
                                arg = strtok(NULL, " ");
                                if (arg)
                                {
                                        chid = (struct count_handler_input_data *)
                                                        &binary_command[binary_command[1] + 3 + sizeof(*args_size_ptr)];
                                        chid->fname_size = htonl(strlen(arg));
                                        strncpy(&binary_command[binary_command[1]
                                                        +sizeof(*args_size_ptr)+sizeof(*chid)+3], arg, ntohl(chid->fname_size)+1);
                                        arg = strtok(NULL, " ");
                                        if (arg)
                                                chid->fsize = htonl(atoi(arg));
                                        else
                                                chid->fsize = 0;
                                }
                                else
                                {
                                        chid = (struct count_handler_input_data *)
                                                        &binary_command[binary_command[1] + 3 + sizeof(*args_size_ptr)];
                                        chid->fname_size = 0;
                                        chid->fsize = 0;
                                }
                                *args_size_ptr = htonl(sizeof(*chid) + ntohl(chid->fname_size));
                                return (3 + binary_command[1] + sizeof(*args_size_ptr) + ntohl(*args_size_ptr));

                        }
                }//CREATE
                else
                {
                        printf("Invalid command for count: %s\n"
                                  "count is a handler, and should be used only with the CREATE command\n"
                                  "\tUsage: CREATE count [<fname> [<fsize>]]", arg);
                        return 0;
                }
        }
        else
        {
                printf("Internal error: No command given??");
                return 0;
        }
}

/*********************
 * Description	: Decode binary code received from 
 * 		  MUSE server responding to command query_handler
 * From		: translator.c <modified>
 *********************/
int decode_query_handler_count(const char *binary_command,
                const char *binary_reply,
                        int size, char *ascii_reply){

        struct count_handler_return_data *chrd;
        int *sizeptr;
        unsigned char *errorptr;
        /*
        int i;
        for (i =0; i < 8; i++)
                printf("%d ", binary_reply[i]);
        printf("\n");
        */
        sizeptr = (int *)binary_reply;
        errorptr = (unsigned char *) (sizeptr+1);

        /* error code testing is done by translator.c, but it could be passed in here
           for specific error codes.
           */
        if(binary_command[0]==3)
        {
                chrd = (struct count_handler_return_data *) (errorptr+1);
                snprintf(ascii_reply, size, "Count: %u\n", ntohl(chrd->count));
        }
        else
        {
                printf("TauMuse.cpp: count translator doesn't understand that command yet\n");
                return 0;
        }
        //return 1;
	return(ntohl(chrd->count));
}

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
                while (b < sizeof(int))
                {
                        if ((b += recv(sockfd, &recv_buffer[b], BUFFERSIZE-b, 0)) == -1) {
                                printf("TauMuse.cpp: recv() failed: %s", strerror(errno));
                                kill(0,SIGTERM);
                        }
                }

                b -= sizeof(int);

                while (b < ntohl(((int *)recv_buffer)[0]))
                {
                        if ((b += recv(sockfd, &recv_buffer[b+sizeof(int)],
                                  BUFFERSIZE-b-sizeof(int), 0)) == -1) {
                                printf("TauMuse.cpp: recv() failed: %s", strerror(errno));
                                kill(0,SIGTERM);
                        }
                }
        }
        return(0);

}

/*********************
 * Description	: Initialize socket connecting to MUSE sever
 * 		  - connect
 * 		  - send command create <handler_hane> <args>
 * 		  - send command start <handlerID>
 *********************/
int TauMuseInit(char* handler_name, char* args,int *sockfd){
  /* creates a socket, connects to the server, creates a MUSE session, adds 
     a filter and starts the session */
        struct sockaddr_in host_addr;
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        char cmdstr[MAX_ARGLEN];
        int command_length;
        unsigned int size_reply;
        int *sizeptr;
        int handlerID;
        unsigned char *byteptr;

        // -----------------------------------
        // Establish socket and connection
        // -----------------------------------
        // host information
        memset(&host_addr,0,sizeof(host_addr));
        host_addr.sin_family = AF_INET;
        host_addr.sin_addr.s_addr = inet_addr(HOST_IP);
        host_addr.sin_port = htons(HOST_PORT);

        // create socket        
        if((*sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
                perror("socket");
                printf("TauMuse.cpp: Unable to create socket: %s\n",
                                strerror(errno));
                return(1);
        }

        // connect to 
#ifdef DEBUG_PROF
        printf("TauMuse.cpp: Connecting to magnetd using AF_INET.\n");
#endif /* DEBUG_PROF */
        if(connect(*sockfd,(struct sockaddr *) &host_addr,
                        sizeof(host_addr)) == -1){
                perror("connect");
                printf("TauMuse.cpp: Unable to connect to server: %s\n",
                                strerror(errno));
                return(1);
        }
        // verify connection
        if(recv(*sockfd, recv_buffer, BUFFERSIZE,0) == -1){
                perror("recv");
                printf("TauMuse.cpp: Unable to establish connection: %s\n",
                                strerror(errno));
                return(1);
        }
        if(recv_buffer[0] == 1){
                printf("TauMuse.cpp: Connection Refused from server.\n");
                close(*sockfd);
                return(1);
        }else if(recv_buffer[0] == 0){
#ifdef DEBUG_PROF
                printf("------------Connection Established-----------\n");
#endif /* DEBUG_PROF */
        }else{
                printf("TauMuse.cpp: Unknown handshake reply.\n");
                close(*sockfd);
                return(1);
        }

        // ------------------------------------
        // command "create <handler_name> <args>"
        // ------------------------------------
        // create command in binary
        sprintf(cmdstr,"create %s %s",handler_name,args);
#ifdef DEBUG_PROF
        printf("cmdstr = %s\n",cmdstr);
#endif /* DEBUG_PROF */
        // Encoder is specific for each handler
        command_length = encode_create_command_count(cmdstr,BUFFERSIZE,send_buffer);

        send_and_check(*sockfd,command_length,send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!handler is created\n");
#endif /* DEBUG_PROF */

        // ------------------------------------
        // command "start <handlerID>"
        // ------------------------------------
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);

        // Command for start
        send_buffer[0] = 5;

        // check size of reply message  
        sizeptr = (int *)recv_buffer;
        size_reply = ntohl(*sizeptr);
#ifdef DEBUG_PROF
        printf("size_reply = %d\n",size_reply);
#endif /* DEBUG_PROF */

        // Check HandlerID ... 
        // HACKY!!!... Endian stuff
        byteptr = (unsigned char *)recv_buffer+5;
        handlerID = (int) *byteptr;
#ifdef DEBUG_PROF
        printf("handlerID is %d\n",handlerID);
#endif /* DEBUG_PROF */
        send_buffer[1] = recv_buffer[5];
        send_and_check(*sockfd,2+sizeof(int),send_buffer,recv_buffer);
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
void TauMuseDestroy(int handlerID, int sockfd){
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        unsigned char *byteptr;
        // ------------------------------------
        // command "stop <handlerID>"
        // ------------------------------------
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 6;
        byteptr = (unsigned char *)&handlerID;
        send_buffer[1] = (char)*byteptr;
        send_and_check(sockfd,2+sizeof(int),send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!handlerID %d is stopped\n",handlerID);
#endif /* DEBUG_PROF */
        // ------------------------------------
        // command "destroy <handlerID>"
        // ------------------------------------
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 4;
        send_buffer[1] = (char)*byteptr;
        send_and_check(sockfd,2+sizeof(int),send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!handlerID %d is destroyed\n",handlerID);
#endif /* DEBUG_PROF */
        // ------------------------------------
        // command "quit"
        // ------------------------------------
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 10;
        send_and_check(sockfd,1+sizeof(int),send_buffer,recv_buffer);

}

/* Get the name of the MUSE handler */
char * get_muse_handler_name(void)
{
	char *handler = getenv("TAU_MUSE_HANDLER");
	if (handler == (char *) NULL)
	{  /* the user has not specified any handler name */
	  return "count";
	}
	else
	  return handler;
}

/*********************
 * Description	: Query_handler from MUSE sever
 * 		  - send command query_handler
 * NOTE: Return double for now.
 *********************/
double TauMuseQuery(void){
        char send_buffer[BUFFERSIZE];
        char recv_buffer[BUFFERSIZE];
        char result_buffer[MAX_REPLY_LENGTH];
        unsigned char *byteptr;
	double result;
	//*********************************		
	// HARDCODED for now for the sake of simplicity 
	// incase TauMuseInit was not called earlier.
	static int global_sockfd; 
	static char *handler_name=get_muse_handler_name();
	static int global_handlerID = TauMuseInit(handler_name,"",&global_sockfd); 
	
	//*********************************		
        // ------------------------------------
        // command "query_handler <handlerID>"
        // ------------------------------------
        // create command in binary
        memset(send_buffer,0,BUFFERSIZE);
        send_buffer[0] = 3;
        byteptr = (unsigned char *)&global_handlerID;
        send_buffer[1] = (char)*byteptr;
        send_and_check(global_sockfd,2+sizeof(int),send_buffer,recv_buffer);
#ifdef DEBUG_PROF
        printf("!!!!!!!!!global_handlerID %d is queried\n",global_handlerID);
#endif /* DEBUG_PROF */
        result = (double)decode_query_handler_count(send_buffer,recv_buffer,
                        MAX_REPLY_LENGTH,result_buffer);
#ifdef DEBUG_PROF
        printf("result: %s\n",result_buffer);
        printf("result value to pass to TAU: %f\n",result);
#endif /* DEBUG_PROF */
	//TauMuseDestroy(global_handlerID,global_sockfd);
        return result;

}

