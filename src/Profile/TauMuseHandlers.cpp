/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.acl.lanl.gov/tau                        **
 * *****************************************************************************
 * **    Copyright 2003                                                       **
 * **    Department of Computer and Information Science, University of Oregon **
 * **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauMuseHandlers.cpp                            **
 * **      Description     : TAU MUSE/MAGNET Interface                      **
 * **      Author          : Suravee Suthikulpanit                          **
 * **      Contact         : Suravee@cs.uoregon.edu                         **
 * **      Flags           : Compile with                                   **
 * **                        -DTAU_MUSE                                     **
 * ****************************************************************************/

#include <Profile/Profiler.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>

//#define DEBUG

#include <endian.h>
#include <byteswap.h>

#if __BYTE_ORDER == __BIG_ENDIAN
#define ntohll(x)	(x)
#define htonll(x)	(x)
#define ntohd(x)	(x)
#define htond(x)	(x)
#else // __BIG_ENDIAN
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define ntohll(x)	bswap_64 (x)
#define htonll(x)	bswap_64 (x)
#define ntohd(x)	ntohd_func (x)
#define htond(x)	htond_func (x)
#endif //__BYTE_ORDER = __LITTLE_ENDIAN
#endif //__BYTE_ORDER = __BIG_ENDIAN

#ifndef bswap_64
#define swap_type(type,a,b) { type t=(a); (a)=(b); (b)=t; }

int64_t bswap_64( int64_t n )
{       
  int8_t* p = (int8_t*)&n;
  swap_type( int8_t, p[0], p[7] );
  swap_type( int8_t, p[1], p[6] );
  swap_type( int8_t, p[2], p[5] );
  swap_type( int8_t, p[3], p[4] );
  return n;             
}                       
#endif /* bswap_64 */  

double ntohd_func(double d)
{
        unsigned long long l;
        l = ntohll(*((long long *)&d));
        d = *((double *)&l);
        return d;
}

double htond_func(double d)
{
        long long l;
        l = htonll(*((long long *)&d));
        d = *((double *)&l);
        return d;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
/*********************
 * Description	: Encode binary code for command create 
 * 		  which wil be sent to Muse server
 * From		: translator.c <modified>
 * NOTE		: To be used with TAU_count handler
 *********************/
int CreateTauCountEncode(char *ascii_command, int size, char *binary_command)
{
        char *arg, temp[MAX_ARGLEN];
        struct count_handler_input_data *chid;
        int *args_size_ptr;

        strncpy(temp, ascii_command, MAX_ARGLEN);
        arg = strtok(temp, " ");
        if (arg) {

                if (strcasecmp(arg, "create")==0) {

                        binary_command[0] = 1;
                        arg = strtok(NULL, " ");
                        if (strcasecmp(arg, "TAU_count")!=0)
                        {
                                printf("Internal error - string count is not in argument from TAU.");
                                return 0;
                        }

                        else
                        {

                                binary_command[1] = strlen(arg);
                                strncpy(&binary_command[2], arg, binary_command[1]+1);
                                args_size_ptr = (int *) &binary_command[binary_command[1]+3];
                                arg = strtok(NULL, " ");
                                if (arg) {
                                        chid = (struct count_handler_input_data *)
						&binary_command[binary_command[1] + 
						3 + sizeof(*args_size_ptr)];
                                        chid->fname_size = htonl(strlen(arg));
                                        strncpy(&binary_command[binary_command[1]
                                                        +sizeof(*args_size_ptr)+
							sizeof(*chid)+3], arg, 
							ntohl(chid->fname_size)+1);
                                        arg = strtok(NULL, " ");
                                        if (arg)
                                                chid->fsize = htonl(atoi(arg));
                                        else
                                                chid->fsize = 0;
                                } else {
                                        chid = (struct count_handler_input_data *)
						&binary_command[binary_command[1] + 
						3 + sizeof(*args_size_ptr)];
                                        chid->fname_size = 0;
                                        chid->fsize = 0;
                                }
                                *args_size_ptr = htonl(sizeof(*chid) + ntohl(chid->fname_size));
                                return (3 + binary_command[1] + sizeof(*args_size_ptr) 
						+ ntohl(*args_size_ptr));
                        }
                }//CREATE
                else {
                        printf("Invalid command for count: %s\n"
                                  "count is a handler, and should be used only with the CREATE command\n"
                                  "\tUsage: CREATE count [<fname> [<fsize>]]", arg);
                        return 0;
                }
        }
        else {
                printf("Internal error: No command given??");
                return 0;
        }
}

/*********************
 * Description	: Decode binary code received from 
 * 		  MUSE server responding to command query_handler
 * From		: translator.c <modified>
 * NOTE		: To be used with TAU_count handler
 *********************/
double QueryTauCountDecode(const char *binary_command,
                const char *binary_reply, int size, char *ascii_reply,double data[]){

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
        if(binary_command[0]==3) {
                chrd = (struct count_handler_return_data *) (errorptr+1);
                snprintf(ascii_reply, size, "Count: %u\n", ntohl(chrd->count));
        } else {
                printf("TauMuse.cpp: count translator doesn't understand that command yet\n");
                return 0;
        }
	data[0]=(ntohl(chrd->count));
	return(data[0]);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/*********************
 * Description	: Encode binary code for command create 
 * 		  which wil be sent to Muse server
 * From		: translator.c <modified>
 * NOTE		: To be used with process_scheduling handler
 *********************/
#if 0
int CreateProcessSchedulingEncode(char *ascii_command, int size, char *binary_command)
{
	char *arg, temp[MAX_ARGLEN];
	struct process_scheduling_handler_input_data *cuhid;
	int *args_size_ptr;

	strncpy(temp, ascii_command, MAX_ARGLEN);
	arg = strtok(temp, " ");
	/* printf("ARG:%s\n",arg); */
	if (arg)
	{

		if (strcasecmp(arg, "create")==0)
		{

			binary_command[0] = 1;
			arg = strtok(NULL, " ");
			if (strcasecmp(arg, "process_scheduling")!=0)
			{
				printf("Internal error");
				return 0;
			}
			else
			{
				/* save length of handler string */
				binary_command[1] = strlen(arg); 
				/* copy handler string into buffer */
				strncpy(&binary_command[2], arg, binary_command[1]+1);
				/* get position of where size of struct input_data goes */
				args_size_ptr = (int *) &binary_command[binary_command[1]+3];
				/* start input_data struct at position in buffer */
				*args_size_ptr = htonl(0);
				return (3 + binary_command[1] + sizeof(*args_size_ptr));
			}
		}
		else
			{
				printf("Invalid command for cpu_usage: %s\n"
							"cpu_usage is a handler, and should be used only with the CREATE command\n"
							"\tUsage: CREATE cpu_usage", arg);
				return 0;
			}
	}
	else
		{
			printf("Internal error: No command given??");
			return 0;
		}
}
#endif
// A new one from JEREMY
int CreateProcessSchedulingEncode(char *ascii_command, int size, char *binary_command)
{
	//*********************************************************
	//ENCODING: 	byte 0  <1>
	//		byte 1  <18>
	//		byte 3+ <process_scheduling>
	//		byte 21 <null>
	//		byte 22 <options_size>
	//		byte 26 <option_flags>
	//		byte 30 <pid>
	//		byte 34 <null>
	//*********************************************************
        char *arg, *val, temp[MAX_ARGLEN];
        unsigned int event, id, pid, sid;
        int *intptr, *options_size;
        unsigned char *byteptr;
        char *option_flags;
        strncpy(temp, ascii_command, MAX_ARGLEN);
        arg = strtok(temp, " ");
        if (arg)
        {
                if (strcasecmp(arg, "create")==0)
                {
                        binary_command[0] = 1;
                        arg = strtok(NULL, " ");
                        if (arg)
                        {
                                if ((!arg)||(strncasecmp(arg, "process_scheduling",18)!=0))
                                {
                                        printf("Internal error arg=%s",arg);
                                        return 0;
                                }
                                else
                                {
					binary_command[1] = strlen(arg);
                                        strncpy(&binary_command[2], arg, binary_command[1]+1);
                                        arg = strtok(NULL, "=");
                                        val = strtok(NULL, " ");
                                        if (arg)
                                        {
                                                intptr=(int *) &binary_command[binary_command[1] + 3];
                                                options_size = intptr++;
                                                *options_size = sizeof(int);
                                                option_flags = (char *)intptr;
						printf("DEBUG: addr of intptr = %x\n",intptr);
                                                intptr++;
						printf("DEBUG: addr of intptr = %x\n",intptr);
                                                *option_flags = 0;
                                                while (arg && val)
                                                {
							if (strcasecmp(arg, "pid")==0) {
								*option_flags = *option_flags|0x20;
								pid = strtoul(val, NULL, 10);
							} else {
								printf("Unknown option: \"%s\"\n", arg);
								return 0;
							}
                                                        *options_size += sizeof(int);
                                                        arg = strtok(NULL, "=");
                                                        val = strtok(NULL, " ");
                                                }
                                                if (arg)
                                                {
                                                        printf("Could not parse option.  Ignoring option %s", arg);

                                                        if (*option_flags == 0)
                                                        {
                                                                printf("No options specified");
                                                                return 0;
                                                        }
                                                }

                                                *options_size = htonl(*options_size);
                                                if (*option_flags&0x20)
                                                        *intptr++ = htonl(pid);
                                                return (4+binary_command[1]+sizeof(int) + ntohl(*options_size));
                                        } else {
                                                intptr=(int *) &binary_command[binary_command[1] + 3];
                                                options_size = intptr++;
                                                *options_size = htonl(0);
                                                //*options_size = 2*sizeof(int);
                                                //option_flags = (char *)intptr;
                                                //intptr++;
                                                //*option_flags = 0;
                                                //*options_size = htonl(*options_size);
						//*intptr++ = htonl(pid);
                                                return (4+binary_command[2]+sizeof(int) + ntohl(*options_size));
		
                                        }
                                }
                        } else {
                                printf("No handler specified");
                        }
                } else {
                                printf("This is not create command.\n");
                }
        } else {
                printf("Internal error.  No command received in encode_command");
        }
        return 0;
}

/*********************
 * Description	: Decode binary code received from 
 * 		  MUSE server responding to command query_handler
 * From		: translator.c <modified>
 * NOTE		: To be used with process_scheduling handler
 *********************/
double QueryProcessSchedulingDecode(const char *binary_command,
                const char *binary_reply, int size, char *ascii_reply,double data[]){

	struct process_scheduling_handler_return_data *cuhrd;
	int *sizeptr;
	unsigned char *errorptr;
        double idleper, busyper, schedper, errorper;
        double midle, mbusy[MAXNUMOFCPU], msched[MAXNUMOFCPU], merror, mtotal;
        double cpu_busy_time_sec[MAXNUMOFCPU];
        double cpu_sched_time_sec[MAXNUMOFCPU];
        double total_time_sec;
        double cpu_speed;
        unsigned int numofcpu;
	double numofcontextswitch[MAXNUMOFCPU];
        int i;
	char *package;
        char ascii_reply_ext[1024];
        sizeptr = (int *)binary_reply;
        errorptr = (unsigned char *) (sizeptr+1);

        /* error code testing is done by translator.c, but it could be passed in here
           for specific error codes.
           */
        if(binary_command[0]==3)
        {
                cuhrd = (struct process_scheduling_handler_return_data *) (errorptr+1);

                cpu_speed = (double)ntohll(cuhrd->cpu_speed);

                mtotal = (double)ntohll(cuhrd->total_time);
                total_time_sec = (double)(mtotal/cpu_speed);
                numofcpu = ntohs(cuhrd->numofcpu);
#ifdef DEBUG
                printf("DEBUG: cpu_speed=%10.10f\n",cpu_speed);
                printf("DEBUG: numofcpu = %u\n",numofcpu);
                printf("DEBUG: total_time = %15.f\n",mtotal);
                printf("DEBUG: total_time_sec = %10.10f\n",total_time_sec);
#endif //DEBUG          

                snprintf(ascii_reply, size, "total_time(sec): %10.10f\n",
                                                 total_time_sec);
                for(i=0;i<numofcpu && i<MAXNUMOFCPU ;i++){
			numofcontextswitch[i]=ntohd(cuhrd->stat[i].numofcontextswitch);

                        mbusy[i] = (double)ntohll(cuhrd->stat[i].time_busy);
                        busyper = mbusy[i] / mtotal;
                        cpu_busy_time_sec[i] = (double)(mbusy[i]/cpu_speed);

                        msched[i] = (double)ntohll(cuhrd->stat[i].time_sched);
                        schedper = msched[i] / mtotal;
                        cpu_sched_time_sec[i] = (double)(msched[i]/cpu_speed);
#ifdef DEBUG
			printf("DEBUG: numofcontextswitch[%d] = %f\n",i,numofcontextswitch[i]);
                        printf("DEBUG: mbusy[%d] = %15.f\n",i,mbusy[i]);
                        printf("DEBUG: cpu_busy_time_sec = %10.10f\n",cpu_busy_time_sec[i]);
                        printf("DEBUG: msched[%d] = %15.f\n",i,msched[i]);
                        printf("DEBUG: cpu_sched_time_sec = %10.10f\n",cpu_sched_time_sec[i]);
#endif //DEBUG          

                /*      //for FUTURE    
                        merror = (double)ntohll(cuhrd->stat[i].error_time);
                        errorper = merror / mtotal;
                        
                        midle = (double)ntohll(cuhrd->stat[i].idle_time);
                        idleper = midle / mtotal;
                */
                        sprintf(ascii_reply_ext,"numofcontextswitch[%d] : %10.10f\ntime_busy[%d](sec) : %10.10f\ntime_sched[%d](sec): %10.10f\n",
                                                        i,numofcontextswitch[i],
                                                        i,cpu_busy_time_sec[i],
                                                        i,cpu_sched_time_sec[i]);
                        strncat(ascii_reply,ascii_reply_ext,size);
                }
                        strncat(ascii_reply,"\n",size);
#ifdef DEBUG
			printf("DEBUG: ascii_reply\n%s",ascii_reply);
#endif //DEBUG
        }
        else
        {
                printf("cpu_usage translator doesn't understand that command yet\n");
                return 0 ;
        }

#if (defined(TAU_MUSE)||defined(TAU_MUSE_EVENT))
	data[0]=0.0;
	package=getenv("TAU_MUSE_PACKAGE");
	// RETURN SINGLE DOUBLE VALUE
	// This result varies according to the TAU_MUSE_PACKAGE
	if(!strcmp(package,"busy_time")){
		// Returning busy time msec
		for(i=0;i<numofcpu;i++)
			data[0]+=(double)cpu_busy_time_sec[i];
		data[0]=data[0]*1000;
		return data[0];
#ifdef DEBUG
		printf("data[0](ms)=%10.10f\n",data[0]);
#endif //DEBUG
	
	}else if(!strcmp(package,"idle_time")){
		// Returning idle (but use busy pid=0 to measure) time in msec
		for(i=0;i<numofcpu;i++)
			data[0]+=cpu_busy_time_sec[i];
		data[0]=data[0]*1000;
		return data[0];
#ifdef DEBUG
		printf("data[0](ms)=%10.10f\n",data[0]*1000);
#endif //DEBUG
	
	}else if(!strcmp(package,"total_time")){
		// Returning total time in msec
		data[0]=total_time_sec*1000; 
		return data[0];
	
	}else if(!strcmp(package,"total_time_debug")){
		// Returning total time in msec
		data[0]=total_time_sec*1000; 
		return data[0];
	
	}else if(!strcmp(package,"context_switch")){
		// Returning total number of context switches for this process.
		for(i=0;i<numofcpu;i++)
			data[0]+=(double)numofcontextswitch[i];
		return data[0];
	}
	return 0;
#endif //(defined(TAU_MUSE)||defined(TAU_MUSE_EVENT))

#ifdef TAU_MUSE_MULTIPLE

	// RETURN ARRAY OF DOUBLE
	// Returning busy time msec
	data[0]=0;
	for(i=0;i<numofcpu;i++)
		data[0]+=(double)cpu_busy_time_sec[i];
#ifdef DEBUG
	printf("data[0](ms)=%10.10f\n",data[0]*1000);
#endif //DEBUG
	data[0]=data[0]*1000;
	
	// Returning total time in msec
	data[1]=total_time_sec*1000; 
	
	// Returning total number of context switches.
	data[2]=0;
	for(i=0;i<numofcpu;i++)
		data[2]+=(double)numofcontextswitch[i];

	return 0;	
#endif //TAU_MUSE_MULTIPLE
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
/*********************
 * Description	: Encode binary code for command create 
 * 		  which wil be sent to Muse server
 * From		: translator.c <modified>
 * NOTE		: To be used with bandwidth handler
 *********************/
int CreateBandwidthEncode(char *ascii_command, int size, char *binary_command)
{
        char *arg, temp[MAX_ARGLEN];
        struct bandwidth_handler_input_data *bhid;
        int *args_size_ptr;

        strncpy(temp, ascii_command, MAX_ARGLEN);

        arg = strtok(temp, " ");
        if (arg)
        {
                if (strcasecmp(arg, "create")==0)
                {
                        binary_command[0] = 1;
                        arg = strtok(NULL, " ");
                        if (strcasecmp(arg, "bandwidth")!=0)
                        {
                                printf("Internal error");
                                return 0;
                        }
                        else
                        {
                                binary_command[1] = strlen(arg);
                                strncpy(&binary_command[2], arg, binary_command[1]+1);
                                arg = strtok(NULL, " ");
                                if (arg)
                                {
                                        args_size_ptr = (int *) &binary_command[binary_command[1]+3];
                                        bhid = (struct bandwidth_handler_input_data *)
                                                        &binary_command[binary_command[1] + 3 + sizeof(*args_size_ptr)];
                                        bhid->delta_time = htond(atof(arg));
                                        arg = strtok(NULL, " ");
                                        if (arg)
                                        {
                                                bhid->fname_size = htonl(strlen(arg));
                                                strncpy(&binary_command[binary_command[1]
                                                                +3*sizeof(int)+sizeof(double)+3], arg, ntohl(bhid->fname_size)+1);
                                                arg = strtok(NULL, " ");
                                                if (arg)
                                                        bhid->fsize = htonl(atoi(arg));
                                                else
                                                        bhid->fsize = 0;
                                        }
                                        else
                                        {
                                                bhid->fname_size = 0;
                                                bhid->fsize = 0;
                                        }
                                        *args_size_ptr = htonl(sizeof(struct bandwidth_handler_input_data) +
                                                        ntohl(bhid->fname_size));
                                        return (3 + binary_command[1] + sizeof(*args_size_ptr) + ntohl(*args_size_ptr));
                                }
                                else
                                {
                                        printf("You must specify delta_time (type \"help bandwidth\" for usage)");
                                        return 0;
                                }
                        }
                }//CREATE
                else
                {
                        printf("Invalid command for bandwidth: %s\n"
                                  "bandwidth is a handler, and should be used only with the CREATE command\n"
                                  "\tUsage: CREATE bandwidth <delta_time> [<fname> [<fsize>]]", arg);
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
 * NOTE		: To be used with bandwidth handler
 *********************/
double QueryBandwidthDecode(const char *binary_command,
                const char *binary_reply, int size, char *ascii_reply,double data[])
{
        struct bandwidth_handler_return_data *bhrd;
        int *sizeptr;
        unsigned char *errorptr;

        sizeptr = (int *)binary_reply;
        errorptr = (unsigned char *)(sizeptr + 1);

        if(binary_command[0]==3)
        {
                bhrd = (struct bandwidth_handler_return_data *) (errorptr + 1);
                snprintf(ascii_reply, size, "Block #: %u\t\tStart time: %f\t\tLost: %u\n"
                                "Send: Average: %f\t\tRecent: %f\n"
                                "Recv: Average: %f\t\tRecent: %f\n",
                                ntohl(bhrd->block_id),
                                ntohd(bhrd->block_start_time),
                                ntohl(bhrd->lost_events),
                                ntohd(bhrd->send_average_bandwidth),
                                ntohd(bhrd->send_recent_bandwidth),
                                ntohd(bhrd->recv_average_bandwidth),
                                ntohd(bhrd->recv_recent_bandwidth));
#ifdef DEBUG
			printf("DEBUG: ascii_reply\n%s",ascii_reply);
#endif //DEBUG
        }
        else
        {
                printf("bandwidth translator doesn't understand that command yet\n");
                return 0;
        }
        data[0]=ntohd(bhrd->recv_average_bandwidth);
        data[1]=ntohd(bhrd->send_average_bandwidth);
        return data[0];

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
/*********************
 * Description	: Encode binary code for command create 
 * 		  which wil be sent to Muse server
 * From		: translator.c <modified>
 * NOTE		: To be used with tcpbandwidth handler
 *********************/
int CreateTcpBandwidthEncode(char *ascii_command, int size, char *binary_command)
{
        char *arg, temp[MAX_ARGLEN];
        struct tcpbandwidth_handler_input_data *bhid;
        int *args_size_ptr;

        strncpy(temp, ascii_command, MAX_ARGLEN);

        arg = strtok(temp, " ");
        if (arg)
        {
                if (strcasecmp(arg, "create")==0)
                {
                        binary_command[0] = 1;
                        arg = strtok(NULL, " ");
                        if (strcasecmp(arg, "tcpbandwidth")!=0)
                        {
                                printf("Internal error");
                                return 0;
                        }
                        else
                        {
                                binary_command[1] = strlen(arg);
                                strncpy(&binary_command[2], arg, binary_command[1]+1);
                                arg = strtok(NULL, " ");
                                if (arg)
                                {
                                        args_size_ptr = (int *) &binary_command[binary_command[1]+3];
                                        bhid = (struct tcpbandwidth_handler_input_data *)
                                                        &binary_command[binary_command[1] + 3 + sizeof(*args_size_ptr)];
                                        bhid->delta_time = htond(atof(arg));
                                        arg = strtok(NULL, " ");
                                        if (arg)
                                        {
                                                bhid->fname_size = htonl(strlen(arg));
                                                strncpy(&binary_command[binary_command[1]
                                                                +3*sizeof(int)+sizeof(double)+3], arg, ntohl(bhid->fname_size)+1);
                                                arg = strtok(NULL, " ");
                                                if (arg)
                                                        bhid->fsize = htonl(atoi(arg));
                                                else
                                                        bhid->fsize = 0;
                                        }
                                        else
                                        {
                                                bhid->fname_size = 0;
                                                bhid->fsize = 0;
                                        }
                                        *args_size_ptr = htonl(sizeof(struct tcpbandwidth_handler_input_data) +
                                                        ntohl(bhid->fname_size));
                                        return (3 + binary_command[1] + sizeof(*args_size_ptr) + ntohl(*args_size_ptr));
                                }
                                else
                                {
                                        printf("You must specify delta_time (type \"help tcpbandwidth\" for usage)");
                                        return 0;
                                }
                        }
                }//CREATE
                else
                {
                        printf("Invalid command for tcpbandwidth: %s\n"
                                  "tcpbandwidth is a handler, and should be used only with the CREATE command\n"
                                  "\tUsage: CREATE tcpbandwidth <delta_time> [<fname> [<fsize>]]", arg);
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
 * NOTE		: To be used with tcpbandwidth handler
 *********************/
double QueryTcpBandwidthDecode(const char *binary_command,
                const char *binary_reply, int size, char *ascii_reply,double data[])
{
        struct tcpbandwidth_handler_return_data *bhrd;
        int *sizeptr;
        unsigned char *errorptr;

        sizeptr = (int *)binary_reply;
        errorptr = (unsigned char *)(sizeptr + 1);

        if(binary_command[0]==3)
        {
                bhrd = (struct tcpbandwidth_handler_return_data *) (errorptr + 1);
                snprintf(ascii_reply, size, "Block #: %u\t\tStart time: %f\t\tLost: %u\n"
                                "Send: Average: %f\t\tRecent: %f\n"
                                "Recv: Average: %f\t\tRecent: %f\n",
                                ntohl(bhrd->block_id),
                                ntohd(bhrd->block_start_time),
                                ntohl(bhrd->lost_events),
                                ntohd(bhrd->send_average_bandwidth),
                                ntohd(bhrd->send_recent_bandwidth),
                                ntohd(bhrd->recv_average_bandwidth),
                                ntohd(bhrd->recv_recent_bandwidth));
        }
        else
        {
                printf("tcpbandwidth translator doesn't understand that command yet\n");
                return 0;
        }
        data[0]=ntohd(bhrd->recv_average_bandwidth);
        data[1]=ntohd(bhrd->send_average_bandwidth);
        return data[0];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
/*********************
 * Description	: Encode binary code for command create 
 * 		  which wil be sent to Muse server
 * From		: translator.c <modified>
 * NOTE		: To be used with accumulator handler
 *********************/
int CreateAccumulatorEncode(char *ascii_command, int size, char *binary_command)
{
        char *arg, temp[MAX_ARGLEN];
        struct accumulator_handler_input_data *ahid;
        int *args_size_ptr;

        strncpy(temp, ascii_command, MAX_ARGLEN);
        arg = strtok(temp, " ");
        if (arg)
        {

                if (strcasecmp(arg, "create")==0)
                {

                        binary_command[0] = 1;
                        arg = strtok(NULL, " ");
                        if (strcasecmp(arg, "accumulator")!=0)
                        {
                                printf("Internal error");
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
                                        // there are no args to accumulator now
                                }
                                else
                                {
                                        // there are no args to accumulator now
                                }
                                *args_size_ptr = htonl(sizeof(*ahid));
                                return (3 + binary_command[1] + sizeof(*args_size_ptr) + ntohl(*args_size_ptr));
         
                        }
                }//CREATE
                else
                {
                        printf("Invalid command for accumulator: %s\n"
                                  "accumulator is a handler, and should be used only with the CREATE command\n"
                                  "\tUsage: CREATE accumulator", arg);
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
 * NOTE		: To be used with accumulator handler
 *********************/
double QueryAccumulatorDecode(const char *binary_command,
                const char *binary_reply, int size, char *ascii_reply,double data[])
{
        struct accumulator_handler_return_data *ahrd;
        int *sizeptr;
        unsigned char *errorptr;
	unsigned long long result;

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
                ahrd = (struct accumulator_handler_return_data *) (errorptr+1);
                snprintf(ascii_reply, size, "Accumulated: %llu\n",(unsigned long long) ntohll(ahrd->sum));
		
		result=(unsigned long long)ntohll(ahrd->sum);
		data[0]=result;
        }
        else
        {
                printf("accumulator translator doesn't understand that command yet\n");
                return 0;
        }
        return result;

}
/* EOF */
