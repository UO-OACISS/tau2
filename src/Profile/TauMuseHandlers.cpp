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

#include <endian.h>
#include <byteswap.h>


//#define DEBUG

#if __BYTE_ORDER == __BIG_ENDIAN
#define ntohll(x)	(x)
#else
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define ntohll(x)	bswap_64 (x)
#endif //__BYTE_ORDER = __LITTLE_ENDIAN
#endif //__BYTE_ORDER = __BIG_ENDIAN

#ifndef bswap64_t
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
#endif /* bswap64_t */  

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
int QueryTauCountDecode(const char *binary_command,
                const char *binary_reply, int size, char *ascii_reply){

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
        //return 1;
	return(ntohl(chrd->count));
}

/*********************
 * Description	: Encode binary code for command create 
 * 		  which wil be sent to Muse server
 * From		: translator.c <modified>
 * NOTE		: To be used with process_scheduling handler
 *********************/
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
				binary_command[1] = strlen(arg); /* copy handler string into buffer */
				strncpy(&binary_command[2], arg, binary_command[1]+1);
				/* get position of where size of struct input_data goes */
				args_size_ptr = (int *) &binary_command[binary_command[1]+2];
				arg = strtok(NULL, " ");
				/* start input_data struct at position in buffer */
				cuhid = (struct process_scheduling_handler_input_data *)
					&binary_command[binary_command[1] + 3 + sizeof(*args_size_ptr)];
				*args_size_ptr = htonl(sizeof(*cuhid));
				return (3 + binary_command[1] + sizeof(*args_size_ptr) +sizeof(cuhid));
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

/*********************
 * Description	: Decode binary code received from 
 * 		  MUSE server responding to command query_handler
 * From		: translator.c <modified>
 * NOTE		: To be used with process_scheduling handler
 *********************/
double QueryProcessSchedulingDecode(const char *binary_command,
                const char *binary_reply,
                        int size, char *ascii_reply){


	struct process_scheduling_handler_return_data *cuhrd;
	int *sizeptr;
	unsigned char *errorptr;
	double idleper, busyper, schedper, errorper;
	double midle, mbusy, msched, merror, mtotal;
	int i;
	unsigned cpu_counter_in[NUMOFCPU], cpu_counter_out[NUMOFCPU];
	double cpu_busy_time[NUMOFCPU];
	double cpu_busy_time_sec[NUMOFCPU];
	double cpu_speed;
	char *package;
        sizeptr = (int *)binary_reply;
        errorptr = (unsigned char *) (sizeptr+1);

        /* error code testing is done by translator.c, but it could be passed in here
           for specific error codes.
           */
        if(binary_command[0]==3)
        {
                cuhrd = (struct process_scheduling_handler_return_data *) (errorptr+1);

                mtotal = (double)ntohll(cuhrd->total_time);
                midle = (double)ntohll(cuhrd->total_idle_time);
                idleper = midle / mtotal;
                mbusy = (double)ntohll(cuhrd->total_busy_time);
                busyper = mbusy / mtotal;
                msched = (double)ntohll(cuhrd->total_sched_time);
                schedper = msched / mtotal;
                merror = (double)ntohll(cuhrd->total_error_time);
                errorper = merror / mtotal;
                cpu_speed = (double)ntohll(cuhrd->cpu_speed);
#ifdef DEBUG
		printf("cpu_speed=%10.10f\n",cpu_speed);
#endif //DEBUG
                for(i=0;i<NUMOFCPU;i++){
                        cpu_counter_in[i] = ntohs(cuhrd->cpu_counter_in[i]);
                        cpu_counter_out[i] = ntohs(cuhrd->cpu_counter_out[i]);
                        cpu_busy_time[i] = (double)ntohll(cuhrd->cpu_busy_time[i]);
                        cpu_busy_time_sec[i] = cpu_busy_time[i]/cpu_speed;
#ifdef DEBUG
                        printf("cpu_busy_time[%d]=%10.10f\n",i,cpu_busy_time[i]);
                        printf("cpu_busy_time_sec[%d]=%10.10f\n",i,cpu_busy_time_sec[i]);
#endif //DEBUG
                }
                snprintf(ascii_reply, size, "Idle(sec) : %15.f %2.3f\nBusy(sec) : %10.10f %2.3f\nSched(sec): %10.10f %2.3f\nError(sec): %10.10f %2.3f\nTotal(sec): %10.10f %2.3f\ncpu_counter_in[0]=%15.u\ncpu_counter_out[0]=%15.u\ncpu_counter_in[1]=%15.u\ncpu_counter_out[1]=%15.u\ncpu_busy_time[0](sec)=%10.10f\ncpu_busy_time[1](sec)=%10.10f\n\n",
                                                 midle/cpu_speed,100*idleper,
                                                 mbusy/cpu_speed,100*busyper,
                                                 msched/cpu_speed,100*schedper,
                                                 merror/cpu_speed,100*errorper,
                                                 mtotal/cpu_speed, 100*(idleper+busyper+schedper),
                                                 cpu_counter_in[0],cpu_counter_out[0],
                                                 cpu_counter_in[1],cpu_counter_out[1],
                                                 cpu_busy_time_sec[0],cpu_busy_time_sec[1]
                                                 );
        }
        else
        {
                printf("cpu_usage translator doesn't understand that command yet\n");
                return 0;
        }
	package=getenv("TAU_MUSE_PACKAGE");
	// This result varies according to the TAU_MUSE_PACKAGE
	if(!strcmp(package,"total_busy_time")){
		// Returning total busy time in millisec
		return (mbusy/cpu_speed)*1000;
	}else if(!strcmp(package,"total_time")){
		// Returning total time in millisec
		return mtotal/cpu_speed*1000;
	}
	return 0;


	
}
/* EOF */
