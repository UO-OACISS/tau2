#ifndef _TAU_MUSE_H_
#define _TAU_MUSE_H_

/* The TAU MAGNET/MUSE API */
#ifdef TAU_MUSE

#define MAX_ARGLEN 		255
#define MAX_REPLY_LENGTH 	1024 
#define NUMOFCPU		2 
#define MAX_FILTER_ARGS		5	

/***************************************************************************************
* This is for handlers stuff
* *************************************************************************************/
//=====================================
// Tau_count handler
//=====================================
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

int CreateTauCountEncode(char *ascii_command, int size, char *binary_command);
int QueryTauCountDecode(const char *binary_command, 
		const char *binary_reply, int size, char *ascii_reply);

//=====================================
// process_scheduling handler
//=====================================

struct process_scheduling_handler_input_data{
};

struct linked_list;

struct process_scheduling_handler_return_data{
  unsigned long long total_time;
  unsigned long long total_idle_time;
  unsigned long long total_busy_time;
  unsigned long long total_sched_time;
  unsigned long long total_error_time;
  unsigned long long current_state;
  unsigned int cpu_counter_in[NUMOFCPU];
  unsigned int cpu_counter_out[NUMOFCPU];
  unsigned long long cpu_busy_time[NUMOFCPU];

};

int CreateProcessSchedulingEncode(char *ascii_command, int size, char *binary_command);
int QueryProcessSchedulingDecode(const char *binary_command, 
		const char *binary_reply, int size, char *ascii_reply);

/***************************************************************************************
* This is for filters stuff
* *************************************************************************************/
struct filter_arguments{
	unsigned int pid;
	char filter_name[50];
	unsigned int event;
};

int AddFilterProcessFilterEncode(char *ascii_command, int size, char *binary_command);

/***************************************************************************************
* This is for TauMuse stuff
* *************************************************************************************/
int TauMuseInit(char *handler_name, int filter_argc,struct filter_arguments *args[]);
double TauMuseQuery(void);
void TauMuseDestroy(void);
#endif /* TAU_MUSE */

#endif /* _TAU_MUSE_H_ */
