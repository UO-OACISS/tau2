/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauKtau.cpp					  **
**	Description 	: TAU Kernel Profiling 				  **
**	Author		: Surave Suthikulpanit				  **
**			: Aroon Nataraj					  **
**	Contact		: suravee@cs.uoregon.edu		 	  **
**			: anataraj@cs.uoregon.edu		 	  **
**	Flags		: Compile with				          **
**			  -DTAUKTAU -DTAUKTAU_MERGE                       **
**	Documentation	:                                                 **
***************************************************************************/

#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <errno.h>

#include <linux/unistd.h>

#include "Profile/Profiler.h"
#include "Profile/TauKtau.h"
#include <ktau_proc_interface.h>
#include "Profile/KtauProfiler.h"

using namespace std;

#define LINE_SIZE		1024	
#define OUTPUT_NAME_SIZE	100

#define DEBUG			0
#define maxof(x,y)		((x>=y)?x:y)

/* For sys_gettid syscall */
//_syscall0(pid_t,gettid);
#include <asm/unistd.h>

extern "C" int ktau_diff_profiles(ktau_output* plist1, int size1, ktau_output* plist2, int size2, ktau_output** outlist);

/*-------------------------- CON/DESTRUCTOR ---------------------------*/
/* 
 * Function 		: TauKtau::TauKtau
 * Description		: Constructor
 */
TauKtau::TauKtau(KtauSymbols& sym):KtauSym(sym)
{
        startBuf = NULL;
        stopBuf  = NULL;
	diffOutput = NULL;

        startSize= 0;
        stopSize = 0;
	outSize = 0;
	ThisKtauOutputInfo.pid = syscall(__NR_gettid);
	ThisKtauOutputInfo.templ_fun_counter = 0;
	ThisKtauOutputInfo.user_ev_counter = 0;
	
        startBufTWO = NULL;
        stopBufTWO  = NULL;
	diffOutputTWO = NULL;

        startSizeTWO = 0;
        stopSizeTWO = 0;
	outSizeTWO = 0;
}

/* 
 * Function 		: TauKtau::~TauKtau
 * Description		: Destructor
 */
TauKtau::~TauKtau()
{
	free(startBuf);
	free(stopBuf);

	if(diffOutput) {
		for(int i=0 ; i<outSize ; i++) 
			free((diffOutput+i)->ent_lst);
		free(diffOutput);
	}

        startSize= 0;
        stopSize = 0;
	outSize = 0;
	startBuf = NULL;
	stopBuf = NULL;
	diffOutput = NULL;

        startSizeTWO = 0;
        stopSizeTWO = 0;
	outSizeTWO = 0;
	startBufTWO = NULL;
	stopBufTWO = NULL;
	diffOutputTWO = NULL;
}

/*----------------------------- PUBLIC --------------------------------*/

int TauKtau::RegisterFork(TauKtau* pKernProf, enum TauFork_t opcode)
{
	int i = 0;

	if(DEBUG)printf("TauKtau::RegisterFork: Enter\n");

	if(DEBUG)printf("TauKtau::RegisterFork: Calling Dest ~TauKtau\n");
	//Call the destructor to cleanup
	pKernProf->~TauKtau();

	if(DEBUG)printf("TauKtau::RegisterFork: Redoing Cons Steps TauKtau\n");
	//Redo stuff done in the constructor
        pKernProf->startBuf = NULL;
        pKernProf->stopBuf  = NULL;
        pKernProf->diffOutput = NULL;
        pKernProf->startSize= 0;
        pKernProf->stopSize = 0;
        pKernProf->outSize = 0;
        pKernProf->ThisKtauOutputInfo.pid = syscall(__NR_gettid);
        pKernProf->ThisKtauOutputInfo.templ_fun_counter = 0;
        pKernProf->ThisKtauOutputInfo.user_ev_counter = 0;

	if(DEBUG)printf("TauKtau::RegisterFork: Calling StartKProfile\n");
	//Do a StartKProfile
	i = pKernProf->StartKProfile();

	if(DEBUG)printf("TauKtau::RegisterFork: Exit\n");

	return i;
}


/* 
 * Function 		: TauKtau::StartKprofile
 * Description		: Read the start profile
 */
int TauKtau::StartKProfile()
{
	//int selfpid = -1;
	int selfpid = 0; //ALL
	//startSize = read_size(KTAU_PROFILE, 1, &selfpid, 1, 0, NULL, -1);
	//startSize = read_size(KTAU_PROFILE, 1, &selfpid, 1, 0, NULL, 2);
	startSize = read_size(KTAU_TYPE_PROFILE, 0 /*ALL*/, &selfpid, 0/*ALL*/, 0, NULL, -1);
	if(startSize <= 0) {
		return -1;
	}
	startBuf = (char*)malloc(startSize*sizeof(ktau_output));
	if(!startBuf) {
		perror("TauKtau::StartKProfile: malloc");
		return -1;
	}

	startSizeTWO = startSize*sizeof(ktau_output);
	startBufTWO = (char*) malloc(startSizeTWO);

	//startSize = read_data(KTAU_PROFILE, 1, &selfpid, 1, startBuf, startSize, 0, NULL);
	startSize = read_data(KTAU_TYPE_PROFILE, 0 /*ALL*/, &selfpid, 0/*ALL*/, startBuf, startSize, 0, NULL);
	if(startSize <= 0) {
		free(startBuf);
		startBuf = NULL;
		return -1;
	}

	memcpy(startBufTWO, startBuf, startSizeTWO);

	return startSize;
}

/* 
 * Function 		: TauKtau::StartKprofile
 * Description		: Read the stop, diff, and dump profile
 */
int TauKtau::StopKProfile()
{
	int i=0;
	int ret = 0;
	//int selfpid = -1;
	int selfpid = 0; //ALL

	//stopSize = read_size(KTAU_PROFILE, 1, &selfpid, 1, 0, NULL, 2);
	stopSize = read_size(KTAU_TYPE_PROFILE, 0/*ALL*/, &selfpid, 0/*ALL*/, 0, NULL, -1);
	if(stopSize <= 0) {
		return -1;
	}
	stopBuf = (char*)malloc(stopSize*sizeof(ktau_output));
	if(!stopBuf) {
		perror("TauKtau::StopKProfile: malloc");
		return -1;
	}

	int stopSizeTWO = stopSize*sizeof(ktau_output);
	stopBufTWO = (char*) malloc(stopSizeTWO);

	//stopSize = read_data(KTAU_PROFILE, 1, &selfpid, 1, stopBuf, stopSize, 0, NULL);
	stopSize = read_data(KTAU_TYPE_PROFILE, 0/*ALL*/, &selfpid, 0/*ALL*/, stopBuf, stopSize, 0, NULL);
	if(stopSize <= 0) {
		free(stopBuf);
		stopBuf = NULL;
		return -1;
	}

	memcpy(stopBufTWO, stopBuf, stopSizeTWO);

	if((outSize = DiffKProfile()) < 0){
		return(-1);	
	}


	if((outSizeTWO = DiffKProfileTWO(startBufTWO, stopBufTWO, startSizeTWO, stopSizeTWO, &diffOutputTWO)) < 0){
		return(-1);	
	}


	return(ret);
}

int TauKtau::DumpKProfileOut() {

	int ret = 0;

	if((ret = DumpKProfileTWO(outSizeTWO, diffOutputTWO, "perprocess")) < 0){
		return(-1);	
	};

	/* AN: TODO: Disabling till a leader election method is settled on
	ktau_output* aggr_profs = NULL;
	AggrKProfiles(startBufTWO, startSizeTWO, stopBufTWO, stopSizeTWO, &aggr_profs);
	if(DumpKProfileTWO(1, aggr_profs, "kernelwide") < 0){
		return(-1);	
	}
	*/
	return 0;
}

int TauKtau::AggrKProfiles(char* start, int startSz, char* stop, int stopSz, ktau_output** aggrprofs)
{
	int i,j,k,diff_count;
	ktau_output *startOutput;
	ktau_output *stopOutput;
	long startProcNum =  0;
	long stopProcNum  = 0;

	startProcNum = unpack_bindata(KTAU_TYPE_PROFILE, start, startSz, &startOutput);
	if(startProcNum < 0) {
		return -1;
	}
       ktau_output* paggr_start_prof = (ktau_output*)calloc(sizeof(ktau_output),1);
	if(!paggr_start_prof) {
		printf("calloc ret null.\n");
		goto free_out;
	}
	paggr_start_prof->ent_lst = (o_ent*)calloc(sizeof(o_ent)*2048,1);
	if(!paggr_start_prof->ent_lst) {
		printf("calloc ret null.\n");
		free(paggr_start_prof);
		goto free_out;
	}
	aggr_many_profiles(startOutput, startProcNum, 2048, paggr_start_prof);
	paggr_start_prof->pid = 0;

	stopProcNum  = unpack_bindata(KTAU_TYPE_PROFILE, stop, stopSz, &stopOutput);
	if(stopProcNum < 0) {
		return -1;
	}

	{ //extra brace to make gcc 4.1 happy
	ktau_output* paggr_prof = (ktau_output*)calloc(sizeof(ktau_output),1);
	if(!paggr_prof) {
		printf("calloc ret null.\n");
		goto free_out;
	}
	paggr_prof->ent_lst = (o_ent*)calloc(sizeof(o_ent)*2048,1);
	if(!paggr_prof->ent_lst) {
		printf("calloc ret null.\n");
		free(paggr_prof);
		goto free_out;
	}
	aggr_many_profiles(stopOutput, stopProcNum, 2048, paggr_prof);
	paggr_prof->pid = 0;

	*aggrprofs = (ktau_output*)calloc(sizeof(ktau_output),1);
	(*aggrprofs)->ent_lst = (o_ent*)calloc(sizeof(o_ent)*2048,1);

	return ktau_diff_profiles(paggr_start_prof, 1, paggr_prof, 1, aggrprofs);
	}//to make gcc 4.1 happy! 
free_out:
	return -1;

}

int TauKtau::DumpKProfileTWO(int outSize, ktau_output* diffOutput, char* tag)
{
	int i=0,j=0;
	char output_path[OUTPUT_NAME_SIZE];
	o_ent *ptr;
	unsigned int cur_index = 0;
	unsigned int user_ev_counter = 0;
	unsigned int templ_fun_counter = 0;

	if((outSizeTWO <= 0)) { // || (stopSizeTWO <= 0) || (startSizeTWO <= 0)) {
		return -1;
	}

	/*
	ktau_output* tmp_diffOutput = diffOutput;
	diffOutput = diffOutputTWO;
       	*/

	/*
	int tmp_outSize = outSize;
	outSize = outSizeTWO;
	int tmp_startSize = startSize;
	startSize = startSizeTWO;
	int tmp_stopSize = stopSize;
	stopSize = stopSizeTWO;
	*/

	/* 
	 * Create output directory ./Kprofile 
	 */
	snprintf(output_path, sizeof(output_path), "./Kprofile.%d.%d.%s", RtsLayer::myNode(), RtsLayer::myThread(), tag);
	if(mkdir(output_path,777) == -1){
		perror("TauKtau::DumpKProfile: mkdir");
		if(errno != EEXIST) { //ignore already-exists errors
			return(-1);
		}
	}
	if(chmod(output_path, S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH ) != 0) {
		perror("TauKtau::DumpKProfile: chmod");
		return(-1);
	}

	/*
	 * Dumping output to the output file "./Kprofile.<rank>/profile.<pid>.0.0"
	 */
	// Data format :
        // %d templated_functions
        // "%s %s" %ld %G %G  
        //  funcname type numcalls Excl Incl
        // %d aggregates
        // <aggregate info>
	
	/* For Each Process */
	for(i=0;i<outSize;i++){
		user_ev_counter = 0;
		templ_fun_counter = 0;
		
		/* Counting Profile */
		for(j=0;j < (diffOutput+i)->size; j++){
			ptr = (((diffOutput)+i)->ent_lst)+j;
			if(ptr->index < 300 || ptr->index > 399){
				templ_fun_counter++;			
			}else{
				user_ev_counter++;
			}
		}
		
		int context = 0;

		if((diffOutput+i)->pid == ThisKtauOutputInfo.pid) {
			context = 1;
		}

		snprintf(output_path, sizeof(output_path), "./Kprofile.%d.%d.%s/profile.%u.0.%d",RtsLayer::myNode(),RtsLayer::myThread(), tag, (diffOutput+i)->pid, context);
		ofstream fs_output (output_path , ios::out);
		if(!fs_output.is_open()){
			cout << "Error opening file: " << output_path << "\n";
			return(-1);
		}
		
		/* OUTPUT: Templated function */
		fs_output << templ_fun_counter << " templated_functions" << endl;
		fs_output << "# Name Calls Subrs Excl Incl ProfileCalls" << endl;

		for(j=0;j < (diffOutput+i)->size; j++){
			ptr = (((diffOutput)+i)->ent_lst)+j;
			string& func_name = KtauSym.MapSym(ptr->entry.addr);
			if(ptr->index < 300 || ptr->index > 399){
				fs_output << "\"" << 
					  func_name << "()\" " 	//Name
					  << ptr->entry.data.timer.count << " "		//Calls
					  << 0 << " "					//Subrs
					  << (double)ptr->entry.data.timer.excl/KTauGetMHz() //Excl
					  //<< (double)ptr->entry.data.timer.excl //Excl
					  << " "		
					  << (double)ptr->entry.data.timer.incl/KTauGetMHz() //Incl
					  //<< (double)ptr->entry.data.timer.incl //Incl
					  << " "	
					  << 0 << " ";					//ProfileCalls
				if(strstr(func_name.c_str(), "schedule")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SCHEDULE\"" << endl;
				//}else if(!strcmp("__run_timers",func_name.c_str())){
				//	fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_RUN_TIMERS\"" << endl;
				}else if(strstr(func_name.c_str(), "page_fault")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_EXCEPTION\"" << endl;
				}else if(strstr(func_name.c_str(), "IRQ")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_IRQ\"" << endl;
				}else if(strstr(func_name.c_str(), "run_timers")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"" << endl;
				}else if(strstr(func_name.c_str(), "workqueue")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"" << endl;
				}else if(strstr(func_name.c_str(), "tasklet")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"" << endl;
				//}else if(!strcmp("__do_softirq",func_name.c_str())){
				//	fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_DO_SOFTIRQ\"" << endl;
				}else if(strstr(func_name.c_str(), "softirq")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"" << endl;
				}else if(strstr(func_name.c_str(), "sys_")){
					if(strstr(func_name.c_str(), "sock")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_SOCK\"" << endl;
					} else if(strstr(func_name.c_str(), "read")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"" << endl;
					} else if(strstr(func_name.c_str(), "write")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"" << endl;
					} else if(strstr(func_name.c_str(), "send")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"" << endl;
					} else if(strstr(func_name.c_str(), "recv")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"" << endl;
					} else {
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL\"" << endl;
					}
				}else if(strstr(func_name.c_str(), "tcp")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_TCP\"" << endl;
				}else if(strstr(func_name.c_str(), "icmp")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_ICMP\"" << endl;
				}else if(strstr(func_name.c_str(), "sock")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SOCK\"" << endl;
				}else{
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_DEFAULT\"" << endl;
				}
			}
		}
		/* OUTPUT: Aggregates*/	
		fs_output << 0 << " aggregates" << endl;
		
		/* OUTPUT: User-events*/
		fs_output << user_ev_counter  << " userevents" << endl;
		fs_output << "# eventname numevents max min mean sumsqr"<< endl;

		for(j=0;j < (diffOutput+i)->size; j++){
			ptr = (((diffOutput)+i)->ent_lst)+j;
			string& ev_name = KtauSym.MapSym(ptr->entry.addr);
			if(ptr->index >= 300 && ptr->index <= 399 ){
				if(ptr->entry.data.timer.count != 0){
					fs_output << "\"Event_"
					  << ev_name << "()\" " 	//eventname
					  << ptr->entry.data.timer.count << " "		//numevents
					  << 1 << " "					//max
					  << 1 << " "					//min
					  << 1 << " "					//mean
					  << 1 << " "					//sumsqr
					  << endl;
				}else{
					fs_output << "\"Event_"
					  << ev_name << "()\" " 	//eventname
					  << ptr->entry.data.timer.count << " "		//numevents
					  << 0 << " "					//max
					  << 0 << " "					//min
					  << 0 << " "					//mean
					  << 0 << " "					//sumsqr
					  << endl;

				}
			}
		}
			
		fs_output.close();
	}

	/*
	diffOutput = tmp_diffOutput;
	startSize = tmp_startSize;
	stopSize = tmp_stopSize;
	outSize = tmp_outSize;
	*/

	return 0;
}
/*----------------------------- PRIVATE -------------------------------*/

/* 
 * Function 		: TauKtau::DumpKprofile
 * Description		: Dump the profile to Kprofile.<nodeid>/
 */
int TauKtau::DumpKProfile(void)
{
	int i=0,j=0;
	char output_path[OUTPUT_NAME_SIZE];
	o_ent *ptr;
	unsigned int cur_index = 0;
	unsigned int user_ev_counter = 0;
	unsigned int templ_fun_counter = 0;

       
	if((outSize <= 0) || (stopSize <= 0) || (startSize <= 0)) {
		return -1;
	}

	/* 
	 * Create output directory ./Kprofile 
	 */
	snprintf(output_path, sizeof(output_path), "./Kprofile.%d.%d", RtsLayer::myNode(), RtsLayer::myThread());
	if(mkdir(output_path,777) == -1){
		perror("TauKtau::DumpKProfile: mkdir");
		if(errno != EEXIST) { //ignore already-exists errors
			return(-1);
		}
	}
	if(chmod(output_path, S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH ) != 0) {
		perror("TauKtau::DumpKProfile: chmod");
		return(-1);
	}

	/*
	 * Dumping output to the output file "./Kprofile.<rank>/profile.<pid>.0.0"
	 */
	// Data format :
        // %d templated_functions
        // "%s %s" %ld %G %G  
        //  funcname type numcalls Excl Incl
        // %d aggregates
        // <aggregate info>
	
	/* For Each Process */
	for(i=0;i<outSize;i++){
		user_ev_counter = 0;
		templ_fun_counter = 0;
		
		/* Counting Profile */
		for(j=0;j < (diffOutput+i)->size; j++){
			ptr = (((diffOutput)+i)->ent_lst)+j;
			if(ptr->index < 300 || ptr->index > 399){
				templ_fun_counter++;			
			}else{
				user_ev_counter++;
			}
		}
		
		int context = 0;

		if((diffOutput+i)->pid == ThisKtauOutputInfo.pid) {
			context = 1;
		}

		snprintf(output_path, sizeof(output_path), "./Kprofile.%d.%d/profile.%u.0.%d",RtsLayer::myNode(),RtsLayer::myThread(),(diffOutput+i)->pid, context);
		ofstream fs_output (output_path , ios::out);
		if(!fs_output.is_open()){
			cout << "Error opening file: " << output_path << "\n";
			return(-1);
		}
		
		/* OUTPUT: Templated function */
		fs_output << templ_fun_counter << " templated_functions" << endl;
		fs_output << "# Name Calls Subrs Excl Incl ProfileCalls" << endl;

		for(j=0;j < (diffOutput+i)->size; j++){
			ptr = (((diffOutput)+i)->ent_lst)+j;
			string& func_name = KtauSym.MapSym(ptr->entry.addr);
			if(ptr->index < 300 || ptr->index > 399){
				fs_output << "\"" << 
					  func_name << "()\" " 	//Name
					  << ptr->entry.data.timer.count << " "		//Calls
					  << 0 << " "					//Subrs
					  << (double)ptr->entry.data.timer.excl/KTauGetMHz() //Excl
					  //<< (double)ptr->entry.data.timer.excl //Excl
					  << " "		
					  << (double)ptr->entry.data.timer.incl/KTauGetMHz() //Incl
					  //<< (double)ptr->entry.data.timer.incl //Incl
					  << " "	
					  << 0 << " ";					//ProfileCalls
				if(strstr(func_name.c_str(), "schedule")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SCHEDULE\"" << endl;
				//}else if(!strcmp("__run_timers",func_name.c_str())){
				//	fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_RUN_TIMERS\"" << endl;
				}else if(strstr(func_name.c_str(), "page_fault")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_EXCEPTION\"" << endl;
				}else if(strstr(func_name.c_str(), "IRQ")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_IRQ\"" << endl;
				}else if(strstr(func_name.c_str(), "run_timers")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"" << endl;
				}else if(strstr(func_name.c_str(), "workqueue")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"" << endl;
				}else if(strstr(func_name.c_str(), "tasklet")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"" << endl;
				//}else if(!strcmp("__do_softirq",func_name.c_str())){
				//	fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_DO_SOFTIRQ\"" << endl;
				}else if(strstr(func_name.c_str(), "softirq")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"" << endl;
				}else if(strstr(func_name.c_str(), "sys_")){
					if(strstr(func_name.c_str(), "sock")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_SOCK\"" << endl;
					} else if(strstr(func_name.c_str(), "read")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"" << endl;
					} else if(strstr(func_name.c_str(), "write")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"" << endl;
					} else if(strstr(func_name.c_str(), "send")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"" << endl;
					} else if(strstr(func_name.c_str(), "recv")){
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"" << endl;
					} else {
						fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL\"" << endl;
					}
				}else if(strstr(func_name.c_str(), "tcp")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_TCP\"" << endl;
				}else if(strstr(func_name.c_str(), "icmp")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_ICMP\"" << endl;
				}else if(strstr(func_name.c_str(), "sock")){
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_SOCK\"" << endl;
				}else{
					fs_output << "GROUP=\"TAU_KERNEL_MERGE | KTAU_DEFAULT\"" << endl;
				}
			}
		}
		/* OUTPUT: Aggregates*/	
		fs_output << 0 << " aggregates" << endl;
		
		/* OUTPUT: User-events*/
		fs_output << user_ev_counter  << " userevents" << endl;
		fs_output << "# eventname numevents max min mean sumsqr"<< endl;

		for(j=0;j < (diffOutput+i)->size; j++){
			ptr = (((diffOutput)+i)->ent_lst)+j;
			string& ev_name = KtauSym.MapSym(ptr->entry.addr);
			if(ptr->index >= 300 && ptr->index <= 399 ){
				if(ptr->entry.data.timer.count != 0){
					fs_output << "\"Event_"
					  << ev_name << "()\" " 	//eventname
					  << ptr->entry.data.timer.count << " "		//numevents
					  << 1 << " "					//max
					  << 1 << " "					//min
					  << 1 << " "					//mean
					  << 1 << " "					//sumsqr
					  << endl;
				}else{
					fs_output << "\"Event_"
					  << ev_name << "()\" " 	//eventname
					  << ptr->entry.data.timer.count << " "		//numevents
					  << 0 << " "					//max
					  << 0 << " "					//min
					  << 0 << " "					//mean
					  << 0 << " "					//sumsqr
					  << endl;

				}
			}
		}
			
		fs_output.close();
	}
	return 0;
}
/*----------------------------- PRIVATE -------------------------------*/

/*
 * Function		: diff_h_ent
 * Description		: Diffing h_ent for each o_ent 
 */
int diff_h_ent(o_ent *o1, o_ent *o2){

	/*
	 * Comparing h_ent of KTAU_TIMER type.
	 */
	if(o1->entry.type == o2->entry.type && o1->entry.type == KTAU_TIMER){
		if(o1->entry.addr == o2->entry.addr){

			o2->entry.data.timer.count -= o1->entry.data.timer.count;
			o2->entry.data.timer.incl -= o1->entry.data.timer.incl;
			o2->entry.data.timer.excl -= o1->entry.data.timer.excl;

			return(0);
		}
	}
	return(-1);
}

/*
 * Function		: diff_ktau_output
 * Description		: Diffing stop and start of each ktau_output and 
 * 			  store the result in out.
 */
int diff_ktau_output(ktau_output *start, ktau_output *stop, ktau_output *out){
	int i,j,k;
	o_ent *st;
	o_ent *sp;

	out->ent_lst = (o_ent*)malloc(sizeof(o_ent)*stop->size);
	/*
	 * For each o_ent, we have to check the hash-index. If the index
	 * doesn't exist in out, then just diff with 0.
	 * */
	for(i=0,j=0,k=0; i<start->size || j<stop->size;){
		st=(start->ent_lst+i);
		sp=(stop->ent_lst+j);
		if(st->index == sp->index){
			if(diff_h_ent(st,sp)){
				return(-1);
			}
			memcpy(((out->ent_lst)+k), ((stop->ent_lst)+j), sizeof(o_ent));
			i++;j++;k++;
		}else{
			memcpy(((out->ent_lst)+k), ((stop->ent_lst)+j), sizeof(o_ent));
			j++;k++;
		}
	}
	return(k);
}

int TauKtau::DiffKProfileTWO(char* startB, char* stopB, int startSz, int stopSz, ktau_output** pdiffOut)
{
	int i,j,k,diff_count;
	ktau_output *startOut;
	ktau_output *stopOut;
	long startProcNum =  0;
	long stopProcNum  = 0;

	if((startB==NULL) || (stopB==NULL) || (startSz<=0) || (stopSz<=0)) {
		return -1;
	}

	startProcNum = unpack_bindata(KTAU_TYPE_PROFILE, startB, startSz, &startOut);
	if(startProcNum < 0) {
		return -1;
	}

	stopProcNum  = unpack_bindata(KTAU_TYPE_PROFILE, stopB, stopSz, &stopOut);
	if(stopProcNum < 0) {
		return -1;
	}

	return ktau_diff_profiles(startOut, startProcNum, stopOut, stopProcNum, pdiffOut);
}

/*
 * Function		: TauKtau::DiffKProfile
 * Description		: Diffing profile for each matched pid in startOutput and 
 * 			  stopOutput. If no match is found, discard the pid.
 * Note			: Might want a different scheme
 */
int TauKtau::DiffKProfile(void)
{
	int i,j,k,diff_count;
	ktau_output *startOutput;
	ktau_output *stopOutput;
	long startProcNum =  0;
	long stopProcNum  = 0;

	if((startBuf==NULL) || (stopBuf==NULL) || (startSize<=0) || (stopSize<=0)) {
		return -1;
	}

	startProcNum = unpack_bindata(KTAU_TYPE_PROFILE, startBuf, startSize, &startOutput);
	if(startProcNum < 0) {
		return -1;
	}

	stopProcNum  = unpack_bindata(KTAU_TYPE_PROFILE, stopBuf, stopSize, &stopOutput);
	if(stopProcNum < 0) {
		return -1;
	}

	if(DEBUG)printf("DiffKProfile: startProcNum = %u\n",startProcNum);
	if(DEBUG)printf("DiffKProfile: stopProcNum = %u\n",stopProcNum);
	diffOutput = (ktau_output*)malloc(sizeof(ktau_output)*maxof(startProcNum,stopProcNum));
	
	/*
	 * For each ktau_output, we have to check pid if it is the same.
	 * Otherwise, we just ignore that one.
	 */
	for(i=0,j=0,k=0; i<startProcNum & j<stopProcNum;){
		if((startOutput+i)->pid == (stopOutput+j)->pid){
			if(DEBUG)printf("DiffKProfile: Diffing pid %d\n",(startOutput+i)->pid);
			if(DEBUG)printf("DiffKProfile: start size = %d\n",(startOutput+i)->size);
			if(DEBUG)printf("DiffKProfile: stop size  = %d\n",(stopOutput+i)->size);
			diff_count = diff_ktau_output((startOutput+i),(stopOutput+j),((diffOutput)+k));
			(diffOutput+k)->size = diff_count;
			(diffOutput+k)->pid = (startOutput+i)->pid;

			if((diffOutput+k)->pid == ThisKtauOutputInfo.pid){
				/* Keep track of information of ktau_output
				 * for this pid
				 */
				for(int x=0;x < (diffOutput+k)->size; x++){
					o_ent *ptr = (((diffOutput)+k)->ent_lst)+x;
					if(ptr->index < 300 || ptr->index > 399){
						ThisKtauOutputInfo.templ_fun_counter++;			
					}else{
						ThisKtauOutputInfo.user_ev_counter++;
					}
				}
			}

			i++;j++;k++;
		}else if((startOutput+i)->pid < (stopOutput+j)->pid){
			if(DEBUG)printf("DiffKProfile: pid unmatch. Increment start.\n");
			diffOutput[k] = startOutput[i];
			i++;k++;
		}else{
			if(DEBUG)printf("DiffKProfile: pid unmatch. Increment stop.\n");
			diffOutput[k] = stopOutput[j];
			j++;k++;
		}
	}
	return(k);
}

int TauKtau::GetNumKProfileFunc(){
	return ThisKtauOutputInfo.templ_fun_counter;
}

int TauKtau::GetNumKProfileEvent(){
	return ThisKtauOutputInfo.user_ev_counter;
}

int TauKtau::MergingKProfileFunc(FILE* fp){
	int i=0,j=0,k=0;
	unsigned int cur_index = 0;

	if(outSize <= 0) {
		return -1;
	}

	for(i=0;i<outSize;i++){
		if(ThisKtauOutputInfo.pid == (diffOutput+i)->pid){
			for(j=0;j < (diffOutput+i)->size; j++){
				o_ent* ptr = (((diffOutput)+i)->ent_lst)+j;
				string& func_name = KtauSym.MapSym(ptr->entry.addr);
				if(ptr->index < 300 || ptr->index > 399){
					fprintf(fp,"\"%s()  \" %u %u %g %g %u ", 
						  func_name.c_str(), //Name
						  ptr->entry.data.timer.count,	//Calls
						  0,				//Subrs
						  (double)(ptr->entry.data.timer.excl)/KTauGetMHz(),	//Excl
						  (double)(ptr->entry.data.timer.incl)/KTauGetMHz(),	//Incl
						  0);				//ProfileCalls
				if(strstr(func_name.c_str(),"schedule")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_SCHEDULE\"\n");
                                //}else if(!strcmp("__run_timers",func_name.c_str())){
                                //      fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_RUN_TIMERS\"\n");
                                }else if(strstr(func_name.c_str(), "page_fault")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_EXCEPTION\"\n");
                                }else if(strstr(func_name.c_str(), "IRQ")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_IRQ\"\n");
                                }else if(strstr(func_name.c_str(), "run_timers")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"\n");
                                }else if(strstr(func_name.c_str(), "workqueue")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"\n");
                                }else if(strstr(func_name.c_str(), "tasklet")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"\n");
                                //}else if(!strcmp("__do_softirq",func_name.c_str())){
                                //      fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_DO_SOFTIRQ\"\n");
                                }else if(strstr(func_name.c_str(), "softirq")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_BH\"\n");
                                }else if(strstr(func_name.c_str(), "sys_")){
                                        if(strstr(func_name.c_str(), "sock")){
                                                fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_SOCK\"\n");
                                        } else if(strstr(func_name.c_str(), "read")){
                                                fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"\n");
                                        } else if(strstr(func_name.c_str(), "write")){
                                                fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"\n");
                                        } else if(strstr(func_name.c_str(), "send")){
                                                fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"\n");
                                        } else if(strstr(func_name.c_str(), "recv")){
                                                fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL | KTAU_IO\"\n");
                                        } else {
                                                fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_SYSCALL\"\n");
                                        }
                                }else if(strstr(func_name.c_str(), "tcp")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_TCP\"\n");
                                }else if(strstr(func_name.c_str(), "icmp")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_ICMP\"\n");
                                }else if(strstr(func_name.c_str(), "sock")){
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_SOCK\"\n");
                                }else{
                                        fprintf(fp, "GROUP=\"TAU_KERNEL_MERGE | KTAU_DEFAULT\"\n");
                                }
				/*if(!strcmp("schedule",func_name.c_str())){
					fprintf(fp,"GROUP=\"KTAU_SCHEDULE\"\n");
				}else if(!strcmp("__run_timers",func_name.c_str())){
					fprintf(fp,"GROUP=\"KTAU_RUN_TIMERS\"\n");
				}else if(!strcmp("__do_softirq",func_name.c_str())){
					fprintf(fp,"GROUP=\"KTAU_DO_SOFTIRQ\"\n");
				}else{
					fprintf(fp,"GROUP=\"KTAU_DEFAULT\"\n");
				}*/
				}
			}
		}
		
	}
	return(0);
}

int TauKtau::MergingKProfileEvent(FILE* fp){
	int i=0,j=0,k=0;
	unsigned int cur_index = 0;

	if(outSize <= 0) {
		return -1;
	}

	for(i=0;i<outSize;i++){
		if(ThisKtauOutputInfo.pid == (diffOutput+i)->pid){
			for(j=0;j < (diffOutput+i)->size; j++){
				o_ent* ptr = (((diffOutput)+i)->ent_lst)+j;
				string& ev_name = KtauSym.MapSym(ptr->entry.addr);
				if(ptr->index >= 300 && ptr->index <= 399)
					fprintf(fp,"\"Event_%s()\" %u %u %u %u %u\n",
					  ev_name.c_str(), 	//eventname
					  ptr->entry.data.timer.count,		//numevents
					  1,					//max
					  1,					//min
					  1,					//mean
					  1);					//sumsqr
					  
			}
		}
	}
	return(0);
}



/***************************************************************************
 * $RCSfile: TauKtau.cpp,v $   $Author: anataraj $
 * $Revision: 1.6 $   $Date: 2007/04/19 03:21:45 $
 ***************************************************************************/

	





