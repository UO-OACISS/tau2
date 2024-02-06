/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: KtauProfiler.cpp				  **
**	Description 	: Kernel-space  FunctionInfo 			  **
**	Author		: Aroon Nataraj					  **
**			: Surave Suthikulpanit				  **
**	Contact		: anataraj@cs.uoregon.edu		 	  **
**			: suravee@cs.uoregon.edu		 	  **
**	Flags		: Compile with				          **
**			  -DTAUKTAU or -DTAUKTAU_MERGE			  **
**	Documentation	:                                                 **
***************************************************************************/

#if defined(TAUKTAU)


#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <errno.h>

#include <Profile/KtauProfiler.h>
#include <Profile/TauKtau.h>
#include <Profile/RtsLayer.h>

#include <Profile/TauAPI.h>

#if defined(TAUKTAU_MERGE)
#include <Profile/KtauFuncInfo.h>
#include <Profile/KtauMergeInfo.h>
// DONT #include <Profile/ktau_atomic.h>
#include <Profile/ktau_proc_interface.h>
//decl
int read_kstate_1buf(ktau_state* pstate, volatile int cur_active, unsigned long long* ptime, unsigned long long* pcalls, unsigned long long *pexcl);
int read_kstate_2buf(ktau_state* pstate, volatile int cur_active, unsigned long long* ptime, unsigned long long* pcalls, unsigned long long *pexcl);

#define read_kstate(pstate, cur_active, ptime, pcalls, pexcl) read_kstate_1buf(pstate, cur_active, ptime, pcalls, pexcl)

#ifdef TAUKTAU_MERGE
char* merge_grp_name[NO_MERGE_GRPS+1] = {
	"KERNEL",
	"SYSCALL",
	"IRQ",
	"BH",
	"SCHED",
	"EXCEPT",
	"SIGNAL",
	"SOCKET",
	"TCP",
	"ICMP",
	"NONE"
};
#endif

#endif /* defined(TAUKTAU_MERGE) */


#define OUTPUT_NAME_SIZE        1024

using namespace std;

//statics
KtauProfiler * KtauProfiler::CurrentKtauProfiler[TAU_MAX_THREADS] = {0};

long long KtauProfiler::refCount[TAU_MAX_THREADS] = {0};

KtauSymbols KtauProfiler::KtauSym(KTAU_KALLSYMS_PATH);


KtauProfiler* KtauProfiler::GetKtauProfiler(int tid) {

	DEBUGPROFMSG(" Entering: tid:"<< tid 
			<< endl; );

	if((tid >= TAU_MAX_THREADS) || (tid < 0)) {
		DEBUGPROFMSG(" Bad tid:"<< tid 
				<< endl; );
		return NULL;
	}

	if(CurrentKtauProfiler[tid] == NULL) {
		CurrentKtauProfiler[tid] = new KtauProfiler(tid);
		DEBUGPROFMSG(" Creating New KtauProfiler:"<< CurrentKtauProfiler[tid]  
				<< endl; );
		refCount[tid] = 0;
	}

	refCount[tid]++;

	DEBUGPROFMSG(" RefCount:"<< refCount[tid]  
			<< endl; );

	DEBUGPROFMSG(" Leaving: tid:"<< tid 
			<< endl; );

	return CurrentKtauProfiler[tid];
}


void KtauProfiler::PutKtauProfiler(int tid) {
	DEBUGPROFMSG(" Entering: tid:"<< tid 
			<< endl; );

	if((tid >= TAU_MAX_THREADS) || (tid < 0)) {
		DEBUGPROFMSG(" Bad tid:"<< tid 
				<< endl; );

		return;
	}
	
	refCount[tid]--;

	if(refCount[tid] == 0) {

		DEBUGPROFMSG(" Deleting KtauProfiler:"<< CurrentKtauProfiler[tid]
				<< "refCount is:" << refCount[tid] 
				<< endl; );

		delete CurrentKtauProfiler[tid];
		CurrentKtauProfiler[tid] = NULL;
	}

	DEBUGPROFMSG(" Leaving: tid:"<< tid 
			<< endl; );
}


//Instrumentation Methods
void KtauProfiler::Start(Profiler *profiler, int tid) {

	DEBUGPROFMSG(" Entering: tid:"<< tid 
			<< "Profiler is: " << profiler
			<< endl; );

	if(refCount[tid] == 1) {

		DEBUGPROFMSG(" 1st Start Call for tid: "<< tid 
				<< endl << "Calling StartKProfile. "
				<< endl; );

		//1. Start call for this thread
		KernProf.StartKProfile();

#if defined TAUKTAU_MERGE
		//2. if merge, then setup merge-data struct

		current_ktau_state = (ktau_state*) calloc(4096,1);
		if(!current_ktau_state) {
			perror("calloc of current_ktau_state:");
			exit(-1);
		}

		DEBUGPROFMSG(" Allocated current_ktau_state: "<< current_ktau_state 
				<< endl; );

		/* Setting ktau_state pointer in kernel-space to point
		 * to the global current_state
		 */
		//AN REMOVED TEMPORSRILY TO CHECK EFFECT - TURN BACK ON ASAP!
		ktau_set_state(NULL,current_ktau_state,NULL);

		DEBUGPROFMSG(" Called ktau_set_state."  
				<< endl; );

#endif /* TAUKTAU_MERGE */

	} 

	//any start
#if defined TAUKTAU_MERGE
	DEBUGPROFMSG(" Calling SetStartState."  
			<< endl; );
	//1. if merge, then read the start-merge-info
	SetStartState(current_ktau_state, profiler, tid, (refCount[tid] == 1));
#endif /* TAUKTAU_MERGE */

	DEBUGPROFMSG(" Leaving: tid:"<< tid 
			<< "Profiler is: " << profiler
			<< endl; );

}


void KtauProfiler::Stop(Profiler *profiler, bool AddInclFlag, int tid) {

	DEBUGPROFMSG(" Entering: tid:"<< tid 
			<< "Profiler is: " << profiler
			<< "AddInclFlag is: " << AddInclFlag
			<< endl; );
	//Any Stop
#if defined TAUKTAU_MERGE

	DEBUGPROFMSG(" Calling SetStopState. " 
			<< endl; );

	//1. if merge, then read the stop-merge-info and put into KtauFuncInfo
	SetStopState(current_ktau_state, AddInclFlag, profiler, tid, (refCount[tid] == 1));
	//2. Also help calc parent's excl-kernel-time etc

#endif /* TAUKTAU_MERGE */

	//Last Stop
        if(refCount[tid] == 1) {

		DEBUGPROFMSG(" Last Stop. tid: " << tid
				<< " profiler: " << profiler
				<< " . Calling StopKProfile." 
				<< endl; );

		//1. Stop call for this thread
                KernProf.StopKProfile();

#if defined TAUKTAU_MERGE

                //2. if merge, then tear-down merge-data struct
                /* Un-Setting ktau_state pointer in kernel-space.
                 */
                ktau_set_state(NULL,NULL,NULL);

		DEBUGPROFMSG(" After Unsetting ktau_set_state. "<< current_ktau_state 
				<< endl; );

                free(current_ktau_state);

		DEBUGPROFMSG(" After free of current_ktau_state. " 
				<< endl; );

		current_ktau_state = NULL;
#endif /* TAUKTAU_MERGE */

        } 

	DEBUGPROFMSG(" Leaving: tid:"<< tid 
			<< "Profiler is: " << profiler
			<< "AddInclFlag is: " << AddInclFlag
			<< endl; );
}

//cons
KtauProfiler::KtauProfiler(int threadid):KernProf(KtauSym) {
	DEBUGPROFMSG(" Entering Constructor: thred: "<< threadid 
			<< endl; );
	tid = threadid;
#ifdef TAUKTAU_MERGE
	current_ktau_state = NULL;
	active_merge_index = 0;
#endif /* TAUKTAU_MERGE */
	//KernProf(KtauSym);
	DEBUGPROFMSG(" Leaving Constructor: thread: "<< threadid 
			<< endl; );
}

//des
KtauProfiler::~KtauProfiler() {
	DEBUGPROFMSG(" Entering Destructor: thred: "<< tid 
			<< endl; );
#ifdef TAUKTAU_MERGE
	current_ktau_state = NULL;
	active_merge_index = 0;
#endif /* TAUKTAU_MERGE */

	DEBUGPROFMSG(" Leaving Destructor: thred: "<< tid 
			<< endl; );
}

/* 
 * Function 		: KtauProfiler::SetStartState
 * Description		: 
 */
int KtauProfiler::SetStartState(ktau_state* pstate, Profiler* pProfiler, int tid, bool stackTop) {

	DEBUGPROFMSG(" Entering: Profiler: "<< pProfiler
			<< "current_ktau_state: " << pstate
			<< endl;)

#if defined TAUKTAU_MERGE
	if(pProfiler) {
		unsigned long long s_ticks[merge_max_grp] = {0};
		unsigned long long s_excl[merge_max_grp] = {0};
		unsigned long long s_calls[merge_max_grp] = {0};

		active_merge_index = read_kstate(pstate, active_merge_index, &(s_ticks[0]), &(s_calls[0]), &(s_excl[0]));

		//If stackTop then update for keeping KERNEL GRP totals
		if(stackTop) {
			for(int i=0; i<merge_max_grp; i++) {
				KtauFuncInfo::kernelGrpCalls[tid][i] = s_calls[i];
				KtauFuncInfo::kernelGrpIncl[tid][i] = s_ticks[i];
				KtauFuncInfo::kernelGrpExcl[tid][i] = s_excl[i];
			}
		}

		KtauMergeInfo* ThisMergeInfo = &(pProfiler->ThisKtauMergeInfo);

		DEBUGPROFMSG(" Previous MergeInfo State: StartTicks: "<< ThisMergeInfo->GetStartTicks()
				<<" StartCalls: " << ThisMergeInfo->GetStartCalls();
				<< endl;)

		for(int i =0; i<merge_max_grp; i++) {
			ThisMergeInfo->SetStartTicks(s_ticks[i], i);
			ThisMergeInfo->SetStartCalls(s_calls[i], i);
			ThisMergeInfo->SetStartKExcl(s_excl[i], i);
		}

		DEBUGPROFMSG(" Latest MergeInfo State: StartTicks: "<< ThisMergeInfo->GetStartTicks()
				<< " StartCalls: " << ThisMergeInfo->GetStartCalls();
				<< endl;)
	}
#endif /* TAUKTAU_MERGE */

	DEBUGPROFMSG(" Leaving: Profiler: "<< pProfiler
			<< "current_ktau_state: " << pstate
			<< endl;)

	return(0);
}


/* 
 * Function 		: KtauProfiler::SetStopState
 * Description		: 
 */
int KtauProfiler::SetStopState(ktau_state* pstate, bool AddInclFlag, Profiler* pProfiler, int tid, bool stackTop) {

	DEBUGPROFMSG(" Entering: Profiler: "<< pProfiler
			<< "current_ktau_state: " << pstate
			<< "AddInclFlag: " << AddInclFlag
			<< endl;)

#if defined TAUKTAU_MERGE
	KtauMergeInfo* ThisMergeInfo = &(pProfiler->ThisKtauMergeInfo);

	DEBUGPROFMSG(" ThisMergeInfo: "<< ThisMergeInfo
			<< endl;)

	//inclusive kernel-mode ticks
	unsigned long long inclticks[merge_max_grp] = {0};
	unsigned long long inclcalls[merge_max_grp] = {0};
	unsigned long long inclExcl[merge_max_grp] = {0};

	active_merge_index = read_kstate(pstate, active_merge_index, &(inclticks[0]), &(inclcalls[0]), &(inclExcl[0]));

	//If stackTop then update for keeping KERNEL GRP totals
	if(stackTop) {
		for(int i=0; i<merge_max_grp; i++) {
			KtauFuncInfo::kernelGrpCalls[tid][i] = inclcalls[i] - KtauFuncInfo::kernelGrpCalls[tid][i];
			KtauFuncInfo::kernelGrpIncl[tid][i] = inclticks[i] - KtauFuncInfo::kernelGrpIncl[tid][i];
			KtauFuncInfo::kernelGrpExcl[tid][i] = inclExcl[i] - KtauFuncInfo::kernelGrpExcl[tid][i];
		}
	}

	DEBUGPROFMSG(" Stop-Ticks: "<< inclticks
			<< endl;)
	
	for(int i=0; i<merge_max_grp; i++) {
		inclticks[i] = inclticks[i] - ThisMergeInfo->GetStartTicks(i);
	}

	DEBUGPROFMSG(" Start-Ticks: "<< ThisMergeInfo->GetStartTicks()
			<< endl;)

	DEBUGPROFMSG(" Incl-Ticks: "<< inclticks
			<< endl;)

	DEBUGPROFMSG(" Child-Ticks: "<< ThisMergeInfo->GetChildTicks()
			<< endl;)

	//exclusive kernel-mode ticks (without children's kernel-mode ticks)
	unsigned long long exclticks[merge_max_grp] = {0};
	for(int i=0; i<merge_max_grp; i++) {
		exclticks[i] = inclticks[i] - ThisMergeInfo->GetChildTicks(i);
	}

	for(int i=0; i<merge_max_grp; i++) {
		if(ThisMergeInfo->GetChildTicks(i) > inclticks[i])
			cout << "KtauProfiler::SetStopState: Kernel: ChildTicks > InclTicks: Child: " 
				<< ThisMergeInfo->GetChildTicks(i) << "  Incl: " << inclticks[i] << endl;

		if(exclticks[i] > inclticks[i])
			cout << "KtauProfiler::SetStopState: Kernel: ExclTicks > InclTicks: Excl: " 
				<< exclticks[i] << "  Incl: " << inclticks[i] << endl;
	}

	for(int i=0; i<merge_max_grp; i++) {
		//inclusive kernel-mode calls
		inclcalls[i] = inclcalls[i] -  ThisMergeInfo->GetStartCalls(i);
	}

	//exclusive kernel-mode ticks (without children's kernel-mode ticks)
	unsigned long long exclcalls[merge_max_grp] = {0};
	for(int i=0; i<merge_max_grp; i++) {
		exclcalls[i] = inclcalls[i] - ThisMergeInfo->GetChildCalls(i);
	}
	
	//ExclTime reported directly from kernel
	for(int i=0; i<merge_max_grp; i++) {
		//inclusive kernel-mode calls
		inclExcl[i] = inclExcl[i] -  ThisMergeInfo->GetStartKExcl(i);
	}
	unsigned long long exclExcl[merge_max_grp] = {0};
	for(int i=0; i<merge_max_grp; i++) {
		exclExcl[i] = inclExcl[i] - ThisMergeInfo->GetChildKExcl(i);
	}


	DEBUGPROFMSG(" ParentProfiler: "<< pProfiler->ParentProfiler
			<< endl;)

	if(pProfiler->ParentProfiler) {

		DEBUGPROFMSG(" Adding to Parent's Child-ticks." 
				<< endl;)
		for(int i=0; i<merge_max_grp; i++) {
			//add our incl-ticks/calls to parent's so that it can calcultae its own excl-ticks
			pProfiler->ParentProfiler->ThisKtauMergeInfo.AddChildTicks(inclticks[i], i);
			pProfiler->ParentProfiler->ThisKtauMergeInfo.AddChildCalls(inclcalls[i], i);
			pProfiler->ParentProfiler->ThisKtauMergeInfo.AddChildKExcl(inclExcl[i], i);
		}
	}

	//Save the state to the KtauFunctionInfo
	KtauFuncInfo * pKFInfo = pProfiler->ThisFunction->GetKtauFuncInfo(tid);

	DEBUGPROFMSG(" KtauFuncInfo Ptr: "  << pKFInfo
			<< endl;)
	
	if(pKFInfo) {
		if(AddInclFlag) {
			for(int i=0; i<merge_max_grp; i++) {
				pKFInfo->AddInclTicks(inclticks[i], i);
				pKFInfo->AddInclCalls(inclcalls[i], i);
				pKFInfo->AddInclKExcl(inclExcl[i], i);
			}
		}
		for(int i=0; i<merge_max_grp; i++) {
			pKFInfo->AddExclTicks(exclticks[i], i);
			pKFInfo->AddExclCalls(exclcalls[i], i);
			pKFInfo->AddExclKExcl(exclExcl[i], i);
		}
	} else {
		cout << "KtauProfiler::SetStopState: Null KtauFuncInfo Found!" << endl;
	}

	/*
	cout << "KtauProfiler::SetStopState" << endl <<
		"   start_ticks:" << start_ticks<<
		"   inclticks:" << inclticks<<
		" , exclticks:" << exclticks<<
		" , child_ticks :" << child_ticks << endl;
	*/

	//reset the counters
	for(int i=0; i>merge_max_grp; i++) {
		ThisMergeInfo->ResetCounters(i);
	}

#endif /* TAUKTAU_MERGE */

	DEBUGPROFMSG(" Leaving: Profiler: "<< pProfiler
			<< "current_ktau_state: " << pstate
			<< "AddInclFlag: " << AddInclFlag
			<< endl;)

	return(0);
}


FILE* KtauProfiler::OpenOutStream(char* tau_dirname, int node, int context, int tid) {
	char output_path[OUTPUT_NAME_SIZE];
	FILE* ktau_fp = NULL;
	char* errormsg = NULL;

	/* 
         * Create output directory <tau-dirname>/Kprofile 
         */
        snprintf(output_path, sizeof(output_path), "%s/Kprofile", tau_dirname);
        if(mkdir(output_path,755) == -1){
		if(errno != EEXIST) {
			perror("KtauProfiler::OpenOutStream: mkdir");
			return(NULL);
		}
        }

	if(chmod(output_path, S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH) != 0) {
		perror("KtauProfiler::OpenOutStream: chmod");
		return NULL;
	}

	char filename[OUTPUT_NAME_SIZE];
        snprintf(filename, sizeof(filename), "%s/profile.%d.%d.%d",output_path, node,
                context, tid);
        if ((ktau_fp = fopen (filename, "w+")) == NULL) {
                errormsg = new char[1024];
                snprintf(errormsg, 1024, "Error: Could not create %s",filename);
                perror(errormsg);
		delete errormsg;
                return NULL;
        }

	return ktau_fp;
}

void KtauProfiler::CloseOutStream(FILE* ktau_fp) {
	if(ktau_fp)
		fclose(ktau_fp);
}


/* returns the updated last active_index value */
/* Single buf implementation */
int read_kstate_1buf(ktau_state* pstate, volatile int cur_active, unsigned long long* ptime, unsigned long long* pcalls, unsigned long long *pexcl) {

        //now read in the values from the 'old' cur_active index
	for(int i =0; i<merge_max_grp; i++) {
		ptime[i] = pstate->state[0].ktime[i];
		pexcl[i] = pstate->state[0].kexcl[i];
		pcalls[i] = pstate->state[0].knumcalls[i];
	}

	return 0;
}

/* Dbl-Buffer Implementation */
int read_kstate_2buf(ktau_state* pstate, volatile int cur_active, unsigned long long* ptime, unsigned long long* pcalls, unsigned long long *pexcl) {

        //1st change the active_index value atomically
        pstate->active_index = 1 - cur_active; //AN-dblbuf

        //now read in the values from the 'old' cur_active index
	for(int i =0; i<merge_max_grp; i++) {
		ptime[i] = pstate->state[cur_active].ktime[i];
		pexcl[i] = pstate->state[cur_active].kexcl[i];
		pcalls[i] = pstate->state[cur_active].knumcalls[i];
	}

        //make cur_active into new cur_actice
        cur_active = 1 - cur_active; //AN-dbl-buf

        //now add in the values from the other buffer after changing the active_index
        pstate->active_index = 1 - cur_active;

        //now read in the values from the 'old' cur_active index
	for(int i =0; i<merge_max_grp; i++) {
		ptime[i] += pstate->state[cur_active].ktime[i];
		pexcl[i] += pstate->state[cur_active].kexcl[i];
		pcalls[i] += pstate->state[cur_active].knumcalls[i];
	}

        //return the chaged cur_state
        return (1 - cur_active);

}

/* Older Double buffer approach - current;ly its not so */
/* returns the updated last active_index value */
/*
int read_kstate(ktau_state* pstate, volatile int cur_active, unsigned long long* ptime, unsigned long long* pcalls, unsigned long long *pexcl) {

        //1st change the active_index value atomically
        pstate->active_index = 1 - cur_active;

        //now read in the values from the 'old' cur_active index
	for(int i =0; i<merge_max_grp; i++) {
		ptime[i] = pstate->state[cur_active].ktime[i];
		pexcl[i] = pstate->state[cur_active].kexcl[i];
		pcalls[i] = pstate->state[cur_active].knumcalls[i];
	}

        //make cur_active into new cur_actice
        cur_active = 1 - cur_active;

        //now add in the values from the other buffer after changing the active_index
        pstate->active_index = 1 - cur_active;

        //now read in the values from the 'old' cur_active index
	for(int i =0; i<merge_max_grp; i++) {
		ptime[i] += pstate->state[cur_active].ktime[i];
		pexcl[i] += pstate->state[cur_active].kexcl[i];
		pcalls[i] += pstate->state[cur_active].knumcalls[i];
	}

        //return the chaged cur_state
        return (1 - cur_active);
}
*/

void KtauProfiler::RegisterFork(Profiler* profiler, int nodeid, int tid, enum TauFork_t opcode) {

	//printf("KtauProfiler::RegisterFork: Enter\n");

	//1. Handle like 1st call to start

#if defined TAUKTAU_MERGE
	//1.a. if merge - then set_ktau_state
	//printf("KtauProfiler::RegisterFork: Freeing Previous current_ktau_state\n");
	if(current_ktau_state != NULL) {
		free(current_ktau_state);
	}
	//printf("KtauProfiler::RegisterFork: Allocating current_ktau_state\n");
	current_ktau_state = (ktau_state*) calloc(4096,1);
	if(!current_ktau_state) {
		printf("KtauProfiler::RegisterFork: Allocating current_ktau_state FAILED.\n");
		perror("calloc of current_ktau_state:");
		exit(-1);
	}

	DEBUGPROFMSG(" Allocated current_ktau_state: "<< current_ktau_state
				<< endl; );

	//printf("KtauProfiler::RegisterFork: Setting Kernel State...\n");
	/* Setting ktau_state pointer in kernel-space to point
	 * to the global current_state
	 */
	ktau_set_state(NULL,current_ktau_state,NULL);
	//printf("KtauProfiler::RegisterFork: Setting Kernel State DONE.\n");

	DEBUGPROFMSG(" Called ktau_set_state."
			<< endl; );

	//1.b. re-init KtauProfiler state - such as active index etc
	active_merge_index = 0;
#endif /* TAUKTAU_MERGE */

	//printf("KtauProfiler::RegisterFork: TauKtau Dest Call.\n");
	//1.c. TauKtau state needs to be re-init - clean-it THEN do a start on it. 
	TauKtau::RegisterFork(&KernProf, opcode);

#ifdef TAUKTAU_MERGE
	if(opcode == TAU_EXCLUDE_PARENT_DATA) {
		//printf("KtauProfiler::RegisterFork: KtauMergeInfo Reseting.\n");
		//2. Run-up Profiler Stack - Re-initing KtauMergeInfo
		//2.a. set all counters in MergeInfo to Zero (for now)
		Profiler* curP = profiler;
		do {
			KtauMergeInfo* ThisMergeInfo = &(curP->ThisKtauMergeInfo);
			for(int i =0; i<merge_max_grp; i++) {
				ThisMergeInfo->ResetCounters(i);
			}
			
			curP = curP->ParentProfiler;

		} while(curP != NULL);
	}//EXCLUDE_PARENT_DATA
#endif /* TAUKTAU_MERGE */

	//printf("KtauProfiler::RegisterFork: Exit\n");
}


int KtauProfiler::VerifyMerge(FunctionInfo* thatFunction) {
#ifdef TAUKTAU_MERGE
	double org_time     = thatFunction->GetExclTime(tid); 
	double kern_time     = (double)(thatFunction->GetKtauFuncInfo(tid)->GetExclTicks(0))/KTauGetMHz();

	if(org_time > kern_time){
	  cout <<"GOOD!!! Kernel space time is less than ExclTime: org_time:"<<
		  org_time << " kern_time:"<< kern_time << endl;
	}else{ 
	  cout <<"ERROR!! Kernel space time is greater than ExclTime: org_time:"<<
		  org_time << " kern_time:"<< kern_time << endl;

	  cout << "KTauGetMHz:" << KTauGetMHz() <<
		  " , Kernel Inc-ticks:" << thatFunction->GetKtauFuncInfo(tid)->GetInclTicks(0) <<
		  " , Kernel IncTime:" << (double)(thatFunction->GetKtauFuncInfo(tid)->GetInclTicks(0))/KTauGetMHz()<<
		  " , Kernel ExclTicks:" << thatFunction->GetKtauFuncInfo(tid)->GetExclTicks(0)<<
		  " , Kernel ExclTime:" << (double)(thatFunction->GetKtauFuncInfo(tid)->GetExclTicks(0))/KTauGetMHz()<<
		  " , User-ExclTime:" << thatFunction->GetExclTime(tid)<<
		  " , User-InclTime:" << thatFunction->GetInclTime(tid)<<
		  " , FuncName: " << thatFunction->GetName() << endl;

		return 0;	
	}
#endif /*TAUKTAU_MERGE*/
	return 1;	
}

#endif /* TAUKTAU */

/***************************************************************************
 * $RCSfile: KtauProfiler.cpp,v $   $Author: anataraj $
 * $Revision: 1.5 $   $Date: 2006/11/10 07:25:42 $
 ***************************************************************************/

