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

#include <Profile/KtauProfiler.h>
#include <Profile/TauKtau.h>
#include <Profile/RtsLayer.h>

#include <Profile/TauAPI.h>

#if defined(TAUKTAU_MERGE)
#include <Profile/KtauFuncInfo.h>
#include <Profile/KtauMergeInfo.h>
#include <Profile/ktau_atomic.h>
#include <Profile/ktau_proc_interface.h>
#endif /* defined(TAUKTAU_MERGE) */


#define OUTPUT_NAME_SIZE        1024

using namespace std;

//statics
KtauProfiler * KtauProfiler::CurrentKtauProfiler[TAU_MAX_THREADS] = {0};

long long KtauProfiler::refCount[TAU_MAX_THREADS] = {0};

KtauSymbols KtauProfiler::KtauSym(KALLSYMS_PATH);


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
	SetStartState(current_ktau_state, profiler);
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
	SetStopState(current_ktau_state, AddInclFlag, profiler);
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
#endif /* TAUKTAU_MERGE */

	DEBUGPROFMSG(" Leaving Destructor: thred: "<< tid 
			<< endl; );
}

/* 
 * Function 		: KtauProfiler::SetStartState
 * Description		: 
 */
int KtauProfiler::SetStartState(ktau_state* pstate, Profiler* pProfiler) {

	DEBUGPROFMSG(" Entering: Profiler: "<< pProfiler
			<< "current_ktau_state: " << pstate
			<< endl;)

#if defined TAUKTAU_MERGE
	if(pProfiler) {
		unsigned long long s_ticks = read_ktime(pstate);
		unsigned long long s_calls = read_kcalls(pstate);
		KtauMergeInfo* ThisMergeInfo = &(pProfiler->ThisKtauMergeInfo);

		DEBUGPROFMSG(" Previous MergeInfo State: StartTicks: "<< ThisMergeInfo->GetStartTicks()
				<<" StartCalls: " << ThisMergeInfo->GetStartCalls();
				<< endl;)

		ThisMergeInfo->SetStartTicks(s_ticks);
		ThisMergeInfo->SetStartCalls(s_calls);

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
int KtauProfiler::SetStopState(ktau_state* pstate, bool AddInclFlag, Profiler* pProfiler) {

	DEBUGPROFMSG(" Entering: Profiler: "<< pProfiler
			<< "current_ktau_state: " << pstate
			<< "AddInclFlag: " << AddInclFlag
			<< endl;)

#if defined TAUKTAU_MERGE
	KtauMergeInfo* ThisMergeInfo = &(pProfiler->ThisKtauMergeInfo);

	DEBUGPROFMSG(" ThisMergeInfo: "<< ThisMergeInfo
			<< endl;)

	//inclusive kernel-mode ticks
	unsigned long long inclticks = read_ktime(pstate);

	DEBUGPROFMSG(" Stop-Ticks: "<< inclticks
			<< endl;)

	inclticks = inclticks - ThisMergeInfo->GetStartTicks();

	DEBUGPROFMSG(" Start-Ticks: "<< ThisMergeInfo->GetStartTicks()
			<< endl;)

	DEBUGPROFMSG(" Incl-Ticks: "<< inclticks
			<< endl;)

	DEBUGPROFMSG(" Child-Ticks: "<< ThisMergeInfo->GetChildTicks()
			<< endl;)

	//exclusive kernel-mode ticks (without children's kernel-mode ticks)
	unsigned long long exclticks = inclticks - ThisMergeInfo->GetChildTicks();

	if(ThisMergeInfo->GetChildTicks() > inclticks)
		cout << "KtauProfiler::SetStopState: Kernel: ChildTicks > InclTicks: Child: " 
			<< ThisMergeInfo->GetChildTicks() << "  Incl: " << inclticks << endl;

	if(exclticks > inclticks)
		cout << "KtauProfiler::SetStopState: Kernel: ExclTicks > InclTicks: Excl: " 
			<< exclticks << "  Incl: " << inclticks << endl;

	//inclusive kernel-mode calls
	unsigned long long inclcalls = read_kcalls(pstate);
	inclcalls = inclcalls -  ThisMergeInfo->GetStartCalls();
	//exclusive kernel-mode ticks (without children's kernel-mode ticks)
	unsigned long long exclcalls = inclcalls - ThisMergeInfo->GetChildCalls();

	DEBUGPROFMSG(" ParentProfiler: "<< pProfiler->ParentProfiler
			<< endl;)

	if(pProfiler->ParentProfiler) {

		DEBUGPROFMSG(" Adding to Parent's Child-ticks." 
				<< endl;)
		//add our incl-ticks/calls to parent's so that it can calcultae its own excl-ticks
		pProfiler->ParentProfiler->ThisKtauMergeInfo.AddChildTicks(inclticks);
		pProfiler->ParentProfiler->ThisKtauMergeInfo.AddChildCalls(inclcalls);
	}

	//Save the state to the KtauFunctionInfo
	KtauFuncInfo * pKFInfo = pProfiler->ThisFunction->GetKtauFuncInfo(tid);

	DEBUGPROFMSG(" KtauFuncInfo Ptr: "  << pKFInfo
			<< endl;)
	
	if(pKFInfo) {
		if(AddInclFlag) {
			pKFInfo->AddInclTicks(inclticks);
			pKFInfo->AddInclCalls(inclcalls);
		}
		pKFInfo->AddExclTicks(exclticks);
		pKFInfo->AddExclCalls(exclcalls);
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
	ThisMergeInfo->ResetCounters();

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
        sprintf(output_path,"%s/Kprofile", tau_dirname);
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
        sprintf(filename,"%s/profile.%d.%d.%d",output_path, node,
                context, tid);
        if ((ktau_fp = fopen (filename, "w+")) == NULL) {
                errormsg = new char[1024];
                sprintf(errormsg,"Error: Could not create %s",filename);
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

#endif /* TAUKTAU */

/***************************************************************************
 * $RCSfile: KtauProfiler.cpp,v $   $Author: anataraj $
 * $Revision: 1.1 $   $Date: 2005/12/01 02:55:08 $
 ***************************************************************************/

