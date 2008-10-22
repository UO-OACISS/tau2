/***************************************************************************************************
This file contains the tracemanger module of TAUPIN. 
This module interacts with both PIN and TAU systems   
****************************************************************************************************/

#include "TraceManager.h"
#include "MpiHooks.h"
extern VOID HookHandleBefore(RtnTrack* rtntr);
extern VOID HookHandleAfter(RtnTrack* rtntr);
extern VOID HookHandle(RtnTrack* rtntr);
extern VOID HookHandle1(RtnTrack* rtntr);
extern SpecManager* mySpec;
VOID MpiInst(IMG img);
VOID MpiInstRtn(RTN myrtn);

TraceManager::TraceManager(SpecManager* spm)
{
	this->Spm=spm;
	this->rtncnt=0;
	mpi_setup=false;
}

TraceManager::TraceManager(SpecManager* spm,string img)
{
	this->Spm=spm;
	this->rtncnt=0;
	this->img=img;
	mpi_setup=false;
	//trace_file.open("debug_trace",ios_base::app);
}

TraceManager::~TraceManager(void)
{
	//trace_file.close();
}

void TraceManager::LogMessage(string msg)
{
	ofstream mtrace_file;
	mtrace_file.open("debug_trace",ios_base::app);
	mtrace_file.write(msg.c_str(),msg.length());
	//trace_file.flush();
	mtrace_file.close();
}

//apply the instrumentation specification
void TraceManager::InstApply()
{	
}

string TraceManager::ImageTrim(string image)
{
	//remove all the path information 
	//get only the module name 
	int imglen=image.length();
	int pos=image.find_last_of('\\',imglen);
	pos++;
	return image.substr(pos,imglen-pos);
}


string unsafeRtns[]={"_RTC_",
  "__CRT_RTC_",
  "pre_c_init",
  "__controlfp_s",
  "mainCRTStartup",
  "__tmainCRTStartup",
  "WinMainCRTStartup",
  "__crt",
  "_initp_misc_winxfltr",
  "memset",
  "_setdefaultprecision",
  "__initterm",
  "pre_cpp_init",
  "atexit",
  "_onexit",
  "_mtinit",
  "_initptd",
  "_except_handler",
  "@_EH4_CallFilterFunc",
  "DebuggerProbe",
  "END" };

bool TraceManager::IsInstSafe(string rtn_name)
{
	int i=0;
	while(unsafeRtns[i]!="END")
	{
		string rtn_pref=unsafeRtns[i];
		int status=rtn_name.compare(0,rtn_pref.length(),rtn_pref);
		if(status==0)
			return false;
		i++;
	}
	
	//more checks here for future
	return true;
}

bool TraceManager::IsNormal(RTN myrtn)
{
	BBL my_bbl=RTN_BblTail(myrtn);	
	if(BBL_Valid(my_bbl))
	{
		INS my_ins=BBL_InsTail(my_bbl);
		while(INS_Valid(my_ins))
		{
			if(INS_IsRet(my_ins))
			{
				//cerr<<"Normal Routine::"<<RTN_Name(myrtn)<<endl;
				return true;
			}
			my_ins=INS_Prev(my_ins);
		}	
		/*if(INS_IsBranch(my_ins)|| INS_IsNop(my_ins))
		{
			cerr<<"!!!Abnormal Routine::"<<RTN_Name(myrtn)<<endl;
			return false;
		}	
		my_bbl=BBL_Prev(my_bbl); */
	}
	//cerr<<"!!!!Abnormal Routine::"<<RTN_Name(myrtn)<<endl;
	return false;
}

bool TraceManager::IsMpiRtn(string rtn_name)
{
	if(rtn_name.at(0)=='_')
		rtn_name=rtn_name.substr(1);
	int indx=rtn_name.find_first_of('@');
	if(indx!=string::npos)
		rtn_name=rtn_name.substr(0,indx);
	if(rtn_name.length()==0)
		return false;
	return mySpec->IsMpiRtn(rtn_name);

	/*
	string rtn_pref="MPI_";
	int status=rtn_name.compare(0,rtn_pref.length(),rtn_pref);
	int status1=rtn_name.compare(1,rtn_pref.length(),rtn_pref);
	if(status==0 || status1==0 )
	{
		if(!mpi_setup)
		{
			MpiSetUp(); 
			mpi_setup=false;
		}
		return true;
	}
	return false;
	*/
}

void TraceManager::InstApply(IMG img)
{

	DBG_TRACE("before check");
	DBG_TRACE(IMG_Name(img));
	
	if(mySpec==NULL){
		cerr<<"Spec manager is Null"<<endl;		
		return;
	}

	string imgname=ImageTrim(IMG_Name(img));
	cerr<<imgname<<endl;
	if(imgname.compare(MPI_LIB)==0)
	{
		//if the module is mpi dll then 
		//MpiInst(img);
		return;
	}
    
	//for rest of the modules it goes ahead here
	//checks if it's to be instrumented 
	//leaves if it's not suppose to be
	if(!mySpec->InstImage(imgname))
	{
		DBG_TRACE("Not to be instrumented");
		return;
	}
	//cerr<<"After approved for image inst"<<endl;	

	//Start enumarting all the routines in the image to be instrumented 
	SEC temp=IMG_SecHead(img);
	while(SEC_Valid(temp))
	{
		//we pick only executable code segment
		if(SEC_Type(temp)==SEC_TYPE_EXEC){
			RTN myrtn=SEC_RtnHead(temp);
			DBG_TRACE("RTNs to be instrumented");
			while(RTN_Valid(myrtn))
			{

				DBG_TRACE(RTN_Name(myrtn));
				//cerr<<RTN_Name(myrtn)<<endl;

				if(IsMpiRtn(RTN_Name(myrtn)))
				{
#ifndef NOMPI
					MpiInstRtn(myrtn);
#endif
					myrtn=RTN_Next(myrtn);
					continue;
				}

				if(RTN_Name(myrtn).length()==0 || !IsInstSafe(RTN_Name(myrtn)))
				{
					myrtn=RTN_Next(myrtn);
					continue;
				}
				
				if(mySpec->InstRtn(RTN_Name(myrtn),imgname))
				{
					//if the routine is to be instrumented 
					//apply the instrumentation here
					RTN_Open(myrtn);
					if(!IsNormal(myrtn))
					{
						RTN_Close(myrtn);
						myrtn=RTN_Next(myrtn);
						continue;
					}
					//DBG_TRACE(RTN_Name(myrtn));
					RtnTrack * brtntr = new RtnTrack;
					brtntr->img=imgname;
					brtntr->rtn=RTN_Name(myrtn);
					brtntr->stage=0;
					brtntr->flag=mySpec->GetProfileFlag(brtntr->rtn,brtntr->img);

					//insert the instrumentation at the exit point
					
					RTN_InsertCall(myrtn, IPOINT_AFTER, (AFUNPTR)HookHandleAfter, IARG_PTR ,brtntr, IARG_END);
					//RTN_InsertCallProbed(myrtn, IPOINT_AFTER, (AFUNPTR)HookHandleAfter, IARG_PTR ,brtntr, IARG_END);
					//insert the instrumentation at the entry point
					RTN_InsertCall(myrtn, IPOINT_BEFORE, (AFUNPTR)HookHandleBefore, IARG_PTR ,brtntr, IARG_END);
					//RTN_InsertCallProbed(myrtn, IPOINT_BEFORE, (AFUNPTR)HookHandleBefore, IARG_PTR ,brtntr, IARG_END);
					//cerr<<"Name::"<<RTN_Name(myrtn)<<" Start::"<<RTN_Address(myrtn)<<"  End::"<<RTN_Address(myrtn)+RTN_Size(myrtn)<<endl;	
					RTN_Close(myrtn);
				}
				myrtn=RTN_Next(myrtn);
			}
		}
		temp=SEC_Next(temp);		
	}	
}
	//before and after execute of the rtn block
void TraceManager::BeforeExec(RtnTrack* rtntr)
{
	//this is called before entry to the function
	int pflag=rtntr->flag;
	//bef_list.push_back(rtntr);
   	//more complicated option checking can done later
	if(pflag&&1)
	{
		DBG_TRACE("Profile Start");
		//start TAU profiling
		StartProfile((char*)rtntr->rtn.c_str(),&rtntr->tau);
	}
}

void TraceManager::AfterExec(RtnTrack* rtntr)
{
	//handles just before exit
	int pflag=rtntr->flag;
	//aft_list.push_back(rtntr);
	if(pflag&&1)
	{
		DBG_TRACE("Profile Stop");
		//calls enf of profile into tau system
		EndProfile((char*)rtntr->rtn.c_str(),rtntr->tau);
	}    
}

//this is the function which will dump everything at the last
void TraceManager::EndTrace()
{
	cerr<<"Tracing ended here"<<endl;
	//this doesnt do anything as of now
	bef_list.sort();
	aft_list.sort();
	list<RtnTrack*>::iterator it1,it2;
	it2=aft_list.begin();
	for(it1=bef_list.begin();it1!=bef_list.end();it1++)
	{
		if(it2!=aft_list.end())
		{
			cerr<<*it2;
			it2++;
		}
	}
	DumpTrace();
}

RTN RTN_FindLoop(IMG img, string rtn_name)
{
	SEC temp=IMG_SecHead(img);
	while(SEC_Valid(temp))
	{
		//we pick only executable code segment
		if(SEC_Type(temp)==SEC_TYPE_EXEC){
			RTN myrtn=SEC_RtnHead(temp);
			DBG_TRACE("RTNs to be instrumented");
			while(RTN_Valid(myrtn))
			{
				cerr<<RTN_Name(myrtn)<<endl;
				if(RTN_Name(myrtn)==rtn_name)
				{
					cerr<<"Rtn Found:"<<rtn_name<<endl;
					return myrtn;
				}
				myrtn=RTN_Next(myrtn);
			}
		}
		temp=SEC_Next(temp);
	}
	RTN empty;
	return empty;
}

#ifndef NOMPI
VOID MpiInst(IMG img)
{
	//the following lists take care of instrumenting 
	// MPI routines in a different way 
	// by using the replace api of PIN
	
	RTN myrtn;
	//myrtn=RTN_FindLoop(img, MPI_Init_FNAME);
	myrtn=RTN_FindByName(img,MPI_Allgather_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Allgather));
	}

	myrtn=RTN_FindByName(img,MPI_Allgatherv_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Allgatherv));
	}

	myrtn=RTN_FindByName(img,MPI_Allreduce_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Allreduce));
	}

	myrtn=RTN_FindByName(img,MPI_Alltoall_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Alltoall));
	}

	myrtn=RTN_FindByName(img,MPI_Alltoallv_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Alltoallv));
	}

	myrtn=RTN_FindByName(img,MPI_Barrier_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Barrier));
	}

	myrtn=RTN_FindByName(img,MPI_Bcast_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Bcast));
	}

	myrtn=RTN_FindByName(img,MPI_Gather_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Gather));
	}

	myrtn=RTN_FindByName(img,MPI_Gatherv_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Gatherv));
	}

	myrtn=RTN_FindByName(img,MPI_Op_create_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Op_create));
	}

	myrtn=RTN_FindByName(img,MPI_Op_free_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Op_free));
	}

	myrtn=RTN_FindByName(img,MPI_Reduce_scatter_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Reduce_scatter));
	}

	myrtn=RTN_FindByName(img,MPI_Reduce_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Reduce));
	}

	myrtn=RTN_FindByName(img,MPI_Scan_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Scan));
	}

	myrtn=RTN_FindByName(img,MPI_Scatter_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Scatter));
	}

	myrtn=RTN_FindByName(img,MPI_Scatterv_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Scatterv));
	}

	myrtn=RTN_FindByName(img,MPI_Attr_delete_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Attr_delete));
	}

	myrtn=RTN_FindByName(img,MPI_Attr_get_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Attr_get));
	}

	myrtn=RTN_FindByName(img,MPI_Attr_put_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Attr_put));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_compare_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_compare));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_create_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_create));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_dup_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_dup));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_free_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_free));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_group_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_group));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_rank_FNAME);
	if (RTN_Valid(myrtn))
	{
		cerr<<"MPI_comm_rank::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_rank));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_remote_group_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_remote_group));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_remote_size_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_remote_size));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_size_FNAME);
	if (RTN_Valid(myrtn))
	{
		cerr<<"MPI_comm_size::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_size));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_split_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_split));
	}

	myrtn=RTN_FindByName(img,MPI_Comm_test_inter_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_test_inter));
	}

	myrtn=RTN_FindByName(img,MPI_Group_compare_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_compare));
	}

	myrtn=RTN_FindByName(img,MPI_Group_difference_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_difference));
	}

	myrtn=RTN_FindByName(img,MPI_Group_excl_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_excl));
	}

	myrtn=RTN_FindByName(img,MPI_Group_free_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_free));
	}

	myrtn=RTN_FindByName(img,MPI_Group_incl_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_incl));
	}

	myrtn=RTN_FindByName(img,MPI_Group_intersection_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_intersection));
	}

	myrtn=RTN_FindByName(img,MPI_Group_rank_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_rank));
	}

	myrtn=RTN_FindByName(img,MPI_Group_range_excl_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_range_excl));
	}

	myrtn=RTN_FindByName(img,MPI_Group_range_incl_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_range_incl));
	}

	myrtn=RTN_FindByName(img,MPI_Group_size_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_size));
	}

	myrtn=RTN_FindByName(img,MPI_Group_translate_ranks_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_translate_ranks));
	}

	myrtn=RTN_FindByName(img,MPI_Group_union_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_union));
	}

	myrtn=RTN_FindByName(img,MPI_Intercomm_create_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Intercomm_create));
	}

	myrtn=RTN_FindByName(img,MPI_Intercomm_merge_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Intercomm_merge));
	}

	myrtn=RTN_FindByName(img,MPI_Keyval_create_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Keyval_create));
	}

	myrtn=RTN_FindByName(img,MPI_Keyval_free_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Keyval_free));
	}

	myrtn=RTN_FindByName(img,MPI_Abort_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Abort));
	}

	myrtn=RTN_FindByName(img,MPI_Error_class_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Error_class));
	}

	myrtn=RTN_FindByName(img,MPI_Errhandler_create_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Errhandler_create));
	}

	myrtn=RTN_FindByName(img,MPI_Errhandler_free_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Errhandler_free));
	}

	myrtn=RTN_FindByName(img,MPI_Errhandler_get_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Errhandler_get));
	}

	myrtn=RTN_FindByName(img,MPI_Error_string_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Error_string));
	}

	myrtn=RTN_FindByName(img,MPI_Errhandler_set_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Errhandler_set));
	}

	myrtn=RTN_FindByName(img,MPI_Finalize_FNAME);
	if (RTN_Valid(myrtn))
	{
		cerr<<"MPI_finalize::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Finalize));
	}

	myrtn=RTN_FindByName(img,MPI_Get_processor_name_FNAME);
	if (RTN_Valid(myrtn))
	{
		cerr<<"MPI_Get_processor::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Get_processor_name));
	}
	cerr<<"Checking:"<<MPI_Init_FNAME<<endl;
	myrtn=RTN_FindByName(img,MPI_Init_FNAME);
	if (RTN_Valid(myrtn))
	{
		cerr<<"MPI_Init::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Init));
	}

	myrtn=RTN_FindByName(img,MPI_Init_thread_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Init_thread));
	}

	myrtn=RTN_FindByName(img,MPI_Wtime_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Wtime));
	}

	myrtn=RTN_FindByName(img,MPI_Address_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Address));
	}

	myrtn=RTN_FindByName(img,MPI_Bsend_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Bsend));
	}

	myrtn=RTN_FindByName(img,MPI_Bsend_init_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Bsend_init));
	}

	myrtn=RTN_FindByName(img,MPI_Buffer_attach_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Buffer_attach));
	}

	myrtn=RTN_FindByName(img,MPI_Buffer_detach_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Buffer_detach));
	}

	myrtn=RTN_FindByName(img,MPI_Cancel_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cancel));
	}

	myrtn=RTN_FindByName(img,MPI_Request_free_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Request_free));
	}

	myrtn=RTN_FindByName(img,MPI_Recv_init_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Recv_init));
	}

	myrtn=RTN_FindByName(img,MPI_Send_init_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Send_init));
	}

	myrtn=RTN_FindByName(img,MPI_Get_elements_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Get_elements));
	}

	myrtn=RTN_FindByName(img,MPI_Get_count_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Get_count));
	}

	myrtn=RTN_FindByName(img,MPI_Ibsend_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Ibsend));
	}

	myrtn=RTN_FindByName(img,MPI_Iprobe_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Iprobe));
	}

	myrtn=RTN_FindByName(img,MPI_Irecv_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Irecv));
	}

	myrtn=RTN_FindByName(img,MPI_Irsend_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Irsend));
	}

	myrtn=RTN_FindByName(img,MPI_Isend_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Isend));
	}

	myrtn=RTN_FindByName(img,MPI_Issend_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Issend));
	}

	myrtn=RTN_FindByName(img,MPI_Pack_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Pack));
	}

	myrtn=RTN_FindByName(img,MPI_Pack_size_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Pack_size));
	}

	myrtn=RTN_FindByName(img,MPI_Probe_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Probe));
	}

	myrtn=RTN_FindByName(img,MPI_Recv_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Recv));
	}

	myrtn=RTN_FindByName(img,MPI_Rsend_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Rsend));
	}

	myrtn=RTN_FindByName(img,MPI_Rsend_init_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Rsend_init));
	}

	myrtn=RTN_FindByName(img,MPI_Send_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Send));
	}

	myrtn=RTN_FindByName(img,MPI_Sendrecv_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Sendrecv));
	}

	myrtn=RTN_FindByName(img,MPI_Sendrecv_replace_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Sendrecv_replace));
	}

	myrtn=RTN_FindByName(img,MPI_Ssend_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Ssend));
	}

	myrtn=RTN_FindByName(img,MPI_Ssend_init_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Ssend_init));
	}

	myrtn=RTN_FindByName(img,MPI_Start_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Start));
	}

	myrtn=RTN_FindByName(img,MPI_Startall_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Startall));
	}

	myrtn=RTN_FindByName(img,MPI_Test_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Test));
	}

	myrtn=RTN_FindByName(img,MPI_Testall_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Testall));
	}

	myrtn=RTN_FindByName(img,MPI_Testany_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Testany));
	}

	myrtn=RTN_FindByName(img,MPI_Test_cancelled_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Test_cancelled));
	}

	myrtn=RTN_FindByName(img,MPI_Testsome_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Testsome));
	}

	myrtn=RTN_FindByName(img,MPI_Type_commit_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_commit));
	}

	myrtn=RTN_FindByName(img,MPI_Type_contiguous_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_contiguous));
	}

	myrtn=RTN_FindByName(img,MPI_Type_extent_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_extent));
	}

	myrtn=RTN_FindByName(img,MPI_Type_free_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_free));
	}

	myrtn=RTN_FindByName(img,MPI_Type_hindexed_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_hindexed));
	}

	myrtn=RTN_FindByName(img,MPI_Type_hvector_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_hvector));
	}

	myrtn=RTN_FindByName(img,MPI_Type_indexed_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_indexed));
	}

	myrtn=RTN_FindByName(img,MPI_Type_lb_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_lb));
	}

	myrtn=RTN_FindByName(img,MPI_Type_size_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_size));
	}

	myrtn=RTN_FindByName(img,MPI_Type_struct_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_struct));
	}

	myrtn=RTN_FindByName(img,MPI_Type_ub_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_ub));
	}

	myrtn=RTN_FindByName(img,MPI_Type_vector_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_vector));
	}

	myrtn=RTN_FindByName(img,MPI_Unpack_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Unpack));
	}

	myrtn=RTN_FindByName(img,MPI_Wait_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Wait));
	}

	myrtn=RTN_FindByName(img,MPI_Waitall_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Waitall));
	}

	myrtn=RTN_FindByName(img,MPI_Waitany_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Waitany));
	}

	myrtn=RTN_FindByName(img,MPI_Waitsome_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Waitsome));
	}

	myrtn=RTN_FindByName(img,MPI_Cart_coords_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_coords));
	}

	myrtn=RTN_FindByName(img,MPI_Cart_create_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_create));
	}

	myrtn=RTN_FindByName(img,MPI_Cart_get_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_get));
	}

	myrtn=RTN_FindByName(img,MPI_Cart_map_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_map));
	}

	myrtn=RTN_FindByName(img,MPI_Cart_rank_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_rank));
	}

	myrtn=RTN_FindByName(img,MPI_Cart_shift_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_shift));
	}

	myrtn=RTN_FindByName(img,MPI_Cart_sub_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_sub));
	}

	myrtn=RTN_FindByName(img,MPI_Cartdim_get_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cartdim_get));
	}

	myrtn=RTN_FindByName(img,MPI_Dims_create_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Dims_create));
	}

	myrtn=RTN_FindByName(img,MPI_Graph_create_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_create));
	}

	myrtn=RTN_FindByName(img,MPI_Graph_get_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_get));
	}

	myrtn=RTN_FindByName(img,MPI_Graph_map_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_map));
	}

	myrtn=RTN_FindByName(img,MPI_Graph_neighbors_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_neighbors));
	}

	myrtn=RTN_FindByName(img,MPI_Graph_neighbors_count_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_neighbors_count));
	}

	myrtn=RTN_FindByName(img,MPI_Graphdims_get_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graphdims_get));
	}

	myrtn=RTN_FindByName(img,MPI_Topo_test_FNAME);
	if (RTN_Valid(myrtn))
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Topo_test));
	}
		
}


VOID MpiInstRtn(RTN myrtn)
{
	//the following lists take care of instrumenting 
	// MPI routines in a different way 
	// by using the replace api of PIN
	if(!RTN_Valid(myrtn))
		return;

	string rtn_name=RTN_Name(myrtn);
	if(rtn_name.at(0)=='_')
		rtn_name=rtn_name.substr(1);
	int indx=rtn_name.find_first_of('@');
	if(indx!=string::npos)
		rtn_name=rtn_name.substr(0,indx);	
	cerr<<"MPI Instrumentation"<<rtn_name<<endl;

	if (rtn_name==MPI_Allgather_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Allgather));
	}

	if (rtn_name==MPI_Allgatherv_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Allgatherv));
	}

	if (rtn_name==MPI_Allreduce_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Allreduce));
	}

	if (rtn_name==MPI_Alltoall_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Alltoall));
	}

	if (rtn_name==MPI_Alltoallv_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Alltoallv));
	}

	if (rtn_name==MPI_Barrier_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Barrier));
	}

	if (rtn_name==MPI_Bcast_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Bcast));
	}

	if (rtn_name==MPI_Gather_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Gather));
	}

	if (rtn_name==MPI_Gatherv_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Gatherv));
	}

	if (rtn_name==MPI_Op_create_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Op_create));
	}

	if (rtn_name==MPI_Op_free_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Op_free));
	}

	if (rtn_name==MPI_Reduce_scatter_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Reduce_scatter));
	}

	if (rtn_name==MPI_Reduce_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Reduce));
	}

	if (rtn_name==MPI_Scan_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Scan));
	}

	if (rtn_name==MPI_Scatter_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Scatter));
	}

	if (rtn_name==MPI_Scatterv_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Scatterv));
	}

	if (rtn_name==MPI_Attr_delete_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Attr_delete));
	}

	if (rtn_name==MPI_Attr_get_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Attr_get));
	}

	if (rtn_name==MPI_Attr_put_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Attr_put));
	}

	if (rtn_name==MPI_Comm_compare_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_compare));
	}

	if (rtn_name==MPI_Comm_create_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_create));
	}

	if (rtn_name==MPI_Comm_dup_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_dup));
	}

	if (rtn_name==MPI_Comm_free_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_free));
	}

	if (rtn_name==MPI_Comm_group_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_group));
	}

	if (rtn_name==MPI_Comm_rank_FNAME)
	
	{
		cerr<<"MPI_comm_rank::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_rank));
	}

	if (rtn_name==MPI_Comm_remote_group_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_remote_group));
	}

	if (rtn_name==MPI_Comm_remote_size_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_remote_size));
	}

	if (rtn_name==MPI_Comm_size_FNAME)
	
	{
		cerr<<"MPI_comm_size::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_size));
	}

	if (rtn_name==MPI_Comm_split_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_split));
	}

	if (rtn_name==MPI_Comm_test_inter_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Comm_test_inter));
	}

	if (rtn_name==MPI_Group_compare_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_compare));
	}

	if (rtn_name==MPI_Group_difference_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_difference));
	}

	if (rtn_name==MPI_Group_excl_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_excl));
	}

	if (rtn_name==MPI_Group_free_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_free));
	}

	if (rtn_name==MPI_Group_incl_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_incl));
	}

	if (rtn_name==MPI_Group_intersection_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_intersection));
	}

	if (rtn_name==MPI_Group_rank_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_rank));
	}

	if (rtn_name==MPI_Group_range_excl_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_range_excl));
	}

	if (rtn_name==MPI_Group_range_incl_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_range_incl));
	}

	if (rtn_name==MPI_Group_size_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_size));
	}

	if (rtn_name==MPI_Group_translate_ranks_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_translate_ranks));
	}

	if (rtn_name==MPI_Group_union_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Group_union));
	}

	if (rtn_name==MPI_Intercomm_create_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Intercomm_create));
	}

	if (rtn_name==MPI_Intercomm_merge_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Intercomm_merge));
	}

	if (rtn_name==MPI_Keyval_create_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Keyval_create));
	}

	if (rtn_name==MPI_Keyval_free_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Keyval_free));
	}

	if (rtn_name==MPI_Abort_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Abort));
	}

	if (rtn_name==MPI_Error_class_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Error_class));
	}

	if (rtn_name==MPI_Errhandler_create_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Errhandler_create));
	}

	if (rtn_name==MPI_Errhandler_free_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Errhandler_free));
	}

	if (rtn_name==MPI_Errhandler_get_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Errhandler_get));
	}

	if (rtn_name==MPI_Error_string_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Error_string));
	}

	if (rtn_name==MPI_Errhandler_set_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Errhandler_set));
	}

	if (rtn_name==MPI_Finalize_FNAME)
	
	{
		cerr<<"MPI_finalize::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Finalize));
	}

	if (rtn_name==MPI_Get_processor_name_FNAME)
	
	{
		cerr<<"MPI_Get_processor::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Get_processor_name));
	}
	cerr<<"Checking:"<<MPI_Init_FNAME<<endl;
	if (rtn_name==MPI_Init_FNAME)
	
	{
		cerr<<"MPI_Init::replaced"<<endl;
		fflush(stdout);
		RTN_Replace(myrtn, AFUNPTR(HMPI_Init));
	}

	if (rtn_name==MPI_Init_thread_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Init_thread));
	}

	if (rtn_name==MPI_Wtime_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Wtime));
	}

	if (rtn_name==MPI_Address_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Address));
	}

	if (rtn_name==MPI_Bsend_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Bsend));
	}

	if (rtn_name==MPI_Bsend_init_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Bsend_init));
	}

	if (rtn_name==MPI_Buffer_attach_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Buffer_attach));
	}

	if (rtn_name==MPI_Buffer_detach_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Buffer_detach));
	}

	if (rtn_name==MPI_Cancel_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cancel));
	}

	if (rtn_name==MPI_Request_free_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Request_free));
	}

	if (rtn_name==MPI_Recv_init_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Recv_init));
	}

	if (rtn_name==MPI_Send_init_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Send_init));
	}

	if (rtn_name==MPI_Get_elements_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Get_elements));
	}

	if (rtn_name==MPI_Get_count_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Get_count));
	}

	if (rtn_name==MPI_Ibsend_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Ibsend));
	}

	if (rtn_name==MPI_Iprobe_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Iprobe));
	}

	if (rtn_name==MPI_Irecv_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Irecv));
	}

	if (rtn_name==MPI_Irsend_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Irsend));
	}

	if (rtn_name==MPI_Isend_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Isend));
	}

	if (rtn_name==MPI_Issend_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Issend));
	}

	if (rtn_name==MPI_Pack_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Pack));
	}

	if (rtn_name==MPI_Pack_size_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Pack_size));
	}

	if (rtn_name==MPI_Probe_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Probe));
	}

	if (rtn_name==MPI_Recv_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Recv));
	}

	if (rtn_name==MPI_Rsend_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Rsend));
	}

	if (rtn_name==MPI_Rsend_init_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Rsend_init));
	}

	if (rtn_name==MPI_Send_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Send));
	}

	if (rtn_name==MPI_Sendrecv_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Sendrecv));
	}

	if (rtn_name==MPI_Sendrecv_replace_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Sendrecv_replace));
	}

	if (rtn_name==MPI_Ssend_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Ssend));
	}

	if (rtn_name==MPI_Ssend_init_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Ssend_init));
	}

	if (rtn_name==MPI_Start_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Start));
	}

	if (rtn_name==MPI_Startall_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Startall));
	}

	if (rtn_name==MPI_Test_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Test));
	}

	if (rtn_name==MPI_Testall_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Testall));
	}

	if (rtn_name==MPI_Testany_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Testany));
	}

	if (rtn_name==MPI_Test_cancelled_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Test_cancelled));
	}

	if (rtn_name==MPI_Testsome_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Testsome));
	}

	if (rtn_name==MPI_Type_commit_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_commit));
	}

	if (rtn_name==MPI_Type_contiguous_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_contiguous));
	}

	if (rtn_name==MPI_Type_extent_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_extent));
	}

	if (rtn_name==MPI_Type_free_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_free));
	}

	if (rtn_name==MPI_Type_hindexed_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_hindexed));
	}

	if (rtn_name==MPI_Type_hvector_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_hvector));
	}

	if (rtn_name==MPI_Type_indexed_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_indexed));
	}

	if (rtn_name==MPI_Type_lb_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_lb));
	}

	if (rtn_name==MPI_Type_size_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_size));
	}

	if (rtn_name==MPI_Type_struct_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_struct));
	}

	if (rtn_name==MPI_Type_ub_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_ub));
	}

	if (rtn_name==MPI_Type_vector_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Type_vector));
	}

	if (rtn_name==MPI_Unpack_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Unpack));
	}

	if (rtn_name==MPI_Wait_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Wait));
	}

	if (rtn_name==MPI_Waitall_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Waitall));
	}

	if (rtn_name==MPI_Waitany_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Waitany));
	}

	if (rtn_name==MPI_Waitsome_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Waitsome));
	}

	if (rtn_name==MPI_Cart_coords_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_coords));
	}

	if (rtn_name==MPI_Cart_create_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_create));
	}

	if (rtn_name==MPI_Cart_get_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_get));
	}

	if (rtn_name==MPI_Cart_map_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_map));
	}

	if (rtn_name==MPI_Cart_rank_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_rank));
	}

	if (rtn_name==MPI_Cart_shift_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_shift));
	}

	if (rtn_name==MPI_Cart_sub_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cart_sub));
	}

	if (rtn_name==MPI_Cartdim_get_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Cartdim_get));
	}

	if (rtn_name==MPI_Dims_create_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Dims_create));
	}

	if (rtn_name==MPI_Graph_create_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_create));
	}

	if (rtn_name==MPI_Graph_get_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_get));
	}

	if (rtn_name==MPI_Graph_map_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_map));
	}

	if (rtn_name==MPI_Graph_neighbors_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_neighbors));
	}

	if (rtn_name==MPI_Graph_neighbors_count_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graph_neighbors_count));
	}

	if (rtn_name==MPI_Graphdims_get_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Graphdims_get));
	}

	if (rtn_name==MPI_Topo_test_FNAME)
	
	{
		RTN_Replace(myrtn, AFUNPTR(HMPI_Topo_test));
	}
		
}
#endif