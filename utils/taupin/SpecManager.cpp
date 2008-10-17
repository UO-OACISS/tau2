/***************************************************************************************************
This file contains the Specmanger module of TAUPIN. 
This module interacts with Tracemanager. Some of the code is left commnted with the view that 
it might be used later. 
****************************************************************************************************/


#include "SpecManager.h"
//#include<Pattern.h>
#include "TraceManager.h"
#include<ctype.h>

/*
This was an initial plan to use the routine instrumentation for MPIs too 
however now it's handled by image instrumentation and routine replacement
but we might light to turn this on in future*/
  string mpiRtns[]={"MPI_Init!1",
  "MPI_Get_processor_name!1",
  "MPI_Comm_rank!1",
  "MPI_Comm_size!1",
  "MPI_Finalize!1",
  "END" };

string my_mpiRtns[]={"MPI_Allgather",
  "MPI_Allgatherv",
  "MPI_Allreduce",
  "MPI_Alltoall",
  "MPI_Alltoallv",
  "MPI_Barrier",
  "MPI_Bcast",
  "MPI_Gather",
  "MPI_Gatherv",
  "MPI_Op_create",
  "MPI_Op_free",
  "MPI_Reduce_scatter",
  "MPI_Reduce",
  "MPI_Scan",
  "MPI_Scatter",
  "MPI_Scatterv",
  "MPI_Attr_delete",
  "MPI_Attr_get",
  "MPI_Attr_put",
  "MPI_Comm_compare",
  "MPI_Comm_create",
  "MPI_Comm_dup",
  "MPI_Comm_free",
  "MPI_Comm_group",
  "MPI_Comm_rank",
  "MPI_Comm_remote_group",
  "MPI_Comm_remote_size",
  "MPI_Comm_size",
  "MPI_Comm_split",
  "MPI_Comm_test_inter",
  "MPI_Group_compare",
  "MPI_Group_difference",
  "MPI_Group_excl",
  "MPI_Group_free",
  "MPI_Group_incl",
  "MPI_Group_intersection",
  "MPI_Group_rank",
  "MPI_Group_range_excl",
  "MPI_Group_range_incl",
  "MPI_Group_size",
  "MPI_Group_translate_ranks",
  "MPI_Group_union",
  "MPI_Intercomm_create",
  "MPI_Intercomm_merge",
  "MPI_Keyval_create",
  "MPI_Keyval_free",
  "MPI_Abort",
  "MPI_Error_class",
  "MPI_Errhandler_create",
  "MPI_Errhandler_free",
  "MPI_Errhandler_get",
  "MPI_Error_string",
  "MPI_Errhandler_set",
  "MPI_Finalize",
  "MPI_Get_processor_name",
  "MPI_Init",
  "MPI_Init_thread",
  "MPI_Wtime",
  "MPI_Address",
  "MPI_Bsend",
  "MPI_Bsend_init",
  "MPI_Buffer_attach",
  "MPI_Buffer_detach",
  "MPI_Cancel",
  "MPI_Request_free",
  "MPI_Recv_init",
  "MPI_Send_init",
  "MPI_Get_elements",
  "MPI_Get_count",
  "MPI_Ibsend",
  "MPI_Iprobe",
  "MPI_Irecv",
  "MPI_Irsend",
  "MPI_Isend",
  "MPI_Issend",
  "MPI_Pack",
  "MPI_Pack_size",
  "MPI_Probe",
  "MPI_Recv",
  "MPI_Rsend",
  "MPI_Rsend_init",
  "MPI_Send",
  "MPI_Sendrecv",
  "MPI_Sendrecv_replace",
  "MPI_Ssend",
  "MPI_Ssend_init",
  "MPI_Start",
  "MPI_Startall",
  "MPI_Test",
  "MPI_Testall",
  "MPI_Testany",
  "MPI_Test_cancelled",
  "MPI_Testsome    ",///****
  "MPI_Type_commit",
  "MPI_Type_contiguous",
  "MPI_Type_extent",
  "MPI_Type_free",
  "MPI_Type_hindexed",
  "MPI_Type_hvector",
  "MPI_Type_indexed",
  "MPI_Type_lb",
  "MPI_Type_size",
  "MPI_Type_struct",
  "MPI_Type_ub",
  "MPI_Type_vector",
  "MPI_Unpack",
  "MPI_Wait",
  "MPI_Waitall",
  "MPI_Waitany",
  "MPI_Waitsome",
  "MPI_Cart_coords",
  "MPI_Cart_create",
  "MPI_Cart_get",
  "MPI_Cart_map",
  "MPI_Cart_rank",
  "MPI_Cart_shift",
  "MPI_Cart_sub",
  "MPI_Cartdim_get",
  "MPI_Dims_create",
  "MPI_Graph_create",
  "MPI_Graph_get",
  "MPI_Graph_map",
  "MPI_Graph_neighbors",
  "MPI_Graph_neighbors_count",
  "MPI_Graphdims_get",
  "MPI_Topo_test", 
  "END" };


RtnInfo::RtnInfo(string ImgName, string RtnName, int ProfileFlag )
{
		  ImgName=ImgName;
		  RtnName=RtnName;
		  ProfileFlag=ProfileFlag;
}

int RtnInfo::GetFlag()
{
	return this->ProfileFlag;
}

string RtnInfo::GetImg()
{
	return this->ImgName;
}

string RtnInfo::GetRtn()
{
	return this->RtnName;
}

RtnInfo::~RtnInfo()
{
}

ImageInfo::ImageInfo(string ImageName, int ProfileFlag)
{
		ImageName=ImageName;
		ProfileFlag=ProfileFlag;
}

void ImageInfo::AddRtn(string rtn)
{
	RtnInfo * robj=new RtnInfo(this->ImgeName,rtn,this->ProfileFlag);
	//put it in the indexed map
	this->rtns[rtn]=robj;
}

void ImageInfo::PrintRtns()
{
	map<string,RtnInfo*>::iterator rit=this->rtns.begin();
	while(rit!=rtns.end())
	{
		cout<<rit->first<<endl;
		rit++;
	}
}

string ImageInfo::GetImage()
{
	return this->ImgeName;
}

RtnInfo * ImageInfo::FindRtn(string rtn)
{
	//find a routine from the map 
	//it should be fast 
	map<string,RtnInfo*>::iterator rtnit=this->rtns.find(rtn);
	if(rtnit==this->rtns.end())
	{
		return NULL;
	}

	return rtnit->second;	
}

ImageInfo::~ImageInfo()
{
		//delete rtns;
}



SpecManager::SpecManager(void)
{
}

SpecManager::SpecManager(string specfile)
{
		SpecFile=specfile;
}

SpecManager::SpecManager(Arguments *args)
{
	this->CurArgs=args;
	list<RuleSet*> rchain= this->CurArgs->GetRuleIt();
	list<RuleSet*>::iterator rit=rchain.begin();

	//populate instrumentaton specification from 
	//argument parser
	while(rit!=rchain.end())
	{
		RuleSet * robj=*rit;
		this->rules.insert(this->rules.begin(),robj);
		rit++;
	}
	//*****this->InitDefaultRules();
}

SpecManager::~SpecManager(void)
{  
}

bool SpecManager::IsMpiRtn(string rtn_name)
{
	int i=0;
	while(my_mpiRtns[i]!="END")
	{
		if(my_mpiRtns[i]==rtn_name)
			return true;
		i++;
	}
	return false;
}

int ParseFlag(string flag)
{
	int flagint=0;
	int prevpos=0,curpos=0;
	int flaglen=flag.length();
	//the flag can come with , seperated anding
	while(prevpos<flaglen)
	{
		curpos=flag.find(',',prevpos);
		if(curpos<flaglen)
		{
			flagint&=atoi(flag.substr(prevpos,flaglen-prevpos).c_str());
			break;
		}
		//perfrom AND operation of the flags
		flagint=flagint&atoi(flag.substr(prevpos,curpos).c_str());
		prevpos=curpos+2;
	}
	return flagint;
}

void SpecManager::AddImage(string img,int flag)
{
	ImageInfo* imobj = new ImageInfo(img,flag);
	this->CurImage=imobj;
	//add the image to the imagemap 
	//easy to search later
	this->imagemap[img]=imobj;
}

void SpecManager::AddRtn(string rtn,string img)
{
   //might have to find the object
	this->CurImage->AddRtn(rtn);
}

//recursive call that matches string
bool SpecManager::MatchRecurse(char *txt,int txt_len ,int txt_indx, char * expr, int expr_len , int expr_indx)
{
	char tc=txt[txt_indx];
	char ec=expr[expr_indx];
	bool status=false;
	//case insensitive convert them all tolower
	tc=tolower(tc);
	ec=tolower(ec);
	if(ec!='*' && ec!='?' && tc!=ec)
		return false;
	
	if(txt_indx==txt_len-1)
	{
		if( expr_len-1 ==expr_indx)
			return true;
		else if(expr_len-2 == expr_indx && expr[expr_indx+1]=='*')
			return true;
		else 
			return false;
	}

	//now switch for recursion
	switch(ec)
	{
	case '*':
		//* based transition can happen once only
		status=MatchRecurse(txt,txt_len ,txt_indx+1, expr,expr_len ,expr_indx+1);
		if(status) return true;
		//* can transition to more inputs 
		status=MatchRecurse(txt,txt_len ,txt_indx+1, expr,expr_len ,expr_indx);
		return status;
	case '?':
		//place holder needs to transition to next regular expression
		status=MatchRecurse(txt,txt_len ,txt_indx+1, expr,expr_len ,expr_indx+1);
		return status;
	default:
		//it's normal character and need to transition to next expression and next input
		status=MatchRecurse(txt,txt_len ,txt_indx+1, expr,expr_len ,expr_indx+1);
		return status;
	}	
}

//wrapper to the above recursive call
bool SpecManager::Match(string my_txt, string my_expr)
{
	char *txt=(char*)my_txt.c_str();
	char *expr=(char*)my_expr.c_str();
	int txt_len=(int)my_txt.length();
	int expr_len=(int)my_expr.length();
	//simply call the recursive matcher
	bool status=MatchRecurse(txt,txt_len ,0, expr,expr_len ,0);
	return status;
}

bool SpecManager::InstImage(string image)
{
	DBG_TRACE("Comes here");
	DBG_TRACE(image);

	//find the ruleset corresponding to the image
	map<string,RuleSet*>::iterator drit=this->imgIndx.find(image);	
	/*map<string,RuleSet*>::iterator myit=this->imgIndx.begin();
	if(myit!=this->imgIndx.end())
	{
		DBG_TRACE("#########");
		DBG_TRACE(myit->first);
	}*/

	if(drit!=this->imgIndx.end())
	{		
		RuleSet *robj=drit->second;
		//add the image to be profiled 
		this->AddImage(image,robj->getProfileFlag());
		return true;
	}
	
	list<RuleSet*>::iterator rit=this->rules.begin();
	while(rit!=this->rules.end())
	{
		
		RuleSet* robj=*rit;
		//Check if the image matches with specification
		//if((robj->GetImagePat(),image))
		if(Match(image,robj->GetImagePat()))
		{
			DBG_TRACE("TRUE returning");
			return true;
		}
	
		//do regular expression here
		rit++;
	}
	DBG_TRACE("FALSE returning");
	return false;			
}

bool SpecManager::InstRtn(string rtn, string image)
{
	DBG_TRACE(rtn);
	DBG_TRACE(image);
	//find the Rule with routine as key
	map<string,RuleSet*>::iterator drit=this->rtnIndx.find(rtn);
	while(drit!=this->rtnIndx.end())
	{
			RuleSet * robj=drit->second;
			if(robj->GetImagePat().compare(image)==0)
			{
				//add to the routine list
				this->AddRtn(rtn,image);
				DBG_TRACE("Returning True");
				return true;
			}
			drit++;
	}
    
	list<RuleSet*>::iterator rit=this->rules.begin();
	//now scan through all the ruleset
	while(rit!=this->rules.end())
	{
		RuleSet* robj=*rit;
        //if the routine matches with specification return true
		//if(Pattern::matches(robj->GetRtnPat(),rtn))
		if(Match(rtn,robj->GetRtnPat()))
		{
			DBG_TRACE("TRUE returning");
			return true;
		}
		//do regular expression here
		rit++;
	}
	DBG_TRACE("Returning False");
	return false;
}

int SpecManager::GetProfileFlag(string rtn, string image)
{
	//give rtn and image retrieve the profile option
	map<string,ImageInfo*>::iterator imgit=this->imagemap.find(image);
	if(imgit==this->imagemap.end())
	{
		return -1;
	}
	ImageInfo* myimg=imgit->second;
	//from the image object retrieve the rtninfo
	RtnInfo* myrtn=myimg->FindRtn(rtn);
	if(myrtn==NULL)
	{
		return -1;
	}
	//return the flag from the rtninfo
	return myrtn->GetFlag();
}

void SpecManager::PrintInsts()
{
	cout<<"instrumented info"<<endl;
	map<string,ImageInfo*>::iterator imit=this->imagemap.begin();
	//print all instrumentable routines
	while(imit!=this->imagemap.end())
	{
		ImageInfo* imobj=imit->second;
		cout<<imobj->GetImage()<<endl;
		imobj->PrintRtns();
		imit++;
	}
}

void SpecManager::InitDefaultRules()
{
	int i=0;
	string rule=mpiRtns[i];
	//scan through all the element in the array 
	//untill END is hit 
	while(rule.compare("END")!=0)
	{
		string rtn;
		string flag;
		//parse the rule fiels here
		int prevpos=rule.find('!');
		if(prevpos<0) return;
		rtn=rule.substr(0,prevpos);
		prevpos++;		
		int curpos=rule.length();
		if(curpos<0)return;
		flag=rule.substr(prevpos,curpos);
		if(rtn.length()<=0 || flag.length()<=0) return;
		//create rule object using the parsed values 
		RuleSet* robj=new RuleSet("msmpi.dll",rtn ,atoi(flag.c_str()));
		//insert it to the maps indexed with rtn and image names 
		this->rtnIndx[rtn]=robj;
		this->imgIndx["msmpi.dll"]=robj;
		rule=mpiRtns[++i];
	}	
}