/***************************************************************************************************
This file contains the entry point of the pintool. The main function here is actually different from
the traditional main. This becomes the exported api of the dll which will be invoked by PIN. 
****************************************************************************************************/


#include "pin.h"
#include <iostream>
#include<stdio.h>
#include "SpecManager.h"
#include "TraceManager.h"
#define H_BEFORE 0
#define H_AFTER 1
#define TAU_WINDOWS
#define PROFILING_ON
#define TAU_DOT_H_LESS_HEADERS

SpecManager* mySpec;
TraceManager* myTracer;
FILE *sout,*sin,*serr;

INT32 Usage()
{
    cout << "pin -t thisdll -r rule -f file -- executable.exe " << endl;
	myTracer->LogMessage("pin -t thisdll -r rule -f file -- executable.exe\n");
    return -1;
}

VOID HookHandleBefore(RtnTrack* myrt)
{
	//cerr<<"Intercept Before:"<<myrt->rtn<<endl;
	myTracer->BeforeExec(myrt);
	myrt->stage++;
}

VOID HookHandleAfter(RtnTrack* myrt)
{
	if(myrt->stage<=0)
	{
		myrt->stage=0;
		return;
	}
	//cerr<<"Intercept After:"<<myrt->rtn<<endl;
	myTracer->AfterExec(myrt);
	myrt->stage--;	
}

VOID HookHandle(RtnTrack* myrt)
{
	cerr<<"Hookhandle"<<endl;
	if(myrt->stage==0)
	{	
		cerr<<"Intercept Before:"<<myrt->rtn<<endl;
		fflush(stdout);
		myTracer->BeforeExec(myrt);
	}else
	{
		cerr<<"Intercept After:"<<myrt->rtn<<endl;
		fflush(stdout);
		myTracer->AfterExec(myrt);		
	}
	delete myrt;
}


VOID ExitHandle(INT32 code, VOID *v)
{
	//simply call into tracemanager
    DBG_TRACE(code);
	myTracer->EndTrace();
}

VOID ImgInstrumentor( IMG img, VOID *v )
{	
	//simply call into tracemanager
	DBG_TRACE(IMG_Name(img));
	myTracer->InstApply(img);
}

bool InitLogging()
{
	//sout=freopen( "taupin_out", "w", stdout);
	//if(sout==NULL)
	//	return false;
	serr=freopen( "taupin_msg", "w", stderr );
	if(serr==NULL)
		return false;
	return true;
}


int main(int argc, char *argv[])
{			
	InitLogging();
    // in the command line or the command line is invalid
	//for(int i=0;i<argc;i++)
	//	cerr<<argv[i]<<endl;

    //initialize the argument and spec objects
	Arguments* myargs= new Arguments(argv,argc);
	//myargs->PrintArgs();
	//myargs->PrintRules();
    
    //recompose the argument 
	//so that PIN doesnt know about the arguments 
	//PINTAU needs otherwise PIN fails 
	char ** argvs=myargs->GetPinArgV();

	//initialize PIN 
	if( PIN_Init(myargs->GetPinArgC(),myargs->GetPinArgV()))
    {
        return Usage();
    }
    cerr<<"Initialized PIN"<<endl;

	//Specmanager created here
	mySpec=new SpecManager(myargs);
    cerr<<"Specmanager Initialized"<<endl;
    mySpec->PrintInsts();

	//Tracemanager Created here
    TraceManager* myTracer=new TraceManager(mySpec);
	cerr<<"Tracemanager Initialized"<<endl;

	//this is required to get the symbols 
    PIN_InitSymbols();
    cerr<<"PIN_Initsymbols"<<endl;
	//register a routine instrumentation callback
    IMG_AddInstrumentFunction( ImgInstrumentor, 0 );

    //registering Exit hook callback
    PIN_AddFiniFunction(ExitHandle, 0);
    cerr<<  "Starting target program after setting up instrumentation" << endl;

	SetupProfileFile();
    PIN_StartProgram();
	//if we want probe mode we need to do the following
	//PIN_StartProgramProbed();
    
    return 0;
}

/*eof*/
