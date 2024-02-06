#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdio.h>
#include <sys/utsname.h>
#include <iostream>
#include <string.h>
#include "pin.H"
#include <math.h>
#include <dlfcn.h>
#include <ctype.h>

extern "C" {
  void Tau_sampling_finalize_if_necessary(int tid) { return ; }
  void Tau_sampling_suspend(int tid){ return; }
  int atexit(void (*function)(void)) { return 0; }
  void Tau_MemMgr_free(int tid, void *addr, size_t size) { return ; }
  void * Tau_MemMgr_malloc(int tid, size_t size) { return NULL; }
  void Tau_sampling_resume(int tid) { return ; }
  bool Tau_MemMgr_initIfNecessary(void) { return false; } 
  void Tau_sampling_init_if_necessary(void) { return; }
  void Tau_MemMgr_finalizeIfNecessary(void) { return; }
  char *cuserid(char *string) { return NULL;} 
  mode_t umask(mode_t cmask) { return cmask; }
  int   clock_gettime(clockid_t  clk_id,  struct  timespec *tp) { return 0; }
  int uname(struct utsname *name) { return 0; } 
  int __isoc99_sscanf(const char *str, const char *format, ...) { return 0; }
  int getrusage(int who, struct rusage *usage) { return 0; }
} 

int Tau_sampling_event_stop(int tid, double *stopTime) { return 0; }
void Tau_sampling_event_start(int tid, void **addresses) { return; }

extern "C" void Tau_profile_exit_all_threads(void);



/* If you comment out the JIT MODE flag, TAU uses PIN's probe mode */
#define TAU_PIN_JIT_MODE 1
//#define TAU_USE_FUNC_NAMES_FOR_START_STOP 1 
/* If you comment this, it uses FunctionInfo pointer instead of strings */

#include <Profile/TauPin.h>
#include <TAU.h> 

typedef struct TauRtnStruct
{
#ifdef TAU_USE_FUNC_NAMES_FOR_START_STOP
    string _name;
    string _image;
#endif /* TAU_USE_FUNC_NAMES_FOR_START_STOP */
    struct TauRtnStruct * _next;
    void *fi; 
} TAU_ROUTINE;

// Linked list of instruction counts for each routine
TAU_ROUTINE * RtnList = 0;
TAU_ROUTINE * FirstEvent = 0; 

extern "C" void Tau_start(const char *); 
extern "C" void Tau_stop(const char *); 

void FunctionEntry(TAU_ROUTINE *rc) {

#ifdef TAU_USE_FUNC_NAMES_FOR_START_STOP
  const char *name = rc->_name.c_str(); 
  TAU_VERBOSE("ENTER: %s\n", name);
  TAU_START(name);
#else
  FunctionInfo *f = (FunctionInfo *) rc->fi; 
  TAU_PROFILER_START(f); 
  //cout <<"Enter: " << f<<" " <<f->GetName()<<endl;
#endif /* TAU_USE_FUNC_NAMES_FOR_START_STOP */
  

}
void FunctionExit(TAU_ROUTINE *rc) {
#ifdef TAU_USE_FUNC_NAMES_FOR_START_STOP
  const char *name = rc->_name.c_str(); 
  TAU_VERBOSE("EXIT : %s\n", name);
  TAU_STOP(name);
#else
  FunctionInfo *f = (FunctionInfo *) rc->fi; 
  TAU_PROFILER_STOP(f); 
  //cout <<"Exit:  "<<f<<" "<<f->GetName()<<endl;
#endif /* TAU_USE_FUNC_NAMES_FOR_START_STOP */
}

void CommRankExit(TAU_ROUTINE *rc, int* rank) { 
  TAU_VERBOSE("Return value from MPI_Comm_rank is %d\n", *rank); 
  TAU_PROFILE_SET_NODE(*rank); 
  FunctionExit(rc); 
}

const char * StripPath(const char * path)
{
    const char * file = strrchr(path,'/');
    if (file)
        return file+1;
    else
        return path;
}


// Pin calls this function to clean up and write profiles
VOID TauPinFinish(INT32 code, VOID *v)
{
  Tau_profile_exit_all_threads(); 
}


// Pin calls this function every time a new rtn is executed
VOID Routine(RTN rtn, VOID *v)
{
    
    // Allocate a counter for this routine
    string path, module, name;
    INT32 line;
    bool mpi_lib = false;
    PIN_GetSourceLocation(RTN_Address(rtn), NULL, &line, &path);
    name = RTN_Name(rtn);
    const char *func  = name.data(); 
    module = StripPath(IMG_Name(SEC_Img(RTN_Sec(rtn))).c_str());
    const string secname= SEC_Name(RTN_Sec(rtn)); 
    if (secname.find(".plt") != std::string::npos) {
      return; // no need to instrument plt stubs.
      //cout <<"func = "<<func<<" secname = "<<secname<<endl;
    }
    if((name.find("PMPI_") == std::string::npos) && 
       (name.find("MPI_") == std::string::npos)) { 
       /* it doesn't have MPI and PMPI in its name */
      if ((line == 0) || path.empty() || 
          (module.find(".so.") != std::string::npos)) {
         TAU_VERBOSE("Not instrumenting: %s\n", func); 
         return;
      }
    }

    TAU_ROUTINE * rc = new TAU_ROUTINE;

   

    char buf[1024]; 
    string func_name; 
    if (line && !path.empty()) {
      snprintf(buf, sizeof(buf),  "%d", line);
      func_name = name +string(" [{") + path + string("}{")+buf+string("}]");
    } else {
      func_name = name; 
    }
    
#ifdef TAU_USE_FUNC_NAMES_FOR_START_STOP
    rc->_name  = func_name;
    rc->_image = StripPath(IMG_Name(SEC_Img(RTN_Sec(rtn))).c_str());
#else
    TAU_PROFILER_CREATE(rc->fi, func_name.c_str(), " ", TAU_USER);
    //cout <<"Creating profiler for "<<func_name<<" " <<rc->fi<<endl; 
#endif 

    rc->_next = RtnList;
    RtnList = rc;

    if (FirstEvent == (TAU_ROUTINE *) 0) { 
      FirstEvent = rc; 
    }
            
#ifdef TAU_PIN_JIT_MODE
    RTN_Open(rtn);
#endif /* TAU_PIN_JIT_MODE */
            
    // Insert a call at the entry and exit points of the routine.

#ifdef TAU_PIN_JIT_MODE
    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)FunctionEntry, IARG_PTR, rc, IARG_END);
    if (func_name.find("MPI_Comm_rank") !=  std::string::npos) { 
      TAU_VERBOSE("Found MPI_Comm_rank\n"); 
      
      RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)CommRankExit, IARG_PTR, rc, IARG_FUNCARG_ENTRYPOINT_VALUE, 1, IARG_END);
    } else {
      RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)FunctionExit, IARG_PTR, rc, IARG_END);
    }
#else

    PROTO pr_entry = PROTO_Allocate(PIN_PARG(void ), CALLINGSTD_DEFAULT, "FunctionEntry", PIN_PARG(TAU_ROUTINE *), PIN_PARG_END()); 
    PROTO pr_exit = PROTO_Allocate(PIN_PARG(void ), CALLINGSTD_DEFAULT, "FunctionExit", PIN_PARG(TAU_ROUTINE *), PIN_PARG_END()); 
    PROTO pr_comm_exit = PROTO_Allocate(PIN_PARG(void ), CALLINGSTD_DEFAULT, "CommRankExit", PIN_PARG(TAU_ROUTINE *), PIN_PARG(int *), PIN_PARG_END()); 

    PROTO pr_main_exit = PROTO_Allocate(PIN_PARG(void ), CALLINGSTD_DEFAULT, "main", PIN_PARG_END()); 
    if (RTN_IsSafeForProbedInsertion(rtn)) { 
      RTN_InsertCallProbed(rtn, IPOINT_BEFORE, (AFUNPTR)FunctionEntry, IARG_PTR, rc, IARG_PROTOTYPE, pr_entry,  IARG_END);
      if (func_name.find("MPI_Comm_rank") !=  std::string::npos) { 
        if (func_name.find("MPI_Comm_rank_f") == std::string::npos) { /* Confirmed it is C */
          TAU_VERBOSE("Found MPI_Comm_rank\n"); 
          RTN_InsertCallProbed(rtn, IPOINT_AFTER, (AFUNPTR)CommRankExit, IARG_PTR, rc, IARG_FUNCARG_ENTRYPOINT_VALUE, 1, IARG_PROTOTYPE, pr_comm_exit, IARG_END);
        }
      } else {
        RTN_InsertCallProbed(rtn, IPOINT_AFTER, (AFUNPTR)FunctionExit, IARG_PTR, rc, IARG_PROTOTYPE, pr_exit,  IARG_END);
      }

      if (func_name.find("main") != std::string::npos) { 
        RTN_InsertCallProbed(rtn, IPOINT_AFTER, (AFUNPTR)TauPinFinish, IARG_UINT32, 0, IARG_PTR, 0, IARG_PROTOTYPE, pr_main_exit, IARG_CALL_ORDER, 250, IARG_END);

      }
    }
    
#endif /* TAU_PIN_JIT_MODE */
#ifdef TAU_PIN_JIT_MODE
    RTN_Close(rtn);
#endif /* TAU_PIN_JIT_MODE */
    
}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

INT32 Usage()
{
    cerr << "mpirun -np <n> tau_exec -pin ./app"<<endl;
    cerr << "This tool instruments the application using TAU"<<endl;
    return -1;
}


typedef int (*CommRankT) (int, int *); 
/* ===================================================================== */
/* TauNewWrapperCommRank                                                 */
/* ===================================================================== */
int NewTauWrapperCommRank(CommRankT orgFuncptr, UINT32 arg0, int *arg1, ADDRINT returnIp) {
  int ret; 
  ret = orgFuncptr(arg0, arg1); 
  int r = *arg1; 
  TAU_VERBOSE("NewTauWrapperCommRank returns %d\n", r); 
  TAU_PROFILE_SET_NODE(r); 
}

/* ===================================================================== */
/* ImageLoad                                                             */
/* ===================================================================== */
VOID ImageLoad(IMG img, VOID *v) {

    TAU_VERBOSE("Image loaded: %s\n", IMG_Name(img).c_str());


}


VOID ImageUnload(IMG img, void *v) {
    TAU_VERBOSE("Image unloaded: %s\n", IMG_Name(img).c_str());

}


/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char * argv[])
{
    // Initialize symbol table code, needed for rtn instrumentation
    PIN_InitSymbols();

    // Initialize pin
    if (PIN_Init(argc, argv)) return Usage();


    IMG_AddInstrumentFunction(ImageLoad, 0);
    IMG_AddUnloadFunction(ImageUnload, 0);

    // Register Routine to be called to instrument rtn
    RTN_AddInstrumentFunction(Routine, 0);


    // Start the program, never returns
#ifdef TAU_PIN_JIT_MODE 
    PIN_AddFiniFunction(TauPinFinish, 0);
    PIN_StartProgram();
#else 
    PIN_StartProgramProbed();
#endif /* TAU_PIN_JIT_MODE */
    
    return 0;
}
