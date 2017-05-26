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
  void Tau_MemMgr_initIfNecessary(void) { return; } 
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



#define TAU_PIN_JIT_MODE 1
//#define TAU_USE_FUNC_NAMES_FOR_START_STOP 1 

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
#endif /* TAU_USE_FUNC_NAMES_FOR_START_STOP */
}

const char * StripPath(const char * path)
{
    const char * file = strrchr(path,'/');
    if (file)
        return file+1;
    else
        return path;
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
    //if (name.find("MPI_") == std::string::npos) {
    if (!((toupper(func[0]) == 'M') && (toupper(func[1]) == 'P') && 
         (toupper(func[2]) == 'I') && (func[3] == '_'))) {
	/* Not an MPI routine */
      if ((line == 0) || path.empty() || 
          (module.find(".so.") != std::string::npos)) {
         TAU_VERBOSE("Not instrumenting: %s\n", func); 
         return;
      }
    } else {
      if ((func[0] == 'm') && (name.find("@plt") != std::string::npos)) { 
        TAU_VERBOSE("Not instrumenting plt: %s\n", func); 
        return; /* do not instrument plts for MPI */
      }
      TAU_VERBOSE("Instrumenting: %s\n", func);
    }

    TAU_ROUTINE * rc = new TAU_ROUTINE;

   

    char buf[1024]; 
    string func_name; 
    if (line && !path.empty()) {
      sprintf(buf, "%d", line);
      func_name = name +string(" [{") + path + string("}{")+buf+string("}]");
    } else {
      func_name = name; 
    }
    
#ifdef TAU_USE_FUNC_NAMES_FOR_START_STOP
    rc->_name  = func_name;
    rc->_image = StripPath(IMG_Name(SEC_Img(RTN_Sec(rtn))).c_str());
#else
    TAU_PROFILER_CREATE(rc->fi, func_name.c_str(), " ", TAU_USER);
#endif 

    rc->_next = RtnList;
    RtnList = rc;

    if (FirstEvent == (TAU_ROUTINE *) 0) { 
      FirstEvent = rc; 
    }
            
    RTN_Open(rtn);
            
    // Insert a call at the entry and exit points of the routine.

#ifdef TAU_PIN_JIT_MODE
    
    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)FunctionEntry, IARG_PTR, rc, IARG_END);
    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)FunctionExit, IARG_PTR, rc, IARG_END);
#else
    if (RTN_IsSafeForProbedInsertion(rtn)) { 
      RTN_InsertCallProbed(rtn, IPOINT_BEFORE, (AFUNPTR)FunctionEntry, IARG_PTR, rc, IARG_END);
      RTN_InsertCallProbed(rtn, IPOINT_AFTER, (AFUNPTR)FunctionExit, IARG_PTR, rc, IARG_END);
    }
    
#endif /* TAU_PIN_JIT_MODE */
    RTN_Close(rtn);
    

    
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

VOID TauPinFinish(INT32 code, VOID *v)
{
  Tau_profile_exit_all_threads(); 
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

    // Register Routine to be called to instrument rtn
    RTN_AddInstrumentFunction(Routine, 0);

    PIN_AddFiniFunction(TauPinFinish, 0);

    // Start the program, never returns
#ifdef TAU_PIN_JIT_MODE 
    PIN_StartProgram();
#else 
    PIN_StartProgramProbed();
#endif /* TAU_PIN_JIT_MODE */
    
    return 0;
}
