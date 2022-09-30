#include <TAU.h>
#include <Profile/TauBfd.h>

#include <starpu_profiling_tool.h>
#include <starpu.h>

#include <sstream>
#include <iostream>
#include <dlfcn.h> // link with -ldl -rdynamic a

#include <stdio.h>

#warning "Compiling StarPU support"

static bool init_done = false;

#define TAU_SET_EVENT_NAME(event_name, str) event_name << str; 
std::map<int,std::string> dev_type;


#ifdef TAU_BFD
#define HAVE_DECL_BASENAME 1
#  if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#    include <demangle.h>
#  endif /* HAVE_GNU_DEMANGLE */
// Add these definitions because the Binutils comedians think all the world uses autotools
#ifndef PACKAGE
#define PACKAGE TAU
#endif
#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION 2.25
#endif
#  include <bfd.h>
#endif /* TAU_BFD */
#define TAU_INTERNAL_DEMANGLE_NAME(name, dem_name)  dem_name = cplus_demangle(name, DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE | DMGL_TYPES); \
        if (dem_name == NULL) { \
          dem_name = name; \
        } \

/*
 *-----------------------------------------------------------------------------
 * Simple hash table to map function addresses to region names/identifier
 *-----------------------------------------------------------------------------
 */

struct StarPUHashNode
{
    StarPUHashNode() : fi(NULL)
  { }

  TauBfdInfo info;		///< Filename, line number, etc.
    FunctionInfo * fi;		///< Function profile information
    std::string location;
    std::string function;
};

struct StarPUHashTable : public TAU_HASH_MAP<unsigned long, StarPUHashNode*>
{
  StarPUHashTable() {
    Tau_init_initializeTAU();
  }
  virtual ~StarPUHashTable() {
    Tau_destructor_trigger();
  }
};

static StarPUHashTable & TheHashTable()
{
  static StarPUHashTable htab;
  return htab;
}

static void Tau_delete_hash_table(void) {
  // clear the hash map to eliminate memory leaks
  StarPUHashTable mytab = StarPUHashTable();
  for ( TAU_HASH_MAP<unsigned long, StarPUHashNode*>::iterator it = mytab.begin(); it != mytab.end(); ++it ) {
    StarPUHashNode * node = it->second;
    /* if (node) {
        	if (node->info) {
        	free (node->info);
            }
            }*/
    delete node;
  }
  mytab.clear();
  Tau_delete_bfd_units();
}

static TAU_HASH_MAP<unsigned long, StarPUHashNode*>& TheLocalHashTable(){
  static thread_local TAU_HASH_MAP<unsigned long, StarPUHashNode*> lhtab;
  return lhtab;
}

static tau_bfd_handle_t & TheBfdUnitHandle()
{
  static tau_bfd_handle_t StarPUbfdUnitHandle = TAU_BFD_NULL_HANDLE;
  if (StarPUbfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    RtsLayer::LockEnv();
    if (StarPUbfdUnitHandle == TAU_BFD_NULL_HANDLE) {
      StarPUbfdUnitHandle = Tau_bfd_registerUnit();
    }
    RtsLayer::UnLockEnv();
  }
  return StarPUbfdUnitHandle;
}


/*
 * Get symbol table by using BFD
 */
static void issueBfdWarningIfNecessary()
{
#ifndef TAU_BFD
  static bool warningIssued = false;
  if (!warningIssued) {
#ifndef __APPLE__
    fprintf(stderr,"TAU Warning: Comp_gnu - "
        "BFD is not available during TAU build. Symbols may not be resolved!\n");
    fflush(stderr);
#endif
    warningIssued = true;
  }
#endif
}

static void updateHashTable(unsigned long addr, const char *funcname)
{
  StarPUHashNode * hn = TheLocalHashTable()[addr];
  if (!hn) {
    RtsLayer::LockDB();
    hn = TheHashTable()[addr];
    if (!hn) {
      hn = new StarPUHashNode;
      TheHashTable()[addr] = hn;
    }
    TheLocalHashTable()[addr] = hn;
    RtsLayer::UnLockDB();
  }
  //  hn->info.funcname = funcname;
  hn->function = std::string( funcname );
}

extern "C" void Tau_profile_exit_all_threads(void);

static int executionFinished = 0;
void runOnExitStarPU()
{
  executionFinished = 1;
  Tau_profile_exit_all_threads();

  // clear the hash map to eliminate memory leaks
  StarPUHashTable & mytab = TheHashTable();
  for ( TAU_HASH_MAP<unsigned long, StarPUHashNode*>::iterator it = mytab.begin(); it != mytab.end(); ++it ) {
  	StarPUHashNode * node = it->second;
    if (node != NULL && node->fi) {
#ifndef TAU_TBB_SUPPORT
// At the end of a TBB program, it crashes here.
		//delete node->fi;
#endif /* TAU_TBB_SUPPORT */
	}
    delete node;
  }
  mytab.clear();

#ifdef TAU_BFD
  Tau_delete_bfd_units();
#endif
  Tau_destructor_trigger();
}


std::pair<std::string,std::string> funcname( void* fun_ptr ){
    TauBfdInfo dinfo;
    
    tau_bfd_handle_t & StarPUbfdUnitHandle = TheBfdUnitHandle();
    
    unsigned long addr = Tau_convert_ptr_to_unsigned_long( fun_ptr );
    StarPUHashNode * node;

    node = TheLocalHashTable()[addr];
    if( !node ) {

        RtsLayer::LockEnv();
        node = TheHashTable()[addr];
        if( !node ) {
            node = new StarPUHashNode;
            TheHashTable()[addr] = node;
        }
          
        TheLocalHashTable()[addr] = node;
        
        Tau_bfd_resolveBfdInfo( StarPUbfdUnitHandle, addr, dinfo );
        RtsLayer::UnLockDB();
           
        std::stringstream ssret;
        ssret << dinfo.filename << ":" << dinfo.lineno;
        node->location = ssret.str();
        node->function = std::string( dinfo.funcname );
    }
    
    //    std::cout << "res " << node->function << "  " << node->location << "  " << toto << " " << titi << " " << false << std::endl;
    
    return  std::pair<std::string,std::string>( node->function, node->location );
}

/* All the callbacks are handled by this function */

void myfunction_cb( struct starpu_prof_tool_info* prof_info,  union starpu_prof_tool_event_info* event_info, struct starpu_prof_tool_api_info* api_info ){

    std::stringstream event_name;
    std::stringstream info;
    const char* name;
    
    int tag = 0;

    //std::cout << "Callback " << prof_info->event_type << " called" << std::endl;
    
    std::pair<std::string, std::string> fun;
    if( nullptr != prof_info->fun_ptr ){
        fun = funcname( prof_info->fun_ptr );
    } else {
        fun.first = "";
        fun.second = "";
    }
    //    std::cout << "Fun : " << fun.first << " " << fun.second << std::endl;
        
    switch(  prof_info->event_type ) {
    case starpu_prof_tool_event_init:
        Tau_create_top_level_timer_if_necessary(); // ???
        TAU_SET_EVENT_NAME( event_name, ">StarPU" );
        break;
    case starpu_prof_tool_event_terminate:
        TAU_SET_EVENT_NAME( event_name, "<StarPU" );
        break;
    case starpu_prof_tool_event_init_begin:
        TAU_SET_EVENT_NAME( event_name, ">StarPU init" );
        break;
    case starpu_prof_tool_event_init_end:
        TAU_SET_EVENT_NAME( event_name, "<StarPU init" );
        break;
    case starpu_prof_tool_event_driver_init:
        TAU_SET_EVENT_NAME( event_name, ">StarPU driver" );
        info << " [{" << dev_type[prof_info->driver_type] << ":" << prof_info->device_number  << "}]";
       break;       
    case starpu_prof_tool_event_driver_deinit:
        TAU_SET_EVENT_NAME( event_name, "<StarPU driver" );
        info << " [{" << dev_type[prof_info->driver_type] << ":" << prof_info->device_number  << "}]";
        break;
    case starpu_prof_tool_event_driver_init_start:
        TAU_SET_EVENT_NAME( event_name, ">StarPU driver init" );
       info << " [{" << dev_type[prof_info->driver_type] << ":" << prof_info->device_number  << "}]";
        break;
    case starpu_prof_tool_event_driver_init_end:
        TAU_SET_EVENT_NAME( event_name, "<StarPU driver init" );
        info << " [{" << dev_type[prof_info->driver_type] << ":" << prof_info->device_number  << "}]";
       break;
    case starpu_prof_tool_event_start_cpu_exec:
    case starpu_prof_tool_event_start_gpu_exec:
        TAU_SET_EVENT_NAME( event_name, ">StarPU exec " );
        info << fun.first.c_str() << " [{" << dev_type[prof_info->driver_type] << ":" << prof_info->device_number << "} function " << prof_info->fun_ptr << " { " << fun.second.c_str() << " }]";
        break;
    case starpu_prof_tool_event_end_cpu_exec:
    case starpu_prof_tool_event_end_gpu_exec:
        TAU_SET_EVENT_NAME( event_name, "<StarPU exec " );
        info << fun.first.c_str() << "[{" << dev_type[prof_info->driver_type] << ":" << prof_info->device_number << "} function " << prof_info->fun_ptr << " { " << fun.second.c_str() << " }]";
        break;
    case starpu_prof_tool_event_start_transfer:
        TAU_SET_EVENT_NAME( event_name, ">StarPU_transfer" );
            //        if( TauEnv_get_track_message() ){
        //            std::cout<<"toto"<<std::endl;
            TAU_TRACE_SENDMSG( tag, prof_info->memnode, prof_info->bytes_transfered );
            //        }
        info << " [{ memnode " << prof_info->memnode << " }]";
        break;
    case starpu_prof_tool_event_end_transfer:
        TAU_SET_EVENT_NAME( event_name, "<StarPU_transfer" );
        // TAU_TRACE_RECVMSG( tag, prof_info->memnode, prof_info->bytes_transfered );
        info << " [{ memnode " << prof_info->memnode << " }]";
        break;
    default:
        std::cout <<  "Unknown callback " <<  prof_info->event_type << std::endl;
        break;
    }
    
    event_name << info.str();
    // std::cout << "Event: " << event_name.str().c_str() << " " << event_name.str()[0] << std::endl;
    // std::cout << "Event1: " << event_name.str().c_str() << std::endl;

    if ( event_name.str()[0] == '>') {
        TAU_VERBOSE("START>>%s\n", &(event_name.str()[1]) );
        TAU_START( &(event_name.str()[1]) );
    }  else if ( event_name.str()[0] == '<' ) {
        TAU_VERBOSE("STOP<<%s\n", &(event_name.str()[1]) );
        // TAU_STOP( &(event_name.str()[1]) );
        Tau_global_stop();
    } else {
        TAU_VERBOSE("event_name = %s\n", &(event_name.str()[0]) );
    }
}

/* Library initialization: callback registration */
extern "C" {
 void starpu_prof_tool_library_register( starpu_prof_tool_entry_register_func reg, starpu_prof_tool_entry_register_func unreg){

    dev_type[starpu_prof_tool_driver_cpu] = "CPU";
    dev_type[starpu_prof_tool_driver_gpu] = "GPU";

    enum  starpu_prof_tool_command info = starpu_prof_tool_command_reg;
    reg( starpu_prof_tool_event_init_begin, &myfunction_cb, info );
    reg( starpu_prof_tool_event_init_end, &myfunction_cb, info );
    reg( starpu_prof_tool_event_init, &myfunction_cb, info );
    reg( starpu_prof_tool_event_terminate, &myfunction_cb, info );
    reg( starpu_prof_tool_event_driver_init, &myfunction_cb, info );
    reg( starpu_prof_tool_event_driver_deinit, &myfunction_cb, info );
    reg( starpu_prof_tool_event_driver_init_start, &myfunction_cb, info );
    reg( starpu_prof_tool_event_driver_init_end, &myfunction_cb, info );
    reg( starpu_prof_tool_event_start_cpu_exec, &myfunction_cb, info );
    reg( starpu_prof_tool_event_end_cpu_exec, &myfunction_cb, info );
    reg( starpu_prof_tool_event_start_gpu_exec, &myfunction_cb, info );
    reg( starpu_prof_tool_event_end_gpu_exec, &myfunction_cb, info );
    reg( starpu_prof_tool_event_start_transfer, &myfunction_cb, info );
    reg( starpu_prof_tool_event_end_transfer, &myfunction_cb, info );

    RtsLayer::LockDB();
     if( !init_done ) {
        tau_bfd_handle_t & bfdUnitHandle = TheBfdUnitHandle();
        issueBfdWarningIfNecessary();
        Tau_bfd_processBfdExecInfo(bfdUnitHandle, updateHashTable);
        init_done = true;
    }
    RtsLayer::UnLockDB();
    
    atexit( runOnExitStarPU );
}
}
#undef TAU_SET_EVENT_NAME


