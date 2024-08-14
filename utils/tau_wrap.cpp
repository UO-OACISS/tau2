/****************************************************************************
 **                      TAU Portable Profiling Package                     **
 **                      http://www.cs.uoregon.edu/research/paracomp/tau    ** 
 *****************************************************************************
 **    Copyright 2007                                                       **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **      File            : tau_wrap.cpp                                    **
 **      Description     : Generates a wrapper library for external pkgs   **
 **                        for instrumentation with TAU.                   **
 **      Author          : Sameer Shende                                   **
 **      Contact         : sameer@cs.uoregon.edu sameer@paratools.com      **
 **      Documentation   :                                                 **
 ***************************************************************************/ 

/* Headers */
#include <stdio.h>
#include <ctype.h>
#include <string.h> 
#include <stdlib.h>
#if (!defined(TAU_WINDOWS))
#include <unistd.h>
#endif //TAU_WINDOWS

#ifdef _OLD_HEADER_
# include <fstream.h>
# include <set.h>
# include <algo.h>
# include <sstream.h>
# include <deque.h>
#else
# include <fstream> 
# include <algorithm>
# include <set> 
# include <list>
# include <string>
# include <sstream>
# include <deque>
using namespace std;
#endif
#include "pdbAll.h"
#include "tau_datatypes.h"



/* defines */
#ifdef TAU_WINDOWS
#define TAU_DIR_CHARACTER '\\' 
#else
#define TAU_DIR_CHARACTER '/' 
#endif /* TAU_WINDOWS */

/* Function call interception types:
 * runtime interception: bar remains bar 
 * preprocessor interception: bar becomes tau_bar
 * wrapper library interception: bar becomes __wrap_bar
 */
#define RUNTIME_INTERCEPT  1
#define PREPROC_INTERCEPT  0
#define WRAPPER_INTERCEPT -1

/* Known UPC environments */
#define UPC_UNKNOWN 0
#define UPC_BERKELEY 1
#define UPC_GNU 2
#define UPC_CRAY 3
#define UPC_XLUPC 4

/* UPC environment */
int upc_env = UPC_UNKNOWN;

/* Note: order is important here */
char const * upc_excluded_functions[][5] = {
    { /* UPC_UNKNOWN */
        NULL
    },
    { /* UPC_BERKELEY */
        "_upcr_alloc",
        "upcri_append_srcloc",
        "upcri_barrier_init",
        NULL
    },
    { /* UPC_GNU */
        NULL,
    },
    { /* UPC_CRAY */
        NULL,
    },
    { /* UPC_XLUPC */
        NULL,
    }
};


//#define DEBUG 1

/* For selective instrumentation */
extern int processInstrumentationRequests(char *fname);
extern bool instrumentEntity(const string& function_name);
extern bool processFileForInstrumentation(const string& file_name);
extern bool isInstrumentListEmpty(void);

/* Prototypes for selective instrumentation */
extern bool addFileInstrumentationRequests(PDB& p, pdbFile *file, vector<itemRef*> & itemvec);


/* Globals */
bool memory_flag = false;   /* by default, do not insert malloc.h in instrumented C/C++ files */
bool strict_typing = false; /* by default unless --strict option is used. */
bool shmem_wrapper = false; /* by default unless --shmem option is used. */
bool pshmem_use_underscore_instead_of_p = false; /* by default unless --pshmem_use_underscore_instead_of_p option is used. */


struct FunctionSignatureInfo
{
  FunctionSignatureInfo(pdbRoutine * r) :
    shmem_fortran_interface(false),
    shmem_len_argcount(0),
    shmem_pe_argcount(0),
    shmem_cond_argcount(0),
    func(r->name()),
    proto(r->name())
  { }

  // For shmem wrapping
  bool shmem_fortran_interface;
  int shmem_len_argcount;
  int shmem_pe_argcount;
  int shmem_cond_argcount;

  // For upc wrapping
  // ...

  string func;
  string funcfort;
  string proto;
  string returntypename;
  string funchandle;
  string rcalledfunc;
  string funcarg;
  string funcargfort;
};


///////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////

/* -------------------------------------------------------------------------- */
/* -- Get a list of instrumentation points for a C program ------------------ */
/* -------------------------------------------------------------------------- */
/* Create a vector of items that need action: such as BODY_BEGIN, RETURN etc.*/
static void getCReferencesForWrapper(vector<itemRef*> & itemvec, PDB& pdb, pdbFile *file) 
{
  /* moved selective instrumentation file processing here */
  if (!isInstrumentListEmpty()) {
    /* there are finite instrumentation requests, add requests for this file */
    addFileInstrumentationRequests(pdb, file, itemvec);
  }
}

static bool isExcluded(pdbRoutine *r) {
    if (r->signature()->hasEllipsis()) {
        // For a full discussion of why vararg functions are difficult to wrap
        // please see: http://www.swig.org/Doc1.3/Varargs.html#Varargs
        return true;
    }
    for(char const ** ptr=upc_excluded_functions[upc_env]; *ptr; ++ptr) {
        if (r->name() == *ptr)
            return true;
    }
    if(r->name().find("shmem_int16_wait_until") != string::npos ||
       r->name().find("shmem_int16_wait") != string::npos ||
       r->name().find("shmem_iget16") != string::npos)
	    return true;
    return false;
}

static bool isReturnTypeVoid(pdbRoutine *r)
{
  string const & rname = r->signature()->returnType()->name();
  return ((rname.compare(0, 4, "void") == 0) &&
          (rname.find("*") == string::npos));
}

static bool doesRoutineNameContainGet(string const & rname)
{
  size_t namelen = rname.size();
  return (((rname.find("get") != string::npos) || 
          ((rname[namelen-2] == '_') && (rname[namelen-1] == 'g')))
          && (rname.find("name") == string::npos
          && rname.find("version") == string::npos));
}

static bool doesRoutineNameContainPut(string const & rname)
{
  size_t namelen = rname.size();
  return ((rname.find("put") != string::npos) || 
          ((rname[namelen-2] == '_') && (rname[namelen-1] == 'p')));
}

/* Fetch and operate operations include swap, fadd and finc */
static bool doesRoutineNameContainFetchOp(string const & rname)
{
  return ((rname.find("swap") != string::npos) || 
          (rname.find("fadd") != string::npos) ||
          (rname.find("finc") != string::npos));
}

/* Fetch and operate operations include swap, fadd and finc */
static bool doesRoutineNameContainCondFetchOp(string const & rname)
{
  return (rname.find("cswap") != string::npos); 
}

static bool isArgumentScalar(string argtype) {
  return (argtype.find("*"));
}

static char const * getMultiplierString(string const & rname) 
{
  static char const * names[] = {
    // List is searched in order, first match returns
    "char", "short", "int",
    "longlong", "longdouble", "long", 
    "double", "float", "16", "32", "64", "128",
    "4", "8",
    (char*)0 /* End of list marker */
  };
  static char const * values[] = {
    "sizeof(char)*", "sizeof(short)*", "sizeof(int)*",
    "sizeof(long long)*", "sizeof(long double)*", "sizeof(long)*", 
    "sizeof(double)*", "sizeof(float)*", "2*", "4*", "8*", "16*",
    "4*", "8*",
    (char*)0 /* End of list marker */
  };

  for(int i=0; names[i]; ++i) {
    if(rname.find(names[i]) != string::npos) {
      return values[i];
    }
  }
  return "";
}

static string upc_mythread()
{
  switch (upc_env) {
    case UPC_BERKELEY: return "upcr_mythread()";
    case UPC_CRAY: return "MYTHREAD";
    case UPC_GNU: return "MYTHREAD";
    case UPC_XLUPC: return "MYTHREAD";
    default: return "MYTHREAD";
  }
}

static string upc_threadof(string const & shared)
{
  switch (upc_env) {
    case UPC_BERKELEY: return "upcr_threadof_shared(" + shared + ")";
    case UPC_CRAY: return "__real_upc_threadof(" + shared + ")";
    case UPC_GNU: return "__real_upc_threadof(" + shared + ")";
    case UPC_XLUPC: return "__real_upc_threadof(" + shared + ")";
    default: return "upc_threadof(" + shared + ")";
  } 
}

static string upc_threads()
{
  switch (upc_env) {
    case UPC_BERKELEY: return "upcr_threads()";
    case UPC_CRAY: return "THREADS";
    case UPC_GNU: return "THREADS";
    case UPC_XLUPC: return "THREADS";
    default: return "THREADS";
  }
}

void printUPCMessageBeforeRoutine(pdbRoutine * r, ofstream & impl, FunctionSignatureInfo sig)
{
  string const & rname = r->name();

  bool isPut = false;
  bool isGet = false;
  bool isCpy = false;
  bool isSig = false;

  // FIXME: list functions not supported at this time
  if ((rname.find("_vlist") != string::npos) ||
      (rname.find("_ilist") != string::npos)) {
    return;
  }
  // FIXME: strided functions not supported at this time
  if (rname.find("strided") != string::npos) {
    return;
  }
  // FIXME: semephore functions not supported at this time
  if (rname.find("_sem_") != string::npos) {
    return;
  }

  if (rname.find("_memput") != string::npos) {
    isPut = true;
    if (rname.find("_signal") != string::npos) {
      isSig = true;
    }
  } else if (rname.find("_memget") != string::npos) {
    isGet = true;
  } else if (rname.find("_memcpy") != string::npos) {
    isCpy = true;
  } else if (rname.find("_memset") != string::npos) {
    isPut = true;
  }

  if (isGet) {
    impl << "  TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, " 
         << upc_mythread() << ", a3, " << upc_threadof("a2") << ");" << endl;
  } else if (isPut) {
    if (isSig) {
      // This is unsafe.... Maybe in future map the semephore to a tag?
      // In any case, we need support for _sem_wait for this to work.
      //impl << "  TAU_TRACE_SENDMSG((int)a4, upcr_threadof_shared(a1), a3);" << endl;
    } else {
      impl << "  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, " << upc_threadof("a1") << ", a3);" << endl;
    }
  } else if (isCpy) {
    impl << "  size_t dst_thread = " << upc_threadof("a1") << ";\n"
         << "  size_t src_thread = " << upc_threadof("a2") << ";\n"
         << "  size_t my_thread = " << upc_mythread() << ";\n"
         << "  if (my_thread == src_thread) {\n"
         << "    TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, dst_thread, a3);\n"
         << "  } else {\n"
         << "    TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, dst_thread, a3, src_thread);\n"
         << "  }\n"
         << endl;
  }

}

void  printUPCMessageAfterRoutine(pdbRoutine * r, ofstream & impl,  FunctionSignatureInfo sig)
{
  string const & rname = r->name();

  bool isPut = false;
  bool isGet = false;
  bool isCpy = false;
  bool isSig = false;

  // FIXME: list functions not supported at this time
  if ((rname.find("_vlist") != string::npos) ||
      (rname.find("_ilist") != string::npos)) {
    return;
  }
  // FIXME: strided functions not supported at this time
  if (rname.find("strided") != string::npos) {
    return;
  }
  // FIXME: semephore functions not supported at this time
  if (rname.find("_sem_") != string::npos) {
    return;
  }

  if (rname.find("_memput") != string::npos) {
    isPut = true;
    if (rname.find("_signal") != string::npos) {
      isSig = true;
    }
  } else if (rname.find("_memget") != string::npos) {
    isGet = true;
  } else if (rname.find("_memcpy") != string::npos) {
    isCpy = true;
  } else if (rname.find("_memset") != string::npos) {
    isPut = true;
  }
  
  if (isGet) {
    impl << "  TAU_TRACE_RECVMSG(TAU_UPC_TAGID, " << upc_threadof("a2") << ", a3);" << endl;
  } else if (isPut && !isSig) {
    impl << "  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, " << upc_mythread() << ", a3, " 
         << upc_threadof("a1") << ");" << endl;
  } else if (isCpy) {
    impl << "  if (my_thread == src_thread) {\n"
         << "    TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, my_thread, a3, dst_thread);\n"
         << "  } else {\n"
         << "    TAU_TRACE_RECVMSG(TAU_UPC_TAGID, src_thread, a3);\n"
         << "  }\n"
         << endl;
  }
}


void printShmemMessageBeforeRoutine(pdbRoutine *r, ofstream& impl, FunctionSignatureInfo sig)
{
  int len_argument_no = sig.shmem_len_argcount;
  int pe_argument_no = sig.shmem_pe_argcount;
  bool fortran_interface = sig.shmem_fortran_interface;
  string const & rname = r->name();
  char length_string[1024];
  char processor_arg[256];

  if (fortran_interface) {
    snprintf(processor_arg, sizeof(processor_arg),  "(*a%d)", pe_argument_no);
  } else {
    snprintf(processor_arg, sizeof(processor_arg),  "a%d", pe_argument_no);
  }

  char const * multiplier_string = getMultiplierString(rname);
#ifdef DEBUG
  printf("Multiplier string = %s\n", multiplier_string);
#endif /* DEBUG */
  if (len_argument_no != 0) {
    if (fortran_interface) {
      snprintf(length_string, sizeof(length_string),  "%s (*a%d)", multiplier_string, len_argument_no);
    } else {
      snprintf(length_string, sizeof(length_string),  "%sa%d", multiplier_string, len_argument_no);
    }
  } else {
    snprintf(length_string, sizeof(length_string),  "%s1", multiplier_string);
  }

  if (doesRoutineNameContainGet(rname) || doesRoutineNameContainFetchOp(rname)) {
#ifdef DEBUG
    cout << "Routine name " << rname << " contains Get variant" << endl;
#endif /* DEBUG */
    impl <<"  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), "<<length_string<<", "<<processor_arg<<");"<<endl;
  }
  if (doesRoutineNameContainPut(rname)) {
#ifdef DEBUG
    cout << "Routine name " << rname << " contains Put variant" << endl;
#endif /* DEBUG */
    impl <<"  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, "<<processor_arg<<", "<<length_string<<");"<<endl;
  }
  if(rname.find("barrier_all") != string::npos) {
    impl << "  TAU_TRACE_BARRIER_ALL_START(TAU_SHMEM_TAGID_NEXT);" << endl;
  } else if(rname.find("barrier") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_BARRIER, a1, a2, a3, 0, 0, -1);" << endl;
  } else if(rname.find("broadcast32") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_BROADCAST, a5, a6, a7, 4*a3, 4*a3, a5);" << endl;
  } else if(rname.find("broadcast4") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_BROADCAST, a5, a6, a7, 4*a3, 4*a3, a4);" << endl;
  } else if(rname.find("broadcast64") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_BROADCAST, a5, a6, a7, 8*a3, 8*a3, a4);" << endl;
  } else if(rname.find("broadcast8") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_BROADCAST, a5, a6, a7, 8*a3, 8*a3, a4);" << endl;
  } else if(rname.find("fcollect4") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHERV, a4, a5, a6, a3*4, a3*4, -1);" << endl;
  } else if(rname.find("fcollect32") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHERV, a4, a5, a6, a3*4, a3*4, -1);" << endl;
  } else if(rname.find("fcollect8") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHERV, a4, a5, a6, a3*8, a3*8, -1);" << endl;
  } else if(rname.find("fcollect64") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHERV, a4, a5, a6, a3*8, a3*8, -1);" << endl;
  } else if(rname.find("collect4") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER, a4, a5, a6, a3*4, a3*4, -1);" << endl;
  } else if(rname.find("collect32") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER, a4, a5, a6, a3*4, a3*4, -1);" << endl;
  } else if(rname.find("collect8") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER, a4, a5, a6, a3*8, a3*8, -1);" << endl;
  } else if(rname.find("collect64") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER, a4, a5, a6, a3*8, a3*8, -1);" << endl;
  } else if(rname.find("to_all") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLREDUCE, a4, a5, a6, 0, 0, -1);" << endl;
  } else if(rname.find("alltoall32") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLTOALL, a4, a5, a6, a3*4*a6, a3*4, -1);" << endl;
  } else if(rname.find("alltoall64") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_BEGIN(TAU_SHMEM_TAGID_NEXT, TAU_TRACE_COLLECTIVE_TYPE_ALLTOALL, a4, a5, a6, a3*8*a6, a3*8, -1);" << endl;
  }
}

void  printShmemMessageAfterRoutine(pdbRoutine *r, ofstream& impl, FunctionSignatureInfo sig)
{
  int len_argument_no = sig.shmem_len_argcount;
  int pe_argument_no = sig.shmem_pe_argcount;
  int cond_argument_no = sig.shmem_cond_argcount;
  bool fortran_interface = sig.shmem_fortran_interface;
  string const & rname = r->name();
  char length_string[1024];
  char processor_arg[256];
  char cond_string[1024];
  bool is_it_a_get = false;
  bool is_it_a_fetchop = false;
  bool is_it_a_cond_fetchop = false;
  bool is_it_a_put = false;

  if (fortran_interface) {
    snprintf(processor_arg, sizeof(processor_arg),  "(*a%d)", pe_argument_no);
  } else {
    snprintf(processor_arg, sizeof(processor_arg),  "a%d", pe_argument_no);
  }

  char const * multiplier_string = getMultiplierString(rname);
#ifdef DEBUG
  printf("Multiplier string = %s\n", multiplier_string);
#endif /* DEBUG */
  if (len_argument_no != 0) {
    if (fortran_interface) {
      snprintf(length_string, sizeof(length_string),  "%s (*a%d)", multiplier_string, len_argument_no);
    } else {
      snprintf(length_string, sizeof(length_string),  "%sa%d", multiplier_string, len_argument_no);
    }
  } else {
    snprintf(length_string, sizeof(length_string),  "%s1", multiplier_string);
  }
  is_it_a_get = doesRoutineNameContainGet(rname);
  is_it_a_fetchop = doesRoutineNameContainFetchOp(rname);
  is_it_a_cond_fetchop = doesRoutineNameContainCondFetchOp(rname);

  if ((rname.find("shmem_init") != string::npos) ||
      (rname.find("start_pes") != string::npos)) {
      impl<<"Tau_set_usesSHMEM(1);"<<endl;
      impl<<"TauTraceOTF2InitShmem_if_necessary();"<<endl;
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
      impl << "  tau_totalnodes(1,__real__num_pes());"<<endl;
      impl << "  TAU_PROFILE_SET_NODE(__real__my_pe());"<<endl;
#else
      impl << "  tau_totalnodes(1,__real_shmem_n_pes());"<<endl;
      impl << "  TAU_PROFILE_SET_NODE(__real_shmem_my_pe());"<<endl;
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  }

  if (is_it_a_get || is_it_a_fetchop ) { /* Get */
#ifdef DEBUG
    cout << "Routine name " << rname << " contains Get variant" << endl;
#endif /* DEBUG */
    impl <<"  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, "<<processor_arg<<", "<<length_string<<");"<<endl;
  }
  if (is_it_a_cond_fetchop || is_it_a_fetchop) { /* add condition */
    if (is_it_a_cond_fetchop && (cond_argument_no == 0)) {
#ifdef DEBUG
      cout << "WARNING: in fetchop function " << rname << ", cond_argument_no is 0???" << endl;
#endif /* DEBUG */
    }
    string indent(""); /* no indent by default */
    bool isVoid = isReturnTypeVoid(r);
    if (is_it_a_cond_fetchop && !isVoid) {
      indent=string("  ");
      if (fortran_interface) {
        snprintf(cond_string, sizeof(cond_string),  "  if (retval == (*a%d)) { ", cond_argument_no);
      } else {
        snprintf(cond_string, sizeof(cond_string),  "  if (retval == a%d) { ", cond_argument_no);
      }
      impl <<cond_string<<endl;; 
    }
    impl <<indent<<"  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, "<<processor_arg<<", "<<length_string<<");"<<endl;
    impl <<indent<<"  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), "<<length_string<<", "<<processor_arg<<");"<<endl;
    if (is_it_a_cond_fetchop && !isVoid) {
      impl<<indent<<"}"<<endl;
    }
  }
  if (doesRoutineNameContainPut(rname)) {
#ifdef DEBUG
    cout << "Routine name " << rname << " contains Put variant" << endl;
#endif /* DEBUG */
    impl <<"  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), "<<length_string<<", "<<processor_arg<<");"<<endl;
  }
  if(rname.find("barrier_all") != string::npos) {
    impl << "  TAU_TRACE_BARRIER_ALL_END(TAU_SHMEM_TAGID);" << endl;
  } else if(rname.find("barrier") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_BARRIER, a1, a2, a3, 0, 0, -1);" << endl;
  } else if(rname.find("broadcast32") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_BROADCAST, a5, a6, a7, 4*a3, 4*a3, a5);" << endl;
  } else if(rname.find("broadcast4") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_BROADCAST, a5, a6, a7, 4*a3, 4*a3, a4);" << endl;
  } else if(rname.find("broadcast64") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_BROADCAST, a5, a6, a7, 8*a3, 8*a3, a4);" << endl;
  } else if(rname.find("broadcast8") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_BROADCAST, a5, a6, a7, 8*a3, 8*a3, a4);" << endl;
  } else if(rname.find("fcollect4") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHERV, a4, a5, a6, a3*4, a3*4, -1);" << endl;
  } else if(rname.find("fcollect32") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHERV, a4, a5, a6, a3*4, a3*4, -1);" << endl;
  } else if(rname.find("fcollect8") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHERV, a4, a5, a6, a3*8, a3*8, -1);" << endl;
  } else if(rname.find("fcollect64") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHERV, a4, a5, a6, a3*8, a3*8, -1);" << endl;
  } else if(rname.find("collect4") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER, a4, a5, a6, a3*4, a3*4, -1);" << endl;
  } else if(rname.find("collect32") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER, a4, a5, a6, a3*4, a3*4, -1);" << endl;
  } else if(rname.find("collect8") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER, a4, a5, a6, a3*8, a3*8, -1);" << endl;
  } else if(rname.find("collect64") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER, a4, a5, a6, a3*8, a3*8, -1);" << endl;
  } else if(rname.find("to_all") != string::npos) {
    impl << "  TAU_TRACE_RMA_COLLECTIVE_END(TAU_SHMEM_TAGID, TAU_TRACE_COLLECTIVE_TYPE_ALLREDUCE, a4, a5, a6, 0, 0, -1);" << endl;
  }
}

void printFunctionNameInOutputFile(pdbRoutine *r, ofstream& impl, char const * prefix, FunctionSignatureInfo & sig)
{
  sig.func = r->name() + "(";
  sig.funcfort = r->name() + "(";
  sig.proto = r->name() + "(";
  if(shmem_wrapper) {
    sig.rcalledfunc = r->name() + "_handle (";
    sig.funchandle = "_handle) (";
  }
  else {
    sig.rcalledfunc = "(*" + r->name() + "_h) (";
    sig.funchandle = "_h) (";
  }
  sig.funcarg = "(";
  sig.funcargfort = "(";

  pdbGroup const * grp = r->signature()->returnType()->isGroup();
  if (grp) { 
    sig.returntypename = grp->name();
  } else {
    sig.returntypename = r->signature()->returnType()->name();
    if (upc_env && (sig.returntypename.compare(0, 10, "shared[1] ") == 0)) {
      sig.returntypename.replace(0, 10, "shared   ");
    }
  }
  impl << "extern " << sig.returntypename << prefix << sig.func; 

#ifdef DEBUG
  cout <<"Examining "<<r->name()<<endl;
  cout <<"Return type :"<<sig.returntypename<<endl;
#endif /* DEBUG */

  int argcount = 1;
  pdbType::argvec const & av = r->signature()->arguments();
  for(pdbType::argvec::const_iterator argsit = av.begin();
      argsit != av.end(); argsit++, argcount++)
  {
#ifdef DEBUG
    cout <<"Argument " << argsit->name() <<" Type " << argsit->type()->name() << endl;
#endif /* DEBUG */

    if (shmem_wrapper) {
      if ((argsit->name().compare("len") == 0) || 
          (argsit->name().compare("nelems") == 0)) {
#ifdef DEBUG
        printf("Argcount = %d for len\n", argcount); 
#endif /* DEBUG */
        sig.shmem_len_argcount = argcount; 
        if (argsit->type()->kind() == pdbItem::TY_PTR) {
          sig.shmem_fortran_interface = true;
        }
      }
      if (argsit->name().compare("pe") == 0) {
#ifdef DEBUG
        printf("Argcount = %d for pe\n", argcount); 
#endif /* DEBUG */
        sig.shmem_pe_argcount = argcount; 
        if (argsit->type()->kind() == pdbItem::TY_PTR) {
          sig.shmem_fortran_interface = true;
        }
      }
      if ((argsit->name().compare("match") == 0) || 
          (argsit->name().compare("cond") == 0)) {
#ifdef DEBUG
        printf("Argcount = %d for match/cond\n", argcount); 
#endif /* DEBUG */
        sig.shmem_cond_argcount = argcount; 
      }
    }

    if (argcount != 1) { /* not a startup */
      sig.func.append(", ");
      sig.funcfort.append(", ");
      sig.proto.append(", ");
      sig.rcalledfunc.append(", ");
      sig.funchandle.append(", ");
      sig.funcarg.append(", ");
      sig.funcargfort.append(", ");
      impl<<", ";
    }

    char number[256];
    snprintf(number, sizeof(number),  "%d", argcount);
    const pdbGroup *gr;
    string argtypename;
    string argtypenamefort;
    if ((gr = argsit->type()->isGroup()) != 0) {
      argtypename = gr->name();
    } else {
      argtypename = argsit->type()->name();
    }

    /* headers sometimes have struct members in the argument name:
     *    const struct upc_filevec {upc_off_t offset;size_t len;}*
     * We need to erase everything between the two curly braces */
    int pos1 = argtypename.find("{");
    int pos2 = argtypename.find("}");
    if (pos1 != string::npos && pos2 != string::npos) {
#ifdef DEBUG
      cout <<"BEFORE ARG type="<<argtypename<<endl;
#endif /* DEBUG */
      argtypename.erase(pos1, pos2-pos1+1);
#ifdef DEBUG
      cout <<"AFTER  ARG type="<<argtypename<<endl;
#endif /* DEBUG */
    }

    if ((upc_env == UPC_GNU) && argtypename.compare(0, 10, "shared[1] ") == 0) {
      argtypename.replace(0, 10, "shared ");
    }

    argtypenamefort = argtypename;
    if(shmem_wrapper) {
      if((argtypenamefort.compare(0, 3, "int") == 0) &&
	((argtypenamefort.length() >= 7) && (argtypenamefort.compare(3,4, "16_t") != 0) && (argtypenamefort.compare(3,4, "32_t") != 0) && (argtypenamefort.compare(3,4, "64_t") != 0))) {
        argtypenamefort.erase(0, 3);
        argtypenamefort.insert(0, "SHMEM_FINT");
      }
      if(argtypenamefort.compare(0, 6, "size_t") == 0) {
        argtypenamefort.erase(0, 6);
        argtypenamefort.insert(0, "SHMEM_FINT");
      }
    }
    int pos3 = argtypenamefort.find("*");
    if(pos3 == string::npos) {
      sig.funcargfort.append(argtypenamefort + " *");
      sig.funcfort.append("*a" + string(number));
    }
    else {
      sig.funcargfort.append(argtypenamefort);
      sig.funcfort.append("a" + string(number));
    }

    sig.func.append("a" + string(number));
    sig.proto.append(argtypename + " a" + string(number));
    sig.funchandle.append(argtypename);
    sig.rcalledfunc.append(" a" + string(number));
    sig.funcarg.append(argtypename);

    /* We need to handle the case int (*)[] separately generating 
       int (*a1)[]  instead of int (*)[] a1 in the impl file */
    const char *found;
    const char *examinedarg = argtypename.c_str();
    if ((found = strstr(examinedarg, "(*)")) != 0) {
      found += 2; /* Reach ) */
      //printf("found = %s diff = found  - examinedarg = %d \n", found, found - examinedarg);
      int i;
      for (i=0; i < found - examinedarg; i++) {
        //printf("Printing %c\n", examinedarg[i]);
        impl << examinedarg[i];
      }
      impl<<"a"<<number;
      sig.funcarg.append("a" + string(number));
      sig.funcargfort.append("a" + string(number));
      for(i=found - examinedarg; i < strlen(examinedarg); i++) {
        //printf("after number Printing %c\n", examinedarg[i]);
        impl << examinedarg[i];
      }
    } else {
      /* print: for (int a, double b), this prints "int" */
      if (upc_env) {
        /* upc headers sometimes have struct members in the argument name:
           const struct upc_filevec {upc_off_t offset;size_t len;}* 
           We need to erase everything between the two curly braces */
        size_t pos1 = argtypename.find("{");
        size_t pos2 = argtypename.find("}");
        if (pos1 != string::npos && pos2 != string::npos) {
#ifdef DEBUG
          cout <<"BEFORE ARG type="<<argtypename<<endl;
#endif /* DEBUG */
          argtypename.erase(pos1, pos2-pos1+1);
#ifdef DEBUG
          cout <<"AFTER  ARG type="<<argtypename<<endl;
#endif /* DEBUG */
        }
      } 
      impl<<argtypename<<" ";
      /* print: for (int a, double b), this prints "a1" in int a1, */
      impl<<"a"<<number;
      sig.funcarg.append(" a" + string(number));
      sig.funcargfort.append(" a" + string(number));
    }
    if (r->signature()->hasEllipsis()) {
      //printf("Has ellipsis...\n");
      impl<<", ...";
    }
  }
  sig.func.append(")");
  sig.funcfort.append(")");
  sig.proto.append(")");
  sig.rcalledfunc.append(")");
  sig.funcarg.append(")");
  sig.funcargfort.append(")");
  impl<<") ";
}

void printRoutineInOutputFile(pdbRoutine *r, ofstream& header, ofstream& impl, string& group_name, int runtime, string& runtime_libname)
{
  FunctionSignatureInfo sig(r);

  string rname = r->name();
  string protoname = r->name() + "_p";
  string macro("#define ");
  string retstring("    return;");
  string dltext;

  if (isExcluded(r)) {
    impl <<"#warning \"TAU: Not generating wrapper for function "<<r->name()<<"\""<<endl;
    cout <<"TAU: Not generating wrapper for function "<<r->name()<<endl;
    return;
  }

  impl << endl; 
  impl << "/**********************************************************"<<endl;
  impl << "   "<<r->name()<< endl;
  impl << " **********************************************************/"<<endl<<endl;

  bool isVoid = isReturnTypeVoid(r);
  if (runtime == WRAPPER_INTERCEPT) { /* linker-based instrumentation */
    printFunctionNameInOutputFile(r, impl, "  __real_", sig);
    impl <<";"<<endl;
  }

  if (runtime == RUNTIME_INTERCEPT) { /* linker-based instrumentation */
    printFunctionNameInOutputFile(r, impl, "  __wrap_", sig);
    impl <<";"<<endl;
  }

  char const * prefix = " ";
//  if (!shmem_wrapper) {
    switch (runtime) {
      case RUNTIME_INTERCEPT: 
        /* for runtime interception, put a blank, the name stays the same*/
        if(shmem_wrapper)
          prefix = "  __real_";
        else
          prefix = "  ";
        break;
      case PREPROC_INTERCEPT:
        /* for standard preprocessor redirection, bar becomes tau_bar */
        prefix = "  tau_";
        break;
      case WRAPPER_INTERCEPT:
        /* for wrapper library interception, it becomes __wrap_bar */
        prefix = "  __wrap_";
        break;
      default: 
        /* hmmm, what about any other case? Just use __wrap_bar */
        prefix = "  __wrap_";
        break;
    }
//  }

  printFunctionNameInOutputFile(r, impl, prefix, sig);
  impl << " {\n" << endl;

  string funcprototype = sig.funchandle + ");";
  string funchandle = sig.funchandle + ") = NULL;";

  if (runtime == RUNTIME_INTERCEPT) {
      ostringstream buff;
    if(shmem_wrapper) {
       std::string type = r->name() + "_t";
       std::string handle =  r->name() + "_handle";
       buff << "  typedef " << (isVoid ? "void" : sig.returntypename)  << " (*" << type << ")" << sig.funcarg << ";\n"
            << "  static " << type << " " << handle << " = (" << type << ")NULL;\n"
            << "  if (!" << handle << ") {\n"
            << "    " << handle << " = get_function_handle(\"" << r->name() << "\");\n"
            << "  }\n" << endl;
    }
    else {
      if (isVoid) {
        if (strict_typing) {
          impl << "  typedef void (*"<<protoname<<funcprototype<<endl;
          impl << "  static "  << protoname << "_h " << r->name() << "_h = NULL;"<<endl;
        } else {
          impl <<"  static void (*"<<r->name()<<funchandle<<endl;
        }
      } else {
        if (strict_typing) {
          impl << "  typedef " << sig.returntypename << " (*"<<protoname<<funcprototype<<endl;
          impl << "  static "  << protoname << "_h " << r->name() << "_h = NULL;"<<endl;
        } else {
          impl <<"  static "<<sig.returntypename<<" (*"<<r->name()<<funchandle<<endl;
        }
        retstring = string("    return retval;");
      }

      buff << "  if (tau_handle == NULL) \n"
           << "    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \n\n"
           << "  if (tau_handle == NULL) { \n"
           << "    perror(\"Error opening library in dlopen call\"); \n"
           << retstring << "\n"
           << "  } else { \n"
           << "    if (" << r->name() << "_h == NULL)\n"
           << "      ";
      if (strict_typing)
        buff << r->name() << "_h = (" << protoname << "_h) dlsym(tau_handle,\"" << r->name() << "\"); \n";
      else
        buff << r->name() << "_h = dlsym(tau_handle,\"" << r->name() << "\"); \n";
      buff << "    if (" << r->name() << "_h == NULL) {\n"
           << "      perror(\"Error obtaining symbol info from dlopen'ed lib\"); \n"
           << "  " << retstring << "\n"
           << "    }\n"
           << "  }\n";
    }
    dltext = buff.str();
  } /* if (runtime == RUNTIME_INTERCEPT) */

#ifdef SHMEM
#ifdef MPI
#ifdef CRAY
  if(runtime == WRAPPER_INTERCEPT && shmem_wrapper && 
      ( rname == "start_pes"  || rname == "shmem_init")) {
    impl << "  MPI_Init();" << endl;
  }
#endif
#endif
#endif

  if (!isVoid) {
    impl<<"  "<<sig.returntypename<< " retval;"<<endl;
  }

  /* Now put in the body of the routine */

  if(upc_env) {
    impl << "  if (tau_upc_node == -1) {\n";
      if (upc_env == UPC_XLUPC) {
        impl << "    TAU_PROFILE_SET_NODE(MYTHREAD); \n";
      }
      impl  << "    tau_upc_node = TAU_PROFILE_GET_NODE();\n"
         << "    if (tau_upc_node == -1) {\n";
    if (isVoid) {
      impl << "      __real_" << sig.func << ";\n"
           << "      return;" << endl;
    } else {
      impl << "      return __real_" << sig.func << ";" << endl;
    }
    impl << "    } else {\n"
         << "      tau_totalnodes(1," << upc_threads() << ");\n"
         << "    }\n"
         << "  }\n"
         << endl;
  } 

  if((shmem_wrapper && runtime != RUNTIME_INTERCEPT) || !shmem_wrapper)
    impl<<"  TAU_PROFILE_TIMER(t,\""<<r->fullName()<<"\", \"\", "<<group_name<<");"<<endl;
  if (runtime == RUNTIME_INTERCEPT)
    impl <<dltext;
  if((shmem_wrapper && runtime != RUNTIME_INTERCEPT) || !shmem_wrapper)
    impl<<"  TAU_PROFILE_START(t);"<<endl;

  if (shmem_wrapper) { /* generate pshmem calls here */
    if(runtime != RUNTIME_INTERCEPT) printShmemMessageBeforeRoutine(r, impl, sig);
    if (!isVoid)
    {
      impl<<"  retval  =";
    }
    if (runtime == RUNTIME_INTERCEPT) {
      impl<<"  "<<sig.rcalledfunc<<";"<<endl;
    } else if(runtime == WRAPPER_INTERCEPT) {
      if(rname.find("shmem_finalize") != string::npos) {
        impl<<"  TauTraceOTF2ShutdownComms_if_necessary(0);"<<endl;
      }
      if(rname.find("shmem_finalize") != string::npos) impl<< "  if(TauEnv_get_profile_format() != TAU_FORMAT_MERGED && TauEnv_get_trace_format() != TAU_TRACE_FORMAT_OTF2)"<<endl<<"  ";
      impl<<"  __real_"<<sig.func<<";"<<endl;
    } else {
      if (pshmem_use_underscore_instead_of_p) {
        impl <<"   _"<<sig.func<<";"<<endl;
      } else {
        impl <<"   p"<<sig.func<<";"<<endl;
      }
    }
    if(runtime != RUNTIME_INTERCEPT) printShmemMessageAfterRoutine(r, impl, sig);
  } else if (upc_env) {
    printUPCMessageBeforeRoutine(r, impl, sig);
    if (!isVoid) {
      impl<<"  retval  =";
    }
    if (runtime == RUNTIME_INTERCEPT) {
      impl<<"  "<<sig.rcalledfunc<<";"<<endl;
    }
    else {
      if (runtime == WRAPPER_INTERCEPT) { /* link time instrumentation using -Wl,-wrap,bar */
        impl<<"  __real_"<<sig.func<<";"<<endl;
      } else { /* default case when we use redirection of bar -> tau_bar */
        impl<<"  "<<sig.func<<";"<<endl;
      }
    }
    printUPCMessageAfterRoutine(r, impl, sig);
  } else {
    if (!isVoid) {
      impl<<"  retval  =";
    }
    if (runtime == RUNTIME_INTERCEPT) {
      impl<<"  "<<sig.rcalledfunc<<";"<<endl;
    }
    else {
      if (runtime == WRAPPER_INTERCEPT) { /* link time instrumentation using -Wl,-wrap,bar */
        impl<<"  __real_"<<sig.func<<";"<<endl;
      } else { /* default case when we use redirection of bar -> tau_bar */
        impl<<"  "<<sig.func<<";"<<endl;
      }
    }
  }

  if((shmem_wrapper && runtime != RUNTIME_INTERCEPT) || !shmem_wrapper)
    impl << "  TAU_PROFILE_STOP(t);" << endl;

  if (!isVoid) {
    impl<<"  return retval;"<<endl;
  }
  impl<<endl;

  impl<<"}\n"<<endl;

  if (runtime == RUNTIME_INTERCEPT) { /* linker-based instrumentation */
#if 0
    printFunctionNameInOutputFile(r, impl, "  ", sig);
    impl << "{" << endl;
    if(sig.returntypename.compare(0, 4, "void") == 0)
            impl << "   __wrap_" << sig.func << ";" << endl;
    else
            impl << "   return __wrap_" << sig.func << ";" << endl;
    impl << "}\n" << endl;
#endif /* 0 */

#if 0
    // Fortran wrapper functions
    impl << "extern " << sig.returntypename << " " << rname << "_" << sig.funcarg << endl;
    impl << "{" << endl;
    impl << "   __wrap_" << sig.func << ";" << endl;
    impl << "}\n" << endl;

    impl << "extern " << sig.returntypename << " " << rname << "__" << sig.funcarg << endl;
    impl << "{" << endl;
    impl << "   __wrap_" << sig.func << ";" << endl;
    impl << "}\n" << endl;

    transform(rname.begin(), rname.end(), rname.begin(), ::toupper);

    impl << "extern " << sig.returntypename << " " << rname << "_" << sig.funcarg << endl;
    impl << "{" << endl;
    impl << "   __wrap_" << sig.func << ";" << endl;
    impl << "}\n" << endl;

    impl << "extern " << sig.returntypename << " " << rname << "__" << sig.funcarg << endl;
    impl << "{" << endl;
    impl << "   __wrap_" << sig.func << ";" << endl;
    impl << "}\n" << endl;
#endif
  }
  if (runtime == WRAPPER_INTERCEPT && shmem_wrapper) {
/*
    printFunctionNameInOutputFile(r, impl, "  ", sig);
    impl << "{" << endl;
    impl << "   __wrap_" << sig.func << ";" << endl;
    impl << "}\n" << endl;
*/

    // Fortran wrapper functions
    impl << "extern " << sig.returntypename << " __wrap_" << rname << "_" << sig.funcargfort << endl;
    impl << "{" << endl;
    if(sig.returntypename.compare(0, 4, "void") == 0)
            impl << "   __wrap_" << sig.funcfort << ";" << endl;
    else
	    impl << "   return __wrap_" << sig.funcfort << ";" << endl;
    impl << "}\n" << endl;

    impl << "extern " << sig.returntypename << " __wrap_" << rname << "__" << sig.funcargfort << endl;
    impl << "{" << endl;
    if(sig.returntypename.compare(0, 4, "void") == 0)
            impl << "   __wrap_" << sig.funcfort << ";" << endl;
    else
            impl << "   return __wrap_" << sig.funcfort << ";" << endl;
    impl << "}\n" << endl;

    transform(rname.begin(), rname.end(), rname.begin(), ::toupper);

    impl << "extern " << sig.returntypename << " __wrap_" << rname << "_" << sig.funcargfort << endl;
    impl << "{" << endl;
    if(sig.returntypename.compare(0, 4, "void") == 0)
            impl << "   __wrap_" << sig.funcfort << ";" << endl;
    else
            impl << "   return __wrap_" << sig.funcfort << ";" << endl;
    impl << "}\n" << endl;

    impl << "extern " << sig.returntypename << " __wrap_" << rname << "__" << sig.funcargfort << endl;
    impl << "{" << endl;
    if(sig.returntypename.compare(0, 4, "void") == 0)
            impl << "   __wrap_" << sig.funcfort << ";" << endl;
    else
            impl << "   return __wrap_" << sig.funcfort << ";" << endl;
    impl << "}\n" << endl;
  }

  if (runtime == PREPROC_INTERCEPT) { /* preprocessor instrumentation */
    macro.append(" "+sig.func+" " +"tau_"+sig.func);
#ifdef DEBUG
    cout <<"macro = "<<macro<<endl;
    cout <<"func = "<<sig.func<<endl;
#endif /* DEBUG */

    /* The macro goes in header file, the implementation goes in the other file */
    header <<macro<<endl;  
    header <<"extern "<<sig.returntypename<<" tau_"<<sig.proto<<";\n"<<endl;
  }

}

/* -------------------------------------------------------------------------- */
/* -- Extract the package name from the header file name:  netcdf.h -> netcdf */
/* -------------------------------------------------------------------------- */
string extractLibName(string const & filename)
{
  return filename.substr(0, filename.find("."));
} 


/* -------------------------------------------------------------------------- */
/* -- Instrumentation routine for a C program ------------------------------- */
/* -------------------------------------------------------------------------- */
bool instrumentCFile(PDB& pdb, pdbFile* f, ofstream& header, ofstream& impl, 
                     ofstream& linkoptsfile, string& group_name, string& header_file, 
                     int runtime, string& runtime_libname, string& libname)
{
  string file(f->name());

  // open source file
  ifstream istr(file.c_str());
  if (!istr) {
    cerr << "Error: Cannot Open '" << file << "'" << endl;
    return false;
  }
#ifdef DEBUG
  cout << "Processing " << file << " in instrumentCFile..." << endl;
#endif

  // initialize reference vector
  vector<itemRef*> itemvec;
  getCReferencesForWrapper(itemvec, pdb, f);
  PDB::croutinevec routines = pdb.getCRoutineVec();
  string rname;
  for (PDB::croutinevec::const_iterator rit=routines.begin(); rit!=routines.end(); ++rit) {
    pdbRoutine::locvec retlocations = (*rit)->returnLocations();
    if ( (*rit)->location().file() == f
        && !(*rit)->isCompilerGenerated()
        && (instrumentEntity((*rit)->fullName())) )
    {
      printRoutineInOutputFile(*rit, header, impl, group_name, runtime, runtime_libname);
      if (runtime == WRAPPER_INTERCEPT) { /* -Wl,-wrap,<func>,-wrap,<func> */
        if (!(*rit)->signature()->hasEllipsis()) { /* does not have varargs */
          linkoptsfile <<"-Wl,-wrap,"<<(*rit)->name()<<" ";
          rname = (*rit)->name();
          linkoptsfile <<"-Wl,-wrap,"<<rname<<"_ ";
          linkoptsfile <<"-Wl,-wrap,"<<rname<<"__ ";
          transform(rname.begin(), rname.end(), rname.begin(), ::toupper);
          linkoptsfile <<"-Wl,-wrap,"<<rname<<"_ ";
          linkoptsfile <<"-Wl,-wrap,"<<rname<<"__ ";
        }
      }
    }
  }
  return true;
} 

/* -------------------------------------------------------------------------- */
/* -- Define a TAU group after <Profile/Profiler.h> ------------------------- */
/* -------------------------------------------------------------------------- */
void defineTauGroup(ofstream& ostr, string & group_name)
{
  if (group_name.compare("TAU_USER") != 0) {
    /* Write the following lines only when -DTAU_GROUP=string is defined */
    ostr<< "#ifndef "<<group_name<<endl;
    ostr<< "#define "<<group_name << " TAU_GET_PROFILE_GROUP(\""<<group_name.substr(10)<<"\")"<<endl;
    ostr<< "#endif /* "<<group_name << " */ "<<endl;
  }
}

void generateMakefile(string const & package, string const & outFileName, 
                      int runtime, string const & runtime_libname, string const & libname, 
                      string const & extradefs)
{
  char const * makefileName = "Makefile";
  char const * compiler_name = "$(TAU_CC)";
  char const * upcprefix = "";

  if (upc_env == UPC_GNU) {
    compiler_name = "$(TAU_UPCC)";
    upcprefix = "$(UPCC_C_PREFIX)";
  }

  char buffer[1024];
  snprintf(buffer, sizeof(buffer),  "%s_wrapper/%s", libname.c_str(), makefileName);

  ofstream makefile(buffer);

  if(shmem_wrapper) {
  // Note: shmem wrapper assumes wr.c and wr_dynamic.c for outFileNames.
        makefile << "include ${TAU_MAKEFILE}\n"
                 << "CC=" << compiler_name << " \n"
                 << "CFLAGS=$(TAU_DEFS) " << extradefs << " $(TAU_INCLUDE) $(TAU_MPI_INCLUDE)  -I.. $(TAU_SHMEM_INC) -fPIC\n"
                 << "EXTRA_FLAGS=$(TAU_CRAY_SHMEM_EXTRA_DEFS)\n"
                 << "\n"
                 << "AR=$(TAU_AR)\n"
                 << "ARFLAGS=rcv \n"
                 << "\n"
                 << "all: lib" << package << "_wrap.a lib" << package << "_wrap.so \n"
                 << "lib" << package << "_wrap.so: " << package << "_wrap_static.o " << package << "_wrap_dynamic.o \n"
                 << "\t$(CC) $(TAU_SHFLAGS) $@ $^ $(TAU_SHLIBS) -ldl\n"
                 << "\n"
                 << "lib" << package << "_wrap.a: " << package << "_wrap_static.o \n"
                 << "\t$(AR) $(ARFLAGS) $@ $^ \n"
                 << "\n"
                 << package << "_wrap_dynamic.o: " << "wr_dynamic.c"<< "\n"
                 << "\t$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@\n"
                 << package << "_wrap_static.o: " << "wr.c" << "\n"
                 << "\t$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@\n"
                 << "\n"
                 << "clean:\n"
                 << "\t/bin/rm -f " << package << "_wrap_dynamic.o " << package << "_wrap_static.o lib" << package << "_wrap.so lib" << package << "_wrap.a\n"
                 << endl;
  } else {
    switch(runtime) {
      case PREPROC_INTERCEPT:
        makefile << "include ${TAU_MAKEFILE}\n"
                 << "CC=" << compiler_name << "\n"
                 << "CFLAGS=$(TAU_DEFS) " << extradefs << " $(TAU_INCLUDE) $(TAU_MPI_INCLUDE) -I.. $(TAU_SHMEM_INC)\n"
                 << "EXTRA_FLAGS=\n"
                 << "\n"
                 << "AR=ar\n"
                 << "ARFLAGS=rcv\n"
                 << "\n"
                 << "lib" << package << "_wrap.a: " << package << "_wrap.o \n"
                 << "\t$(AR) $(ARFLAGS) $@ $<\n"
                 << "\n"
                 << package << "_wrap.o: " << outFileName << "\n"
                 << "\t$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@\n"
                 << "clean:\n"
                 << "\t/bin/rm -f " << package << "_wrap.o lib" << package << "_wrap.a\n"
                 << endl;
        break;
      case RUNTIME_INTERCEPT:
        makefile << "include ${TAU_MAKEFILE}\n"
                 << "CC=" << compiler_name << " \n"
                 << "CFLAGS=$(TAU_DEFS) " << extradefs << " $(TAU_INCLUDE) $(TAU_MPI_INCLUDE)  -I.. $(TAU_SHMEM_INC) -fPIC\n"
                 << "EXTRA_FLAGS=\n"
                 << "\n"
                 << "lib" << package << "_wrap.so: " << package << "_wrap.o \n"
                 << "\t$(CC) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl\n"
                 << "\n"
                 << package << "_wrap.o: " << outFileName << "\n"
                 << "\t$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@\n"
                 << "clean:\n"
                 << "\t/bin/rm -f " << package << "_wrap.o lib" << package << "_wrap.so\n"
                 << endl;
        break;
      case WRAPPER_INTERCEPT:
        makefile << "include ${TAU_MAKEFILE} \n"
                 << "CC=" << compiler_name << " \n"
                 << "CFLAGS=$(TAU_DEFS) " << extradefs << " $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I.. $(TAU_SHMEM_INC)\n"
                 << "EXTRA_FLAGS=\n"
                 << "\n"
                 << "AR=$(TAU_AR)\n"
                 << "ARFLAGS=rcv \n"
                 << "\n"
                 << "lib" << package << "_wrap.a: " << package << "_wrap.o \n"
                 << "\t$(AR) $(ARFLAGS) $@ $< \n"
                 << "\n"
                 << package << "_wrap.o: " << outFileName << "\n"
                 << "\t$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@\n"
                 << "clean:\n"
                 << "\t/bin/rm -f " << package << "_wrap.o lib" << package << "_wrap.a\n"
                 << endl;
        break;
      default:
        // Unknown runtime flag!
        break;
    }
  }
}


void show_usage(char const * argv0)
{
  cout <<"Usage : "<< argv0 <<" <pdbfile> <sourcefile> [-o <outputfile>] [-w librarytobewrapped] [-r runtimelibname] [-g groupname] [-i headerfile] [-c|-c++|-fortran] [-f <instr_req_file> ] [--strict]"<<endl;
  cout <<" To use runtime library interposition, -r <name> must be specified\n"<<endl;
  cout <<" --strict enforces strict typing (no dynamic function pointer casting). \n"<<endl;
  cout <<" e.g., "<<endl;
  cout <<"   tau_wrap hdf5.h.pdb hdf5.h libhdf5.a -o wrap_hdf5.c -w /usr/lib/libhdf5.a "; 
  cout <<"----------------------------------------------------------------------------------------------------------"<<endl;
}


/* -------------------------------------------------------------------------- */
/* -- Instrument the program using C, C++ or F90 instrumentation routines --- */
/* -------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  string outFileName("out.ins.C");
  string group_name("TAU_USER"); /* Default: if nothing else is defined */
  string runtime_libname("libc.so"); /* Default: if nothing else is defined */
  string header_file("Profile/Profiler.h");
  string extradefs("");
  bool retval;
  bool outFileNameSpecified = false;

  /* by default generate PDT based re-direction library*/
  int runtime = PREPROC_INTERCEPT;

  if (argc < 3) {
    show_usage(argv[0]);
    return 1;
  }

  PDB p(argv[1]); 
  if ( !p ) {
    show_usage(argv[0]);
    cout << "Invalid PDB file: " << argv[1] << endl;
    return 1;
  }

  string filename = argv[2];

#ifdef DEBUG
  cout << "Name of pdb file = " << argv[1] << endl;
  cout << "Name of source file = " << argv[2] << endl;
#endif /* DEBUG */

  for(int i=3; i<argc; i++) {
    if (strcmp(argv[i], "-o") == 0) {
      outFileName = argv[i+1];
#ifdef DEBUG
      cout << "output file = " << outFileName << endl;
#endif /* DEBUG */
      outFileNameSpecified = true;
    }
    else if (strcmp(argv[i], "--upc") == 0) {
      if(i == (argc - 1)) {
        cout << "ERROR: --upc requires an argument" << endl;
        exit(1);
      }
      char const * arg = argv[i+1];
      if (strncmp(arg, "berkeley", 4) == 0) {
        upc_env = UPC_BERKELEY;
      } else if (strncmp(arg, "gnu", 4) == 0) {
        upc_env = UPC_GNU;
      } else if (strncmp(arg, "xlupc", 5) == 0) {
        upc_env = UPC_XLUPC;
      } else if (strncmp(arg, "cray", 4) == 0) {
        upc_env = UPC_CRAY;
        extradefs = "$(TAU_UPC_COMPILER_OPTIONS)";
      } else {
        cout << "ERROR: invalid --upc argument: " << arg << endl;
        exit(1);
      } 
#ifdef DEBUG
      cout << "upc_env = " << upc_env << endl;
#endif /* DEBUG */
    } 
    else if (strcmp(argv[i], "--shmem") == 0) {
      shmem_wrapper = true;
#ifdef DEBUG
      cout << "shmem_wrapper = true" << endl;
#endif /* DEBUG */
    } 
    else if (strcmp(argv[i], "--pshmem_use_underscore_instead_of_p") == 0) {
      pshmem_use_underscore_instead_of_p = true;
#ifdef DEBUG
      cout << "pshmem_use_underscore_instead_of_p = true" << endl;
#endif /* DEBUG */
    } 
    else if (strcmp(argv[i], "-r") == 0) {
      runtime_libname = argv[i+1];
      runtime = RUNTIME_INTERCEPT;
#ifdef DEBUG
      cout << "Runtime library name: " << runtime_libname << endl;
#endif /* DEBUG */
    }
    else if (strcmp(argv[i], "-w") == 0) {
      runtime_libname = argv[i+1];
      runtime = WRAPPER_INTERCEPT;
#ifdef DEBUG
      cout << "Link time -Wl,-wrap library name: " << runtime_libname << endl;
#endif /* DEBUG */
    }
    else if (strcmp(argv[i], "-g") == 0) {
      group_name = string("TAU_GROUP_")+string(argv[i+1]);
#ifdef DEBUG
      printf("Group %s\n", group_name.c_str());
#endif /* DEBUG */
    }
    else if (strcmp(argv[i], "-i") == 0) {
      header_file = string(argv[i+1]);
#ifdef DEBUG
      printf("Header file %s\n", header_file.c_str());
#endif /* DEBUG */
    }
    else if (strcmp(argv[i], "-f") == 0) {
      processInstrumentationRequests(argv[i+1]);
#ifdef DEBUG
      printf("Using instrumentation requests file: %s\n", argv[i]);
#endif /* DEBUG */
    }
    else if (strcmp(argv[i], "--strict") == 0) {
      strict_typing = true;	
#ifdef DEBUG
      printf("Using strict typing. \n");
#endif /* DEBUG */
    }
  }

  if (!outFileNameSpecified) {
    /* if name is not specified on the command line */
    outFileName = filename + string(".ins");
  }

  /* should we make a directory and put it in there? */
  string libname = extractLibName(filename);
  string dircmd("mkdir -p "+libname+"_wrapper");
  system(dircmd.c_str());

  ostringstream buff;
  buff << libname << "_wrapper/link_options.tau";
  string linkoptsfileName = buff.str();

  ofstream linkoptsfile(linkoptsfileName.c_str());
  if (!linkoptsfile) {
    cerr << "Error: Cannot open: '" << linkoptsfileName << "'" << endl;
    return false;
  }

  system(dircmd.c_str());
  ofstream impl(string(libname+"_wrapper/"+outFileName).c_str()); /* actual implementation goes in here */
  ofstream header(string(libname+"_wrapper/"+filename).c_str()); /* use the same file name as the original */
  if (!impl) {
    cerr << "Error: Cannot open output file '" << outFileName << "'" << endl;
    return false;
  }
  if (!header) {
    cerr << "Error: Cannot open wrapper/" << filename << endl;
    return false;
  }

  /* files created properly */
  if (shmem_wrapper) {
    impl << "#ifndef _GNU_SOURCE\n"
         << "#define _GNU_SOURCE\n"
         << "#endif\n"
         << endl;
    if (runtime == WRAPPER_INTERCEPT) {
      impl << "#ifndef SHMEM_FINT\n"
           << "#define SHMEM_FINT int\n"
           << "#endif\n"
           << "#ifndef SHMEM_FINT8_t\n"
           << "#define SHMEM_FINT8_t int8_t\n"
           << "#endif\n"
           << endl;
    }
  }
  impl << "#include <" << filename << ">\n"
       << "#include <" << header_file << ">\n"
       << "#include <stdio.h>\n"
       << "#include <stdlib.h>\n"
       << endl;
  if (shmem_wrapper) {
    if ( runtime == WRAPPER_INTERCEPT) 
      impl << "#include <Profile/TauEnv.h>\n"
           << "#include <Profile/TauAPI.h>\n"
           << "#include <Profile/TauTrace.h>\n"
           << endl;
    impl << "int TAUDECL tau_totalnodes(int set_or_get, int value);\n"
         << "int TAUDECL Tau_set_usesSHMEM(int value);\n"
         << "int __real_shmem_n_pes(void);\n"
         << "int __real_shmem_my_pe(void);\n"
         << "static int tau_shmem_tagid_f=0;\n"
         << "#define TAU_SHMEM_TAGID (tau_shmem_tagid_f = (tau_shmem_tagid_f & 255))\n"
         << "#define TAU_SHMEM_TAGID_NEXT ((++tau_shmem_tagid_f) & 255)\n"
         << endl;
  }
  if (upc_env) {
    impl << "#pragma pupc off\n"
         << "\n"
         << "#ifdef __BERKELEY_UPC__\n"
         << "#pragma UPCR NO_SRCPOS \n"
         << "#endif\n"
         << "\n"
         << "static int tau_upc_node = -1;\n"
         << "static int tau_upc_tagid_f = 0;\n"
         << "#define TAU_UPC_TAGID (tau_upc_tagid_f = (tau_upc_tagid_f & 255))\n"
         << "#define TAU_UPC_TAGID_NEXT ((++tau_upc_tagid_f) & 255)\n"
         << "\n"
         << "void tau_totalnodes(int, int);\n"
         << endl;
  }

  if (runtime == RUNTIME_INTERCEPT) {
    /* add the runtime library calls */
    impl <<"#include <dlfcn.h>"<<endl<<endl;
    impl <<"static const char * tau_orig_libname = "<<"\""<<
      runtime_libname<<"\";"<<endl; 
    impl <<"static void *tau_handle = NULL;"<<endl<<endl<<endl;

    if(shmem_wrapper){
      /* add get_function_handle function */
      impl << "static void * get_function_handle(char const * name)"<<endl;
      impl << "{"<<endl;
      impl << "  char const * err;"<<endl;
      impl << "  void * handle;"<<endl<<endl;;
      impl << "  // Reset error pointer"<<endl;
      impl << "  dlerror();"<<endl<<endl;
      impl << "  // Attempt to get the function handle"<<endl;
      impl << "  handle = dlsym(RTLD_NEXT, name);"<<endl<<endl;
      impl << "  // Detect errors"<<endl;
      impl << "  if ((err = dlerror())) {"<<endl;
      impl << "    // These calls are unsafe, but we're about to die anyway.     "<<endl;
      impl << "    fprintf(stderr, \"Error getting %s handle: %s\\n\", name, err);  "<<endl;
      impl << "    fflush(stderr);"<<endl;
      impl << "    exit(1);"<<endl;
      impl << "  }"<<endl<<endl;;
      impl << "  return handle;"<<endl;
      impl << "}"<<endl;
    }
  }

  defineTauGroup(impl, group_name); 

#ifdef DEBUG
  cout <<"Library name is "<<libname<<endl;
#endif /* DEBUG */

  header <<"#ifndef _TAU_"<<libname<<"_H_"<<endl;
  header <<"#define _TAU_"<<libname<<"_H_"<<endl<<endl;
  header <<"#include <../"<<filename<<">"<<endl<<endl;
  header <<"#ifdef __cplusplus"<<endl;
  header <<"extern \"C\" {"<<endl;
  header <<"#endif /*  __cplusplus */"<<endl<<endl;

  bool fuzzyMatchResult;
  bool fileInstrumented = false;
  if (processFileForInstrumentation(filename)) {
    for (PDB::filevec::const_iterator it=p.getFileVec().begin(); it!=p.getFileVec().end(); it++) {
#ifdef DEBUG
      cout <<"Instrument file: "<<filename<<" "<< (*it)->name()<<endl;
#endif /* DEBUG */
      instrumentCFile(p, *it, header, impl, linkoptsfile, group_name, header_file, runtime, runtime_libname, libname);
    }
  }
  if (runtime == WRAPPER_INTERCEPT && !shmem_wrapper) {
    char dirname[1024]; 
    getcwd(dirname, sizeof(dirname)); 
    linkoptsfile <<"-L"<<dirname<<"/"<<libname<<"_wrapper/ -l"<< libname<<"_wrap "<<runtime_libname<<endl;
  }
  header <<"#ifdef __cplusplus"<<endl;
  header <<"}"<<endl;
  header <<"#endif /* __cplusplus */"<<endl<<endl;
  header <<"#endif /*  _TAU_"<<libname<<"_H_ */"<<endl;

  header.close();

  if (runtime != PREPROC_INTERCEPT) { /* 0 is for default preprocessor based wrapping */
    string hfile = libname+"_wrapper/"+filename;
#ifdef DEBUG
    cout <<"Deleting " << hfile << endl;
#endif /* DEBUG */
    /* delete the header file, we do not need it */
    unlink(hfile.c_str());
  }

  generateMakefile(libname, outFileName, runtime, runtime_libname, libname, extradefs);

} /* end of main */






///////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////

/* EOF */

/***************************************************************************
 * $RCSfile: tau_wrap.cpp,v $   $Author: sameer $
 * $Revision: 1.18 $   $Date: 2009/10/27 22:47:54 $
 * VERSION_ID: $Id: tau_wrap.cpp,v 1.18 2009/10/27 22:47:54 sameer Exp $
 ***************************************************************************/

