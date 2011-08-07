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
 



//#define DEBUG 1
/* For selective instrumentation */
extern int processInstrumentationRequests(char *fname);
extern bool instrumentEntity(const string& function_name);
extern bool processFileForInstrumentation(const string& file_name);
extern bool isInstrumentListEmpty(void);
/* Prototypes for selective instrumentation */
extern bool addFileInstrumentationRequests(PDB& p, pdbFile *file, vector        <itemRef *>& itemvec);



/* Globals */
bool memory_flag = false;   /* by default, do not insert malloc.h in instrumented C/C++ files */
bool strict_typing = false; /* by default unless --strict option is used. */
bool shmem_wrapper = false; /* by default unless --shmem option is used. */
bool pshmem_use_underscore_instead_of_p = false; /* by default unless --pshmem_use_underscore_instead_of_p option is used. */


///////////////////////////////////////////////////////////////////////////



/* -------------------------------------------------------------------------- */
/* -- Fuzzy Match. Allows us to match files that don't quite match properly, 
 * but infact refer to the same file. For e.g., /home/pkg/foo.cpp and ./foo.cpp
 * or foo.cpp and ./foo.cpp. This routine allows us to match such files! 
 * -------------------------------------------------------------------------- */
bool fuzzyMatch(const string& a, const string& b)
{ /* This function allows us to match string like ./foo.cpp with
     /home/pkg/foo.cpp */
  if (a == b)
  { /* the two files do match */
#ifdef DEBUG
    cout <<"fuzzyMatch returns true for "<<a<<" and "<<b<<endl;
#endif /* DEBUG */
    return true;
  }
  else 
  { /* fuzzy match */
    /* Extract the name without the / character */
    int loca = a.find_last_of(TAU_DIR_CHARACTER);
    int locb = b.find_last_of(TAU_DIR_CHARACTER);

    /* truncate the strings */
    string trunca(a,loca+1);
    string truncb(b,locb+1);
    /*
    cout <<"trunca = "<<trunca<<endl;
    cout <<"truncb = "<<truncb<<endl;
    */
    if (trunca == truncb) 
    {
#ifdef DEBUG
      cout <<"fuzzyMatch returns true for "<<a<<" and "<<b<<endl;
#endif /* DEBUG */
      return true;
    }
    else
    {
#ifdef DEBUG
      cout <<"fuzzyMatch returns false for "<<a<<" and "<<b<<endl;
#endif /* DEBUG */
      return false;
    }
  }
}
///////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////

/* -------------------------------------------------------------------------- */
/* -- Get a list of instrumentation points for a C program ------------------ */
/* -------------------------------------------------------------------------- */
/* Create a vector of items that need action: such as BODY_BEGIN, RETURN etc.*/
void getCReferencesForWrapper(vector<itemRef *>& itemvec, PDB& pdb, pdbFile *file) {

  /* moved selective instrumentation file processing here */
  if (!isInstrumentListEmpty()) 
  { /* there are finite instrumentation requests, add requests for this file */
    addFileInstrumentationRequests(pdb, file, itemvec);
  }
}

#ifdef OLD
{
  /* we used to keep the selective instrumentation file processing at the
     entry. But, when a routine is specified as a phase, we need to annotate
     its itemRef accordingly. This needs the entry/exit records to be created
     prior to processing the selective instrumentation file. N/A for wrappers
     as there are no entry/exit records created.*/

  PDB::croutinevec routines = pdb.getCRoutineVec();
  for (PDB::croutinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit)
  {
    pdbRoutine::locvec retlocations = (*rit)->returnLocations();
    if ( (*rit)->location().file() == file && !(*rit)->isCompilerGenerated()
	 && (instrumentEntity((*rit)->fullName())) )
    {
        itemvec.push_back(new itemRef(*rit, BODY_BEGIN,
                (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col()));
#ifdef DEBUG
        cout <<" Location begin: "<< (*rit)->location().line() << " col "
             << (*rit)->location().col() <<endl;
        cout <<" Location head Begin: "<< (*rit)->headBegin().line() << " col "             << (*rit)->headBegin().col() <<endl;
        cout <<" Location head End: "<< (*rit)->headEnd().line() << " col "
             << (*rit)->headEnd().col() <<endl;
        cout <<" Location body Begin: "<< (*rit)->bodyBegin().line() << " col "             << (*rit)->bodyBegin().col() <<endl;
        cout <<" Location body End: "<< (*rit)->bodyEnd().line() << " col "
             << (*rit)->bodyEnd().col() <<endl;
#endif /* DEBUG */
        for(pdbRoutine::locvec::iterator rlit = retlocations.begin();
           rlit != retlocations.end(); rlit++)
        {
#ifdef DEBUG 
          cout <<" Return Locations : "<< (*rlit)->line() << " col "
             << (*rlit)->col() <<endl;
#endif /* DEBUG */
          itemvec.push_back(new itemRef(*rit, RETURN,
                (*rlit)->line(), (*rlit)->col()));
        }
        itemvec.push_back(new itemRef(*rit, BODY_END,
                (*rit)->bodyEnd().line(), (*rit)->bodyEnd().col()));
#ifdef DEBUG 
        cout <<" Return type: " << (*rit)->signature()->returnType()->name()<<endl;
        cout <<" Routine name: "<<(*rit)->name() <<" Signature: " <<
                (*rit)->signature()->name() <<endl;
#endif /* DEBUG */

	/* See if the current routine calls exit() */
	pdbRoutine::callvec c = (*rit)->callees(); 
    }
  }

  /* All instrumentation requests are in. Sort these now and remove duplicates */
#ifdef DEBUG
  for(vector<itemRef *>::iterator iter = itemvec.begin(); iter != itemvec.end();
   iter++)
  {
    cout <<"Before SORT: Items ("<<(*iter)->line<<", "<<(*iter)->col<<")"
	 <<"snippet = "<<(*iter)->snippet<<endl;
  }
#endif /* DEBUG */
#ifdef DEBUG
  for(vector<itemRef *>::iterator iter = itemvec.begin(); iter != itemvec.end();
   iter++)
  {
    cout <<"Items ("<<(*iter)->line<<", "<<(*iter)->col<<")"
	 <<"snippet = "<<(*iter)->snippet<<endl;
  }
#endif /* DEBUG */
}
#endif /* OLD - delete */

bool isReturnTypeVoid(pdbRoutine *r)
{
  if (strcmp(r->signature()->returnType()->name().c_str(), "void") == 0)
  {
#ifdef DEBUG
     cout <<"Return type is void for "<<r->name()<<endl;
#endif /* DEBUG */
     return true;
  }
  else 
     return false;
}

bool doesRoutineNameContainGet(const char *rname, int namelen) {
  if ((strstr(rname, "get") != 0 ) || 
     ((rname[namelen-2] == '_') && (rname[namelen-1] == 'g')))
    return true;
  else
    return false;
}

bool doesRoutineNameContainPut(const char *rname, int namelen) {
  if ((strstr(rname, "put") != 0 ) || 
     ((rname[namelen-2] == '_') && (rname[namelen-1] == 'p')))
    return true;
  else
    return false;
}

/* Fetch and operate operations include swap, fadd and finc */
bool doesRoutineNameContainFetchOp(const char *rname, int namelen) {
  if ((strstr(rname, "swap") != 0 ) || (strstr(rname, "fadd") != 0) ||
      (strstr(rname, "finc") != 0 ))
    return true;
  else
    return false;
}

/* Fetch and operate operations include swap, fadd and finc */
bool doesRoutineNameContainCondFetchOp(const char *rname, int namelen) {
  if (strstr(rname, "cswap") != 0 ) 
    return true;
  else
    return false;
}

void getMultiplierString(const char *rname, string& multiplier_string) {
  if (strstr(rname, "char") !=0) { // char is found 
    multiplier_string=string("sizeof(char)*"); return;
  }
  if (strstr(rname, "short") !=0) { // short is found 
    multiplier_string=string("sizeof(short)*"); return;
  }
  if (strstr(rname, "int") !=0) { // int is found 
    multiplier_string=string("sizeof(int)*"); return;
  }
  if (strstr(rname, "longlong") !=0) { // long long is found 
    multiplier_string=string("sizeof(long long)*"); return;
  }
  if (strstr(rname, "longdouble") !=0) { // float is found 
    multiplier_string=string("sizeof(long double)*"); return;
  }
  if (strstr(rname, "long") !=0) { // long is found 
    multiplier_string=string("sizeof(long)*"); return;
  }
  if (strstr(rname, "double") !=0) { // double is found 
    multiplier_string=string("sizeof(double)*"); return;
  }
  if (strstr(rname, "float") !=0) { // float is found 
    multiplier_string=string("sizeof(float)*"); return;
  }
  if (strstr(rname, "16") !=0) { 
    multiplier_string=string("2*"); return;
  }
  if (strstr(rname, "32") !=0) {
    multiplier_string=string("4*"); return;
  }
  if (strstr(rname, "64") !=0) {
    multiplier_string=string("8*"); return;
  }
  if (strstr(rname, "128") !=0) {
    multiplier_string=string("16*"); return;
  }
  if (strstr(rname, "4") !=0) { // INT4_SWAP uses 4 bytes not 4 bits. 
    multiplier_string=string("4*"); return;
  }
  if (strstr(rname, "8") !=0) { // INT8_SWAP uses 8 bytes not 8 bits. = 64
    multiplier_string=string("8*"); return;
  }
}

void  printShmemMessageBeforeRoutine(pdbRoutine *r, ofstream& impl, int len_argument_no, int pe_argument_no, bool fortran_interface) {
  const char *rname = r->name().c_str();
  int routine_len = r->name().size();
  string multiplier_string(""); 
  char length_string[1024];
  char processor_arg[256];

  if (fortran_interface) {
    sprintf(processor_arg, "(*a%d)", pe_argument_no);
  } else {
    sprintf(processor_arg, "a%d", pe_argument_no);
  }

  printf("Size = %d, name = %s\n", routine_len, rname);
  getMultiplierString(rname, multiplier_string); 
  printf("Multiplier string = %s\n", multiplier_string.c_str());
  if (len_argument_no != 0) {
    if (fortran_interface) {
      sprintf(length_string, "%s (*a%d)", multiplier_string.c_str(), len_argument_no);
    } else {
      sprintf(length_string, "%sa%d", multiplier_string.c_str(), len_argument_no);
    }
  } else {
    sprintf(length_string, "%s1", multiplier_string.c_str());
  }
  
  if ((doesRoutineNameContainGet(rname, routine_len) == true) || 
      (doesRoutineNameContainFetchOp(rname, routine_len) == true)) { /* Get */
    printf("Routine name %s contains Get variant\n", rname);
    impl <<"  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), "<<length_string<<", "<<processor_arg<<");"<<endl;
  }
  if (doesRoutineNameContainPut(rname, routine_len) == true) {
    printf("Routine name %s contains Put variant\n", rname);
    impl <<"  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, "<<processor_arg<<", "<<length_string<<");"<<endl;
  }
   

}

void  printShmemMessageAfterRoutine(pdbRoutine *r, ofstream& impl, int len_argument_no, int pe_argument_no, int cond_argument_no, bool fortran_interface) {
  const char *rname = r->name().c_str();
  int routine_len = r->name().size();
  string multiplier_string("");
  char length_string[1024];
  char processor_arg[256];
  char cond_string[1024];
  bool is_it_a_get = false;
  bool is_it_a_fetchop = false;
  bool is_it_a_cond_fetchop = false;
  bool is_it_a_put = false;

  if (fortran_interface) {
    sprintf(processor_arg, "(*a%d)", pe_argument_no);
  } else {
    sprintf(processor_arg, "a%d", pe_argument_no);
  }

  printf("Size = %d, name = %s\n", routine_len, rname);
  getMultiplierString(rname, multiplier_string);
  printf("Multiplier string = %s\n", multiplier_string.c_str());
  if (len_argument_no != 0) {
    if (fortran_interface) {
      sprintf(length_string, "%s (*a%d)", multiplier_string.c_str(), len_argument_no);
    } else {
      sprintf(length_string, "%sa%d", multiplier_string.c_str(), len_argument_no);
    }
  } else {
    sprintf(length_string, "%s1", multiplier_string.c_str());
  }
  is_it_a_get = doesRoutineNameContainGet(rname, routine_len);
  is_it_a_fetchop = doesRoutineNameContainFetchOp(rname, routine_len); 
  is_it_a_cond_fetchop = doesRoutineNameContainCondFetchOp(rname, routine_len); 

  if (strstr(rname, "start_pes") != 0) {
     impl << "  tau_totalnodes(1,pshmem_n_pes());"<<endl;
     impl << "  TAU_PROFILE_SET_NODE(pshmem_my_pe());"<<endl;
  }

  if (is_it_a_get || is_it_a_fetchop ) { /* Get */
    printf("Routine name %s contains Get variant\n", rname);
    impl <<"  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, "<<processor_arg<<", "<<length_string<<");"<<endl;
  }
  if (is_it_a_cond_fetchop || is_it_a_fetchop) { /* add condition */
    if (is_it_a_cond_fetchop && (cond_argument_no == 0)) {
      printf("WARNING: in fetchop function %s, cond_argument_no is 0???\n", 
	rname); 
    }
    string indent(""); /* no indent by default */
    if (is_it_a_cond_fetchop) {
      indent=string("  ");
      if (fortran_interface) {
        sprintf(cond_string, "  if (retval == (*a%d)) { ", cond_argument_no);
      } else {
        sprintf(cond_string, "  if (retval == a%d) { ", cond_argument_no);
      }
      impl <<cond_string<<endl;; 
    }
    impl <<indent<<"  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, "<<processor_arg<<", "<<length_string<<");"<<endl;
    impl <<indent<<"  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), "<<length_string<<", "<<processor_arg<<");"<<endl;
    if (is_it_a_cond_fetchop) {
      impl<<indent<<"}"<<endl;
    }
  }
  if (doesRoutineNameContainPut(rname, routine_len) == true) {
    printf("Routine name %s contains Put variant\n", rname);
    impl <<"  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), "<<length_string<<", "<<processor_arg<<");"<<endl;
  }

}

void  printRoutineInOutputFile(pdbRoutine *r, ofstream& header, ofstream& impl, string& group_name, int runtime, string& runtime_libname)
{
  string macro("#define ");
  string func(r->name());
  string proto(r->name());
  string protoname(r->name());
  string funchandle("_h) (");
  string rcalledfunc("(*"+r->name()+"_h)");
  string wcalledfunc("__real_"+r->name());
  string dltext;
  string returntypename;
  string retstring("    return;");
  const pdbGroup *grp;
  func.append("(");
  rcalledfunc.append("(");
  proto.append("(");
  protoname.append("_p");
  bool fortran_interface = false; /* if *len or *pe appears in the arglist */

  if (r->signature()->hasEllipsis()) {
    // For a full discussion of why vararg functions are difficult to wrap
    // please see: http://www.swig.org/Doc1.3/Varargs.html#Varargs
 
    impl <<"#warning \"TAU: Not generating wrapper for vararg function "<<r->name()<<"\""<<endl;
    cout <<"TAU: Not generating wrapper for vararg function "<<r->name()<<endl;
    return;
  }
  if ((grp = r->signature()->returnType()->isGroup()) != 0) { 
    returntypename = grp->name();
  } else {
    returntypename = r->signature()->returnType()->name();
  }

  impl << endl; 
  impl << "/**********************************************************"<<endl;
  impl << "   "<<r->name()<< endl;
  impl << " **********************************************************/"<<endl<<endl;

  if (shmem_wrapper == true) {
    impl << returntypename << " "; /* nothing else */
  }
  else {
    switch (runtime) {
    case 1: /* for runtime interception, put a blank, the name stays the same*/
      impl<<returntypename<<" "; /* put in return type */
      break;
    case 0: /* for standard preprocessor redirection, bar becomes tau_bar */
      impl<<returntypename<<"  tau_"; /* put in return type */
      break;
    case -1: /* for wrapper library interception, it becomes __wrap_bar */
      impl<<returntypename<<"  __wrap_"; /* put in return type */
      break;
    default: /* hmmm, what about any other case? Just use __wrap_bar */
      impl<<returntypename<<"  __wrap_"; /* put in return type */
      break;
    }
  }
  impl<<func;
#ifdef DEBUG
  cout <<"Examining "<<r->name()<<endl;
  cout <<"Return type :"<<returntypename<<endl;
#endif /* DEBUG */
  pdbType::argvec av = r->signature()->arguments();
  int argcount = 1;
  bool isVoid = isReturnTypeVoid(r);
  int shmem_len_argcount = 0; 
  int shmem_pe_argcount = 0; 
  int shmem_cond_argcount = 0; 
  for(pdbType::argvec::const_iterator argsit = av.begin();
      argsit != av.end(); argsit++, argcount++)
  {
    char number[256];
#ifdef DEBUG
    cout <<"Argument "<<(*argsit).name()<<" Type "<<(*argsit).type()->name()<<endl;
#endif /* DEBUG */

    if (shmem_wrapper) {
      cout <<"Argument "<<(*argsit).name()<<" Type "<<(*argsit).type()->name()<<endl;
      if (strcmp((*argsit).name().c_str(), "len") == 0) {
        printf("Argcount = %d for len\n", argcount); 
        shmem_len_argcount = argcount; 
        if ((*argsit).type()->kind() == pdbItem::TY_PTR) {
          fortran_interface = true;
        }
      }
      if (strcmp((*argsit).name().c_str(), "pe") == 0) {
        printf("Argcount = %d for pe\n", argcount); 
        shmem_pe_argcount = argcount; 
        if ((*argsit).type()->kind() == pdbItem::TY_PTR) {
          fortran_interface = true;
        }
      }
      if ((strcmp((*argsit).name().c_str(), "match") == 0) || 
         (strcmp((*argsit).name().c_str(), "cond") == 0)) {
        printf("Argcount = %d for match/cond\n", argcount); 
        shmem_cond_argcount = argcount; 
      }
    }
    if (argcount != 1) { /* not a startup */
      func.append(", ");
      proto.append(", ");
      funchandle.append(", ");
      rcalledfunc.append(", ");
      impl<<", ";
    }
    sprintf(number, "%d", argcount);
    const pdbGroup *gr;
    string argtypename;
    if ( (gr=(*argsit).type()->isGroup()) != 0) {
      argtypename=gr->name();
    } else {
      argtypename=(*argsit).type()->name();
    }
    proto.append(argtypename);
    funchandle.append(argtypename);

    proto.append(" " + string("a")+string(number));
    rcalledfunc.append(" " + string("a")+string(number));
    func.append(string("a")+string(number));

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
      for(i=found - examinedarg; i < strlen(examinedarg); i++) {
        //printf("after number Printing %c\n", examinedarg[i]);
        impl << examinedarg[i];
      }

    }
    else {
      /* print: for (int a, double b), this prints "int" */
      impl<<argtypename<<" ";
      /* print: for (int a, double b), this prints "a1" in int a1, */
      impl<<"a"<<number;
    }
    if (r->signature()->hasEllipsis()) {
      //printf("Has ellipsis...\n");
      impl<<", ...";
    }
  }
  func.append(")");
  proto.append(")");
  rcalledfunc.append(")");
  impl<<") {" <<endl<<endl;
	string funcprototype = funchandle + string(");");
 	funchandle.append(") = NULL;");
  if (runtime == 1) {
    if (!isVoid) {
			if (strict_typing)
			{
				impl << "  typedef " << returntypename << " (*"<<protoname<<funcprototype<<endl;
				impl << "  static "  << protoname << "_h " << r->name() << "_h = NULL;"<<endl;
			}
			else
			{
      	impl <<"  static "<<returntypename<<" (*"<<r->name()<<funchandle<<endl;
			}
      retstring = string("    return retval;");
    } else {
			if (strict_typing)
			{
				impl << "  typedef void (*"<<protoname<<funcprototype<<endl;
				impl << "  static "  << protoname << "_h " << r->name() << "_h = NULL;"<<endl;
			}
			else
			{
      	impl <<"  static void (*"<<r->name()<<funchandle<<endl;
			}
    }
		string dlsym = "";
		if (strict_typing)
		{
		  dlsym = r->name() + string("_h = (") + protoname + string("_h) dlsym(tau_handle,\"")+r->name() + string("\"); \n");
		}
		else	
		{
			dlsym = r->name() + string("_h = dlsym(tau_handle,\"")+r->name() +
			string("\"); \n");
		}
    dltext = string("  if (tau_handle == NULL) \n") + 
    string("    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \n\n") + 
    string("  if (tau_handle == NULL) { \n") + 
    string("    perror(\"Error opening library in dlopen call\"); \n")+ retstring + string("\n") + 
    string("  } \n") + 
    string("  else { \n") + 
    string("    if (") + r->name() + string("_h == NULL)\n\t") + dlsym + 
    string("    if (") + r->name() + string ("_h == NULL) {\n") + 
    string("      perror(\"Error obtaining symbol info from dlopen'ed lib\"); \n") + string("  ")+ retstring + string("\n    }\n");

  }

  if (!isVoid) {
    impl<<"  "<<returntypename<< " retval;"<<endl;
  }
  /* Now put in the body of the routine */
  
  impl<<"  TAU_PROFILE_TIMER(t,\""<<r->fullName()<<"\", \"\", "
      <<group_name<<");"<<endl;
  if (runtime == 1)
    impl <<dltext;
  impl<<"  TAU_PROFILE_START(t);"<<endl;
  if (shmem_wrapper) { /* generate pshmem calls here */
    printShmemMessageBeforeRoutine(r, impl, shmem_len_argcount, shmem_pe_argcount, fortran_interface);
    if (!isVoid)
    {
      impl<<"  retval  =";
    }
    if (pshmem_use_underscore_instead_of_p) {
      impl <<"   _"<<func<<";"<<endl;
    } else {
      impl <<"   p"<<func<<";"<<endl;
    }
    printShmemMessageAfterRoutine(r, impl, shmem_len_argcount, shmem_pe_argcount, shmem_cond_argcount, fortran_interface);
  }
  else {
    if (!isVoid)
    {
      impl<<"  retval  =";
    }
    if (runtime == 1) {
      impl<<"  "<<rcalledfunc<<";"<<endl;
    }
    else {
      if (runtime == -1) { /* link time instrumentation using -Wl,-wrap,bar */
        impl<<"  __real_"<<func<<";"<<endl;
      } else { /* default case when we use redirection of bar -> tau_bar */
        impl<<"  "<<func<<";"<<endl;
      }
    }
  }
  impl<<"  TAU_PROFILE_STOP(t);"<<endl;

  if (runtime == 1) {
    impl<<"  }"<<endl;
  }

  if (!isVoid)
  {
    impl<<"  return retval;"<<endl;
  }
  impl<<endl;

  impl<<"}"<<endl<<endl;
  if (runtime == 0) { /* preprocessor instrumentation */
    macro.append(" "+func+" " +"tau_"+func);
#ifdef DEBUG
    cout <<"macro = "<<macro<<endl;
    cout <<"func = "<<func<<endl;
#endif /* DEBUG */

  /* The macro goes in header file, the implementation goes in the other file */
    header <<macro<<endl;  
    header <<"extern "<<returntypename<<" tau_"<<proto<<";"<<endl<<endl;
  }

}

/* -------------------------------------------------------------------------- */
/* -- Extract the package name from the header file name:  netcdf.h -> netcdf */
/* -------------------------------------------------------------------------- */
void extractLibName(const char *filename, string& libname)
{
  char *name = strdup(filename);
  int len = strlen(name); /* length */
  int i;

  for (i=0; i < len; i++)
  {
    if (name[i] == '.') name[i] = '\0'; /* truncate it if . is found */
  }
  libname=string(name);
} 


/* -------------------------------------------------------------------------- */
/* -- Instrumentation routine for a C program ------------------------------- */
/* -------------------------------------------------------------------------- */
bool instrumentCFile(PDB& pdb, pdbFile* f, ofstream& header, ofstream& impl, ofstream& linkoptsfile, string& group_name, string& header_file, int runtime, string& runtime_libname, string& libname)
{
  //static int firsttime=0;
  string file(f->name());
  
  // open source file
  ifstream istr(file.c_str());
  if (!istr) {
    cerr << "Error: Cannot open '" << file << "'" << endl;
    return false;
  }
#ifdef DEBUG
  cout << "Processing " << file << " in instrumentCFile..." << endl;
#endif

  // initialize reference vector
  vector<itemRef *> itemvec;
  getCReferencesForWrapper(itemvec, pdb, f);
  PDB::croutinevec routines = pdb.getCRoutineVec();
  for (PDB::croutinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit)   {
    pdbRoutine::locvec retlocations = (*rit)->returnLocations();
    if ( (*rit)->location().file() == f && !(*rit)->isCompilerGenerated()
         && (instrumentEntity((*rit)->fullName())) )
    {
       printRoutineInOutputFile(*rit, header, impl, group_name, runtime, runtime_libname);
       if (runtime == -1) { /* -Wl,-wrap,<func>,-wrap,<func> */
	 if (!(*rit)->signature()->hasEllipsis()) { /* does not have varargs */
           linkoptsfile <<"-Wl,-wrap,"<<(*rit)->name()<<" ";
         }
       }

    }
  }
  return true;

} 
/* -------------------------------------------------------------------------- */
/* -- Define a TAU group after <Profile/Profiler.h> ------------------------- */
/* -------------------------------------------------------------------------- */
void defineTauGroup(ofstream& ostr, string& group_name)
{
  if (strcmp(group_name.c_str(), "TAU_USER") != 0)
  { /* Write the following lines only when -DTAU_GROUP=string is defined */
    ostr<< "#ifndef "<<group_name<<endl;
    ostr<< "#define "<<group_name << " TAU_GET_PROFILE_GROUP(\""<<group_name.substr(10)<<"\")"<<endl;
    ostr<< "#endif /* "<<group_name << " */ "<<endl;
  }
}

void generateMakefile(string& package, string &outFileName, int runtime, string& runtime_libname, string& libname)
{
  string makefileName("Makefile");
  ofstream makefile(string(libname+"_wrapper/"+string(makefileName)).c_str());
  
  if (runtime == 0) {
    string text("include ${TAU_MAKEFILE} \n\
CC=$(TAU_CC) \n\
CFLAGS=$(TAU_DEFS) $(TAU_INCLUDE) $(TAU_MPI_INCLUDE) -I.. \n\
\n\
AR=ar \n\
ARFLAGS=rcv \n\
\n\
lib"+package+"_wrap.a: "+package+"_wrap.o \n\
	$(AR) $(ARFLAGS) $@ $<\n\
\n\
"+package+"_wrap.o: "+outFileName+"\n\
	$(CC) $(CFLAGS) -c $< -o $@\n\
clean:\n\
	/bin/rm -f "+package+"_wrap.o lib"+package+"_wrap.a\n\
");
    makefile <<text<<endl;
  }
  else { 
    if (runtime == 1) { 
      string text("include ${TAU_MAKEFILE} \n\
CC=$(TAU_CC) \n\
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(TAU_MPI_INCLUDE)  -I.. \n\
\n\
lib"+package+"_wrap.so: "+package+"_wrap.o \n\
	$(CC) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl\n\
\n\
"+package+"_wrap.o: "+outFileName+"\n\
	$(CC) $(CFLAGS) -c $< -o $@\n\
clean:\n\
	/bin/rm -f "+package+"_wrap.o lib"+package+"_wrap.so\n\
");

      makefile <<text<<endl;
    } else { 
      if (runtime == -1) {
        string text("include ${TAU_MAKEFILE} \n\
CC=$(TAU_CC) \n\
ARFLAGS=rcv \n\
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I.. \n\
\n\
lib"+package+"_wrap.a: "+package+"_wrap.o \n\
	$(TAU_AR) $(ARFLAGS) $@ $< \n\
\n\
"+package+"_wrap.o: "+outFileName+"\n\
	$(CC) $(CFLAGS) -c $< -o $@\n\
clean:\n\
	/bin/rm -f "+package+"_wrap.o lib"+package+"_wrap.a\n\
");
        makefile <<text<<endl;

      }
    }
  }
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
  int runtime = 0; /* by default generate PDT based re-direction library*/
  bool retval;
        /* Default: if nothing else is defined */

  if (argc < 3)
  {
    cout <<"Usage : "<<argv[0] <<" <pdbfile> <sourcefile> [-o <outputfile>] [-w librarytobewrapped] [-r runtimelibname] [-g groupname] [-i headerfile] [-c|-c++|-fortran] [-f <instr_req_file> ] [--strict]"<<endl;
    cout <<" To use runtime library interposition, -r <name> must be specified\n"<<endl;
    cout <<" --strict enforces strict typing (no dynamic function pointer casting). \n"<<endl;
    cout <<" e.g., "<<endl;
    cout <<" % tau_wrap hdf5.h.pdb hdf5.h libhdf5.a -o wrap_hdf5.c -w /usr/lib/libhdf5.a "; 
    cout<<"----------------------------------------------------------------------------------------------------------"<<endl;
  }
  PDB p(argv[1]); if ( !p ) return 1;
  /* setGroupName(p, group_name); */
  bool outFileNameSpecified = false;
  int i;
  const char *filename;
  for(i=0; i < argc; i++)
  {
    switch(i) 
    {
      case 0:
#ifdef DEBUG
        printf("Name of pdb file = %s\n", argv[1]);
#endif /* DEBUG */
        break;
      case 1:
#ifdef DEBUG
        printf("Name of source file = %s\n", argv[2]);
#endif /* DEBUG */
        filename = argv[2];
        break;
      default:
        if (strcmp(argv[i], "-o")== 0)
        {
          ++i;
#ifdef DEBUG
          printf("output file = %s\n", argv[i]);
#endif /* DEBUG */
          outFileName = string(argv[i]);
          outFileNameSpecified = true;
        }

        if (strcmp(argv[i], "--shmem") == 0)
        {
	  shmem_wrapper = true;
        } 

        if (strcmp(argv[i], "--pshmem_use_underscore_instead_of_p") == 0)
        {
	  pshmem_use_underscore_instead_of_p = true;
        } 

        if (strcmp(argv[i], "-r") == 0)
        {
          ++i;
          runtime_libname = string(argv[i]);
          runtime = 1; /* 1 is for runtime interposition LD_PRELOAD */
#ifdef DEBUG
          printf("Runtime library name: %s\n", runtime_libname.c_str());
#endif /* DEBUG */
        }

        if (strcmp(argv[i], "-w") == 0)
        {
          ++i;
          runtime_libname = string(argv[i]);
          runtime = -1; /* -1 is for link time -Wl,-wrap,func interposition */
#ifdef DEBUG
          printf("Link time -Wl,-wrap library name: %s\n", runtime_libname.c_str());
#endif /* DEBUG */
        }

        if (strcmp(argv[i], "-g") == 0)
        {
          ++i;
          group_name = string("TAU_GROUP_")+string(argv[i]);
#ifdef DEBUG
          printf("Group %s\n", group_name.c_str());
#endif /* DEBUG */
        }
        if (strcmp(argv[i], "-i") == 0)
        {
          ++i;
          header_file = string(argv[i]);
#ifdef DEBUG
          printf("Header file %s\n", header_file.c_str());
#endif /* DEBUG */
        }
        if (strcmp(argv[i], "-f") == 0)
        {
          ++i;
          processInstrumentationRequests(argv[i]);
#ifdef DEBUG
          printf("Using instrumentation requests file: %s\n", argv[i]);
#endif /* DEBUG */
        }
        if (strcmp(argv[i], "--strict") == 0)
				{
					strict_typing = true;	
#ifdef DEBUG
          printf("Using strict typing. \n");
#endif /* DEBUG */
				}
        break; /* end of default case */
    }
  }

  if (!outFileNameSpecified)
  { /* if name is not specified on the command line */
    outFileName = string(filename + string(".ins"));
  }
  /* should we make a directory and put it in there? */
  string libname;
  extractLibName(filename, libname);
  string dircmd("mkdir -p "+libname+"_wrapper");
  //system("mkdir -p wrapper");
  system(dircmd.c_str());
  ofstream linkoptsfile(string(libname+"_wrapper/link_options.tau").c_str()); 
  if (!linkoptsfile) {
    cerr << "Error: Cannot open '" << libname+"_wrapper/link_options.tau" << "'" << endl;
    return false;
  }

  system(dircmd.c_str());
  ofstream impl(string(libname+"_wrapper/"+outFileName).c_str()); /* actual implementation goes in here */
  ofstream header(string(libname+"_wrapper/"+string(filename)).c_str()); /* use the same file name as the original */
  if (!impl) {
    cerr << "Error: Cannot open '" << outFileName << "'" << endl;
    return false;
  }
  if (!header) {
    cerr << "Error: Cannot open wrapper/" <<filename  << "" << endl;
    return false;
  }
  /* files created properly */
  //header <<"#include <"<<filename<<">"<<endl;
  impl <<"#include <"<<filename<<">"<<endl;
  impl <<"#include <"<<header_file<<">"<<endl; /* Profile/Profiler.h */
  if (shmem_wrapper) {
    impl <<"int TAUDECL tau_totalnodes(int set_or_get, int value);"<<endl;
    impl <<"int tau_shmem_tagid=0 ; "<<endl;
    impl <<"#define TAU_SHMEM_TAGID tau_shmem_tagid"<<endl;
    impl <<"#define TAU_SHMEM_TAGID_NEXT (++tau_shmem_tagid) % 256 "<<endl;
  }


  if (runtime == 1) {
  /* add the runtime library calls */
     impl <<"#include <dlfcn.h>"<<endl<<endl;
     impl <<"const char * tau_orig_libname = "<<"\""<<
	runtime_libname<<"\";"<<endl; 
     impl <<"static void *tau_handle = NULL;"<<endl<<endl<<endl;
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
  bool instrumentThisFile;
  bool fuzzyMatchResult;
  bool fileInstrumented = false;
  for (PDB::filevec::const_iterator it=p.getFileVec().begin();
       it!=p.getFileVec().end(); ++it)
  {
     /* reset this variable at the beginning of the loop */
     instrumentThisFile = false;

     /* for headers, we should only use the processFileForInstrumentation check */
/*
     if ((fuzzyMatchResult = fuzzyMatch((*it)->name(), string(filename))) &&
         (instrumentThisFile = processFileForInstrumentation(string(filename))))     
*/
     if (instrumentThisFile = processFileForInstrumentation(string(filename)))
     { /* should we instrument this file? Yes */
       instrumentCFile(p, *it, header, impl, linkoptsfile, group_name, header_file, runtime, runtime_libname, libname);
     }
  }
  if (runtime == -1) {
    char * dirname = new char[1024]; 
    char *dirnameptr; 
    dirnameptr=getcwd(dirname, 1024); 
    linkoptsfile <<"-L"<<dirnameptr<<"/"<<libname<<"_wrapper/ -l"<< libname<<"_wrap "<<runtime_libname<<endl;
    delete[] dirname;
  }
  header <<"#ifdef __cplusplus"<<endl;
  header <<"}"<<endl;
  header <<"#endif /* __cplusplus */"<<endl<<endl;
  header <<"#endif /*  _TAU_"<<libname<<"_H_ */"<<endl;

  header.close();

  if (runtime != 0) { /* 0 is for default preprocessor based wrapping */
    string hfile = string(libname+"_wrapper/"+string(filename));
#ifdef DEBUG
    cout <<"Deleting foo.h"<<endl;
#endif /* DEBUG */
    /* delete the header file, we do not need it */
    unlink(hfile.c_str());
  }

  generateMakefile(libname, outFileName, runtime, runtime_libname, libname);

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

