//#define DEBUG 1
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


/* defines */
#ifdef TAU_WINDOWS
#define TAU_DIR_CHARACTER '\\'
#else
#define TAU_DIR_CHARACTER '/'
#endif /* TAU_WINDOWS */


#ifdef TAU_INSTRUMENT_PURE
bool instrumentPure = true;
#else
bool instrumentPure = false;
#endif

/* The IBM xlf compiler does not support sizeof(A) to find the size of an object */
#ifdef TAU_USE_SIZE_INSTEAD_OF_SIZEOF
string tau_size_tok("size");
#else /* TAU_USE_SIZE_INSTEAD_OF_SIZEOF */
string tau_size_tok("sizeof");
#endif /* TAU_USE_SIZE_INSTEAD_OF_SIZEOF */

/* For selective instrumentation */
extern int processInstrumentationRequests(char *fname);
extern bool instrumentEntity(const string& function_name);
extern bool processFileForInstrumentation(const string& file_name);
extern bool isInstrumentListEmpty(void);
extern void writeAdditionalDeclarations(ostream& ostr, const pdbRoutine *routine);
extern void writeAdditionalFortranInvocations(ostream& ostr, const pdbRoutine *routine);
extern bool addMoreInvocations(int routine_id, string& snippet);
extern bool isVoidRoutine(const pdbItem* r);

#include "tau_datatypes.h"

const int INBUF_SIZE = 65536;


#define EXIT_KEYWORD_SIZE 256
/* For Pooma, add a -noinline flag */
bool noinline_flag = false; /* instrument inlined functions by default */
bool noinit_flag = false;   /* initialize using TAU_INIT(&argc, &argv) by default */
bool memory_flag = false;   /* by default, do not insert malloc.h in instrumented C/C++ files */
bool lang_specified = false; /* implicit detection of source language using PDB file */
bool process_this_return = false; /* for C instrumentation using a different return keyword */
char exit_keyword[EXIT_KEYWORD_SIZE] = "exit"; /* You can define your own exit keyword */
bool using_exit_keyword = false; /* By default, we don't use the exit keyword */
tau_language_t tau_language; /* language of the file */
bool use_perflib = false;   /* by default, do not insert calls for perflib package */

/* This variable should actually be defined here. However, tau_wrap should not
   depend on this (and it currently would). */
extern bool use_spec;

list<string> current_timer; /* for Fortran loop level instrumentation. */

/* not needed anymore */
#ifdef OLD
  const pdbItem *item;
  itemKind_t kind; /* For C instrumentation */ 
  bool     isTarget;
  int      line;
  int      col;
  string   snippet;
#endif /* OLD */

/* Prototypes for selective instrumentation */
extern bool addFileInstrumentationRequests(PDB& p, pdbFile *file, vector<itemRef *>& itemvec);


void processExitOrAbort(vector<itemRef *>& itemvec, const pdbItem *i, pdbRoutine::callvec & c); /* in this file below */

/* A strict weak ordering is a binary predicate that compares two objects, returning true if the first preceeds the second */
static bool locCmp(const itemRef* r1, const itemRef* r2) {

  if (r1 == r2) { // strict weak ordering requires false on equal elements
#ifdef DEBUG 
    printf("locCmp: r1 == r2, returning false\n");
#endif /* DEBUG */
    return false;
  }

  if ( r1->line == r2->line )
  {
    if (r1->col == r2->col)
    { /* they're both equal */

      
      // we must first check if they are both BODY_BEGIN and return false in that case
      // otherwise it just depends on which is r1 and which is r2
      // STL sort requires strict weak ordering, meaning that two elements are equal
      // then this routine must return (x,y) as false and (y,x) as false
      if (r1->kind == r2->kind) return false;

      if (r1->kind == BODY_BEGIN) return true; 
      if (r2->kind == BODY_BEGIN) return false; /* body begin must always come first */
      if (r1->kind == INSTRUMENTATION_POINT) return false; 
      return true; 
    }
    else
    { /* col numbers are different */
      return r1->col < r2->col;
    }
  }
  else
  {
    return r1->line < r2->line;
  }
}

static bool itemEqual(const itemRef* r1, const itemRef* r2) {
#ifdef DEBUG
  printf("Comparing <%d:%d> with <%d:%d> kind = %d vs %d, target %d vs %d, attribute %d vs %d\n",
	r1->line, r1->col, r2->line, r2->col, r1->kind, r2->kind, r1->isTarget, r2->isTarget, r1->attribute, r2->attribute);
#endif /* DEBUG */
  /* two loops on the same line shouldn't be instrumented twice -- happens with templates with different instantiations. */
  if ((r1->line == r2->line) && (r1->col == r2->col) && (r1->kind == r2->kind) &&
	(r1->isTarget == r2->isTarget) && (r1->attribute == r2->attribute) &&
	((r1->kind == START_LOOP_TIMER ) || (r1->kind == STOP_LOOP_TIMER))) {
#ifdef DEBUG
    printf("Items are equal returning true!\n");
#endif /* DEBUG */
    return true; /* they are equal -- don't bother checking the snippet part.*/
  }
  else 
    return ( (r1->line == r2->line) &&
           (r1->col  == r2->col) && 
           (r1->kind == r2->kind) && 
           (r1->isTarget == r2->isTarget) && 
	   (r1->attribute == r2->attribute) && 
	   (r1->snippet == r2->snippet)); 
}
 


string getInstrumentedName(const pdbItem *item) {
  // create the instrumented routine name
  std::ostringstream oss;
  pdbRoutine *pdbr = (pdbRoutine*)item;
  if (!item) return string("");
  pdbLoc loc = item->location();
  const char *fullfile = item->location().file()->name().c_str();

  while (strchr(fullfile,TAU_DIR_CHARACTER)) { // remove path
    fullfile = strchr(fullfile,TAU_DIR_CHARACTER)+1;
  }

  // we only have fat item data for C/C++ right now
  pdbFatItem *fatItem = (pdbFatItem*)item;

  if (fatItem->headBegin().line() == 0) {
    oss << item->fullName() << " [{" << fullfile << "} {" << loc.line() << "," << loc.col() << "}]";
  } else {
    oss << item->fullName() << " [{" << fullfile 
	<< "} {" 
	<< fatItem->headBegin().line() << "," << fatItem->headBegin().col() 
	<< "}-{" 
	<< fatItem->bodyEnd().line() << "," << fatItem->bodyEnd().col() 
	<< "}]";
  }

  string instrumentedName(oss.str());
  
  return instrumentedName;
}


/* -------------------------------------------------------------------------- */
/* -- Sometimes a routine is created by a macro expansion and we shouldn't    */
/* -- instrument it. identicalBeginEnd returns true if body begin/end line    */
/* -- and columns match perfectly, false otherwise.                           */
/* -------------------------------------------------------------------------- */
bool identicalBeginEnd(const pdbCRoutine *rit)
{
  if (rit && (rit->bodyBegin().line() == rit->bodyEnd().line()) && 
	  (rit->bodyBegin().col() == rit->bodyBegin().col())) 
  {
    return true;
  }
  else 
  {
    return false;
  }
}
  
/* -------------------------------------------------------------------------- */
/* -- getMeasurementEntity(i) returns TAU_PROFILE/TAU_PHASE based on i  ----- */
/* -------------------------------------------------------------------------- */
const char * getMeasurementEntity(itemRef *i)
{ /* for C++ */
  if (i->isPhase) { /* static or dynamic phase */
     if (i->isDynamic) {
       return "TAU_DYNAMIC_PHASE";
     } else { /* ! dynamic phase */
       return "TAU_PHASE";
     }
  } else {  /* timer */
     if (i->isDynamic) {
       return "TAU_DYNAMIC_PROFILE";  
     } else { /* ! dynamic timer -- plain old static timer */
       return "TAU_PROFILE";
     }
  } /* phase/timer */
}

/* -------------------------------------------------------------------------- */
/* -- getCreateMeasurementEntity(i) returns TAU_PROFILE_TIMER/TAU_PHASE_TIMER */
/* -------------------------------------------------------------------------- */
const char * getCreateMeasurementEntity(itemRef *i)
{ /* NOTE: THIS ROUTINE IS ONLY MEANT FOR C/C++. See writeFortranTimer for F90. */
  if (i && i->isPhase) { /* static or dynamic phase */
     if (i->isDynamic) {
       return "TAU_PHASE_CREATE_DYNAMIC_AUTO";  /* NOTE: Change this! */
     } else { /* ! dynamic phase */
       return "TAU_PHASE_CREATE_STATIC";
     }
  } else {  /* timer. We currently do not support a dynamic timer for the 
               full routine. Hence, we use the default static timer.*/
     if (i->isDynamic) {
       return "TAU_PROFILE_CREATE_DYNAMIC_AUTO"; /* NOTE: Change this! */
     } else { /* ! dynamic timer */
       return "TAU_PROFILE_TIMER";
     }
  }
   
}

/* -------------------------------------------------------------------------- */
/* -- getStartMeasurementEntity(i) returns TAU_PROFILE_START/TAU_PHASE_START */
/* -------------------------------------------------------------------------- */
const char * getStartMeasurementEntity(itemRef *i)
{

   if (i && i->isPhase) 
     return "TAU_PHASE_START";
   else 
     return "TAU_PROFILE_START";
}

/* -------------------------------------------------------------------------- */
/* -- getStopMeasurementEntity(i) returns TAU_PROFILE_STOP/TAU_PHASE_STOP --- */
/* -------------------------------------------------------------------------- */
const char * getStopMeasurementEntity(itemRef *i)
{

#ifdef DEBUG
   printf("Inside getStopMeasurementEntity: isPhase = %d\n", i->isPhase);
#endif /* DEBUG */
   if (i && i->isPhase) 
     return "TAU_PHASE_STOP";
   else 
     return "TAU_PROFILE_STOP";
}

/* -------------------------------------------------------------------------- */
/* -- Merge instrumentation requests of same kind for same location --------- */
/* -------------------------------------------------------------------------- */
void mergeInstrumentationRequests(vector<itemRef *>& itemvec) {
  /* Now merge objects of the same kind at the same location */
  if (itemvec.size() > 1)
  {
    vector<itemRef *>::iterator iter = itemvec.begin()+1;
    while (iter != itemvec.end())
    {
      itemRef* item1 = *(iter - 1);
      itemRef* item2 = *iter;

      if (item1->kind == item2->kind &&
          item1->line == item2->line &&
          item1->col  == item2->col)
      {
        if (item1->snippet.empty())
          item1->snippet = item2->snippet;
        else if (!item2->snippet.empty())
          item1->snippet += "\n\t" + item2->snippet;
        iter = itemvec.erase(iter);
      }
      else
      {
        ++iter;
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/* -- Merge instrumentation requests of same kind for same location --------- */
/* -------------------------------------------------------------------------- */
void postprocessInstrumentationRequests(vector<itemRef *>& itemvec)
{
  /* It is assumed that all instrumentation requests are in the vector at this
     point. Now the requests can be sorted, duplicates removed, and requests
     of the same type on the same location merged. */
  stable_sort(itemvec.begin(), itemvec.end(), locCmp);
  itemvec.erase(unique(itemvec.begin(), itemvec.end(),itemEqual),itemvec.end());
  mergeInstrumentationRequests(itemvec);
#ifdef DEBUG
  for(vector<itemRef *>::iterator iter = itemvec.begin(); iter != itemvec.end();
   iter++)
  {
    cout <<"Items ("<<(*iter)->line<<", "<<(*iter)->col<<")"<<endl;
  }
#endif /* DEBUG */
}

/* -------------------------------------------------------------------------- */
/* -- Get a list of instrumentation points for a C++ program ---------------- */
/* -------------------------------------------------------------------------- */
bool getCXXReferences(vector<itemRef *>& itemvec, PDB& pdb, pdbFile *file) {
/* get routines, templates and member templates of classes */
bool retval;

  /* we used to keep the selective instrumentation file processing at the
     entry. But, when a routine is specified as a phase, we need to annotate
     its itemRef accordingly. This needs the entry/exit records to be created
     prior to processing the selective instrumentation file. */
  

  PDB::croutinevec routines = pdb.getCRoutineVec();
  for (PDB::croutinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit) 
  {
    if ( (*rit)->location().file() == file && !(*rit)->isCompilerGenerated() && 
	 (((*rit)->bodyBegin().line() != 0) && (*rit)->kind() != pdbItem::RO_EXT) && 
	 (instrumentEntity((*rit)->fullName())) && !identicalBeginEnd(*rit) ) 
    {
	/* See if the current routine calls exit() */
	pdbRoutine::callvec c = (*rit)->callees(); 

	processExitOrAbort(itemvec, *rit, c); 
#ifdef DEBUG
        cout <<"Routine "<<(*rit)->fullName() <<" body Begin line "
             << (*rit)->bodyBegin().line() << " col "
             << (*rit)->bodyBegin().col() <<endl;
#endif /* DEBUG */
	if ((*rit)->isInline())
  	{ 
	  if (noinline_flag)
	  {
#ifdef DEBUG
	    cout <<"Dont instrument "<<(*rit)->fullName()<<endl;
#endif /* DEBUG */
	    continue; /* Don't instrument it*/
	  }
	}
	
	if ((*rit)->isStatic()) 
	{
#ifdef DEBUG
	  cout <<" STATIC "<< (*rit)->fullName() <<endl;
#endif /* DEBUG */
	}
        itemvec.push_back(new itemRef(*rit, true, 
	    (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col()));
        /* parameter is always true: no need to add CT(*this) for non templates */
#ifdef DEBUG
      cout << (*rit)->fullName() <<endl;
#endif
    }
  }

  /* templates are not included in this. Get them */
  PDB::templatevec u = pdb.getTemplateVec();
  for(PDB::templatevec::iterator te = u.begin(); te != u.end(); ++te)
  { 
    if ( (*te)->location().file() == file)
    {
      pdbItem::templ_t tekind = (*te)->kind();
      if (((tekind == pdbItem::TE_MEMFUNC) || 
	  (tekind == pdbItem::TE_STATMEM) ||
	  (tekind == pdbItem::TE_FUNC)) && ((*te)->bodyBegin().line() != 0) &&
	  (instrumentEntity((*te)->fullName())) )
      { 
	/* Sometimes a compiler generated routine shows up in a template.
	   These routines (such as operator=) do not have a body position. 
 	   Instrument only if it has a valid body position.  */
  	// templates need some processing. Give it a false for isTarget arg.
	// target helps identify if we need to put a CT(*this) in the type
	// old: 
        //if ((((*te)->parentGroup()) == 0) && (tekind != pdbItem::TE_STATMEM)) 

	      /* TEMPLATES DO NOT HAVE CALLEES ONLY ROUTINES DO */
	/* See if the current template calls exit() */
	      /*
	pdbRoutine::callvec c = (*te)->callees(); 
	processExitOrAbort(itemvec, *te, c); 
	      */

        if ((tekind == pdbItem::TE_FUNC) || (tekind == pdbItem::TE_STATMEM))
	{ 
	  // There's no parent class. No need to add CT(*this)
          itemvec.push_back(new itemRef(*te, true, 
            (*te)->bodyBegin().line(), (*te)->bodyBegin().col())); 
	  // False puts CT(*this)
	}
	else 
	{ 
#ifdef DEBUG
	  cout <<"Before adding false to the member function, we must verify that it is not static"<<endl;
#endif /* DEBUG */
	  const pdbCRoutine *tr = (*te)->funcProtoInst();
	  if (!tr || (((tekind == pdbItem::TE_FUNC) || (tekind == pdbItem::TE_MEMFUNC))
	      && ((tr) && (tr->isStatic()))))

	  { /* check to see if there's a prototype instantiation entry */
	    /* it is indeed a static member function of a class template */
	    /* DO NOT add CT(*this) to the static member function */
	    
            itemvec.push_back(new itemRef(*te, true, 
	      (*te)->bodyBegin().line(), (*te)->bodyBegin().col()));
	  }
	  else
	  {
	    // it is a member function add the CT macro
            itemvec.push_back(new itemRef(*te, false, 
	      (*te)->bodyBegin().line(), (*te)->bodyBegin().col()));
	  }
	}
      }
      else 
      {
#ifdef DEBUG
	cout <<"T: "<<(*te)->fullName()<<endl;
#endif // DEBUG
      }
    }
  }

  return true; /* everything is ok */
}

/* -------------------------------------------------------------------------- */
/* -- Get a list of instrumentation points for a C program ------------------ */
/* -------------------------------------------------------------------------- */
/* Create a vector of items that need action: such as BODY_BEGIN, RETURN etc.*/
void getCReferences(vector<itemRef *>& itemvec, PDB& pdb, pdbFile *file) {


  /* we used to keep the selective instrumentation file processing at the
     entry. But, when a routine is specified as a phase, we need to annotate
     its itemRef accordingly. This needs the entry/exit records to be created
     prior to processing the selective instrumentation file. */

  PDB::croutinevec routines = pdb.getCRoutineVec();
  for (PDB::croutinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit)
  {
    pdbRoutine::locvec retlocations = (*rit)->returnLocations();
    if ( (*rit)->location().file() == file && !(*rit)->isCompilerGenerated() &&
         ((*rit)->kind() != pdbItem::RO_EXT) && ((*rit)->bodyBegin().line() != 0) 
	 && ((*rit)->bodyEnd().line() != 0) && 
	 (instrumentEntity((*rit)->fullName())) )
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

	processExitOrAbort(itemvec, *rit, c); 
    }
  }
}

/* -------------------------------------------------------------------------- */
/* -- Get a list of instrumentation points for a F90 program ---------------- */
/* -------------------------------------------------------------------------- */
void getFReferences(vector<itemRef *>& itemvec, PDB& pdb, pdbFile *file) {
/* get routines, templates and member templates of classes */
  PDB::froutinevec routines = pdb.getFRoutineVec();


  /* we used to keep the selective instrumentation file processing at the
     entry. But, when a routine is specified as a phase, we need to annotate
     its itemRef accordingly. This needs the entry/exit records to be created
     prior to processing the selective instrumentation file. */

  for (PDB::froutinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit) 
  {
    pdbRoutine::locvec retlocations = (*rit)->returnLocations();
    pdbRoutine::locvec stoplocations = (*rit)->stopLocations();

    if ( (*rit)->location().file() == file &&  
         ((*rit)->kind() != pdbItem::RO_FSTFN) &&  
	 ((*rit)->firstExecStmtLocation().file()) && 
	 (instrumentEntity((*rit)->fullName())) )
    {
#ifdef DEBUG
	cout <<"Routine " << (*rit)->fullName() <<endl;
#endif /* DEBUG */
        itemvec.push_back(new itemRef(*rit, BODY_BEGIN,
                (*rit)->firstExecStmtLocation().line(), 
		(*rit)->firstExecStmtLocation().col()));
#ifdef DEBUG
        cout <<" firstExecStatement: "<< 
		(*rit)->firstExecStmtLocation().line() << " col "
             << (*rit)->firstExecStmtLocation().col() <<endl;
#endif /* DEBUG */
	/* First process the return locations */
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
	/* Next process the stop locations */
        for(pdbRoutine::locvec::iterator slit = stoplocations.begin();
           slit != stoplocations.end(); slit++)
        {
#ifdef DEBUG 
          cout <<" Stop Locations : "<< (*slit)->line() << " col "
             << (*slit)->col() <<endl;
#endif /* DEBUG */
          itemvec.push_back(new itemRef(*rit, EXIT,
                (*slit)->line(), (*slit)->col()));
        }
    }
  }
}



/* -------------------------------------------------------------------------- */
/* -- process exit or abort statement and generate a record for itemRef       */
/* -------------------------------------------------------------------------- */
void processExitOrAbort(vector<itemRef *>& itemvec, const pdbItem *rit, pdbRoutine::callvec & c)
{
	for (pdbRoutine::callvec::iterator cit = c.begin(); cit !=c.end(); cit++)
	{ 
	   const pdbRoutine *rr = (*cit)->call(); 
#ifdef DEBUG 
	   cout <<"Callee " << rr->name() << " location line " << (*cit)->line() << " col " << (*cit)->col() <<endl; 
#endif /* DEBUG */
	   /* we do not want to call TAU_PROFILE_EXIT before obj->exit or 
	      obj->abort. Ignore the routines that have a parent group */
           if ((rr->parentGroup() == (const pdbGroup *) NULL) && 
	       (strcmp(rr->name().c_str(), exit_keyword)== 0))
	   { /* routine name matches and it is not a member of a class */
	     /* routine calls exit */
#ifdef DEBUG
             cout <<"Exit keyword matched"<<endl;
#endif /* DEBUG */
	     itemvec.push_back(new itemRef(rit, EXIT, (*cit)->line(), 
		(*cit)->col()));
	   } 
	   else if (using_exit_keyword)
	   { /* also check for "exit" where it occurs */
	     if (strcmp(rr->name().c_str(), "exit")== 0)
	     {
	       /* routine calls exit */
	       itemvec.push_back(new itemRef(rit, EXIT, (*cit)->line(), 
		(*cit)->col()));
	     }
	   } /* using exit keyword */

	   if (strcmp(rr->name().c_str(), "abort") == 0)
	   { /* routine calls abort */
	     itemvec.push_back(new itemRef(rit, EXIT, (*cit)->line(), 
		(*cit)->col()));
	   }
	}
}	

/* -------------------------------------------------------------------------- */
/* -- Returns true is return type is a reference else returns false --------- */
/* -------------------------------------------------------------------------- */
bool isReturnTypeReference(itemRef * i)
{
  const pdbType *t = ((pdbRoutine *)(i->item))->signature()->returnType();
  if ((t->kind() == pdbItem::TY_REF) || (t->isGroup())) 
    return true;
  else
    return false;
}
  
/* -------------------------------------------------------------------------- */
/* -- When int main(int, char **) is used, the argname is -. We can't use it  */
/* -- with TAU_INIT(&argname,...); So, we check to see that no arg has a -    */
/* -------------------------------------------------------------------------- */
bool okToPrintTauInit(pdbType::argvec& av)
{
  for(pdbType::argvec::const_iterator argsit = av.begin();
	argsit != av.end(); argsit++)
  {
    if (strcmp((*argsit).name().c_str(), "-") == 0) return false;
  }
  return true;  /* its ok. No argument is -. */
}

/* -------------------------------------------------------------------------- */
/* -- Prints TAU_PROFILE_INIT ----------------------------------------------- */
/* -------------------------------------------------------------------------- */
void print_tau_profile_init(ostream& ostr, pdbCRoutine *main_routine)
{
   if ( noinit_flag == false )
   { /* Put TAU_INIT */
     pdbType::argvec av = main_routine->signature()->arguments();
     if (av.size() == 2) {
       int arg_count = 0;
       if (okToPrintTauInit(av)) {
         ostr<<"  TAU_INIT(";
         for(pdbType::argvec::const_iterator argsit = av.begin();
           argsit != av.end(); argsit++, arg_count++)
         {
           ostr<<"&"<<(*argsit).name();
           if (arg_count == 0) ostr<<", ";
         }
         ostr<<"); "<<endl;
       }
     }
   }
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

/* -------------------------------------------------------------------------- */
/* -- Instrumentation routine for a C++ program ----------------------------- */
/* -------------------------------------------------------------------------- */
bool instrumentCXXFile(PDB& pdb, pdbFile* f, string& outfile, string& group_name, string &header_file)
{
  int inbufLength, k;
  bool retval;
  bool print_cr; 
  int write_upto, i, space; 
  string file(f->name());
  static char inbuf[INBUF_SIZE]; // to read the line
  // open outfile for instrumented version of source file
  ofstream ostr(outfile.c_str());
  if (!ostr) {
    cerr << "Error: Cannot open '" << outfile << "'" << endl;
    return false;
  }
  // open source file
  ifstream istr(file.c_str());
  if (!istr) {
    cerr << "Error: Cannot open '" << file << "'" << endl;
    return false;
  }
#ifdef DEBUG
  cout << "Processing " << file << " ..." << endl;
#endif 
  memset(inbuf, INBUF_SIZE, 0); // reset to zero


  // initialize reference vector
  vector<itemRef *> itemvec;
  if (!use_spec)
  {
    /* In "spec" mode, only the file instrumentation requests are used */
    retval = getCXXReferences(itemvec, pdb, f);
    if (!retval){
#ifdef DEBUG
    cout <<"instrumentCXXFile: propagating error from getCXXReferences..."<<endl;
#endif /* DEBUG */
      return retval; /* return error if we catch one */
    }
  }
  /* check if the given file has line/routine level instrumentation requests */
  if (!isInstrumentListEmpty()) 
  { /* there are finite instrumentation requests, add requests for this file */
    retval = addFileInstrumentationRequests(pdb, f, itemvec);
    if (!retval)
      return retval;
  }
  /* All instrumentation requests are in. Now do postprocessing. */
  postprocessInstrumentationRequests(itemvec);

  // put in code to insert <Profile/Profiler.h>
  if (use_spec)
  {
    /* XXX Insert code here */
  }
  else if (use_perflib)
  {
    /* XXX Insert code here */
  }
  else
    ostr<< "#include <"<<header_file<<">"<<endl;
  if (memory_flag)
    ostr<< "#include <malloc.h>"<<endl;

  defineTauGroup(ostr, group_name); 
  
  int inputLineNo = 0;
  int lastInstrumentedLineNo = 0;
  for(vector<itemRef *>::iterator it = itemvec.begin(); it != itemvec.end();
	++it)
  {
    // Read one line each till we reach the desired line no. 
#ifdef DEBUG
    if ((*it) && (*it)->item)
      cout <<"S: "<< (*it)->item->fullName() << " line "<< (*it)->line << " col " << (*it)->col << endl;
#endif 
    bool instrumented = false;
    /* NOTE: We need to change this for line level instrumentation. It can 
     * happen that the routine's entry line is also specified for line level
     * instrumentation */
    if (lastInstrumentedLineNo >= (*it)->line )
    {
      // Hey! This line has already been instrumented. Go to the next
      // entry in the func
#ifdef DEBUG
      cout <<"Entry already instrumented or brace not found - reached next routine! line = "<<(*it)->line <<endl;
#endif
      if ((*it)->kind == INSTRUMENTATION_POINT)
      {
#ifdef DEBUG
	cout <<"Instrumentation Point: inbuf = "<<inbuf<<endl;
#endif /* DEBUG */
	ostr << (*it)->snippet<<endl;
      }
      continue; // takes you to the next iteration in the for loop
    }

    while((instrumented == false) && (istr.getline(inbuf, INBUF_SIZE)) )
    { /* This assumes only one instrumentation request per line. Not so! */
      inputLineNo ++;
      if (inputLineNo < (*it)->line) 
      {
	// write the input line in the output stream
        ostr << inbuf <<endl;
      }
      else 
      { 
        switch((*it)->kind) {
	  case ROUTINE: 
          // we're at the desired line no. search for an open brace
  	  inbufLength = strlen(inbuf);
  
  	  for(i=0; i< inbufLength; i++)
  	  { 
  	    if ((inbuf[i] == '{') && (instrumented == false))
  	    {
#ifdef DEBUG
  	      cout <<"found the *first* { on the line inputLineNo=" <<inputLineNo<< endl;
#endif 
  	      ostr << inbuf[i] <<endl; // write the open brace and '\n'
  	      // put in instrumentation for the routine HERE
  	      //ostr <<"/*** INSTRUMENTATION ***/\n"; 
#ifdef SPACES
  	      // leave some leading spaces for formatting...
  	      for (space = 0; space < (*it)->col ; space++) ostr << " " ; 
#endif
              if (use_spec)
              {
                /* XXX Insert code here */
              }
              else if (use_perflib)
              {
                /* XXX Insert code here */
              }
              else {
  	        ostr <<"  "<<getMeasurementEntity((*it))<<"(\"" << getInstrumentedName((*it)->item) ;
  	        if (!((*it)->isTarget))
  	        { // it is a template member. Help it by giving an additional ()
  	        // if the item is a member function or a static member func give
  	        // it a class name using CT
  	          ostr <<"\", CT(*this), ";
  	        } 
   	        else // it is not a class member 
  	        { 
  	          ostr << "\", \" \", "; // null type arg to TAU_PROFILE 
  	        }

  	        if (strcmp((*it)->item->name().c_str(), "main")==0) 
  	        { /* it is main() */
  	          ostr << "TAU_DEFAULT);" <<endl; // give an additional line 
#ifdef SPACES
  	          for (space = 0; space < (*it)->col ; space++) ostr << " " ; 
#endif 
  	          // leave some leading spaces for formatting...

  	          print_tau_profile_init(ostr, (pdbCRoutine *) (*it)->item);
  	          ostr <<"#ifndef TAU_MPI" <<endl; // set node 0
  	          ostr <<"#ifndef TAU_SHMEM" <<endl; // set node 0
  	          ostr <<"  TAU_PROFILE_SET_NODE(0);" <<endl; // set node 0
  	          ostr <<"#endif /* TAU_SHMEM */" <<endl; // set node 0
  	          ostr <<"#endif /* TAU_MPI */" <<endl; // set node 0
  	        }
  	        else 
  	        {
  	          ostr <<group_name<<");" <<endl; // give an additional line 
  	        }
              }
  	      // if its a function, it already has a ()
  	      instrumented = true;
  	      lastInstrumentedLineNo = inputLineNo; 
  	      // keep track of which line was instrumented last
   	      // and then finish writing the rest of the line 
  	      // this will be used for templates and instantiated functions
  	    } // if open brace
  	    else 
  	    {  // if not open brace
  	      // ok to write to ostr 
#ifdef DEBUG
	      cout <<"DUMPING inbuf[i]"<<inbuf[i]<<endl;
/* We need to check here if there are other items to be written out like
entry instrumentation points  for the iteration right after the writing of
the open brace. */
#endif /* DEBUG */
  	      ostr << inbuf[i]; 
  	      if (inbuf[i] == ';')
  	      { // Hey! We've got a ; before seeing an open brace. That means 
  	        // we're dealing with a template declaration. we can't instrument
  	        // a declaration, only a definition. 
  	        instrumented = true; 	
  	        // by setting instrumented as true, it won't look for an open
  	        // brace on this line 
  	      }
            } // if open brace 
  	  } // for i loop
  	  ostr <<endl;
   	  // if we didn't find the open brace on the desired line, if its in the 
  	  // next line go on with the while loop till we return instrumented 
     	  // becomes true. 
          break;

	  case EXIT:

#ifdef DEBUG 
		cout <<"Exit" <<endl;
		cout <<"using_exit_keyword = "<<using_exit_keyword<<endl;
		cout <<"exit_keyword = "<<exit_keyword<<endl;
		cout <<"infbuf[(*it)->col-1] = "<<inbuf[(*it)->col-1]<<endl;
#endif /* DEBUG */
		/* first flush out all characters till our column */
		for (k = 0; k < (*it)->col-1; k++) ostr<<inbuf[k];
		if ((strncmp(&inbuf[(*it)->col-1], "abort", strlen("abort")) == 0) 
		  ||(strncmp(&inbuf[(*it)->col-1], "exit", strlen("exit")) == 0) 
		  ||(using_exit_keyword && (strncmp(&inbuf[(*it)->col-1], 
				exit_keyword, strlen(exit_keyword)) == 0) ))
                {
#ifdef DEBUG
		  cout <<"WRITING EXIT RECORD "<<endl;
#endif /* DEBUG */
                  if (use_spec)
                  {
                    /* XXX Insert code here */
                  }
                  else if (use_perflib)
                  {
                    /* XXX Insert code here */
                  }
                  else {
		    ostr <<"{ TAU_PROFILE_EXIT(\"exit\"); ";
                  }
		  for (k = (*it)->col-1; k < strlen(inbuf) ; k++)
		    ostr<<inbuf[k]; 
                  ostr <<endl;
                  if (use_spec)
                  {
                    /* XXX Insert code here */
                  }
                  else if (use_perflib)
                  {
                    /* XXX Insert code here */
                  }
                  else {
		    ostr <<"      }";
                  }
		  ostr <<endl;
		  instrumented = true; 
		} else {
		  fprintf (stderr, "Warning: exit was found at line %d, column %d, but wasn't found in the source code.\n",(*it)->line, (*it)->col);
		  fprintf (stderr, "If the exit call occurs in a macro (likely), make sure you place a \"TAU_PROFILE_EXIT\" before it (note: this warning will still appear)\n");
		  for (k = (*it)->col-1; k < strlen(inbuf); k++)
		    ostr<<inbuf[k]; 
		  instrumented = true;
		  // write the input line in the output stream
		}            
            break;

	  case START_LOOP_TIMER:
		for (k = 0; k < (*it)->col-1; k++) ostr<<inbuf[k];
                if (use_spec)
                {
                  /* XXX Insert code here */
                }
                else if (use_perflib)
                {
                  /* Insert code here */
                }
                else
		  ostr<<"\n{ TAU_PROFILE(\""<<(*it)->snippet<<"\", \" \", TAU_USER);"<<endl;
		for (k = 0; k < (*it)->col-1; k++) ostr<<" "; /* put spaces */
		/* if there is another instrumentation request on the same line */
		
		instrumented = true;
	        if ((it+1) != itemvec.end())
	        { /* there are other instrumentation requests */
	          if (((*it)->line == (*(it+1))->line) && ((*(it+1))->kind == STOP_LOOP_TIMER))
		  {
                    write_upto = (*(it+1))->col - 1 ; 
#ifdef DEBUG
		    cout <<"There was a stop timer on the same line: "<<(*it)->line<<endl;
#endif /* DEBUG */
		    for (k=(*it)->col-1; k < write_upto; k++) ostr<<inbuf[k];
                    if (use_spec)
                    {
                      /* XXX Insert code here */
                    }
                    else if (use_perflib)
                    {
                      /* XXX Insert code here */
                    }
                    else
                    {
		      ostr <<" } ";
                    }
		    for (k=write_upto; k < strlen(inbuf); k++) ostr<<inbuf[k];
		    ostr<<endl;
		    it++; /* increment the iterator so this request is not processed again */
		    break; /* finish the processing for this special case */
		  } /* there is no stop timer on the same line */
	        } /* or there are no more instrumentation requests -- flush out */
		for (k = (*it)->col-1; k < strlen(inbuf) ; k++)
		  ostr<<inbuf[k]; 
                ostr<<endl;
	    	break;
	  case STOP_LOOP_TIMER:
		for (k = 0; k < (*it)->col-1; k++) ostr<<inbuf[k];
                if (use_spec)
                {
                  /* XXX Insert code here */
                }
                else if (use_perflib)
                {
                  /* XXX Insert code here */
                }
                else
                {
		  ostr <<"}"<<endl;
                }
		for (k = (*it)->col-1; k < strlen(inbuf) ; k++)
		  ostr<<inbuf[k]; 
		instrumented = true;
	    	break;
	  case GOTO_STOP_TIMER:
		/* first flush all the characters till we reach the goto */
		for (k = 0; k < (*it)->col-1; k++) ostr<<inbuf[k];
#ifdef DEBUG
		cout <<"WRITING STOP LOOP TIMER RECORD "<<endl;
#endif /* DEBUG */
                if (use_spec)
                {
                  /* XXX Insert code here */
                }
                else if (use_perflib)
                {
                  /* XXX Insert code here */
                }
                else
                {
		  ostr <<"{ TAU_PROFILE_STOP(lt); "; 
                }
		for (k = (*it)->col-1; k < strlen(inbuf) ; k++)
		  ostr<<inbuf[k]; 
                if (use_spec)
                {
                  /* XXX Insert code here */
                }
                else if (use_perflib)
                {
                  /* XXX Insert code here */
                }
                else
                {
		  ostr <<" }";
                }
		ostr <<endl;
		instrumented = true; 
	    break;
	  case INSTRUMENTATION_POINT:
#ifdef DEBUG
	    cout <<"Instrumentation point -> line = "<< (*it)->line<<endl;
	    cout <<"col = "<<(*it)->col<<endl;
	    cout <<"inbuf = "<<inbuf<<endl;
#endif /* DEBUG */
  	    for (i = 0; i < (*it)->col-1 ; i++) ostr << inbuf[i];
	    if ((*it)->attribute == BEFORE)
            {
	      ostr << (*it)->snippet<<endl;
            }
	    else 
            { /* after */
	      ostr << endl;
	    }
	    /* We need to add code to write the rest of the buffer */
	    if ((it+1) != itemvec.end())
	    { /* there are other instrumentation requests */
		if ((*it)->line == (*(it+1))->line)
		{
                  write_upto = (*(it+1))->col - 1 ; 
#ifdef DEBUG
		  cout <<"There were other requests for the same line write_upto = "<<write_upto<<endl;
#endif /* DEBUG */
		  print_cr = true;
		  instrumented = true; /* let it get instrumented in the next round */
		}
                else
		{
                  write_upto = strlen(inbuf);
#ifdef DEBUG
		  cout <<"There were no other requests for the same line write_upto = "<<write_upto<<endl;
#endif /* DEBUG */
		  print_cr = true;
		  instrumented = true; /* let it get instrumented in the next round */
		}
	    } 
	    else 
	    { /* this was the last request - flush the inbuf */
	      write_upto = strlen(inbuf);
	      print_cr = true;
              instrumented = true; 
            }
	    for (space = 0; space < (*it)->col-1; space++) ostr <<" ";
	    /* write out the snippet! */
	    if ((*it)->attribute == AFTER)
            {
	      ostr << (*it)->snippet<<endl;
	      for (space = 0; space < (*it)->col-1; space++) ostr <<" ";
            }
#ifdef DEBUG
            printf("it col -1 = %d, write_upto = %d\n", (*it)->col-1, write_upto);
#endif /* DEBUG */
	    for (i = (*it)->col-1; i < write_upto; i++)
            {
#ifdef DEBUG
              printf("Writing (3.1) inbuf[%d] = %c\n", i, inbuf[i]);
#endif /* DEBUG */
              ostr << inbuf[i]; 
            }
	    if (print_cr) ostr <<endl; 
	    break;
	  default:
	    cout <<"Unknown option in instrumentCXXFile:"<<(*it)->kind<<endl;
	    instrumented = true;
	    break;
        }
      } // else      

      memset(inbuf, INBUF_SIZE, 0); // reset to zero
    } // while loop

  } // For loop
  // For loop is over now flush out the remaining lines to the output file
  while (istr.getline(inbuf, INBUF_SIZE) ) 
  { 
    ostr << inbuf <<endl;
  }
  // written everything. quit and debug!
  ostr.close();

#ifdef DEBUG
  cout <<"Everything is ok!"<<endl;
#endif /* DEBUG */
  return true; /* everything is ok */
}

char return_nonvoid_string[256] = "return";
char return_void_string[256] = "return";
char use_return_void[256] = "return";
char use_return_nonvoid[256] = "return";




/* -------------------------------------------------------------------------- */
/* -- writeLongFortranStatement breaks up the suffix in one or more lines --- */
/* -------------------------------------------------------------------------- */
void writeLongFortranStatement(ostream &ostr, string& prefix, string suffix) {
  string fullstring = prefix + suffix + string("')");

  if (fullstring.length() <= 72) {
    ostr << fullstring << endl;
    return;
  }

  string s1 = prefix;
  string s2 = "";
  int length = s1.length();
  for (int i=length; i < 72; i++) {
    s2 = s2 + " ";
  }

  string full = s1 + s2 + "&\n";

  // continue to break lines in the correct spot
  while (suffix.length() > 64) {
    string first = suffix.substr(0,64);
    suffix.erase(0,64);
    full = full + "     &"+first+"&\n";
  }
  
  full = full + "     &"+suffix+"')";

  ostr << full << endl;

}

/* -------------------------------------------------------------------------- */
/* -- writeFortranTimer writes the long timer name in two or more statements  */
/* -- if necessary. It invokes writeLongFortranStatement -------------------  */
/* -------------------------------------------------------------------------- */
void writeFortranTimer(ostream &ostr, string timername, itemRef *i) {
  string prefix;

  if (i->isDynamic) {
    ostr << "      tau_iter = tau_iter + 1"<<endl;
    if (i->isPhase) { /* dynamic phase */
      prefix = string("      call TAU_PHASE_DYNAMIC_ITER(tau_iter, profiler, '");
      
    } else { /* dynamic timer */
      prefix = string("      call TAU_PROFILE_DYNAMIC_ITER(tau_iter, profiler, '");
    } /* end of dynamic */
  } else { /* it is static */
      if (i->isPhase) { /* static phase */
        prefix = string("      call TAU_PHASE_CREATE_STATIC(profiler, '");
      } else { /* static timer */
        prefix = string("      call TAU_PROFILE_TIMER(profiler, '");
      }  /* static timer */
  } /* is static? */
  writeLongFortranStatement(ostr, prefix, timername);
}



/* -------------------------------------------------------------------------- */
/* -- BodyBegin for a routine that does return some value ------------------- */
/* -------------------------------------------------------------------------- */
void processNonVoidRoutine(ostream& ostr, string& return_type, itemRef *i, string& group_name)
{
  int space; 
#ifdef DEBUG
  cout <<"Return type :" << return_type<<endl;
#endif /* DEBUG */
  ostr << "{\n\t" << return_type << " tau_ret_val;\n";
  writeAdditionalDeclarations(ostr, (pdbRoutine *)(i->item));
  ostr << "\n\t";
  if (use_spec)
  {
    ostr << i->snippet << endl;
    /* XXX Insert code here */
  }
  else if (use_perflib)
  {
    ostr<<"Perf_Update(\"" <<((pdbRoutine *)(i->item))->name() << "\", 1);"<<endl;
  }
  else
  {
    ostr <<getCreateMeasurementEntity(i)<<"(tautimer, \""<<
      getInstrumentedName(i->item) << "\", \" " << "\",";
      // ((pdbRoutine *)(i->item))->signature()->name() << "\", ";

    if (strcmp(i->item->name().c_str(), "main")==0)
    { /* it is main() */
      ostr << "TAU_DEFAULT);" <<endl; // give an additional line
#ifdef SPACES
      for (space = 0; space < (*it)->col ; space++) ostr << " " ;
#endif
      // leave some leading spaces for formatting...

      print_tau_profile_init(ostr, (pdbCRoutine *) (i->item));
      ostr <<"#ifndef TAU_MPI" <<endl; // set node 0
      ostr <<"#ifndef TAU_SHMEM" <<endl; // set node 0
      ostr <<"  TAU_PROFILE_SET_NODE(0);" <<endl; // set node 0
      ostr <<"#endif /* TAU_SHMEM */" <<endl; // set node 0
      ostr <<"#endif /* TAU_MPI */" <<endl; // set node 0
    }
    else
    {
      ostr <<group_name<<");" <<endl; // give an additional line
    }

    ostr <<"\t"<<getStartMeasurementEntity(i)<<"(tautimer); "<<endl;
    ostr << i->snippet << endl;
  }
}

/* -------------------------------------------------------------------------- */
/* -- Body Begin for a void C routine --------------------------------------- */
/* -------------------------------------------------------------------------- */
void processVoidRoutine(ostream& ostr, string& return_type, itemRef *i, string& group_name)
{
  int space;
  ostr << "{\n";
  writeAdditionalDeclarations(ostr, (pdbRoutine *)(i->item));
  ostr << "\n\t";
  if (use_spec)
  {
    ostr << i->snippet << endl;
    /* XXX Insert code here */
  }
  else if (use_perflib)
  {
    ostr<<"Perf_Update(\"" <<((pdbRoutine *)(i->item))->name() << "\", 1);"<<endl;
  }
  else
  {
    ostr <<getCreateMeasurementEntity(i)<<"(tautimer, \""<<
      getInstrumentedName(i->item) << "\", \" " << "\", ";
      //((pdbRoutine *)(i->item))->signature()->name() << "\", ";
  
    if (strcmp(i->item->name().c_str(), "main")==0)
    { /* it is main() */
       ostr << "TAU_DEFAULT);" <<endl; // give an additional line
#ifdef SPACES
       for (space = 0; space < (*it)->col ; space++) ostr << " " ;
#endif
       // leave some leading spaces for formatting...
  
       print_tau_profile_init(ostr, (pdbCRoutine *) (i->item));
       ostr <<"#ifndef TAU_MPI" <<endl; // set node 0
       ostr <<"  TAU_PROFILE_SET_NODE(0);" <<endl; // set node 0
       ostr <<"#endif /* TAU_MPI */" <<endl; // set node 0
    }
    else
    {
      ostr <<group_name<<");" <<endl; // give an additional line
    }
  
    ostr <<"\t"<<getStartMeasurementEntity(i)<<"(tautimer);"<<endl;
    ostr << i->snippet << endl;
  }
}

/* -------------------------------------------------------------------------- */
/* -- Checks to see if string is blank  ------------------------------------- */
/* -------------------------------------------------------------------------- */
bool isBlankString(string& s)
{
  int i;
  const char * chr = s.c_str();
  if (!chr) /* if chr is 0 or null, it is blank. */
  {
    return true;
  }
  else
  { /* it is not null, we need to examine each character */
    i = 0;
    while (chr[i] != '\0')
    { /* keep going to the end ... */
      if (! (chr[i] == ' ' || chr[i] == '\t')) return false;
      /* if it is not a space, just return a false -- it is not a blank */
      i++; /* see next character */
    } /* reached the end, and didn't find any non-white space characters */
  }
  return true; /* the string has only white space characters */
}


/* -------------------------------------------------------------------------- */
/* -- Close the loop timer before the return                 ---------------- */
/* -------------------------------------------------------------------------- */
void processCloseLoopTimer(ostream& ostr)
{
  for (list<string>::iterator siter = current_timer.begin(); 
    siter != current_timer.end(); siter++)
  { /* it is not empty -- we must shut the timer before exiting! */
#ifdef DEBUG 
    cout <<"Shutting timer "<<(*siter)<<" before stopping the profiler "<<endl;
#endif /* DEBUG */
    ostr <<" TAU_PROFILE_STOP(lt); ";
  }
 
}
/* -------------------------------------------------------------------------- */
/* -- Writes the return expression to the instrumented file  ---------------- */
/* -------------------------------------------------------------------------- */
void processReturnExpression(ostream& ostr, string& ret_expression, itemRef *it, char *use_string)
{
  if (isReturnTypeReference(it) ||
      isBlankString(ret_expression) ||
      (use_spec && isBlankString(it->snippet)))
  {
    ostr <<"{ ";
    processCloseLoopTimer(ostr);
    ostr << it->snippet << " ";
    if (use_spec)
    {
      /* XXX Insert code here */
    }
    else if (use_perflib)
    { 
      ostr<<"Perf_Update(\""<< ((pdbRoutine *)(it->item))->name()<<"\", 0);";
    }
    else
    {
      ostr <<getStopMeasurementEntity(it)<<"(tautimer);";
    }
    ostr << use_string << " " << (ret_expression) << "; }" << endl;
  }
  else 
  {
    ostr <<"{ tau_ret_val = " << ret_expression << "; ";
    processCloseLoopTimer(ostr);
    ostr << it->snippet << " ";
    if (use_spec)
    {
      /* XXX Insert code here */
    }
    else if (use_perflib)
    {
      ostr<<"Perf_Update(\""<< ((pdbRoutine *)(it->item))->name()<<"\", 0); ";
    }
    else
      ostr<<getStopMeasurementEntity(it)<<"(tautimer); ";
    ostr << use_string << " " << "(tau_ret_val); }" << endl;
  }
}

/* -------------------------------------------------------------------------- */
/* -- Writes the exit expression to the instrumented file  ------------------ */
/* -------------------------------------------------------------------------- */
void processExitExpression(ostream& ostr, string& exit_expression, itemRef *it, char *use_string, bool abort_used)
{

  ostr <<"{ ";
  if (abort_used){
    ostr<<"int tau_exit_val = 0;";
    if (use_spec)
    {
      /* XXX Insert code here */
    }
    else if (use_perflib)
    {
     ostr<<"Perf_Update(\""<< ((pdbRoutine *)(it->item))->name()<<"\", 0);}";
    }
    else 
    {
      ostr<<"TAU_PROFILE_EXIT("<<"\""<<use_string<<"\");";
    }
    ostr << " " << use_string << " (); }" << endl;
  }
  else 
  {
    ostr<<"int tau_exit_val = "<<exit_expression<<"; ";
    if (use_spec)
    {
      /* XXX Insert code here */
    }
    else if (use_perflib)
    {
      ostr<<"Perf_Update(\""<< ((pdbRoutine *)(it->item))->name()<<"\", 0);";
    }
    else
    {
      ostr<<"TAU_PROFILE_EXIT("<<"\""<<use_string<<"\");";
    }
    ostr << " " << use_string << " (tau_exit_val); }" << endl;
  }
}


/* -------------------------------------------------------------------------- */
/* -- Instrumentation routine for a C program ------------------------------- */
/* -------------------------------------------------------------------------- */
bool instrumentCFile(PDB& pdb, pdbFile* f, string& outfile, string& group_name, string& header_file) 
{ 
  int inbufLength, i, j, space;
  string file(f->name());
  static char inbuf[INBUF_SIZE]; // to read the line
  static char exit_type[EXIT_KEYWORD_SIZE]; // to read the line
  string exit_expression;
  bool abort_used = false;
  char newline;
  newline = '\n'; /* for C \ processing in return statements */
  // open outfile for instrumented version of source file
  ofstream ostr(outfile.c_str());
  string timercode; /* for outer-loop level timer-based instrumentation */
  if (!ostr) {
    cerr << "Error: Cannot open '" << outfile << "'" << endl;
    return false;
  }
  // open source file
  ifstream istr(file.c_str());
  if (!istr) {
    cerr << "Error: Cannot open '" << file << "'" << endl;
    return false;
  }
#ifdef DEBUG
  cout << "Processing " << file << " in instrumentCFile..." << endl;
#endif


  memset(inbuf, INBUF_SIZE, 0); // reset to zero
  // initialize reference vector
  vector<itemRef *> itemvec;
  if (!use_spec)
  {
    /* In "spec" mode, only the file instrumentation requests are used */
    getCReferences(itemvec, pdb, f);
  }
  /* check if the given file has line/routine level instrumentation requests */
  if (!isInstrumentListEmpty()) 
  { /* there are finite instrumentation requests, add requests for this file */
    addFileInstrumentationRequests(pdb, f, itemvec);
  }
  /* All instrumentation requests are in. Now do postprocessing. */
  postprocessInstrumentationRequests(itemvec);


  // Begin Instrumentation
  // put in code to insert <Profile/Profiler.h>
  if (use_spec)
  {
    /* XXX Insert code here */
  }
  else if (use_perflib)
    ostr<< "void Perf_Update(char *name, int entry);"<<endl;
  else 
    ostr<< "#include <"<<header_file<<">"<<endl;
  if (memory_flag)
    ostr<< "#include <malloc.h>"<<endl;
  defineTauGroup(ostr, group_name); 

  int inputLineNo = 0;
  vector<itemRef *>::iterator lit = itemvec.begin();
  while (lit != itemvec.end() && !istr.eof())
  {
    // Read one line each till we reach the desired line no.
#ifdef DEBUG
    if ((*lit) && (*lit)->item)
      cout <<"S: "<< (*lit)->item->fullName() << " line "<< (*lit)->line << " col " << (*lit)->col << endl;
#endif
    bool instrumented = false;
    while((instrumented == false) && (istr.getline(inbuf, INBUF_SIZE)) )
    {
      inputLineNo ++;
#ifdef DEBUG
        cout <<"In while: inbuf: "<<inbuf<<" inputline no "
	<<inputLineNo<< endl;
#endif /* DEBUG */
      if (inputLineNo < (*lit)->line)
      {
#ifdef DEBUG
	cout <<"Writing(3): "<<inbuf<<endl;
#endif /* DEBUG */
        // write the input line in the output stream
        ostr << inbuf <<endl;
      }
      else
      { /* We're at the desired line no. */
        for(i=0; i< ((*lit)->col)-1; i++)
	{ 
#ifdef DEBUG
	  cout <<"Writing(1): "<<inbuf[i]<<endl;
#endif /* DEBUG */
	  ostr << inbuf[i];
	}
        vector<itemRef *>::iterator it;
        for (it = lit; ((it != itemvec.end()) && ((*it)->line == (*lit)->line)); ++it) 
        { /* it/lit */
          inbufLength = strlen(inbuf);

#ifdef DEBUG 
	  cout <<"Line " <<(*it)->line <<" Col " <<(*it)->col <<endl;
#endif /* DEBUG */
	  /* set instrumented = true after inserting instrumentation */
	  string return_string; 
	  int write_from, write_upto;
	  int k;
	  write_from = ((*it)->col)-1; 

        /* Examine the instrumentation request */
	  switch ((*it)->kind) {
	    case BODY_BEGIN: 
#ifdef DEBUG 
		cout <<"Body Begin" <<endl;
#endif /* DEBUG */
/*
		return_string = ((pdbRoutine *)((*it)->item))->signature()->returnType()->name() ;
*/
		{
   		  const pdbType *t = ((pdbRoutine *)((*it)->item))->signature()->returnType();
   		  if ( const pdbGroup* gr = t->isGroup() )
		  {
     		    return_string = gr->name();
/* IT WAS gr->fullName(); we changed it to account for unnamed namespaces */
#ifdef TAU_SPECIFY_FULL_NAMES_IN_RETURN_TYPE
     		    return_string = gr->fullName();
#endif /* TAU_SPECIFY_FULL_NAMES_IN_RETURN_TYPE */
		  }
   		  else
		  {
     		    return_string = t->name();
/* IT WAS t->fullName(); we changed it to account for unnamed namespaces */
/* Unnamed namespaces pose a unique problem! We get the name as 
  <unnamed@6000000000074ab8>::COLORS instead of COLORS. We need to get rid of 
  this part */
#ifdef TAU_SPECIFY_FULL_NAMES_IN_RETURN_TYPE
     		    return_string = t->fullName();
#endif /* TAU_SPECIFY_FULL_NAMES_IN_RETURN_TYPE */
		  }
		}

		/* If return type is a reference, treat it as a void */
	        if (isVoidRoutine((*it)->item) || isReturnTypeReference(*it))
		{
#ifdef DEBUG 
		  cout <<"Void return value "<<endl;
#endif /* DEBUG */
		  processVoidRoutine(ostr, return_string, *it, group_name);
		}
		else
		{
		  processNonVoidRoutine(ostr, return_string, *it, group_name);
		}
		instrumented = true; 
		break;
	    case RETURN: 
#ifdef DEBUG 
		cout <<"Return "<<endl;
#endif /* DEBUG */
 		process_this_return = false;
		if (strncmp((const char *)&inbuf[((*it)->col)-1], 
			return_void_string, strlen(return_void_string))==0)
		{
		  if (!isalnum(inbuf[((*it)->col)-1 + strlen(return_void_string)])) {
		    process_this_return = true; 
		    strcpy(use_return_void, return_void_string);
		  }
	        }
		if (strncmp((const char *)&inbuf[((*it)->col)-1], 
			return_nonvoid_string, strlen(return_nonvoid_string))==0)
		{
		  if (!isalnum(inbuf[((*it)->col)-1 + strlen(return_nonvoid_string)])) {
		    process_this_return = true; 
		    strcpy(use_return_nonvoid, return_nonvoid_string);
		  }
	        }
		if (strncmp((const char *)&inbuf[((*it)->col)-1], 
			"return", strlen("return")) == 0)
		{
		  if (!isalnum(inbuf[((*it)->col)-1 + strlen("return")])) {
		      process_this_return = true;
		      strcpy(use_return_void, "return");
		      strcpy(use_return_nonvoid, "return");
		  }
		}

		if (process_this_return)
		{
		  if (isVoidRoutine((*it)->item))
		  {	
#ifdef DEBUG 
		    cout <<" Return for a void routine" <<endl;
#endif /* DEBUG */
		    /* instrumentation code here */
	            if (use_spec)
                    {
                      ostr << "{ " << (*it)->snippet << " " << use_return_void << "; }" << endl;
                      /* XXX Insert code here */
                    }
                    else if (use_perflib)
		    {
		      ostr<<"{ Perf_Update(\""<< ((pdbRoutine *)((*it)->item))->name()<<"\", 0);"<<use_return_void<<";}"<<endl;
		    }
		    else 
		        ostr << "{ "<< (*it)->snippet << " " <<getStopMeasurementEntity(*it)<<"(tautimer); "<<use_return_void<<"; }" <<endl;
		    for (k=((*it)->col)-1; inbuf[k] !=';'; k++)
		     ;
		    write_from = k+1;
		  }
		  else
		  {
		    string ret_expression; 
#ifdef DEBUG 
		    cout <<"Return for a non void routine "<<endl;
#endif /* DEBUG */
		    for (k = (*it)->col+strlen(use_return_nonvoid)-1; (inbuf[k] != ';') && (k<inbufLength) ; k++)
		    {
                      char current_char = inbuf[k];
		      ret_expression.append(&current_char, 1);
                      if ((inbuf[k] == '\\' ) && (inbuf[k+1] == '\0')) {
			ret_expression.append(&newline, 1);
                      }
		    }
#ifdef DEBUG
		    cout <<"k = "<<k<<" inbuf = "<<inbuf[k]<<endl;
#endif /* DEBUG */
		    if (inbuf[k] == ';')
		    { /* Got the semicolon. Return expression is in one line. */
#ifdef DEBUG
		      cout <<"No need to read in another line"<<endl;
#endif /* DEBUG */
	              write_from = k+1;
		    }
 		    else	
		    {
		      int l;   
		      do {
#ifdef DEBUG
 		        cout <<"Need to read in another line to get ';' "<<endl;
#endif /* DEBUG */
			if(istr.getline(inbuf, INBUF_SIZE)==NULL)
			{   
			  perror("ERROR in reading file: looking for ;"); 
			  exit(1); 
			}
			inbufLength = strlen(inbuf);
                        inputLineNo ++;
			/* Now search for ; in the string */
			for(l=0; (inbuf[l] != ';') && (l < inbufLength); l++)
			{
			  ret_expression.append(&inbuf[l], 1);
                          if (inbuf[l] == '\\') ret_expression.append(&newline, 1);
			}
		      } while(inbuf[l] != ';');
			/* copy the buffer into inbuf */
		      write_from = l+1; 
		    }
			 
#ifdef DEBUG 
		    cout <<"ret_expression = "<<ret_expression<<endl;
#endif /* DEBUG */
		    processReturnExpression(ostr, ret_expression, *it, use_return_nonvoid); 
		    /* instrumentation code here */
		  }
		}
		else 
		{ 
		  /* if there was no return */
		  write_from =  (*it)->col - 1;
#ifdef DEBUG
		  cout <<"WRITE FROM (no return found) = "<<write_from<<endl;
		  cout <<"inbuf = "<<inbuf<<endl;
#endif /* DEBUG */
		}

		instrumented = true; 
		break;
	    case BODY_END: 
#ifdef DEBUG 
		cout <<"Body End "<<endl;
#endif /* DEBUG */
                if (use_spec)
                {
                  ostr << "\n}\n\t" << (*it)->snippet << endl;
                  /* XXX Insert code here */
                }
                else if (use_perflib)
		{
		  ostr<<"\n}\n\tPerf_Update(\""<< ((pdbRoutine *)((*it)->item))->name()<<"\", 0);"<<endl;
		}
		else 
		{
                  ostr << "\n}\n\t" << (*it)->snippet << endl;
		  ostr<<"\t"<<getStopMeasurementEntity((*it))<<"(tautimer);\n"<<endl; 
		}
		instrumented = true; 
		break;
	    case EXIT:
#ifdef DEBUG 
		cout <<"Exit" <<endl;
		cout <<"using_exit_keyword = "<<using_exit_keyword<<endl;
		cout <<"exit_keyword = "<<exit_keyword<<endl;
		cout <<"infbuf[(*it)->col-1] = "<<inbuf[(*it)->col-1]<<endl;
#endif /* DEBUG */
                memset(exit_type, EXIT_KEYWORD_SIZE, 0); // reset to zero
		abort_used = false; /* initialize it */
		if (strncmp(&inbuf[(*it)->col-1], "abort", strlen("abort")) == 0) 
	        {
                   strcpy(exit_type, "abort");
		   abort_used  = true; /* abort() takes void */
	        }
		if (strncmp(&inbuf[(*it)->col-1], "exit", strlen("exit")) == 0) 
	        {
		   strcpy(exit_type, "exit");
		}
		if (using_exit_keyword && (strncmp(&inbuf[(*it)->col-1], 
				exit_keyword, strlen(exit_keyword)) == 0) )
                {
		   strcpy(exit_type, exit_keyword);
		}
#ifdef DEBUG 
		    cout <<"Return for a non void routine "<<endl;
#endif /* DEBUG */
		exit_expression.clear();
		if (exit_type != '\0')
		{ /* is it null, or did we copy something into this string? */
		  for (k = (*it)->col+strlen(exit_type)-1; (inbuf[k] != ';') && (k<inbufLength) ; k++)
		    exit_expression.append(&inbuf[k], 1);
#ifdef DEBUG
		    cout <<"k = "<<k<<" inbuf = "<<inbuf[k]<<endl;
#endif /* DEBUG */
		    if (inbuf[k] == ';')
		    { /* Got the semicolon. Return expression is in one line. */
#ifdef DEBUG
		      cout <<"No need to read in another line"<<endl;
#endif /* DEBUG */
	              write_from = k+1;
		    }
 		    else	
		    {
		      int l;   
		      do {
#ifdef DEBUG
 		        cout <<"Need to read in another line to get ';' "<<endl;
#endif /* DEBUG */
			if(istr.getline(inbuf, INBUF_SIZE)==NULL)
			{   
			  perror("ERROR in reading file: looking for ;"); 
			  exit(1); 
			}
			inbufLength = strlen(inbuf);
                        inputLineNo ++;
			/* Now search for ; in the string */
			for(l=0; (inbuf[l] != ';') && (l < inbufLength); l++)
			{
			  exit_expression.append(&inbuf[l], 1);
			}
		      } while(inbuf[l] != ';');
			/* copy the buffer into inbuf */
		      write_from = l+1; 
		    }
			 
#ifdef DEBUG 
		    cout <<"exit_expression = "<<exit_expression<<endl;
#endif /* DEBUG */
		    processExitExpression(ostr, exit_expression, *it, exit_type, abort_used); 
		}
		else { /* exit_type was null! Couldn't find anything here */
		  fprintf (stderr, "Warning: exit was found at line %d, column %d, but wasn't found in the source code.\n",(*it)->line, (*it)->col);
		  fprintf (stderr, "If the exit call occurs in a macro (likely), make sure you place a \"TAU_PROFILE_EXIT\" before it (note: this warning will still appear)\n");
		  instrumented = true;
		  // write the input line in the output stream
		}            
		break; 
	
    case INSTRUMENTATION_POINT:
#ifdef DEBUG
	cout <<"Instrumentation point in C -> line = "<< (*it)->line<<endl;
#endif /* DEBUG */
	if ((*it)->attribute == AFTER) ostr<<endl;
	ostr << (*it)->snippet<<endl;
	instrumented = true;
	break;

    case START_LOOP_TIMER: 
	if ((*it)->attribute == AFTER) ostr<<endl;
        if (use_spec)
        {
          timercode = "{ ";
          /* XXX Insert code here */
        }
        else if (use_perflib)
        {
          timercode = string(string("{ Perf_Update(\"" )+(*it)->snippet+", 1); ");
        }
	else
	{
	  timercode = string(string("{ TAU_PROFILE_TIMER(lt, \"")+(*it)->snippet+"\", \" \", TAU_USER); TAU_PROFILE_START(lt); ");
        }
#ifdef DEBUG
	cout <<"Inserting timercode: "<<timercode<<endl;
#endif /* DEBUG */
	ostr <<timercode <<endl;
	/* insert spaces to make it look better */
	for(space = 0; space < (*it)->col-1; space++) ostr<<" ";
	instrumented = true;
	current_timer.push_front((*it)->snippet); 
	/* Add this timer to the list of currently open timers */
	break;

    case STOP_LOOP_TIMER: 
	if ((*it)->attribute == AFTER) ostr<<endl;
	/* insert spaces to make it look better */
	for(space = 0; space < (*it)->col-1; space++) ostr<<" ";
        if (use_spec)
        {
          timercode = " }";
          /* XXX Insert code here */
        }
        else if (use_perflib)
        {
          timercode = string(string(" Perf_Update(\"" )+(*it)->snippet+", 0); } ");
        }
	else
        {
          ostr << "TAU_PROFILE_STOP(lt); } "<<endl;
        }
	instrumented = true;
        /* pop the current timer! */
	if (!current_timer.empty()) current_timer.pop_front();
	break;

    case GOTO_STOP_TIMER:
        ostr << "{ ";
        if (use_spec)
        {
          /* XXX Insert code here */
        }
        else if (use_perflib)
        {
          /* XXX Insert code here */
        }
        else {
	  ostr <<"TAU_PROFILE_STOP(lt);";
        }
	for (k = (*it)->col-1; k < strlen(inbuf); k++)
	  ostr<<inbuf[k];
	ostr <<" }";
	write_from = k+1;
	instrumented = true;
	break;

    default:
	cout <<"Unknown option in instrumentCFile:"<<(*it)->kind<<endl;
	instrumented = true; 
	break;
  } /* Switch statement */
  if (it+1 != itemvec.end())
  {
    write_upto = (*(it+1))->line == (*it)->line ? (*(it+1))->col-1 : inbufLength; 
#ifdef DEBUG
    cout <<"CHECKING write_from "<<write_from <<" write_upto = "<<write_upto<<endl;
	    cout <<"it = ("<<(*it)->line<<", "<<(*it)->col<<") ;";
	    cout <<"it+1 = ("<<(*(it+1))->line<<", "<<(*(it+1))->col<<") ;"<<endl;
#endif /* DEBUG */
	  }
	  else
	    write_upto = inbufLength; 

#ifdef DEBUG
   	  cout <<"inbuf: "<<inbuf<<endl;
#endif /* DEBUG */
	  for (j=write_from; j < write_upto; j++)
	  {
#ifdef DEBUG 
   	    cout <<"Writing(4): "<<inbuf[j]<<endl;
#endif /* DEBUG */
	    ostr <<inbuf[j];
	  }
	  ostr <<endl;
	
          } /* for it/lit */
        lit=it; 
      } /* else line no*/
      memset(inbuf, INBUF_SIZE, 0); // reset to zero
    } /* while */
  } /* while lit != end */
  // For loop is over now flush out the remaining lines to the output file
  while (istr.getline(inbuf, INBUF_SIZE) )
  {
    ostr << inbuf <<endl;
  }
  // written everything. quit and debug!
  ostr.close();

  return true;
} /* End of instrumentCFile */ 


#define WRITE_SPACE(os, c) { char ch = c; if (!((ch == ' ') || (ch == '\t'))) ch = ' '; \
				os << ch; }

#define WRITE_TAB(os, column) if (column < 7) os<<"\t";

/* The macro below assumes that ostr, inbuf and print_cr are defined */
#define WRITE_SNIPPET(attr, col, write_upto, snippetcode) { \
	int i; /* the index into the array */ \
                if (attr == BEFORE)  \
		{ \
		  for(i = 0; i < col-1; i++) \
		  { \
                    ostr <<inbuf[i]; \
		  } \
                  ostr << snippetcode <<endl; \
		  ostr <<"\t";  /* write a tab */ \
                  for (i=col-1; i < write_upto; i++) \
                    ostr <<inbuf[i]; \
                  if (print_cr) ostr<<endl; \
		} \
                if (attr == AFTER) \
 		{ \
		  for(i = 0; i < write_upto; i++) \
                    ostr <<inbuf[i]; \
                  if (print_cr) ostr<<endl; \
                  ostr << snippetcode <<endl; \
		} \
	} 

/* In Fortran programs, it is a bad idea to begin the first column with a C
 * as it can be confused with a comment. So, we should check and see if the
 * "call TAU_PROFILE..." statement starts on the first column and if so, we 
 * should introduce a tab there. Hence, the need for the WRITE_TAB macro */




int CPDB_GetSubstringCol(const char *haystack, const char *needle)
{
  char *local_haystack = strdup(haystack);
  int length = strlen(local_haystack); 
  int i;
  for (i = 0; i < length; i++)
  { /* make it all lowercase */
    local_haystack[i] = tolower(haystack[i]);
    if (local_haystack[i] == '!') 
    { /* ignore all comments -- everything after ! on a line ... */
      local_haystack[i] = '\0';
      /* break out of the loop */
      break;
    }
  }
  /* We need to ensure that characters immediately before/after are not
     alpha numeric. i.e., it does not return the column associated with
     rcToReturn when we are searching for return. Same applies for after */

  bool check_before = false;
  bool check_after = false;
  const char *res;
  int col_found; 
  int needle_length = strlen(needle); /* how many characters are we searching */
  while (check_before == false || check_after == false)
  { /* keep searching until before and after are ok */
#ifdef DEBUG
    printf("Examining: %s, and %s\n", local_haystack, needle); 
#endif /* DEBUG */
    res = strstr(local_haystack, needle);
    /* is res non null? Did it find a match? */
    if (!res) break; /* out of the loop - it is null */
    /* we found something, examine the columns before and after it */
    col_found = res - local_haystack; 
#ifdef DEBUG 
    printf("Found col_found %d value= %c\n", col_found, local_haystack[col_found]);
    printf("local_haystack[col_found - 1] = %c, isalnum(): %d\n",
	local_haystack[col_found - 1], isalnum(local_haystack[col_found - 1] ));
#endif /* DEBUG */
    if (col_found == 0) break; /* no need to continue, didn't find anything! */
    if ((col_found) &&isalnum(local_haystack[col_found - 1])) {
      /* hey, the string has another needle which shouldn't match - rcToReturn */	
      local_haystack[col_found] = '^'; /* set it to something else rcTo^eturn*/
      continue; /* searching */
    }
    else {
      if (local_haystack[col_found - 1 ] == '_') { /* corner case foo_return */
	local_haystack[col_found] = '^';  /* set it to foo_^eturn */
	continue;
      }
      
      check_before = true; /* its ok, proceed */
#ifdef DEBUG
      printf("check_before = true, local_haystack[col_found+needle_length] = %c, isalnum(): %d, ispunct(): %d\n", local_haystack[col_found+needle_length],
	isalnum(local_haystack[col_found+needle_length]), 
	ispunct(local_haystack[col_found+needle_length]));
#endif /* DEBUG */
    }
    /* what about after return? */

    int punctuation = ispunct(local_haystack[col_found+needle_length]);
    if ((col_found) && (isalnum(local_haystack[col_found+needle_length]) ||
	ispunct(local_haystack[col_found+needle_length]))){
      /* hey, there is text after the return -- end subroutine return_foo */
      /* but, if it is a ( then it is ok as in allocate(foo(2,3)) */
      if (local_haystack[col_found+needle_length] != '(') {
        local_haystack[col_found + needle_length - 1 ] = '^'; 
      /* set it to  retur^_foo */
#ifdef DEBUG
      printf("local_haystack = %s\n", local_haystack);
#endif /* DEBUG */
        continue; /* searching */
      }
    }
    /* is it safe to examine after the column? */
    int hay_length = strlen(local_haystack);  /* we modified local_haystack, tolower,
	so we need to measure this length again : rctoreturn */
    
    if (col_found+needle_length >= hay_length) {
      /* it's fine */
      check_after = true; /* ok */
      continue;
    }
    /* aha! there is something else going on -- returned for instance */ 
#ifdef DEBUG
    printf("Col after return = %d, value  = %c, length=%d\n",
	col_found +needle_length, local_haystack[col_found+needle_length], hay_length);
#endif /* DEBUG */
    if (isalnum(local_haystack[col_found+needle_length])) {
      local_haystack[col_found] = '^'; /* check again ^eturned */
      continue; /* searching again */
    }
    else { /* character is ok */
      check_after = true; 
      continue; 
    }

  } /* keep searching until before and after are ok */
 
  int diff = 0;
  if (res)
  {
    diff = res - local_haystack + 1 ;  /* columns start from 1, not 0 */
#ifdef DEBUG 
    printf("needle:%s\n", needle);
    printf("haystack:%s\n", local_haystack);
    printf("diff = %d\n", diff);
#endif /* DEBUG */ 
  }
  free((void *)local_haystack);
  return diff;
}


/* -------------------------------------------------------------------------- */
/* -- Does the statement contain this keyword? ------------------------------ */
/* -------------------------------------------------------------------------- */
bool isKeywordPresent(char *line, const char *keyword)
{
   bool present;
   if ((!((line[0] == 'c') || (line[0] == 'C') || (line[0] == '!'))) && 
	  	    ( CPDB_GetSubstringCol(line, keyword) > 0 ))
     present = true;
   else
     present = false;

   /* is it present? */
#ifdef DEBUG
   cout <<"isKeywordPresent:"<<line<<" keyword: "<<keyword<<"? "<<present<<endl;
#endif /* DEBUG */
   return present; /* is it present? */
}

/* -------------------------------------------------------------------------- */
/* -- remove ! from the line and replace it with \0 ------------------------- */
/* -------------------------------------------------------------------------- */
void removeCommentFromLine(char * & inbuf)
{
  char *line = inbuf;
  int i, len;
    
  len = strlen(line);
  for (i=0 ; i < len, *line; i++, line++) {
    if ((*line == '"') || (*line == '\'')) {
      char quote = *line;
      line++;
      while (*line != NULL && *line != quote) {
	i++;
	line++;
      }
    }
    if (*line == '!') {
      *line = '\0'; 
      break;
    }
  }
#ifdef DEBUG
  printf("removeCommentFromLine: after cleaning inbuf= %s\n", inbuf);
#endif /* DEBUG */
  return ;
}

void removeContinuation(char* &line) {
  char *ptr = line + strlen(line) - 1;
  while (*ptr == ' ') ptr--;
  if (*ptr == '&') { 
    // remove trailing continuation char
    *ptr = 0;
  }
}


/* -------------------------------------------------------------------------- */
/* -- is it a continuation line? Checking the previous line for an &          */
/* -------------------------------------------------------------------------- */
bool continuedFromLine(char *line)
{
  int length = strlen(line);

  if (length == 0) return false; /* no it is not continued -- the prev. line is null */
  int i = length - 1;
#ifdef DEBUG 
  cout <<"CONTINUEDFROMLINE: Line = "<<line<<":Starting check from "<<line[i]<<endl;
  cout <<"length = "<<length <<endl;
#endif /* DEBUG */
  for (i = length -1; i != 0; i--)
  { 
#ifdef DEBUG
    printf("continuedFromLine: checking ... line[%d] = %c\n", i, line[i]);
#endif /* DEBUG */
    if (line[i] == ' ') continue;
    if (line[i] == '&') return true; /* it is continued */
    else return false; /* nope. something else! */
  }
/* OLD CODE :
  if (line[i] == '&') return true; 
  else return false; */ /* no continuation characters found */
  return false; 
}
/* -------------------------------------------------------------------------- */
/* -- is it a continuation line? Check for a ) from the end of current line
 * -- and check for a & at the end of the previous line (preferrably after
 * -- removing comments from the line i.e., all characters before ! comm      */
/* -------------------------------------------------------------------------- */
bool isContinuationLine(char *currentline, char *previousline, int columnToCheckFrom)
{
  int c;
  /* Check to see if return is in a continuation line */
  /* such as :
  *     if(  (ii/=jj  .or. kk<=0)  .and. &
  *           & (kcheck==0  .or. ii/=lsav+1 .or. kk>0) ) return
  */
#ifdef DEBUG
  cout <<"columnToCheckFrom = "<<columnToCheckFrom<<endl;
  cout <<"currentline = "<<currentline <<" prevline = "<<previousline<<" colToCheck "<<columnToCheckFrom<<endl;
#endif /* DEBUG */
  for(c = columnToCheckFrom; c > 0; c--)
  {
#ifdef DEBUG
    cout <<"c = "<<c<<"currentline[c] = "<<currentline[c]<<endl;
#endif /* DEBUG */
    if (currentline[c] == ' ' || currentline[c] == '\t') continue;
    if (currentline[c] == ')' || currentline[c] == '&' || 
       (c+1 == 6 && (currentline[c] != ' ' && currentline[c] != '0' && 
        currentline[4] == ' ' && currentline[3] == ' ' && currentline[2] == ' '
        && currentline[1] == ' ' && currentline[0] == ' ')) )
    { /* return is in a continuation line - has " ) return" */
      /* if there's a non blank, non zero character in 6th column and columns
       * 1 through 5 are blank, then it is a continuation line in Fixed form! */
      /* or something like:
       * if (x .gt. 3) &
       *   & return 
       * if (x .gt. 3) 
       *. return
       * is there something in the 6th column besides the space? 
       */
#ifdef DEBUG
      cout <<"currentline[c] = "<<currentline[c]<<endl;
#endif /* DEBUG */
      return true; /* it is a continuation line. Well, that it found a ) from if */
    }
    else
    { /* we found a character that was not a ) or a blank. Hmm could it be in the 6th column? */
      return false; /* nope there was something else */
    }
  } /* keep checking all the columns from the end */
  /* we haven't checked the previous line yet... */
#ifdef DEBUG
  cout <<"Reached here: currentline= "<<currentline<<endl;
  cout <<"Reached here: previousline= "<<previousline<<endl;
#endif /* DEBUG */
  if (continuedFromLine(previousline))
    return true; /* put in a then/endif clause as there is continuation */
  else
    return false; /* by default return a false */
}

/* -------------------------------------------------------------------------- */
/* -- Should we add a then and endif clause --------------------------------- */
/* -------------------------------------------------------------------------- */
bool addThenEndifClauses(char *currentline, char *previousline, int currentcol)
{
  char *checkbuf;  
  bool is_if_stmt = false; /* initialized to a false value */
  int i; 

  /* create a copy of the currentline */
  checkbuf = new char [strlen(currentline)+1];
  if (checkbuf == (char *) NULL) 
  {
    perror("ERROR: addThenEndifClauses: new returns NULL while creating checkbuf");
    exit(1);
  }

  if (currentcol < 0) currentcol = strlen(currentline);
  /* fill in checkbuf until the current construct */
  for  (i = 0; i < currentcol; i++)
  { /* currentcol is (*it)->col - 1; */
    checkbuf[i] = currentline[i];
  }
  checkbuf[i] = '\0'; 

  /* now that it is filled in, let us see if it has a "if" in it */
  is_if_stmt = isKeywordPresent(checkbuf, "if");

  /* Before we declare that we should insert the then clause,
   * we need to ensure that a then does not appear in the statement already */

  if (is_if_stmt == true)
  {
    /* does a then appear? */
    if (isKeywordPresent(checkbuf, "then"))
    is_if_stmt = false;

    /* here we are merely checking if we are inside a single-if
     * statement, one that does not have a then clause. If there
     * is a then clause in the same statement, then we classify
     * is_if_stmt as false */
   }

   /* Check to see if return is in a continuation line */
   /* such as :
    *     if(  (ii/=jj  .or. kk<=0)  .and. &
    *           & (kcheck==0  .or. ii/=lsav+1 .or. kk>0) ) return
    */

   if (is_if_stmt == false) 
   {
     //removeCommentFromLine(currentline);
     removeCommentFromLine(previousline);
     if(isContinuationLine(currentline, previousline, currentcol - 1 ))
     { /* check from one less than the current column number: (*it)->col - 2. */
       is_if_stmt = true; 
     }
   }
   /* Here, either is_if_stmt is true or it is a plain return*/
#ifdef DEBUG
   cout <<"if_stmt = "<<is_if_stmt<<endl;
#endif /* DEBUG */
  delete[] checkbuf;  /* checkbuf has served its purpose - get rid of it! */
  return is_if_stmt; 
}

/* -------------------------------------------------------------------------- */
/* -- Strip the module name from the routine name. M1::f2 is returned as f2 - */
/* -------------------------------------------------------------------------- */
const char * stripModuleFromName(const string functionname)
{
  char *s = NULL;
  char *nm  = const_cast<char*>(functionname.c_str());

  /* go back -- Jeff doesn't like the stripped name anymore */
  return nm;

  /* Here is the code to strip the code */
  /* traverse the string till you get rid of the :: using strstr */
  while (s = strstr(nm, "::"))
  {
    nm += strlen(s)+1;
  }
  return nm;
}


/* -------------------------------------------------------------------------- */
/* -- does it continue onto the next line? ---------------------------------- */
/* -------------------------------------------------------------------------- */
int doesStmtContinueOntoNextLine(char * & inbuf, int openparens)
{ /* This checks for ( and ) -- so it works for alloc/dealloc but not IO where
     a statement can also continue using , instead of ( ). */
  char *line = inbuf;
  int i, len;
    
  len = strlen(line);
  for (i=0 ; i < len, *line; i++, line++)
  {
    if (*line == '(') openparens ++;
    if (*line == ')') openparens --;
    if (*line == '!') {
      *line = '\0'; 
      break;
    }
  }
  return openparens;
 
} 

/* -------------------------------------------------------------------------- */
/* -- isFreeFormat returns true if it has a & in the current line ----------- */
/* -------------------------------------------------------------------------- */
bool isFreeFormat(char inbuf[])
{ /* get rid of ! */
  if (strstr(inbuf, "&"))
  {
#ifdef DEBUG
    printf("Continuation character found -- Free format\n");
#endif /* DEBUG */
    return true;
  }  
  else return false;
}
/* -------------------------------------------------------------------------- */
/* -- getNextToken returns true if it is done, and has no more variables ---- */
/* -- to process in the same line. Extracts variable name from line.        - */
/* -------------------------------------------------------------------------- */

bool getNextToken(char* &line, char* &varname)
{ /* if I pass it
      "hi ", Ary(x,y)%b(2), B(2)
      it should return successively "hi "
         then                       Ary(x,y)%b(2)
         then                       B(2)
      In contrast, getVariableName returns "hi ", Ary, B for alloc/dealloc */
   /* We need to traverse the string till we reach a comma and return it. If
      we instead see a (, we should continue processing till the closing parens ) */

   int openparens = 0;
   int opensinglequotes, openquotes=0;
   int i = 0;
   int len = 0;

   len = strlen(line);
   while (i < len) {

     if (*line == ',' && openparens == 0) {
         line++;
 	break; /* print *, ... */
     }

     varname[i] = *line;

     if ((*line == '"') || (*line == '\'')) {
       char quote = *line;
       i++;
       line++;
       while (*line != quote) {
 	varname[i] = *line;
 	i++;
 	line++;
       }
       varname[i] = *line;
     }


     if (*line == '(' ) /* first open paren. Search for close parenthesis */
       openparens++;
     if (*line == ')' ) {
       openparens --;
     }
     i++;
     line++; /* increment and go on */
   }

   varname[i] = '\0';
   if (line && *line == ',') line++;  /* get rid of comma */
#ifdef DEBUG
   printf("varname retrieved = %s, line = %s\n", varname, line);
#endif /* DEBUG */
   len = strlen(line);
   // printf("length remaining = %d\n", len);
   if (len == 0) return true; /* done = true! no more to retrieve */
   else return false; /* there are more strings to process... */
}



static void removeWhitespace(char *str) {
  int w=0;
  int len = strlen(str);
  for (int r=0;r<len;r++) {
    if (str[r] != ' ') {
      str[w]=str[r];
      w++;
    }
  }
  str[w] = 0;
}


static void blankQuote(char *str) {
  char *p = str;

  while (*p) {
    // blank everything in quotes
    if (*p == '\'' || *p == '"') {
      char quote = *p;
      p++;
      while (*p && *p != quote) {
	*p = 'x';
	p++;
      }
    }
    p++;
  }
}

/*
 * Heuristic to determine if any element of an IO statement contains an implied-do statement
 * The logic says that after you remove the whitespace, the first character must be a '(' and there
 * must be an '=' somewhere in it.
 *
 * Some examples:
 *     print *, "a","b","c",(A(1, I), I= 1,10), "d","e"
 * Nested:
 *     print *, "a","b","c",( (A(I,J), I=1, 10, 2), J= 1, 10), "d","e"
 */
bool isImpliedDo(char *str) {
  char tmp[4096];
  strcpy (tmp, str);

  blankQuote(tmp);
  removeWhitespace(tmp);

  if (tmp[0] == '(' && strchr(tmp,'=')) {
    return true;
  }

  return false;
}



void getImpliedToken(char* &str, char* &token) {
  if (*str == ',') {
    str++;
  }

  if (*str == '(') {
    // see if we find an equals sign before the matching ')'
    char *p = str+1;
    int paren = 1;
    int eqSeen = 0;
    while (*p && paren != 0) {
      if (*p == '=') {
	eqSeen = 1;
      }
      if (*p == '(') {
	paren++;
      }
      if (*p == ')') {
	paren--;
      }
      p++;
    }

    if (eqSeen) {
      token[0] = *str;
      token[1] = 0;
      str++;
      return;
    }
  }

  if (*str == ')') {
    token[0] = *str;
    token[1] = 0;
    str++;
    return;
  }

  int eqSeen = 0;
  int idx = 0;
  int paren = 0;
  while (*str) {

    if (*str == ',' && paren == 0) {
      break;
    }

    if (!(paren != 0 || (*str != ',' && *str != ')'))) {
      break;
    }

    if (*str == '=') {
      eqSeen = 1;
    }
    if (*str == '(') {
      paren++;
    }
    if (*str == ')') {
      paren--;
    }
    token[idx++] = *str;
    str++;
  }

  if (eqSeen) {
    while (*str != ')') {
      token[idx++] = *str;
      str++;
    }
  }

  token[idx] = 0;
}



class treeElement {
public:
  //  virtual void f();
  virtual ~treeElement() {}
  virtual void print() = 0;
  virtual void output(char *iostmt, int id) = 0;
  virtual const char *getString() = 0;
};

class listTreeElement : public treeElement {
public:
  vector<treeElement*> list;
  virtual void print() {
    //printf ("size=%d(",list.size());
    printf ("( ");
    for (int i=0; i<list.size()-1; i++) {
      list[i]->print();
      printf (", ");
    }
    list[list.size()-1]->print();
    printf (" )");
  }

  virtual const char *getString() {
    return 0;
  }

  virtual void output(char *iostmt, int id) {
    if (list.size() == 0) {
      fprintf (stderr, "Warning: list.size() == 0? (Failed to parse implied-do)\n");
    } else if (list.size() == 1) {
      list[0]->output(iostmt,id);
    } else {
      treeElement *iterElement = list[list.size()-1];

      strcat(iostmt, "      DO ");
      strcat(iostmt, iterElement->getString());
      strcat(iostmt, "\n");

      for (int i=0; i<list.size()-1; i++) {
	list[i]->output(iostmt,id);
      }
      strcat(iostmt, "      END DO\n");
   }

  }
};

class stringTreeElement : public treeElement {
public:
  string str;
  virtual void print() {
    printf ("'%s'", str.c_str());
  }

  virtual const char *getString() {
    return str.c_str();
  }

  virtual void output(char *iostmt, int id) {
    char phrase[4096];
    sprintf (phrase, "       tio_%d_sz = tio_%d_sz + sizeof(%s)\n", id, id, str.c_str());
    strcat(iostmt, phrase);
  }

};



void recurseCrap(char* &buf, listTreeElement *element) {
  char *token = new char[4096];
  while (*buf) {
    getImpliedToken(buf,token);
//      printf ("token = %s\n", token);
    if (strlen(token) == 1 && token[0] == '(') {
      listTreeElement *newElement = new listTreeElement();
      recurseCrap(buf,newElement);
      element->list.push_back(newElement);
    } else if (strlen(token) == 1 && token[0] == ')') {
      delete[] token;
      return;
    } else {
      stringTreeElement *newElement = new stringTreeElement();
      newElement->str = token;
      element->list.push_back(newElement);
    }
  }
  delete[] token;
  return;
}


/*
 * Process an implied-do construct, see above for definition and examples
 * We insert DO loops to compute the size
 */
void processImpliedDo(char *iostmt, char *element, int id) {

  char tmp[4096];
  strcpy (tmp, element);

  blankQuote(tmp);
  removeWhitespace(tmp);
  char *p = tmp;

  char key[4096];
  int nest = 0;
  int first = 1;
  int count = 0;

  listTreeElement *elements = new listTreeElement();

  recurseCrap(p, elements);

  strcat(iostmt,"\n");

//   elements->print();
//   printf ("\n");
  elements->output(iostmt,id);
}




/* -------------------------------------------------------------------------- */
/* -- getVariableName returns true if it is done, and has no more variables - */
/* -- to process in the same line. Extracts variable name from line.        - */
/* -------------------------------------------------------------------------- */
bool getVariableName(char * &line, char * & varname)
{
  int len, i, openparens, varlen;
  bool done = false;
  bool foundcompoundstmt = false;
  char *lastptr;
  char *firstptr = line;
  char token = ' ';
  do {
    len = strlen(line);
#ifdef DEBUG
    printf("AT THE BEGINNING : line=%s\n", line);
#endif

    for (i = 0; i < len && *line != ',' && *line != ')' && *line != '('; i++, line++)
    { /* check for the variable name. parsing shouldn't reach a  , ) or ( */
      varname[i] = *line;
      lastptr = line;
#ifdef DEBUG
      printf("varname[%d] = %c\n", i, varname[i]);
#endif /* DEBUG */
      if (varname[i] == '&') varname[i] = ' '; /* we don't want &B */
    }
    varname[i] = '\0';
    lastptr ++; /* increment to the end of the variable name */

    if (*line == ',') {
#ifdef DEBUG
      printf("HEY! Found a , when we were looking for a (, varname=%s\n", varname);
#endif /* DEBUG */
      /* go over the , and return from the routine. We have found what we were
         looking for. e.g., ALLOCATE(tmp_a(ispin)%pw, tmp_b(ispin)%pw) or
       ALLOCATE(pw1, b(23), stat=stat) 
       in these cases we need to return tmp_a(ispin)% pw and pw1 respectively*/

      line++;
      break;
    
    }
    while (*line && *line != '(') line++; /* go to the first ( */

#ifdef DEBUG 
    printf("after going for ( -- *line = %c\n", *line);
#endif /* DEBUG */

    if (!*line) {
#ifdef DEBUG
     printf("There were no ( in the allocate statement. Look for , as in allocate(x,y)\n");
#endif /* DEBUG */
     line = lastptr; /* reset it */
    }
    /* next count the number of ( opened before we reach a ) */
    len = strlen(line);
    for (i = 0, openparens = 0; i < len, *line; i++, line++)
    {
      if (*line == '(') openparens ++;
      if (*line == ')') openparens --;
      if (openparens == 0) break;
#ifdef DEBUG
      printf("line = %c, openparens = %d\n", *line, openparens);
#endif /* DEBUG */
    }
    //printf("after loop: openparens = %d\n", openparens);
    if (*line == ')') line ++;
    while (*line && *line == ' ') line++; /* skip whitespaces */
#ifdef DEBUG
    if (*line)
      printf("GETVARNAME: line = %c\n", *line); 
#endif /* DEBUG */

    token = *line; /* assign token here! */

    if (token == '%') 
    {
#ifdef DEBUG 
      printf("FOUND a percentage symbol AFTER the variable parsing is completed!! line = %s\n", line);
      /* implies a compound object. A variable that is part of a user defined type is present */
#endif /* DEBUG */
      line++; /* advance one token*/
      foundcompoundstmt = true; /* found % */
    }
    else 
    { /* look for the comma */
      while (*line && *line != ',') line++; /* go to the first , */

#ifdef DEBUG 
      if (*line) 
        printf("After looking for , line = %c, next = %c, done = %d\n", *line, *(line+1), done);
#endif /* DEBUG */
      if (*line) {
#ifdef DEBUG
        printf("NOT a NULL *line = %c\n", *line);
#endif /* DEBUG */
        line++; /* skip , */
      }
      else  {
#ifdef DEBUG 
       printf("NULL??? *line = %c\n", *line);
#endif /* DEBUG */
       done = true;
#ifdef DEBUG 
       printf("And after that: looking for , line = %c, done = %d\n", *line, done);
#endif /* DEBUG */
      }
    }
#ifdef DEBUG 
    printf("After IF token=%c, foundcompoundstmt=%d\n", token, foundcompoundstmt);
#endif /* DEBUG */

  } while (token == '%');
#ifdef DEBUG
  printf("foundcompoundstmt = %d, lastptr = %c\n",foundcompoundstmt, *lastptr);
#endif /* DEBUG */
  if (foundcompoundstmt)
  {
    for (i=0; firstptr != lastptr; i++)
    {
      varname[i] = *firstptr;
#ifdef DEBUG
      printf("ASSIGNING varname[%d] = %c\n", i, varname[i]);
#endif /* DEBUG */
      if (varname[i] == '&') varname[i] = ' '; /* we don't want &B */
      firstptr++;
    }
    varname[i] = '\0';
  }
#ifdef DEBUG
    printf("Got: line = %s, varname=%s, done=%d\n", line, varname, done);
#endif /* DEBUG */
  return done;


}
/* -------------------------------------------------------------------------- */
/* -- isFreeFormat returns true if it has a & in the current line ----------- */
/* -------------------------------------------------------------------------- */
bool isRequestOnSameLineAsPreviousRequest(vector<itemRef *>::iterator& it, vector<itemRef *>& itemvec)
{
  int currentLine = (*it)->line;
  int prevLine = it == itemvec.begin()? 0 : (*(it-1))->line; /* if it is null, prevLine = 0 */
  if (currentLine == prevLine) {
#ifdef DEBUG
   printf("isRequestOnSameLineAsPreviousRequest returns true line = %d\n", currentLine);
#endif /* DEBUG */
    return true;
  }
  else return false;
}

/* -------------------------------------------------------------------------- */
/* -- Write call TAU_ALLOC(...) statement ----------------------------------- */
/* -------------------------------------------------------------------------- */
int printTauAllocStmt(ifstream& istr, ofstream& ostr, char inbuf[], vector<itemRef *>::iterator& it, char *& laststatement)
{
 /* consider the string: allocate(A(100), stat=ierr) */
#ifdef DEBUG
  cout <<"Allocate Stmt: line ="<<(*it)->line<<endl;
  cout <<"inbuf ="<<inbuf<<endl;
#endif /* DEBUG */
  char suffixstmt[64*1024];
  char *allocstmt = new char [INBUF_SIZE];
  int i, openparens, len;
  bool done = false; 
  string prefix, suffix;
  int linesread = 0;
  bool isfree;
  char *start;
  char *line;
  char *varname = new char [INBUF_SIZE]; 

  removeCommentFromLine(inbuf);
  string nextline(inbuf);
  line = inbuf;
  /* count the number of open parentheses present on a line */
  if (openparens=doesStmtContinueOntoNextLine(inbuf, 0))
  { 
#ifdef DEBUG
    printf("Contination line: %s\n", inbuf);
#endif /* DEBUG */
    isfree = isFreeFormat(inbuf);
    do {
       if (istr.getline(allocstmt, INBUF_SIZE) == NULL)
       { 
         perror("ERROR in reading file: looking for ) for continuation line instrumentation of alloc/dealloc");
         exit(1);
       }
       strcpy(laststatement, allocstmt); /* save buffer and pass it out of the routine */
       removeCommentFromLine(allocstmt);
       /* if the file is in free format, start the next line by getting rid of the
          first six columns */
       if (!isfree) start = &allocstmt[6];
       else start = allocstmt; 
       while (start && *start == ' ') start ++; /* eat up leading spaces */
       len = strlen(start); 
       nextline.append(start, len);
#ifdef DEBUG
       printf("nextline=%s\n", nextline.c_str()); 
#endif /* DEBUG */
       ostr <<allocstmt<<endl;
       linesread ++; /* the number of lines processed. We need to return this */
    } while (openparens = doesStmtContinueOntoNextLine(allocstmt, openparens)); 
    line = (char *) nextline.c_str();     
#ifdef DEBUG
    printf("AFTER PROCESSING: line = %s\n", line);
#endif /* DEBUG */
  }
  else
  {
#ifdef DEBUG 
    printf("NOT Continuation line: %s\n", inbuf);
#endif /* DEBUG */
  }

  ostr<<endl; /* start with a new line. Clears up residue from TAU_PROFILE_START*/

/* NEW CODE! */
#ifdef DEBUG
  printf("after joining lines, line = %s\n", line);
#endif /* DEBUG */
  while (*line && *line != '(') line++;
  line++; /* skip first ( */

  while (!done)
  {
    done = getVariableName(line, varname);
#ifdef DEBUG
    printf("After getVariableName: done = %d, line = %s, varname = %s\n", done, line, varname);
#endif /* DEBUG */
  
    /* what about ! comment */
    if (!strstr(varname, "="))
    {
    /* we don't want stat=ierr argument */
    /* We need to break up this into a continuation line if it exceeds 72 chars */
/*
      ostr<<"\t call TAU_ALLOC("<<varname<<", "<<(*it)->line<< ", "
          <<tau_size_tok<<"("<<varname<<"), '"<< (*it)->snippet<< ", var="
          <<varname<<"')"<<endl;
*/
     char *p = varname;
     while (p && *p == ' ') p++; /* eat up leading space */
     sprintf(allocstmt, "       call TAU_ALLOC(%s, %d, %s(%s), '",
	p, (*it)->line, tau_size_tok.c_str(), p);
     sprintf(suffixstmt, "%s, variable=%s", (*it)->snippet.c_str(), p);
     string prefix=string(allocstmt);
     string suffix=string(suffixstmt);
     writeLongFortranStatement(ostr, prefix, suffix);
#ifdef DEBUG
      printf("Putting in file: varname=%s, line = %s\n", varname, line);
#endif /* DEBUG */
    }
    else break; /* end of processing */
  }
  delete[] allocstmt;
  delete[] varname;
  return linesread;

}
/* -------------------------------------------------------------------------- */
/* -- Write call TAU_DEALLOC(...) statement --------------------------------- */
/* -------------------------------------------------------------------------- */
int printTauDeallocStmt(ifstream& istr, ofstream& ostr, char inbuf[], vector<itemRef *>::iterator& it, bool writetab, char *& laststatement)
{
  int i, len;
  char suffixstmt[64*1024];
  int openparens, linesread=0; /* how many additional lines (cont) did we read? */
  char *deallocstmt = new char[INBUF_SIZE];
  int isfree;
  bool done = false; 
  char *varname = new char [INBUF_SIZE]; 
  char *start;
  char *line;
  list<string> statements;
  list<string>::iterator sit;

  ostr<<endl; /* start with a new line. Clears up residue from TAU_PROFILE_START*/
  statements.push_back(inbuf); /* initialize the list of statements to inbuf */
#ifdef DEBUG
  cout <<"Deallocate Stmt: line ="<<(*it)->line<<endl;
  printf("Deallocate Stmt: line = %d... \n",(*it)->line);
  cout <<"inbuf ="<<*inbuf<<endl;
#endif /* DEBUG */

  removeCommentFromLine(inbuf);
  string nextline(inbuf);
  line = inbuf;

  /* first we need to figure out if this line is a continuation line */
  if (openparens=doesStmtContinueOntoNextLine(inbuf, 0))
  { /* yes it does! */
#ifdef DEBUG
    printf("Contination line: %s\n", inbuf);
#endif /* DEBUG */
    isfree = isFreeFormat(inbuf);
    do {
       if (istr.getline(deallocstmt, INBUF_SIZE) == NULL)
       {
         perror("ERROR in reading file: looking for ) for continuation line instrumentation of alloc/dealloc");
         exit(1);
       }
       strcpy(laststatement, deallocstmt); /* copy it in */
#ifdef DEBUG
       printf("LASTSTATEMENT = %s\n", laststatement);
#endif /* DEBUG */
       removeCommentFromLine(deallocstmt);
       statements.push_back(deallocstmt);
       /* if the file is in free format, start the next line by getting rid of the
          first six columns */
       if (!isfree) start = &deallocstmt[6];
       else start = deallocstmt;
       while (start && *start == ' ') start ++; /* eat up leading spaces */
       len = strlen(start);
       nextline.append(start, len);
       //nextline.append(string("\n", 1));
#ifdef DEBUG
       printf("nextline=%s\n", nextline.c_str());
#endif /* DEBUG */
       linesread ++; /* the number of lines processed. We need to return this */
    } while (openparens = doesStmtContinueOntoNextLine(deallocstmt, openparens));
  }

  line = (char *) nextline.c_str();     


  while (*line && *line != '(') line++;
  line++; /* skip first ( */

  while (!done)
  {
    done = getVariableName(line, varname);

    /* what about ! comment */
    if (!strstr(varname, "="))
    {
    /* we don't want stat=ierr argument */
    /* We need to break up this into a continuation line if it exceeds 72 chars */
/*
      ostr<<"\t call TAU_ALLOC("<<varname<<", "<<(*it)->line<< ", "
          <<tau_size_tok<<"("<<varname<<"), '"<< (*it)->snippet<< ", var="
          <<varname<<"')"<<endl;
*/
     char *p = varname;
     while (p && *p == ' ') p++; /* eat up leading space */

/* new */
     sprintf(deallocstmt, "       call TAU_DEALLOC(%s, %d, '",
        p, (*it)->line);
     sprintf(suffixstmt, "%s, variable=%s", (*it)->snippet.c_str(), p);
     string prefix=string(deallocstmt);
     string suffix=string(suffixstmt);
     writeLongFortranStatement(ostr, prefix, suffix);

#ifdef DEBUG
      printf("Putting in file: varname=%s, line = %s\n", varname, line);
#endif /* DEBUG */
    }
    else break; /* end of processing */
  }
//  ostr<<"\t call TAU_DEALLOC(A, "<<(*it)->line<< ", '"<< (*it)->snippet<< ", var=A')"<<endl;
  if (writetab) ostr<<"\t";
  for (sit = statements.begin(); sit != statements.end();
	sit++)
    ostr <<(*sit)<<endl;
  delete [] deallocstmt;
  delete[] varname;
  return linesread;

}

/* -------------------------------------------------------------------------- */
/* -- Write TAU's IO tracking calls and the statement ----------------------- */
/* -------------------------------------------------------------------------- */
int printTauIOStmt(ifstream& istr, ofstream& ostr, char inbuf[], vector<itemRef *>::iterator& it, bool writetab, char *& laststatement)
{
  int i, len, origlen, sizeoflen;
  char string_containing_sizeof[64*1024];
  int openparens, linesread=0; /* how many additional lines (cont) did we read? */
  char *iostmt = new char[INBUF_SIZE];
  int isfree;
  bool done = false; 
  char *varname = new char [INBUF_SIZE]; 
  char *start;
  char *line;
  int lineno, numlines;
  list<string> statements;
  list<string>::iterator sit;

  ostr<<endl; /* start with a new line. Clears up residue from TAU_PROFILE_START*/
  statements.push_back(inbuf); /* initialize the list of statements to inbuf */
  lineno = (*it)->line;
  numlines = (*it)->end.line() - (*it)->begin.line();
/* Now there are two points -- begin and end that are available for this IO statement.
   By checking if the line numbers differ, we can tell if it has a continuation line */
#ifdef DEBUG
  printf("INSIDE printTauIOStmt: inbuf=%s  --> line= %d, spanning %d lines \n", 
	inbuf, lineno, numlines);
  cout <<"IO Stmt: line ="<<lineno<<endl;
  cout <<"inbuf ="<<*inbuf<<endl;
#endif /* DEBUG */

  removeCommentFromLine(inbuf);
  removeContinuation(inbuf);

  string nextline(inbuf);
  line = inbuf;

  /* first we need to figure out if this statement spills over multiple lines */
  for (i =0; i < numlines; i++)
  { /* yes it does! */
#ifdef DEBUG
    printf("Contination line: %s\n", inbuf);
#endif /* DEBUG */
    isfree = isFreeFormat(inbuf);
    if (istr.getline(iostmt, INBUF_SIZE) == NULL)
    {
      perror("ERROR in reading file: looking for continuation line instrumentation of io");
      exit(1);
    }
    strcpy(laststatement, iostmt); /* copy it in */
#ifdef DEBUG
    printf("LASTSTATEMENT = %s, isfree = %d\n", laststatement, isfree);
#endif /* DEBUG */
    removeCommentFromLine(iostmt);
    statements.push_back(iostmt);
    removeContinuation(iostmt);
    /* if the file is in fixed format, start the next line by getting rid of the
       first six columns */
    if (!isfree) start = &iostmt[6];
    else start = iostmt;
    while (start && *start == ' ') start ++; /* eat up leading spaces */
    len = strlen(start);
    nextline.append(start, len);
#ifdef DEBUG
    printf("nextline=%s\n", nextline.c_str());
#endif /* DEBUG */
    linesread ++; /* the number of lines processed. We need to return this */
  } /* for loop */

  line = (char *) nextline.c_str();     

  /* Now we have the statement in the line */

#ifdef DEBUG 
  printf ("AFTER merging lines: line = %s\n", line);
  printf("NEXT!\n");
#endif /* DEBUG */

/* THIS needs to be changed */
  /* Consider two types of IO statements:
     print *, "string = ", value
     write (6,*, ERR=24) ary(2,4), b, c
     Either a , can appear or a ( can appear. If ( appears, we need to reach 
     the corresponding ) before reading in the variables and strings from the 
     next stage */
  openparens = 0; 
  while (line) 
  {
#ifdef DEBUG
    printf("Loop: (2.3) *line = %c\n", *line);
#endif /* DEBUG */

    if (*line == ',' && openparens == 0) {
	line ++; break; /* print *, ... */
    }
    if (*line == '(' ) /* first open paren. Search for close parenthesis */
      openparens++;
    if (*line == ')' )
    {
      openparens --; 
      if (openparens == 0) { 
	line++;  /* increment ( before bailing out */
        break; 
      }
    }
    line ++; /* increment and go on */
  }

#ifdef DEBUG
  printf ("After checking format string: line = %s\n", line);
#endif /* DEBUG */
  
  sprintf(iostmt, "      tio_%d_sz = 0",lineno);
  while (!done)
  {
    done = getNextToken(line, varname);
    
    if (isImpliedDo(varname)) {
      processImpliedDo(iostmt, varname, lineno);
    } else {
      /* what about ! comment */
      /* We need to break up this into a continuation line if it exceeds 72 chars */
      
      char *p = varname;
      while (p && *p == ' ') p++; /* eat up leading space */
      if (strlen(p) == 0) continue ; /* don't put sizeof() */
      
      sprintf(string_containing_sizeof, "+sizeof(%s)", p); 
      origlen = strlen(iostmt);
      sizeoflen = strlen(string_containing_sizeof);
      
      if (origlen+sizeoflen >= 72) { /* exceeds 72 columns -- break it up! */
	sprintf(string_containing_sizeof, "\n      tio_%d_sz = tio_%d_sz+sizeof(%s)",
		lineno, lineno, p);
      }
      strcat(iostmt, string_containing_sizeof);
    }
  }
  ostr <<iostmt<<endl;
  ostr <<"      call TAU_CONTEXT_EVENT(tio_"<<lineno<<", tio_"<<lineno<<"_sz)"<<endl;
     // writeLongFortranStatement(ostr, prefix, suffix);

#ifdef DEBUG
      printf("Putting in file: varname=%s, line = %s\n", varname, line);
#endif /* DEBUG */
  if (writetab) ostr<<"\t";
  for (sit = statements.begin(); sit != statements.end();
	sit++)
    ostr <<(*sit)<<endl;
  delete [] iostmt;
  delete[] varname;
  return linesread;

}

/* -------------------------------------------------------------------------- */
/* -- Get column number of read/write/print statement. 0 if none are present- */
/* -------------------------------------------------------------------------- */
int getIOColumnNumber(const char *inbuf)
{
  int col = 0;

  col = CPDB_GetSubstringCol(inbuf, "print");
  if (!col)
  {
    col = CPDB_GetSubstringCol(inbuf, "write");
    if (!col)
      col = CPDB_GetSubstringCol(inbuf, "read");
  }
 
#ifdef DEBUG
  printf("getIOColumnNumber %s: returning %d\n", inbuf, col);
#endif /* DEBUG */
  return col;
}

/* -------------------------------------------------------------------------- */
/* -- Get a list of instrumentation points for a C++ program ---------------- */
/* -------------------------------------------------------------------------- */
bool instrumentFFile(PDB& pdb, pdbFile* f, string& outfile, string& group_name) 
{ 
  string file(f->name());
  string codesnippet; /* for START_TIMER, STOP_TIMER */
  static char inbuf[INBUF_SIZE]; // to read the line
  static char * previousline = new char [INBUF_SIZE]; // to read the line
  char *checkbuf=NULL; // Assign inbuf to checkbuf for return processing
  // open outfile for instrumented version of source file
  ofstream ostr(outfile.c_str());
  int space, i, j, k, c, additionalLinesRead;
  int docol, ifcol, thencol, gotocol, alloccol, dealloccol, startcol, iocol;
  if (!ostr) {
    cerr << "Error: Cannot open '" << outfile << "'" << endl;
    return false;
  }
  // open source file
  ifstream istr(file.c_str());
  if (!istr) {
    cerr << "Error: Cannot open '" << file << "'" << endl;
    return false;
  }
#ifdef DEBUG
  cout << "Processing " << file << " in instrumentFFile..." << endl;
#endif

  memset(previousline, INBUF_SIZE, 0); // reset to zero
  memset(inbuf, INBUF_SIZE, 0); // reset to zero
  // initialize reference vector
  vector<itemRef *> itemvec;
  if (!use_spec)
  {
    /* In "spec" mode, only file instrumentation requests are used */
    getFReferences(itemvec, pdb, f);
  }
  /* check if the given file has line/routine level instrumentation requests */
  if (!isInstrumentListEmpty() || memory_flag) 
  { /* there are finite instrumentation requests, add requests for this file */
    addFileInstrumentationRequests(pdb, f, itemvec);
  }
  /* All instrumentation requests are in. Now do postprocessing. */
  postprocessInstrumentationRequests(itemvec);


  int inputLineNo = 0;
  bool is_if_stmt;
  vector<itemRef *>::iterator lit = itemvec.begin();

  /* Iterate through the list of instrumentation requests */
  while (lit != itemvec.end() & !istr.eof())
  {
    // Read one line each till we reach the desired line no.
#ifdef DEBUG
    if ((*lit) && (*lit)->item)
      cout <<"S: "<< (*lit)->item->fullName() << " line "<< (*lit)->line << " col " << (*lit)->col << endl;
#endif
    bool instrumented = false;

    while((instrumented == false) && (istr.getline(inbuf, INBUF_SIZE)) )
    {
      inputLineNo ++;
      if (inputLineNo < (*lit)->line)
      {
        // write the input line in the output stream
#ifdef DEBUG
	cout <<"Writing (3): "<<inbuf<<endl;
#endif /* DEBUG */
        ostr << inbuf <<endl;
      }
      else
      { /* reached line */
        int inbufLength = strlen(inbuf);
        // we're at the desired line no. go to the specified col
#ifdef DEBUG 
	cout <<"Line " <<(*lit)->line <<" Col " <<(*lit)->col <<endl;
#endif /* DEBUG */
	/* Now look at instrumentation requests for that line */
        vector<itemRef *>::iterator it;
	int write_upto = 0;
	bool print_cr = true;
	for(it = lit; ((it != itemvec.end()) && ((*it)->line == (*lit)->line));
 	    ++it)
        { /* it/lit */
          if (it+1 != itemvec.end())
	  {
	    if ((*(it+1))->line == (*it)->line)
	    {
 	      write_upto = (*(it+1))->col - 1; 
	      print_cr = false;
	    }
	    else
	    {
	      write_upto = inbufLength;
	      print_cr = true;
	    }
	  /* 
          write_upto = (*(it+1))->line == (*it)->line ? (*(it+1))->col-1 : inbufLength; 
	  */
  	
#ifdef DEBUG
	  cout <<"CHECKING write_upto = "<<write_upto<<endl;
	  cout <<"inbuf = "<<inbuf<<endl;
	  cout <<"it = "<<(*it)->line<<", "<<(*it)->col<<") ; ";
	  cout <<"it+1 = "<<(*(it+1))->line <<", "<<(*(it+1))->col <<") ;"<<endl;
#endif /* DEBUG */
	  }
	  else
	  {
	    write_upto = inbufLength; 
	    print_cr = true;
	  }

	  int pure = 0;
#ifdef PDT_PURE
	  /* When INSTRUMENTATION_POINT is used, sometimes the item is null */
	  if ( (*it)->item &&
             (((pdbRoutine *)(*it)->item)->fprefix() == pdbItem::FP_PURE ||
	        ((pdbRoutine *)(*it)->item)->fprefix() == pdbItem::FP_ELEM)) {
	       pure = 1;
	  }
#endif

	  // get the instrumented routine name
	  string instrumentedName = getInstrumentedName((*it)->item);

	  /* set instrumented = true after inserting instrumentation */
	  switch((*it)->kind)
	  {
	    case BODY_BEGIN:

	      // write out the line up to the desired column
	      for (i=0; i< ((*it)->col)-1; i++) {
		ostr << inbuf[i];
	      }

	      // break the line
	      ostr << endl;

              if (use_spec)
              {
                writeAdditionalDeclarations(ostr, (pdbRoutine *)((*it)->item));
                ostr << "\t" << (*it)->snippet << endl;
		ostr << "      ";
		
		// write the rest of the original statement
		for (k = (*it)->col-1; k < write_upto ; k++) {
		  ostr << inbuf[k];
		}
		if (print_cr) {
		  ostr << endl;
		}
                /* XXX Insert code here */
                break;
              }
              else if (use_perflib)
              {
		if (pure && instrumentPure) {
		  ostr << "      interface\n";
		  ostr << "      pure subroutine f_perf_update(name, flag)\n";
		  ostr << "      character(*), intent(in) :: name\n";
		  ostr << "      logical, intent(in) :: flag\n";
		  ostr << "      end subroutine f_perf_update\n";
		  ostr << "      end interface\n";
		}
		
		// should we call MPI_Init? Only if it is the main program
/*
		if (((pdbRoutine *)(*it)->item)->kind() == pdbItem::RO_FPROG) {
		  ostr << "      INTEGER tau_mpi_init_err"<<endl;
		  ostr << "      call MPI_Init(tau_mpi_init_err)"<<endl;
		  ostr << "      call f_perf_init('"<<(*it)->item->fullName()<<"', 0, 0, 'UNKNOWN')"<<endl;
		  ostr << "      ";
		}
*/
		if (!pure || instrumentPure)
		{
		  ostr << "      call f_perf_update('"<<stripModuleFromName((*it)->item->fullName())<<"', .true.)"<<endl;
		}
		ostr << "      ";
		
		// write the rest of the original statement
		for (k = (*it)->col-1; k < write_upto ; k++) {
		  ostr << inbuf[k];
		}
		if (print_cr) {
		  ostr << endl;
		}
		instrumented = true;
		break;
	      }
	      
		
	      if (pure) {
		if (instrumentPure) {
		  ostr << "      interface\n";
		  ostr << "      pure subroutine TAU_PURE_START(name)\n";
		  ostr << "      character(*), intent(in) :: name\n";
		  ostr << "      end subroutine TAU_PURE_START\n";
		  ostr << "      pure subroutine TAU_PURE_STOP(name)\n";
		  ostr << "      character(*), intent(in) :: name\n";
		  ostr << "      end subroutine TAU_PURE_STOP\n";
		  ostr << "      end interface\n";
		}
	      } else {
		
#ifdef TAU_ALIGN_FORTRAN_INSTRUMENTATION
		// alignment issues on solaris2-64 and IRIX64 require a value that will be properly aligned
		ostr << "      DOUBLE PRECISION profiler / 0 /"<<endl;
		// a possible alternative
		//ostr << "      integer*8 profiler / 0 /"<<endl;
#else
#ifdef TAU_ALT_FORTRAN_INSTRUMENTATION
		// nagware and g95 require this
		ostr << "      integer, dimension(2) :: profiler = (/ 0, 0 /)"<<endl;
#else
		ostr << "      integer profiler(2) / 0, 0 /"<<endl;
#endif /*TAU_ALT_FORTRAN_INSTRUMENTATION*/
#endif /*TAU_ALIGN_FORTRAN_INSTRUMENTATION*/ 
		ostr << "      save profiler"<<endl<<endl;
	      }
	      
                writeAdditionalDeclarations(ostr, (pdbRoutine *)((*it)->item));
		if (((pdbRoutine *)(*it)->item)->kind() == pdbItem::RO_FPROG) {
		  // main
		  ostr << "      call TAU_PROFILE_INIT()"<<endl;
		  writeFortranTimer(ostr, instrumentedName, (*it));
		} else { 
		  // For all routines
		  
		  if (!pure) {
		    if (strcmp(group_name.c_str(), "TAU_USER") != 0) { 
		      // Write the following lines only when -DTAU_GROUP=string is defined
                      string groupInstrumentedName = group_name.substr(10)+">"+instrumentedName;
		      writeFortranTimer(ostr, groupInstrumentedName, (*it));
		    } else { 
		      /* group_name is not defined, write the default fullName of the routine */
		      writeFortranTimer(ostr, instrumentedName, (*it));
		    }
		  }
  		}
		/* spaces */
     		for (space = 0; space < (*it)->col-1 ; space++) 
		  WRITE_SPACE(ostr, inbuf[space]) 

		WRITE_TAB(ostr,(*it)->col);
		if (pure) {
		  if (instrumentPure) {
		    ostr << "call TAU_PURE_START('" << instrumentedName << "')"<<endl;
		  }
		} else {
		  ostr <<"call "<<getStartMeasurementEntity((*it))<<"(profiler)"<<endl;
		}
                writeAdditionalFortranInvocations(ostr, (pdbRoutine *)((*it)->item));
                if (!(*it)->snippet.empty())
                  ostr << "\n\t" << (*it)->snippet << "\n\t";

#ifdef DEBUG
		printf("Before 2.1: (*it)->col = %d, write_upto=%d\n", (*it)->col, write_upto);
#endif /* DEBUG */
		if ((*it)->col != 1)
		  ostr << "      ";
		// IMPORTANT: If the formatting of the next statement is wrong, please remove the above comment!
		// write the rest of the original statement
     		for (k = (*it)->col-1; k < write_upto ; k++) {
		  ostr << inbuf[k];
#ifdef DEBUG
		  printf("WRITING 2.1 : inbuf[%d] = %c\n", k, inbuf[k]);
#endif /* DEBUG */
		}

		/* should we write the carriage return? */
	 	if (print_cr) {
		  ostr<< endl;
		}

		instrumented = true;
		break;
	  case EXIT:
	  case RETURN:
#ifdef DEBUG
	    cout <<"RETURN/EXIT statement "<<endl;
	    cout <<"inbuf = "<<inbuf<<endl;
	    cout <<"line ="<<(*it)->line<<" col = "<<(*it)->col<<endl;
#endif /* DEBUG */
	    /* Check to see if it is not a comment and has a "if" in the string */

	    /* search for 'return', since preprocessing may have 
	       moved it and given us a bogus column, if we can't find it
	       revert back since this may be an exit and not a 'return'
	    */
	    int col;
	    col = CPDB_GetSubstringCol(inbuf,"return");
	    
            if (col > 1 && (*it)->kind == EXIT) {
#ifdef DEBUG
	       cout <<"RETURN/EXIT statement was EXIT but return was found!"<<endl;
	       cout <<"setting column to 0"<<endl;
	       
#endif /* DEBUG */
               col = 0;
            }
	       
            /* STOP 'COMMENT HAS RETURN IN IT' */
            /* When the return is at an incorrect location in the pdb file,
               we need to flush the buffer from the current location to the 
	       correct location */
	    /* Also check to see if lit is the same as it or in other 
	       words, has the body begin written the statement? */
	    /* printf("Return: row = %d col = %d\n", (*it)->line, col); */
	    if (col && col > (*it)->col && lit!=it)
	    {
	      for(i=(*it)->col-1; i < col-1; i++)
	      {
		ostr<<inbuf[i];
#ifdef DEBUG
		cout <<"WRITING (5:RET..): "<<inbuf[i]<<endl;
#endif /* DEBUG */
	      }

	    }

		
	    if (col != 0) {
	      (*it)->col = col;
	    }

#ifdef DEBUG
	    cout <<"Return is found at "<< (*it)->line<<" Column: "<<(*it)->col<<endl;
#endif /* DEBUG */

	    if ((*it)->col > strlen(inbuf)) {
	      fprintf(stderr, "ERROR: specified column number (%d) is beyond the end of the line (%d in length)\n",(*it)->col,strlen(inbuf));
	      fprintf(stderr, "line = %s (%d)\n",inbuf,(*it)->line);
	      exit(-1);
	    } 
            is_if_stmt = addThenEndifClauses(inbuf, previousline, (*it)->col - 1);
	    if (lit == it)
	    { /* Has body begin already written the beginning of the statement? */
	      /* No. Write it (since it is same as lit) */
              for(i=0; i< ((*it)->col)-1; i++)
	      { 
#ifdef DEBUG
	  	cout << "Writing (1): "<<inbuf[i]<<endl;
#endif /* DEBUG */
	  	ostr <<inbuf[i]; 
	      }
	    }

	    if (is_if_stmt)
	    { 
	       ostr << "then"<<endl;
	       ostr << "      ";
#ifdef DEBUG
		cout <<"WRITING (6:then..): then"<<endl;
#endif /* DEBUG */
	    }
	
	    WRITE_TAB(ostr,(*it)->col);
	    /* before writing stop/exit examine the kind */
		if ((*it)->kind == EXIT)
		{ /* Turn off the timers. This is similar to abort/exit in C */
                  if (use_spec)
                  {
                    /* XXX Insert code here */
                  }
                  else if (use_perflib)
		    ostr <<"call f_perf_update('"<<stripModuleFromName((*it)->item->fullName())<<"', .false.)"<<endl;
		  else
		    ostr <<"call TAU_PROFILE_EXIT('exit')"<<endl;
		}
		else
		{ /* it is RETURN */
		  if (pure) {
		    if (instrumentPure) {
	              if (use_spec)
                      {
                        /* XXX Insert code here */
                      }
                      else if (use_perflib)
			ostr <<"call f_perf_update('" << stripModuleFromName((*it)->item->fullName())<< "', .false.)"<<endl;
		      else
			ostr <<"call TAU_PURE_STOP('" << instrumentedName << "')"<<endl;
		    }
		  } else {
		    /* we need to check if the current_timer (outer-loop level instrumentation) is set */
		    for (list<string>::iterator siter = current_timer.begin(); 
			siter != current_timer.end(); siter++)
		    { /* it is not empty -- we must shut the timer before exiting! */
#ifdef DEBUG 
			cout <<"Shutting timer "<<(*siter)<<" before stopping the profiler "<<endl;
#endif /* DEBUG */
			ostr <<"call TAU_PROFILE_STOP("<<(*siter)<<")"<<endl<<"\t";
		    }
                    if (!(*it)->snippet.empty())
                      ostr << (*it)->snippet << "\n\t";
		    if (use_spec)
                    {
                      /* XXX Insert code here */
                    }
                    else if (use_perflib)
		      ostr <<"call f_perf_update('"<<stripModuleFromName((*it)->item->fullName())<<"', .false.)"<<endl;
		    else
		      ostr <<"call "<<getStopMeasurementEntity((*it))<<"(profiler)"<<endl;
		  }
		}

     		for (space = 0; space < (*it)->col-1 ; space++) 
		  WRITE_SPACE(ostr, inbuf[space])

		instrumented = true;
	
        	for(j= ((*it)->col)-1; j <write_upto; j++)
		{ 
#ifdef DEBUG
	  	  cout <<"Writing(2): "<<inbuf[j]<<endl;
#endif /* DEBUG */
	  	  ostr << inbuf[j];
		}

		ostr<<endl;

		if (is_if_stmt)
		{
	  	  ostr <<"         endif"<<endl;
		}      

		break;
		 
	    case INSTRUMENTATION_POINT:
#ifdef DEBUG
		cout <<"Instrumentation point Fortran -> Line = "<< (*it)->line<<" col "<<(*it)->col <<" write_upto = "<<write_upto<<" snippet = "<<(*it)->snippet<<endl;
#endif /* DEBUG */
		/* are we at the entry of the routine. Are there other items that need to be done at this point? */
	        /* if so, we should really add this snippet to other invocations. */
		if (((it+1) != itemvec.end()) && ((*it)->line == (*(it+1))->line))
	        {
                  int rid = -1; 
#ifdef DEBUG
		  cout <<"NEXT Instrumentation point on the same line!"<<(*it)->line<<endl;
#endif /* DEBUG */
		  /* find out what routine no. is associated with it. iterate through 
		     records till you find one that has a non-null item. Get its routine id */
		  vector<itemRef *>::iterator institer = it; 	
		  while ((*institer) && (*institer)->line == (*it)->line)
                  {
                    if ((*institer)->item && (*institer)->kind== BODY_BEGIN) 
		    {
                      rid = ((pdbRoutine *)(*institer)->item)->id();
#ifdef DEBUG
		      cout <<"Found routine = "<<rid <<endl;
#endif /* DEBUG */
		      break;
		    }
		    institer++;
		  }
		  /* push snippet to the additionalInvocations */
		  if (rid != -1)
		  {
		    addMoreInvocations(rid, (*it)->snippet); /* assign the list of strings to the list */
		  }
		  else
                  {
		    WRITE_SNIPPET((*it)->attribute, (*it)->col, 0, (*it)->snippet);
                  }
		  /* if there is another instrumentation point on the same line, it will take care of the write_upto part */
		}
		else {
		  WRITE_SNIPPET((*it)->attribute, (*it)->col, write_upto, (*it)->snippet);
		}
		instrumented = true;
		break;
            case START_DO_TIMER:
            case START_TIMER:
		docol = (*it)->col;

		/* I've commented this section out because it produces incorrect results
		   for named loops, e.g.
		   
		   loopone: do i=1,6
		     ...
		   enddo loopone

		   I believe it was intented to fix incorrect pdb files where the start
		   of the do loop for a labeled do was pointing to the label instead
		   of the D in DO.  This has been fixed in flint now though.
		*/

// 		if ((*it)->kind == START_DO_TIMER)
// 		{

// 		  docol = CPDB_GetSubstringCol(inbuf,"do");
// #ifdef DEBUG
//                 cout <<"START_DO_TIMER point Fortran -> line = "<< (*it)->line
// 		     <<" col = "<< (*it)->col <<" write_upto = "<< write_upto
// 		     <<" timer = "<<(*it)->snippet<< " docol = "<<docol<<endl;
// 		cout <<"inbuf = "<<inbuf<<endl;
// #endif /* DEBUG */
// 		}
		codesnippet = string("       call TAU_PROFILE_START(")+(*it)->snippet+")";
		WRITE_SNIPPET((*it)->attribute, docol, write_upto, codesnippet);
                instrumented = true;
		/* Push the current timer name on the stack */
	 	current_timer.push_front((*it)->snippet); 
                break;

	    case GOTO_STOP_TIMER:

#ifdef DEBUG
                cout <<"GOTO_STOP_TIMER point Fortran -> line = "<< (*it)->line
		     <<" timer = "<<(*it)->snippet<<endl;
#endif /* DEBUG */
		/* we need to check if this goto occurs on a line with an if */
		gotocol = CPDB_GetSubstringCol(inbuf,"goto");
                /* check for "go to <label>" instead of "goto <label>" */
		if (gotocol == 0) 
		{
		  gotocol = CPDB_GetSubstringCol(inbuf,"go");
		} 
#ifdef DEBUG
		printf("gotocol = %d, inbuf = %s\n", gotocol, inbuf);
#endif /* DEBUG */
                if (addThenEndifClauses(inbuf, previousline, gotocol-1))
	        {
			/* old code: 
		ifcol = CPDB_GetSubstringCol(inbuf,"if");
		thencol = CPDB_GetSubstringCol(inbuf,"then");
		if (ifcol && gotocol && !thencol)
		{
#ifdef DEBUG
		  cout <<"ifcol = "<<ifcol<<" goto col = "<<gotocol
		        <<" thencol = " << thencol<<endl;
#endif */ /* DEBUG */
#ifdef DEBUG
                  cout <<"GOTO_STOP_TIMER INSIDE SINGLE_IF "<<endl;
#endif /* DEBUG */
		  
		  codesnippet = string("\t then \n       call TAU_PROFILE_STOP(")+(*it)->snippet+")";
		  WRITE_SNIPPET(BEFORE, gotocol, write_upto, codesnippet);
		  ostr <<"\t endif"<<endl;
		}
		else
		{
		  codesnippet = string("       call TAU_PROFILE_STOP(")+(*it)->snippet+")";
		  WRITE_SNIPPET((*it)->attribute, (*it)->col, write_upto, codesnippet);
		}
                instrumented = true;
	        /* We maintain the current outer loop level timer active. We need to pop the stack */
		break; /* no need to close the timer in the list */
            case STOP_TIMER:
#ifdef DEBUG
                cout <<"STOP_TIMER point Fortran -> line = "<< (*it)->line
		     <<" timer = "<<(*it)->snippet<<" col "<<(*it)->col
		     <<" write_upto = "<<write_upto<<endl;
#endif /* DEBUG */

		codesnippet = string("       call TAU_PROFILE_STOP(")+(*it)->snippet+")";
		WRITE_SNIPPET((*it)->attribute, (*it)->col, write_upto, codesnippet);
                instrumented = true;
	        /* We maintain the current outer loop level timer active. We need to pop the stack */
		if (!current_timer.empty()) current_timer.pop_front();
                break;

	    case ALLOCATE_STMT:
                alloccol = CPDB_GetSubstringCol(inbuf,"allocate");
		if (addThenEndifClauses(inbuf, previousline, alloccol - 1)  && (alloccol != 0))
		{
		/* only write till the alloc column. This assumes statement 
		begins on col 1? even if it is "20 if (x.gt.2) allocate(A)" */
		  startcol = (alloccol == (*it)->col ? (*it)->col: 1);
		  if(!isRequestOnSameLineAsPreviousRequest(it, itemvec))
		    startcol = 1;
		  /* the previous instrumentation request does not write upto our current column number */
#ifdef DEBUG
		  printf("TAB:: line = %d, alloccol = %d, it->col = %d, startcol=%d, last line = %d last col = %d \n", (*it)->line, alloccol, (*it)->col, startcol, (*(it-1))->line, (*(it-1))->col);
#endif /* DEBUG */
		  //if ((*it)->col-1) ostr<<"\t"; /* bump it up if it is col 1 */
		  for(i=startcol - 1; i< alloccol - 1; i++) {
#ifdef DEBUG
                    cout << "Writing (1.4): "<<inbuf[i]<<" startcol="<<startcol<<endl;
#endif /* DEBUG */
                    ostr <<inbuf[i];
                  }
		  ostr<<"\t then \n\t";
		  for(i=alloccol-1; i< strlen(inbuf); i++) {
#ifdef DEBUG
                    cout << "Writing (1.5): "<<inbuf[i]<<endl;
#endif /* DEBUG */
                    ostr <<inbuf[i];
                  }
		  ostr<<endl;
		  additionalLinesRead=printTauAllocStmt(istr, ostr, &inbuf[alloccol-1], it, previousline);
		  inputLineNo += additionalLinesRead; 
                  if (additionalLinesRead)
		    strcpy(inbuf, previousline); /* update last line read */
                  ostr<<"\t endif"<<endl;
                }
		else
		{ /* there is no if clause */
		  /* If the PDB file puts the continued if statement on the 
                     same line as allocate, we need to take care of it. e.g.,
     6         if ( value .gt. 3) &
     7           allocate(A(2))
    PDB:  rstmt st#2 fallocate so#1 6 8 so#1 7 23 NA NA
    PDB:  rstmt st#1 fsingle_if so#1 6 8 so#1 7 23 st#3 st#2
		  */

                  is_if_stmt = addThenEndifClauses(inbuf, previousline, alloccol - 1);
#ifdef DEBUG
                  printf("IS IT IF??? %d alloccol = %d\n", is_if_stmt, alloccol);
#endif /* DEBUG */
		  if (is_if_stmt && (alloccol == 0)) { 
			/* handle this separately. write the current statement */
		     printf("TAU ERROR: <file=%s,line=%d>:  Currently we cannot handle allocate statements in this version of PDT that are on the same line as a single-if statement that uses a continuation character. Please modify the source to put an explicit then/endif clause around the allocate statement and re-try, or upgrade your PDT package.\n", f->name().c_str(), inputLineNo);
		       ostr<<inbuf<<endl;
/* matching of ( and ) does not work in this case if the line is split up. 
		     do {
		      printf("INSIDE IS_IT_IF\n");
		       ostr<<inbuf<<endl;
		       alloccol = CPDB_GetSubstringCol(inbuf,"allocate");
		       istr.getline(inbuf, INBUF_SIZE); 
		       inputLineNo++;
                     } while (alloccol != 0);
		     ostr <<"\t then \n\t";
		     inputLineNo+= printTauAllocStmt(istr, ostr, inbuf, it);
		     ostr <<"\t endif"<<endl;
*/
		
		  }
		  else {
		    ostr<<"\n"<<inbuf<<endl;
		    additionalLinesRead = printTauAllocStmt(istr, ostr, inbuf, it, previousline); 
		    inputLineNo += additionalLinesRead;
                    if (additionalLinesRead)
                      strcpy(inbuf, previousline); /* update last line read */
		  }
		}	
                instrumented = true;
		break;
	    case DEALLOCATE_STMT:
                dealloccol = CPDB_GetSubstringCol(inbuf,"deallocate");
                if (addThenEndifClauses(inbuf, previousline, dealloccol - 1) && (dealloccol != 0))
                {
                /* only write till the alloc column. This assumes statement
                begins on col 1? even if it is "20 if (x.gt.2) allocate(A)" */
		  //if ((*it)->col-1) ostr<<"\t"; /* bump it up if it is col 1 */
		  startcol = (dealloccol == (*it)->col ? (*it)->col: 1);
		  if(!isRequestOnSameLineAsPreviousRequest(it, itemvec))
		    startcol = 1;
		  /* the previous instrumentation request does not write upto our current column number */
#ifdef DEBUG
		  printf("TAB:: dealloccol = %d, it->col = %d, startcol=%d\n", dealloccol, (*it)->col, startcol);
#endif /* DEBUG */
		  for(i=startcol - 1; i< dealloccol - 1; i++) {
#ifdef DEBUG
                    printf("Writing (1.7):: inbuf[%d] = %c\n",i, inbuf[i]);
#endif /* DEBUG */
                    ostr <<inbuf[i];
                  }
                  ostr<<"\t then \n";
		  /* first write TAU_DEALLOC, then the deallocate stmt */
                  additionalLinesRead=printTauDeallocStmt(istr, ostr, &inbuf[dealloccol-1], it, true, previousline);
		  inputLineNo+=additionalLinesRead;
		  if (additionalLinesRead)
		    strcpy(inbuf, previousline); /* update last line read */
		  /* now the deallocate stmt */
                  ostr<<"\t endif"<<endl;
                }
                else
                { /* there is no if clause, write TAU_DEALLOC, then stmt */
                  /* If the PDB file puts the continued if statement on the
                     same line as allocate, we need to take care of it. e.g.,
     6         if ( value .gt. 3) &
     7           deallocate(A   )
    PDB:  rstmt st#2 fdeallocate so#1 6 8 so#1 7 23 NA NA
    PDB:  rstmt st#1 fsingle_if so#1 6 8 so#1 7 23 st#3 st#2
                  */

                  is_if_stmt = addThenEndifClauses(inbuf, previousline, dealloccol - 1);
#ifdef DEBUG
                  printf("IS IT IF STMT??? %d dealloccol = %d\n", is_if_stmt, dealloccol);
#endif /* DEBUG */
                  if (is_if_stmt && (dealloccol == 0)) {
                        /* handle this separately. write the current statement */
                     printf("TAU ERROR: <file=%s,line=%d>: Currently we cannot handle de-allocate statements in this version of PDT that are on the same line as a single-if statement that uses a continuation character. Please modify the source to put an explicit then/endif clause around the de-allocate statement and re-try, or upgrade your PDT package.\n", f->name().c_str(), inputLineNo);
                       ostr<<inbuf<<endl;
                  }
                  else {

                    additionalLinesRead=printTauDeallocStmt(istr, ostr, inbuf, it, false, previousline);
		    inputLineNo+=additionalLinesRead;
		    if (additionalLinesRead)
		      strcpy(inbuf, previousline); /* update last line read */
/* 
		    ostr<<inbuf<<endl;
*/
                  }
                }
                instrumented = true;
		break;
	    case IO_STMT:
#ifdef DEBUG
                printf("I/O statement line= %d\n",(*it)->line);
#endif /* DEBUG */
		iocol = getIOColumnNumber(inbuf);
		/* This logic differs from dealloc because there may be some IO
		requests that have a column of 0 (open/close). In this case where
		we do not match read/write/print, we need to just write out the 
		original statement as it is */
                if (iocol)  { /* found read/write/print */

                if (addThenEndifClauses(inbuf, previousline, iocol - 1))
                { 
                /* only write till the io column. This assumes statement
                begins on col 1? even if it is "20 if (x.gt.2) write (3) A" */
		  //if ((*it)->col-1) ostr<<"\t"; /* bump it up if it is col 1 */
		  startcol = (iocol == (*it)->col ? (*it)->col: 1);
		  if(!isRequestOnSameLineAsPreviousRequest(it, itemvec))
		    startcol = 1;
		  /* the previous instrumentation request does not write upto our current column number */
#ifdef DEBUG
		  printf("TAB:: iocol = %d, it->col = %d, startcol=%d\n", iocol, (*it)->col, startcol);
#endif /* DEBUG */
		  for(i=startcol - 1; i< iocol - 1; i++) {
#ifdef DEBUG
                    printf("Writing (1.9):: inbuf[%d] = %c\n",i, inbuf[i]);
#endif /* DEBUG */
                    ostr <<inbuf[i];
                  }
                  ostr<<"\t then \n";
		  /* first write TAU's IO statement, then the IO stmt */
                  additionalLinesRead=printTauIOStmt(istr, ostr, &inbuf[iocol-1], it, true, previousline);
		  inputLineNo+=additionalLinesRead;
		  if (additionalLinesRead)
		    strcpy(inbuf, previousline); /* update last line read */
		  /* now the IO stmt */
                  ostr<<"\t endif"<<endl;
                }
                else
                { /* there is no if clause, write TAU's IO statement, then stmt */
                  /* If the PDB file puts the continued if statement on the
                     same line as IO, we need to take care of it. e.g.,
     6         if ( value .gt. 3) &
     7           write (4) A
                  */

                  is_if_stmt = addThenEndifClauses(inbuf, previousline, iocol - 1);
#ifdef DEBUG
                  printf("IS IT IF STMT??? %d iocol = %d\n", is_if_stmt, iocol);
#endif /* DEBUG */
                  if (is_if_stmt && (iocol == 0)) {
                        /* handle this separately. write the current statement */
                     printf("TAU ERROR: <file=%s,line=%d>: Currently we cannot handle IO statements in this version of PDT that are on the same line as a single-if statement that uses a continuation character. Please modify the source to put an explicit then/endif clause around the IO statement and re-try, or upgrade your PDT package.\n", f->name().c_str(), inputLineNo);
                       ostr<<inbuf<<endl;
                  }
                  else {

                    additionalLinesRead=printTauIOStmt(istr, ostr, inbuf, it, false, previousline);
		    inputLineNo+=additionalLinesRead;
		    if (additionalLinesRead)
		      strcpy(inbuf, previousline); /* update last line read */
                  }
                }
#ifdef DEBUG
                  printf("I/O statement at col %d: %s\n",iocol, inbuf);
#endif /* DEBUG */
                }
		else { /* just write out the statement (open/close) as it is */
  		  ostr<<endl; 
		/* start with a new line. Clears up residue from TAU_PROFILE_START*/
                  ostr<<inbuf<<endl;
                }
                instrumented = true;
                break;
	    default:
		cout <<"Unknown option in instrumentFFile:"<<(*it)->kind<<endl;
		instrumented = true;
		break;
	  } /* end of switch statement */
        } /* for it/lit */
        lit = it;		
      } /* reached line */
      strcpy(previousline, inbuf); /* save the current line */
#ifdef DEBUG
      printf("SAVING %s\n", previousline);
#endif /* DEBUG */
      memset(inbuf, INBUF_SIZE, 0); // reset to zero
    } /* while */
  } /* while lit!= end */
  // For loop is over now flush out the remaining lines to the output file
  while (istr.getline(inbuf, INBUF_SIZE) )
  {
    ostr << inbuf <<endl;
    strcpy(previousline, inbuf); /* save the current line. Hmm... not necessary here?  */
  }
  // written everything. quit and debug!
  ostr.close();
  delete [] previousline;
  return true; /* end of instrumentFFile */
}

/* -------------------------------------------------------------------------- */
/* -- Determine the TAU profiling group name associated with the pdb file --- */
/* -------------------------------------------------------------------------- */
void setGroupName(PDB& p, string& group_name)
{

 PDB::macrovec m = p.getMacroVec();
 string search_string = "TAU_GROUP"; 

 for (PDB::macrovec::iterator it = m.begin(); it != m.end(); ++it)
 {
   string macro_name = (*it)->fullName();
   if (macro_name.find(search_string) != string::npos)
   { 
#ifdef DEBUG
     cout <<"Found group:"<<macro_name<<endl;
#endif /* DEBUG */
     if ((*it)->kind() == pdbItem::MA_DEF)
     {
	string needle = string("#define TAU_GROUP ");
	string haystack = (*it)->text(); /* the complete macro text */
        
	/* To extract the value of TAU_GROUP, search the macro text */
        string::size_type pos = haystack.find(needle); 
	if (pos != string::npos)
        {
	  group_name = string("TAU_GROUP_") + haystack.substr(needle.size(),
	    haystack.size() - needle.size() - pos); 
#ifdef DEBUG
	  cout <<"Extracted group name:"<<group_name<<endl;
#endif /* DEBUG  */
	}
     }
     
   }

 }


}

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

/* -------------------------------------------------------------------------- */
/* -- Instrument the program using C, C++ or F90 instrumentation routines --- */
/* -------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  string outFileName("out.ins.C");
  string group_name("TAU_USER"); /* Default: if nothing else is defined */
  string header_file("Profile/Profiler.h"); 
  bool retval;
	/* Default: if nothing else is defined */

  if (argc < 3) 
  { 
    cout <<"Usage : "<<argv[0] <<" <pdbfile> <sourcefile> [-o <outputfile>] [-noinline] [-noinit] [-memory] [-g groupname] [-i headerfile] [-c|-c++|-fortran] [-f <instr_req_file> ] [-rn <return_keyword>] [-rv <return_void_keyword>] [-e <exit_keyword>] [-p] [-check <filename>]"<<endl;
    cout<<"----------------------------------------------------------------------------------------------------------"<<endl;
    cout <<"-noinline: disables the instrumentation of inline functions in C++"<<endl;
    cout <<"-noinit: does not call TAU_INIT(&argc,&argv). This disables a.out --profile <group[+<group>]> processing."<<endl;
    cout <<"-memory: calls #include <malloc.h> at the beginning of each C/C++ file for malloc/free replacement and traps Fortran 90 allocate/deallocate statements."<<endl;
    cout <<"-g groupname: puts all routines in a profile group. "<<endl;
    cout <<"-i headerfile: instead of <Profile/Profiler.h> a user can specify a different header file for TAU macros"<<endl;
    cout<<"-c : Force a C++ program to be instrumented as if it were a C program with explicit timer start/stops"<<endl;
    cout<<"-c++ : Force instrumentation of file using TAU's C++ API in case it cannot infer the language"<<endl;
    cout<<"-fortran : Force instrumentation using TAU's Fortran API in case it cannot infer the language"<<endl;
    cout<<"-f <inst_req_file>: Specify an instrumentation specification file"<<endl;
    cout<<"-rn <return_keyword>: Specify a different keyword for return (e.g., a  macro that calls return"<<endl;
    cout<<"-rv <return_void_keyword>: Specify a different keyword for return in a void routine"<<endl;
    cout<<"-e <exit_keyword>: Specify a different keyword for exit (e.g., a macro that calls exit)"<<endl;
    cout<<"-p : Generate instrumentation calls for perflib [LANL] instead of TAU" <<endl;
    cout<<"-spec <spec_file>: Use instrumentation commands from <spec_file>"<<endl;
    cout<<"-check <filename>: Check match of filename in selective instrumentation file"<<endl;
    cout<<"----------------------------------------------------------------------------------------------------------"<<endl;
    cout<<"e.g.,"<<endl;
    cout<<"% "<<argv[0]<<" foo.pdb foo.cpp -o foo.inst.cpp -f select.tau"<<endl;
    cout<<"----------------------------------------------------------------------------------------------------------"<<endl;
    return 1;
  }
  bool outFileNameSpecified = false;
  int i; 

  const char *filename; 
  for(i=0; i < argc; i++)
  {
    switch(i) {
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
        if (strcmp(argv[i], "-noinline")==0)
 	{
#ifdef DEBUG
          printf("Noinline flag\n");
#endif /* DEBUG */
          noinline_flag = true;
        }
        if (strcmp(argv[i], "-noinit")==0)
 	{
#ifdef DEBUG
          printf("Noinit flag\n");
#endif /* DEBUG */
          noinit_flag = true;
        }
        if (strcmp(argv[i], "-memory")==0)
 	{
#ifdef DEBUG
          printf("Memory profiling flag\n");
#endif /* DEBUG */
          memory_flag = true;
        }
        if (strcmp(argv[i], "-c")==0)
 	{
#ifdef DEBUG
          printf("Language explicitly specified as C\n");
#endif /* DEBUG */
          lang_specified = true;
	  tau_language = tau_c;
        }
        if (strcmp(argv[i], "-c++")==0)
 	{
#ifdef DEBUG
          printf("Language explicitly specified as C++\n");
#endif /* DEBUG */
          lang_specified = true;
	  tau_language = tau_cplusplus;
        }
        if (strcmp(argv[i], "-fortran")==0)
 	{
#ifdef DEBUG
          printf("Language explicitly specified as Fortran\n");
#endif /* DEBUG */
          lang_specified = true;
	  tau_language = tau_fortran;
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
        if (strcmp(argv[i], "-p") == 0)
        {
          use_perflib=true;
        }
        if (strcmp(argv[i], "-rn") == 0)
	{
	  ++i;
	  strcpy(return_nonvoid_string,argv[i]);
#ifdef DEBUG
          printf("Using non void return keyword: %s\n", return_nonvoid_string);
#endif /* DEBUG */
  	}
        if (strcmp(argv[i], "-rv") == 0)
	{
	  ++i;
	  strcpy(return_void_string,argv[i]);
#ifdef DEBUG
          printf("Using void return keyword: %s\n", return_void_string);
#endif /* DEBUG */
  	}
        if (strcmp(argv[i], "-e") == 0)
	{
	  ++i;
	  strcpy(exit_keyword,argv[i]);
	  using_exit_keyword = true;
#ifdef DEBUG
          printf("Using exit_keyword keyword: %s\n", exit_keyword);
#endif /* DEBUG */
  	}
        if (strcmp(argv[i], "-spec") == 0)
        {
          ++i;
          processInstrumentationRequests(argv[i]);
          use_spec = true;
#ifdef DEBUG
          printf("Using instrumentation code from spec file: %s\n", argv[i]);
#endif /* DEBUG */
        }
        if (strcmp(argv[i], "-check") == 0) {
          ++i;
	  if (processFileForInstrumentation(argv[i])) {
	    printf ("yes\n");
	  } else {
	    printf ("no\n");
	  }
	  return 0;
        }
        break;
      }

   }
  if (!outFileNameSpecified)
  { /* if name is not specified on the command line */
    outFileName = string(filename + string(".ins"));
  }


  PDB p(argv[1]); if ( !p ) return 1;
  setGroupName(p, group_name);

  bool instrumentThisFile;
  bool fuzzyMatchResult;
  bool fileInstrumented = false;
  for (PDB::filevec::const_iterator it=p.getFileVec().begin();
       it!=p.getFileVec().end(); ++it) 
  {
     /* reset this variable at the beginning of the loop */
     instrumentThisFile = false;

     if ((fuzzyMatchResult = fuzzyMatch((*it)->name(), string(filename))) && 
         (instrumentThisFile = processFileForInstrumentation(string(filename))))
     { /* should we instrument this file? Yes */
       PDB::lang_t l = p.language();
       fileInstrumented = true; /* We will instrument this file */

#ifdef DEBUG
       cout <<" *** FILE *** "<< (*it)->name()<<endl;
       cout <<"Language "<<l <<endl;
#endif
       if (lang_specified)
       { /* language explicitly specified on command line*/
	 switch (tau_language) { 
	   case tau_cplusplus :
                if (use_spec)
                  instrumentCFile(p, *it, outFileName, group_name, header_file);
                else
                {
         	  retval = instrumentCXXFile(p, *it, outFileName, group_name, header_file);
	          if (!retval) {
		    cout <<"Uh Oh! There was an error in instrumenting with the C++ API, trying C next... Please do not force a C++ instrumentation API on this file: "<<(*it)->name()<<endl;
         	    instrumentCFile(p, *it, outFileName, group_name, header_file);
	 	  }
		}
		break;
	   case tau_c :
         	instrumentCFile(p, *it, outFileName, group_name, header_file);
		break;
	   case tau_fortran : 
         	instrumentFFile(p, *it, outFileName, group_name);
		break;
	   default:
		printf("Language unknown\n ");
		break;
	 }
       }
       else 
       { /* implicit detection of language */
         if (l == PDB::LA_CXX)
	 {
           if (use_spec)
             instrumentCFile(p, *it, outFileName, group_name, header_file);
           else
           {
             retval = instrumentCXXFile(p, *it, outFileName, group_name, header_file);
             if (!retval)
	     {
#ifdef DEBUG
	       cout <<"Uh Oh! There was an error in instrumenting with the C++ API, trying C next... "<<endl;
#endif /* DEBUG */
               instrumentCFile(p, *it, outFileName, group_name, header_file);
	     }
           }
	 }
         if (l == PDB::LA_C)
           instrumentCFile(p, *it, outFileName, group_name, header_file);
         if (l == PDB::LA_FORTRAN)
           instrumentFFile(p, *it, outFileName, group_name);
       }
     } /* don't instrument this file. Should we copy in to out? */
     else
     { 
       if ((fuzzyMatchResult == true) && (instrumentThisFile == false))
       { /* we should copy the file to outFile */
         ifstream ifs(filename);
         ofstream ofs(outFileName.c_str());
         /* copy ifs to ofs */
         if (ifs.is_open() && ofs.is_open())
           ofs << ifs.rdbuf(); /* COPY */ 
	 instrumentThisFile = true; /* sort of like instrumentation,
		more like processed this file. Later we need to know
		if no files were processed */
       }
     }
  }
  if (fileInstrumented == false)
  { /* no files were processed */
#ifdef DEBUG
    cout <<"No files were processed"<<endl;
#endif /* DEBUG */
    /* We should copy this file to outfile */
    ifstream ifsc(filename);
    ofstream ofsc(outFileName.c_str());
    /* copy ifsc to ofsc */
    if (ifsc.is_open() && ofsc.is_open())
       ofsc << ifsc.rdbuf(); /* COPY */ 
  }

  /* start with routines */
/* 
  for (PDB::croutinevec::iterator r=p.getCRoutineVec().begin();
       r != p.getCRoutineVec().end(); r++)
  {
    
#ifdef DEBUG
    cout << (*r)->fullName() <<endl;
#endif

  }
*/
#ifdef DEBUG
  cout <<"Done with instrumentation!" << endl;
#endif 

  return 0;
} 

  
  
/***************************************************************************
 * $RCSfile: tau_instrumentor.cpp,v $   $Author: amorris $
 * $Revision: 1.200 $   $Date: 2008/10/22 00:19:53 $
 * VERSION_ID: $Id: tau_instrumentor.cpp,v 1.200 2008/10/22 00:19:53 amorris Exp $
 ***************************************************************************/


