#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OLD_HEADER_
# include <fstream.h>
# include <set.h>
# include <algo.h>
#else
# include <fstream>
  using std::ifstream;
  using std::ofstream;
# include <set>
  using std::set;
# include <algorithm>
  using std::sort;
  using std::unique;
#endif
#include "pdbAll.h"

/* For C instrumentation */
enum itemKind_t { ROUTINE, BODY_BEGIN, FIRST_EXECSTMT, BODY_END, RETURN, EXIT};

/* For Pooma, add a -noinline flag */
bool noinline_flag = false; /* instrument inlined functions by default */

struct itemRef {
  itemRef(const pdbItem *i, bool isT) : item(i), isTarget(isT) {
    line = i->location().line();
    col  = i->location().col();
    kind = ROUTINE; /* for C++, only routines are listed */ 
  }
  itemRef(const pdbItem *i, itemKind_t k, int l, int c) : 
	line (l), col(c), item(i), kind(k) {
#ifdef DEBUG
    cout <<"Added: "<<i->name() <<" line " << l << " col "<< c <<" kind " 
	 << k <<endl;
#endif /* DEBUG */
    isTarget = true; 
  }
  itemRef(const pdbItem *i, bool isT, int l, int c)
         : item(i), isTarget(isT), line(l), col(c) {
    kind = ROUTINE; 
  }
  const pdbItem *item;
  itemKind_t kind; /* For C instrumentation */ 
  bool     isTarget;
  int      line;
  int      col;
};

static bool locCmp(const itemRef* r1, const itemRef* r2) {
  if ( r1->line == r2->line )
    return r1->col < r2->col;
  else
    return r1->line < r2->line;
}

static bool itemEqual(const itemRef* r1, const itemRef* r2) {
  return ( (r1->line == r2->line) &&
           (r1->col  == r2->col)); 
}
 
void getCXXReferences(vector<itemRef *>& itemvec, PDB& pdb, pdbFile *file) {
/* get routines, templates and member templates of classes */
  PDB::croutinevec routines = pdb.getCRoutineVec();
  for (PDB::croutinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit) 
  {
    if ( (*rit)->location().file() == file && !(*rit)->isCompilerGenerated() && 
	 ((*rit)->kind() != pdbItem::RO_EXT) ) 
    {
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
	if ((((*rit)->parentGroup()) == 0) || (*rit)->isStatic())
	{ // If it is a static function or if 
	  // there's no parent class. No need to add CT(*this)
          itemvec.push_back(new itemRef(*rit, true));
	}
	else
	{
          itemvec.push_back(new itemRef(*rit, false));
	  // false puts CT(*this)
	}
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
	  (tekind == pdbItem::TE_FUNC)) && ((*te)->bodyBegin().line() != 0))
      { 
	/* Sometimes a compiler generated routine shows up in a template.
	   These routines (such as operator=) do not have a body position. 
 	   Instrument only if it has a valid body position.  */
  	// templates need some processing. Give it a false for isTarget arg.
	// target helps identify if we need to put a CT(*this) in the type
	// old: 
        //if ((((*te)->parentGroup()) == 0) && (tekind != pdbItem::TE_STATMEM)) 
        if ((tekind == pdbItem::TE_FUNC) || (tekind == pdbItem::TE_STATMEM))
	{ 
	  // There's no parent class. No need to add CT(*this)
          itemvec.push_back(new itemRef(*te, true)); // False puts CT(*this)
	}
	else 
	{ 
	  // it is a member function add the CT macro
          itemvec.push_back(new itemRef(*te, false));
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
  sort(itemvec.begin(), itemvec.end(), locCmp);
}

/* Create a vector of items that need action: such as BODY_BEGIN, RETURN etc.*/
void getCReferences(vector<itemRef *>& itemvec, PDB& pdb, pdbFile *file) {
  PDB::croutinevec routines = pdb.getCRoutineVec();
  for (PDB::croutinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit)
  {
    pdbRoutine::locvec retlocations = (*rit)->returnLocations();
    if ( (*rit)->location().file() == file && !(*rit)->isCompilerGenerated() &&
         ((*rit)->kind() != pdbItem::RO_EXT))
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
	for (pdbRoutine::callvec::iterator cit = c.begin(); cit !=c.end(); cit++)
	{ 
	   const pdbRoutine *rr = (*cit)->call(); 
#ifdef DEBUG 
	   cout <<"Callee " << rr->name() << " location line " << (*cit)->line() << " col " << (*cit)->col() <<endl; 
#endif /* DEBUG */
	   if (strcmp(rr->name().c_str(), "exit")== 0)
	   {
	     /* routine calls exit */
	     itemvec.push_back(new itemRef(*rit, EXIT, (*cit)->line(), 
		(*cit)->col()));
	   } 
	   if (strcmp(rr->name().c_str(), "abort") == 0)
	   { /* routine calls abort */
	     itemvec.push_back(new itemRef(*rit, EXIT, (*cit)->line(), 
		(*cit)->col()));
	   }
	}
    }
  }
  sort(itemvec.begin(), itemvec.end(), locCmp);
  itemvec.erase(unique(itemvec.begin(), itemvec.end(),itemEqual),itemvec.end());
#ifdef DEBUG
  for(vector<itemRef *>::iterator iter = itemvec.begin(); iter != itemvec.end();
   iter++)
  {
    cout <<"Items ("<<(*iter)->line<<", "<<(*iter)->col<<")"<<endl;
  }
#endif /* DEBUG */
}

void getFReferences(vector<itemRef *>& itemvec, PDB& pdb, pdbFile *file) {
/* get routines, templates and member templates of classes */
  PDB::froutinevec routines = pdb.getFRoutineVec();

  for (PDB::froutinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit) 
  {
    pdbRoutine::locvec retlocations = (*rit)->returnLocations();
    pdbRoutine::locvec stoplocations = (*rit)->stopLocations();
    if ( (*rit)->location().file() == file &&  
	 ((*rit)->firstExecStmtLocation().file()))
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
          itemvec.push_back(new itemRef(*rit, RETURN,
                (*slit)->line(), (*slit)->col()));
        }
    }

  }
/* Now sort all these locations */
  sort(itemvec.begin(), itemvec.end(), locCmp);

}


const int INBUF_SIZE = 2048;

/* returns true is string is void else returns false */
bool isVoidRoutine(itemRef * i)
{
  string return_string = ((pdbRoutine *)(i->item))->signature()->returnType()->name() ;
  if (return_string.compare("void") == 0)
	return true; 
  else
	return false;
}
	

/* to instrument the file */
int instrumentCXXFile(PDB& pdb, pdbFile* f, string& outfile) 
{
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


  // initialize reference vector
  vector<itemRef *> itemvec;
  getCXXReferences(itemvec, pdb, f);

  // put in code to insert <Profile/Profiler.h> 
  ostr<< "#include <Profile/Profiler.h>"<<endl;
  
  int inputLineNo = 0;
  int lastInstrumentedLineNo = 0;
  for(vector<itemRef *>::iterator it = itemvec.begin(); it != itemvec.end();
	++it)
  {
    // Read one line each till we reach the desired line no. 
#ifdef DEBUG
    cout <<"S: "<< (*it)->item->fullName() << " line "<< (*it)->line << " col " << (*it)->col << endl;
#endif 
    bool instrumented = false;
    if (lastInstrumentedLineNo >= (*it)->line )
    {
      // Hey! This line has already been instrumented. Go to the next
      // entry in the func
#ifdef DEBUG
      cout <<"Entry already instrumented or brace not found - reached next routine! line = "<<(*it)->line <<endl;
#endif
      continue; // takes you to the next iteration in the for loop
    }

    while((instrumented == false) && (istr.getline(inbuf, INBUF_SIZE)) )
    {
      inputLineNo ++;
      if (inputLineNo < (*it)->line) 
      {
	// write the input line in the output stream
        ostr << inbuf <<endl;
      }
      else 
      { 
	// we're at the desired line no. search for an open brace
	int inbufLength = strlen(inbuf);

	for(int i=0; i< inbufLength; i++)
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
	    for (int space = 0; space < (*it)->col ; space++) ostr << " " ; 
#endif
	    // leave some leading spaces for formatting...

	    ostr <<"  TAU_PROFILE(\"" << (*it)->item->fullName() ;
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

	    if (strstr((*it)->item->name().c_str(), "main")) 
	    { /* it is main() */
	      ostr << "TAU_DEFAULT);" <<endl; // give an additional line 
#ifdef SPACES
	      for (int space = 0; space < (*it)->col ; space++) ostr << " " ; 
#endif 
	      // leave some leading spaces for formatting...
	
	      ostr <<"#ifndef TAU_MPI" <<endl; // set node 0
	      ostr <<"  TAU_PROFILE_SET_NODE(0);" <<endl; // set node 0
	      ostr <<"#endif /* TAU_MPI */" <<endl; // set node 0
	    }
	    else 
	    {
	      ostr <<"TAU_USER);" <<endl; // give an additional line 
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
	
      } // else      

    } // while loop

  } // For loop
  // For loop is over now flush out the remaining lines to the output file
  while (istr.getline(inbuf, INBUF_SIZE) ) 
  { 
    ostr << inbuf <<endl;
  }
  // written everything. quit and debug!
  ostr.close();

  return 0;
}

/* BodyBegin for a routine that does returns some value */
void processNonVoidRoutine(ostream& ostr, string& return_type, itemRef *i)
{

#ifdef DEBUG
  cout <<"Return type :" << return_type<<endl;
#endif /* DEBUG */
  ostr <<"{\n\t"<<return_type<< " tau_ret_val; "<<endl;
  ostr <<"\tTAU_PROFILE_TIMER(tautimer, \""<<
    ((pdbRoutine *)(i->item))->name() << "\", \"" <<
    ((pdbRoutine *)(i->item))->signature()->name() << "\", ";

  if (strstr(i->item->name().c_str(), "main"))
  { /* it is main() */
     ostr << "TAU_DEFAULT);" <<endl; // give an additional line
#ifdef SPACES
     for (int space = 0; space < (*it)->col ; space++) ostr << " " ;
#endif
     // leave some leading spaces for formatting...

     ostr <<"#ifndef TAU_MPI" <<endl; // set node 0
     ostr <<"  TAU_PROFILE_SET_NODE(0);" <<endl; // set node 0
     ostr <<"#endif /* TAU_MPI */" <<endl; // set node 0
  }
  else
  {
    ostr <<"TAU_USER);" <<endl; // give an additional line
  }

  ostr <<"\tTAU_PROFILE_START(tautimer); "<<endl;
	
}
void processVoidRoutine(ostream& ostr, string& return_type, itemRef *i)
{
  ostr <<"{ \n\tTAU_PROFILE_TIMER(tautimer, \""<<
    ((pdbRoutine *)(i->item))->name() << "\", \"" <<
    ((pdbRoutine *)(i->item))->signature()->name() << "\", ";

  if (strstr(i->item->fullName().c_str(), "main("))
  { /* it is main() */
     ostr << "TAU_DEFAULT);" <<endl; // give an additional line
#ifdef SPACES
     for (int space = 0; space < (*it)->col ; space++) ostr << " " ;
#endif
     // leave some leading spaces for formatting...

     ostr <<"#ifndef TAU_MPI" <<endl; // set node 0
     ostr <<"  TAU_PROFILE_SET_NODE(0);" <<endl; // set node 0
     ostr <<"#endif /* TAU_MPI */" <<endl; // set node 0
  }
  else
  {
    ostr <<"TAU_USER);" <<endl; // give an additional line
  }

  ostr <<"\tTAU_PROFILE_START(tautimer);"<<endl;
}

void processReturnExpression(ostream& ostr, string& ret_expression)
{
  ostr <<"{ tau_ret_val = " << ret_expression << "; TAU_PROFILE_STOP(tautimer); return tau_ret_val; }"<<endl;
}




int instrumentCFile(PDB& pdb, pdbFile* f, string& outfile) 
{ 
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
  cout << "Processing " << file << " in instrumentCFile..." << endl;
#endif


  // initialize reference vector
  vector<itemRef *> itemvec;
  getCReferences(itemvec, pdb, f);

  // Begin Instrumentation
  // put in code to insert <Profile/Profiler.h>
  ostr<< "#include <Profile/Profiler.h>"<<endl;

  int inputLineNo = 0;
  vector<itemRef *>::iterator lit = itemvec.begin();
  while (lit != itemvec.end())
  {
    // Read one line each till we reach the desired line no.
#ifdef DEBUG
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
        for(int i=0; i< ((*lit)->col)-1; i++)
	{ 
#ifdef DEBUG
	  cout <<"Writing(1): "<<inbuf[i]<<endl;
#endif /* DEBUG */
	  ostr << inbuf[i];
	}
        vector<itemRef *>::iterator it;
        for (it = lit; ((it != itemvec.end()) && ((*it)->line == (*lit)->line)); ++it) 
        { /* it/lit */
          int inbufLength = strlen(inbuf);

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
		return_string = ((pdbRoutine *)((*it)->item))->signature()->returnType()->name() ;
		if (isVoidRoutine(*it))
		{
#ifdef DEBUG 
		  cout <<"Void return value "<<endl;
#endif /* DEBUG */
		  processVoidRoutine(ostr, return_string, *it);
		}
		else
		{
		  processNonVoidRoutine(ostr, return_string, *it);
		}
		instrumented = true; 
		break;
	    case RETURN: 
#ifdef DEBUG 
		cout <<"Return "<<endl;
#endif /* DEBUG */
		if (strncmp((const char *)&inbuf[((*it)->col)-1], "return", 6)==0)
		{
		/* currently, the return statement cannot be a macro. It *has* 
		   to contain "return <...> ;" syntax. */
		  if (isVoidRoutine(*it))
		  {	
#ifdef DEBUG 
		    cout <<" Return for a void routine" <<endl;
#endif /* DEBUG */
		    /* instrumentation code here */
		    ostr << "{ TAU_PROFILE_STOP(tautimer); return; }" <<endl;
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
		    for (k = (*it)->col+5; (inbuf[k] != ';') && (k<inbufLength) ; k++)
		      ret_expression.append(&inbuf[k], 1);
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
			}
		      } while(inbuf[l] != ';');
			/* copy the buffer into inbuf */
		      write_from = l+1; 
		    }
			 
#ifdef DEBUG 
		    cout <<"ret_expression = "<<ret_expression<<endl;
#endif /* DEBUG */
		    processReturnExpression(ostr, ret_expression); 
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
		ostr<<"\n}\n\tTAU_PROFILE_STOP(tautimer);\n"<<endl; 
		instrumented = true; 
		break;
	    case EXIT:
#ifdef DEBUG 
		cout <<"Exit" <<endl;
#endif /* DEBUG */
		ostr <<"{ TAU_PROFILE_EXIT(\"exit\"); ";
		for (k = (*it)->col-1; inbuf[k] != ';' ; k++)
		  ostr<<inbuf[k]; 
		ostr <<"; }";
		instrumented = true; 
		write_from = k+1;
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
	  for (int j=write_from; j < write_upto; j++)
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
    } /* while */
  } /* while lit != end */
  // For loop is over now flush out the remaining lines to the output file
  while (istr.getline(inbuf, INBUF_SIZE) )
  {
    ostr << inbuf <<endl;
  }
  // written everything. quit and debug!
  ostr.close();


} /* End of instrumentCFile */ 




int instrumentFFile(PDB& pdb, pdbFile* f, string& outfile) 
{ /* To be implemented */
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
  cout << "Processing " << file << " in instrumentFFile..." << endl;
#endif

  // initialize reference vector
  vector<itemRef *> itemvec;
  getFReferences(itemvec, pdb, f);

  int inputLineNo = 0;
  int lastInstrumentedLineNo = 0;
  bool is_if_stmt;
  for(vector<itemRef *>::iterator it = itemvec.begin(); it != itemvec.end();
        ++it)
  {
    // Read one line each till we reach the desired line no.
#ifdef DEBUG
    cout <<"S: "<< (*it)->item->fullName() << " line "<< (*it)->line << " col " << (*it)->col << endl;
#endif
    bool instrumented = false;
    bool isProgram = false;
    if (lastInstrumentedLineNo >= (*it)->line )
    {
      // Hey! This line has already been instrumented. Go to the next
      // entry in the func
#ifdef DEBUG
      cout <<"Entry already instrumented - reached next routine! line = "<<(*it)->line <<endl;
#endif
      continue; // takes you to the next iteration in the for loop
    }

    while((instrumented == false) && (istr.getline(inbuf, INBUF_SIZE)) )
    {
      inputLineNo ++;
      is_if_stmt = false; /* initialize variable now. Check for it later */
      if (inputLineNo < (*it)->line)
      {
        // write the input line in the output stream
        ostr << inbuf <<endl;
      }
      else
      { /* reached line */
        int inbufLength = strlen(inbuf);
        // we're at the desired line no. go to the specified col
#ifdef DEBUG 
	cout <<"Line " <<(*it)->line <<" Col " <<(*it)->col <<endl;
#endif /* DEBUG */
        for(int i=0; i< ((*it)->col)-1; i++)
	{ 
	  ostr << inbuf[i];
	}
	/* set instrumented = true after inserting instrumentation */
	switch((*it)->kind)
	{
	  case BODY_BEGIN:

#ifdef DEBUG
	  cout <<"Body Begin: Routine " <<(*it)->item->fullName()<<endl;
#endif /* DEBUG */
		ostr <<"integer profiler(2)"<<endl;
		/* spaces */
     		for (int space = 0; space < (*it)->col-1 ; space++) 
		  ostr << inbuf[space] ;
		ostr <<"save profiler"<<endl<<endl;
     		for (int space = 0; space < (*it)->col-1 ; space++) 
		    ostr << inbuf[space]; 
		if (((pdbRoutine *)(*it)->item)->kind() == pdbItem::RO_FPROG)
		{
#ifdef DEBUG
	  	  cout <<"Routine is main fortran program "<<endl;
#endif /* DEBUG */
		  ostr <<"call TAU_PROFILE_INIT()"<<endl;
		  isProgram = true;
		  /* put spaces on the next line */
     		  for (int space = 0; space < (*it)->col-1 ; space++) 
		    ostr << inbuf[space]; 
		}
		ostr <<"call TAU_PROFILE_TIMER(profiler,'" <<
		  (*it)->item->fullName()<< "')"<<endl;
		/* spaces */
     		for (int space = 0; space < (*it)->col-1 ; space++) 
		  ostr << inbuf[space]; 
		ostr <<"call TAU_PROFILE_START(profiler)"<<endl;
/* DON'T CALL TAU_PROFILE_SET_NODE here. 
   Call it in TAU_PROFILE_INIT if MPI is not used */ 		
		/* 
	        if (isProgram) 
		{
     		  for (int space = 0; space < (*it)->col-1 ; space++) 
		    ostr << inbuf[space]; 
		  ostr <<"call TAU_PROFILE_SET_NODE(0)"<<endl;
		}
		*/
		/* spaces */
     		for (int space = 0; space < (*it)->col-1 ; space++) 
		  ostr << inbuf[space]; 
		instrumented = true;
		break;
	  case RETURN:
#ifdef DEBUG
	        cout <<"RETURN statement "<<endl;
#endif /* DEBUG */
		if (strstr(inbuf,"if") != NULL)
		{ 
		  is_if_stmt = true;
		  ostr << "then"<<endl;
		  ostr << "          ";
		}
	
		ostr <<"call TAU_PROFILE_STOP(profiler)"<<endl;
     		for (int space = 0; space < (*it)->col-1 ; space++) 
		{
 		  char c = inbuf[space]; 
		  if (!((c == ' ') || (c == '\t')))
		    c = ' ';
		  ostr << c; 
		}
		instrumented = true;
		break;
		 
	  default:
		cout <<"Unknown option in instrumentFFile:"<<(*it)->kind<<endl;
		instrumented = true;
		break;
	}

        for(int j= ((*it)->col)-1; j <inbufLength; j++)
	{ 
	  ostr << inbuf[j];
	}
	ostr<<endl;
	if (is_if_stmt) 
	{
	  ostr <<"         endif"<<endl;
	}      
      } /* reached line */
    } /* while */
  } /* for */
  // For loop is over now flush out the remaining lines to the output file
  while (istr.getline(inbuf, INBUF_SIZE) )
  {
    ostr << inbuf <<endl;
  }
  // written everything. quit and debug!
  ostr.close();
}


int main(int argc, char **argv)
{
  string outFileName("out.ins.C");
  if (argc < 3) 
  { 
    cout <<"Usage : "<<argv[0] <<" <pdbfile> <sourcefile> [-o <outputfile>] [-noinline]"<<endl;
    return 1;
  }
  PDB p(argv[1]); if ( !p ) return 1;
  const char * filename = argv[2];  

  if ((argc == 5) || (argc == 6))
  {
#ifdef DEBUG
    cout <<"5 argc "<<endl;
#endif
    if (strcmp(argv[3], "-o") == 0) 
    { 
#ifdef DEBUG
      cout <<"checks out... -o option"<<endl;
#endif 
      outFileName = string(argv[4]);
    }
    else 
    {
      cout<<"Hey! 5/6 args but -o doesn't show up as 4th arg." <<endl;
      cout <<"argv[4] is "<<argv[4] <<endl;
    }
    if (argc == 6) 
    {
      if (strcmp(argv[5], "-noinline") == 0)	
      {
        noinline_flag = true;
#ifdef DEBUG
        cout <<"Setting noinline_flag to true "<<endl;
#endif /* DEBUG */
      }
    }
  }
  else 
  {
    outFileName = string(filename + string(".ins"));
  }


  for (PDB::filevec::const_iterator it=p.getFileVec().begin();
       it!=p.getFileVec().end(); ++it) 
  {
     if ((*it)->name() == string(filename)) 
     {
       PDB::lang_t l = p.language();

#ifdef DEBUG
       cout <<" *** FILE *** "<< (*it)->name()<<endl;
       cout <<"Language "<<l <<endl;
#endif
       if (l == PDB::LA_CXX)
         instrumentCXXFile(p, *it, outFileName);
       if (l == PDB::LA_C)
         instrumentCFile(p, *it, outFileName);
       if (l == PDB::LA_FORTRAN)
         instrumentFFile(p, *it, outFileName);
     }
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

  
  
