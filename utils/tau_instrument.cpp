/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2005  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: tau_instrument.cpp				  **
**	Description 	: Provides selective instrumentation support in   **
**                        TAU.                                            **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@paratools.com      **
**	Documentation	:                                                 **
***************************************************************************/

/* Headers */
#include <string> 
#include <iostream>
using namespace std;
#include "tau_instrument.h"
#ifdef _OLD_HEADER_
# include <fstream.h>
# include <algo.h>
# include <list.h>
#else
# include <fstream>
# include <algorithm>
# include <list>
# include <map>
#endif
#include "pdbAll.h"

//#define DEBUG 1
extern bool wildcardCompare(char *wild, char *string, char kleenestar);
extern bool instrumentEntity(const string& function_name);
extern bool fuzzyMatch(const string& a, const string& b);
extern bool memory_flag;

/* Globals */
///////////////////////////////////////////////////////////////////////////
vector<tauInstrument *> instrumentList; 
list<pair<int, list<string> > > additionalDeclarations; 
list<pair<int, list<string> > > additionalInvocations; 
/* In this list resides a list of variable declarations that must be added before
the first executable statement. It has a list of strings with the routine no. as 
the first element of the pair (the second is the list of strings). */
///////////////////////////////////////////////////////////////////////////

/* Constructors */
///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor which sets all the items
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, string r, int l, string c, 
	instrumentKind_t k): filename(f), fileSpecified(true), routineName(r),
	lineno(l), lineSpecified (true), code(c), codeSpecified(true), kind (k) 
{}


///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// file = "foo.cpp" line=245 code = "TAU_NODE(0);" 
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, int l, string c, instrumentKind_t k) : 
      	filename(f), fileSpecified(true), lineno(l), lineSpecified(true), 
      	routineSpecified(false), code(c), codeSpecified(true), kind(k) 
{}

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// entry routine="foo" code="print *, 'Hi'; " */
//    /* FOR THIS TYPE, you must specify the codeSpecified argument */
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string r, string c, bool cs, instrumentKind_t k) :
      	routineName(r), routineSpecified(true), code(c), codeSpecified(cs), 
  	kind(k), fileSpecified(false), lineSpecified(false)
{}

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
//    /* entry/exit file = "foo.f90" routine = "foo" code = "printf" */
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, string r, string c, instrumentKind_t k) 
	: filename(f), fileSpecified(true), routineName(r), 
	routineSpecified(true), code (c), codeSpecified(true), 
	lineSpecified(false), kind (k) 
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// loops routine = "foo" 
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string r, instrumentKind_t k ) : 
	routineName(r), routineSpecified (true), kind (k), 
	lineSpecified(false), fileSpecified(false), codeSpecified(false) 
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// loops file = "f1.cpp" routine = "foo" 
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, string r, instrumentKind_t k ) : 
	filename (f), fileSpecified(true), routineName(r), 
	routineSpecified (true), kind (k), lineSpecified(false), 
	codeSpecified(false) 
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() dtor
///////////////////////////////////////////////////////////////////////////
tauInstrument::~tauInstrument() { }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::print writes the contents of tauInstrument to ostr
///////////////////////////////////////////////////////////////////////////
ostream& tauInstrument::print(ostream& ostr) const 
{
       if (fileSpecified) ostr << "filename: "<<filename<< " ";
       if (routineSpecified) ostr <<"routine: "<<routineName<< " " ;
       if (lineSpecified) ostr<<"line no: "<<lineno<< "  " ;
       if (codeSpecified) ostr<<"code: "<<code<<" " ;
       switch (kind) 
       {
	 case TAU_LOOPS:
		 ostr<<"loops: ";
		 break;
	 case TAU_IO:
		 ostr<<"io: ";
		 break;
	 case TAU_MEMORY:
		 ostr<<"memory: ";
		 break;
	 case TAU_LINE:
		 ostr<<"line:";
		 break;
	 case TAU_ROUTINE_ENTRY:
		 ostr<<"entry: ";
		 break;
	 case TAU_ROUTINE_EXIT:
		 ostr<<"exit: ";
		 break;
	 case TAU_NOT_SPECIFIED:
		 ostr<<"ERROR: NOT SPECIFIED KIND";
		 break;
	 default:
		 ostr<<"default: ???";
		 break;
       }
       ostr<<endl;
       return ostr;
}
    
///////////////////////////////////////////////////////////////////////////
// tauInstrument::getFileSpecified() accesses private data member
///////////////////////////////////////////////////////////////////////////
bool 	tauInstrument::getFileSpecified(void) { return fileSpecified; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getFileName() accesses private data member
///////////////////////////////////////////////////////////////////////////
string& tauInstrument::getFileName(void) { return filename; } 

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getRoutineSpecified() accesses private data member
///////////////////////////////////////////////////////////////////////////
bool 	tauInstrument::getRoutineSpecified(void) { return routineSpecified; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getRoutineName() accesses private data member
///////////////////////////////////////////////////////////////////////////
string& tauInstrument::getRoutineName(void) { return routineName; } 

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getLineSpecified() accesses private data member
///////////////////////////////////////////////////////////////////////////
bool 	tauInstrument::getLineSpecified(void) { return lineSpecified; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getLineNo() accesses private data member
///////////////////////////////////////////////////////////////////////////
int 	tauInstrument::getLineNo(void) { return lineno; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getCodeSpecified() accesses private data member
///////////////////////////////////////////////////////////////////////////
bool 	tauInstrument::getCodeSpecified(void) { return codeSpecified; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getCode() accesses private data member
///////////////////////////////////////////////////////////////////////////
string& tauInstrument::getCode(void) { return code; } 

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getKind() accesses private data member
///////////////////////////////////////////////////////////////////////////
instrumentKind_t tauInstrument::getKind(void) { return kind; }


/* Instrumentation section */

///////////////////////////////////////////////////////////////////////////
// parseError
// input: line number and column 
///////////////////////////////////////////////////////////////////////////
void parseError(char *message, char *line, int lineno, int column)
{
  printf("ERROR: %s: parse error at selective instrumentation file line %d col %d\n",
	message, lineno, column);
  printf("line=%s\n", line);
  exit(0);
}

#define WSPACE(line) while ( line[0] == ' ' || line[0] == '\t')  \
    { \
      if (line[0] == '\0') parseError("EOL found", line, lineno, line - original);  \
      line++;  \
    } 

#define TOKEN(k) if (line[0] != k || line[0] == '\0') parseError("token not found", line, lineno, (int ) (line - original)); \
		 else line++; 

#define RETRIEVESTRING(pname, line) i = 0; \
  while (line[0] != '"') { \
  if (line [0] == '\0') parseError("EOL", line, lineno, line - original); \
    pname[i++] = line[0]; line++; \
  } \
  pname[i] = '\0';  \
  line++; /* found closing " */

#define RETRIEVECODE(pname, line) i = 0; \
  while (line[0] != '"') { \
    if (line [0] == '\0') parseError("EOL", line, lineno, line - original); \
    if ((line[0] == '\\') && (line[1] == '"')) line++; \
    pname[i++] = line[0]; line++; \
  } \
  pname[i] = '\0';  \
  line++; /* found closing " */

#define RETRIEVENUMBER(pname, line) i = 0; \
  while (line[0] != ' ' && line[0] != '\t' ) { \
  if (line [0] == '\0') parseError("EOL", line, lineno, line - original); \
    pname[i++] = line[0]; line++; \
  } \
  pname[i] = '\0';  \
  line++; /* found closing " */

///////////////////////////////////////////////////////////////////////////
// parseInstrumentationCommand
// input: line -  character string containing a line of text from the selective 
// instrumentation file 
// input: lineno - integer line no. (for reporting parse errors if any)
//
///////////////////////////////////////////////////////////////////////////
void parseInstrumentationCommand(char *line, int lineno)
{
  char  *original;
  int i, ret, value; 
  bool filespecified = false; 
  char pname[INBUF_SIZE]; /* parsed name */
  char pfile[INBUF_SIZE]; /* parsed filename */
  char plineno[INBUF_SIZE]; /* parsed lineno */
  char pcode[INBUF_SIZE]; /* parsed code */
  int m1, m2, m3; 
  instrumentKind_t kind = TAU_NOT_SPECIFIED;
  m1 = m2 = m3 = 1; /* does not match by default -- for matching loops/io/mem */

#ifdef DEBUG
  printf("Inside parseInstrumentationCommand: line %s lineno: %d\n",
		  line, lineno);
#endif /* DEBUG */

  original = line; 
  /* check the initial keyword */
  if (strncmp(line, "file", 4) == 0)
  { 
    /* parse: file = "foo.cc" line = 245 code = "TAU_NODE(0);" */
    line+=4; /* start checking from here */
    /* WHITE SPACES */
    WSPACE(line);
    TOKEN('=');
    WSPACE(line);
    TOKEN('"');
    RETRIEVESTRING(pfile, line);
    filespecified = true; 
#ifdef DEBUG
    printf("Got name = %s\n", pfile);
#endif 

    WSPACE(line); /* space  */
    if (strncmp(line, "line", 4) == 0)
    { /* got line token, get line no. */
      line += 4; 
      WSPACE(line);
      TOKEN('=');
      WSPACE(line);
      RETRIEVENUMBER(plineno, line);
      ret = sscanf(plineno, "%d", &value); 
#ifdef DEBUG
      printf("got line no = %d\n", value);
#endif /* DEBUG */
    }
    else parseError("<line> token not found", line, lineno, line - original);
    WSPACE(line); 
    /* go to code */
    if (strncmp(line, "code", 4) == 0)
    { 
      line+= 4; /* move 4 spaces */
      /* check for = <WSPACE> " */
      WSPACE(line);
      TOKEN('=');
      WSPACE(line);
      TOKEN('"');
      RETRIEVECODE(pcode, line);
#ifdef DEBUG
      printf("Got code = %s\n", pcode);
#endif /* DEBUG */
    }
    else parseError("<code> token not found", line, lineno, line - original); 
#ifdef DEBUG
    printf("file=%s, code = %s, line no = %d\n", pfile, pcode, value);
#endif /* DEBUG */
    instrumentList.push_back(new tauInstrument(string(pfile), value, string(pcode), TAU_LINE)); 
    
  }
  else
  { /* parse: entry routine="foo()", code = "TAU_SET_NODE(0)" */
    if (strncmp(line, "entry", 5) == 0)
    {
      line+=5; 
#ifdef DEBUG
      printf("Found ENTRY!\n");
#endif /* DEBUG */
      WSPACE(line);
      if (strncmp(line, "file", 4) == 0)
      {
	line+= 4;
	WSPACE(line);
	TOKEN('=');
	WSPACE(line);
	TOKEN('"');
	RETRIEVESTRING(pfile, line);
	WSPACE(line);
	filespecified = true; 
#ifdef DEBUG
	printf("GOT FILE = %s\n", pfile);
#endif /* DEBUG */
      }
      if (strncmp(line, "routine", 7) == 0)
      {
	line+=7; 
	WSPACE(line);
	TOKEN('=');
	WSPACE(line);
	TOKEN('"');
	RETRIEVESTRING(pname, line);
	WSPACE(line);
        if (strncmp(line, "code", 4) == 0)
        { 
          line+= 4; /* move 4 spaces */
          /* check for = <WSPACE> " */
          WSPACE(line);
          TOKEN('=');
          WSPACE(line);
          TOKEN('"');
          RETRIEVECODE(pcode, line);
#ifdef DEBUG
          printf("Got code = %s\n", pcode);
#endif /* DEBUG */
	}
        else parseError("<code> token not found", line, lineno, line - original); 
#ifdef DEBUG 
	printf("Got entry routine = %s code =%s\n", pname, pcode);
#endif /* DEBUG */
        if (filespecified)
	{
          instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_ROUTINE_ENTRY)); 
	}
        else 
	{
	  bool codespecified = true; 
          instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_ROUTINE_ENTRY)); 
	} /* file and routine are both specified for entry */
      }
      else parseError("<routine> token not found", line, lineno, line - original);
    } /* end of entry token */
    else 
    { /* parse exit, and loops */
      if (strncmp(line, "exit", 4) == 0)
      {
        line+=4; 
        WSPACE(line);
        if (strncmp(line, "file", 4) == 0)
        {
     	  line+= 4;
	  WSPACE(line);
	  TOKEN('=');
	  WSPACE(line);
	  TOKEN('"');
	  RETRIEVESTRING(pfile, line);
	  WSPACE(line);
	  filespecified = true; 
#ifdef DEBUG
	  printf("GOT FILE = %s\n", pfile);
#endif /* DEBUG */
	}
        if (strncmp(line, "routine", 7) == 0)
        {
          line+=7; 
	  WSPACE(line);
	  TOKEN('=');
	  WSPACE(line);
	  TOKEN('"');
	  RETRIEVESTRING(pname, line);
	  WSPACE(line);
          if (strncmp(line, "code", 4) == 0)
          { 
            line+= 4; /* move 4 spaces */
            /* check for = <WSPACE> " */
            WSPACE(line);
            TOKEN('=');
            WSPACE(line);
            TOKEN('"');
            RETRIEVECODE(pcode, line);
#ifdef DEBUG
            printf("Got code = %s\n", pcode);
#endif /* DEBUG */
            if (filespecified)
	    {
              instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_ROUTINE_EXIT)); 
	    }
            else 
	    {
	      bool codespecified = true; 
              instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_ROUTINE_EXIT)); 
	    } /* file and routine are both specified for entry */
	  }
	  else parseError("<code> token not found", line, lineno, line - original);
	}
	else parseError("<routine> token not found", line, lineno, line - original);
#ifdef DEBUG
	printf("exit routine = %s code = %s\n", pname, pcode);
#endif /* DEBUG */
      } /* end of exit */
      else 
      { /* loops */
	m1 = strncmp(line, "loops", 5);
	m2 = strncmp(line, "io", 2);
	m3 = strncmp(line, "memory", 6);
        if ((m1 == 0) || (m2 == 0) || (m3 == 0)) {
	  if (m1 == 0) { 
	    kind = TAU_LOOPS; 
	    line += 5; /* move the pointer 5 spaces (loops) for next token */
	  }
	  else {  
            if (m2 == 0) {
	      kind = TAU_IO;
	      line += 2;/* move the pointer 2 spaces (io) for next token */
	    }
	    else {
	      if (m3 == 0) {
	        kind = TAU_MEMORY;
	        line += 6;/* move the pointer 6 spaces (memory) for next token */
	      }
	   }
	 }

	  /* check for WSPACE */
	  WSPACE(line);
          if (strncmp(line, "file", 4) == 0)
          {
     	    line+= 4;
	    WSPACE(line);
	    TOKEN('=');
	    WSPACE(line);
	    TOKEN('"');
	    RETRIEVESTRING(pfile, line);
	    WSPACE(line);
	    filespecified = true; 
#ifdef DEBUG
	    printf("GOT FILE = %s\n", pfile);
#endif /* DEBUG */
	  }
	  if (strncmp(line, "routine", 7) == 0)
	  {
	    line+=7;
	    /* found routine */ 
	    WSPACE(line);
	    TOKEN('=');
	    WSPACE(line);
	    TOKEN('"');
	    RETRIEVESTRING(pname, line);
#ifdef DEBUG
	    printf("got loops routine = %s\n", pname);
#endif /* DEBUG */
	    if (filespecified)
	    {
	      instrumentList.push_back(new tauInstrument(string(pfile), string(pname), kind));
	    }
	    else
	    {
	      instrumentList.push_back(new tauInstrument(string(pname), kind));
	    }
	  }
	  else parseError("<routine> token not found", line, lineno, line - original);
	}
      } /* end of loops directive */
    }
  }


}

///////////////////////////////////////////////////////////////////////////
// isInstrumentListEmpty() returns true if there are no entries in 
// instrumentList
///////////////////////////////////////////////////////////////////////////

bool isInstrumentListEmpty(void)
{
  return instrumentList.empty();
}

///////////////////////////////////////////////////////////////////////////
// printInstrumentList() lists all entries in instrumentList
///////////////////////////////////////////////////////////////////////////


void printInstrumentList(void)
{
  char orig[INBUF_SIZE];
  vector<tauInstrument *>::iterator it;
#ifdef DEBUG
  if (!isInstrumentListEmpty())
  { /* the list is not empty! */ 
    for (it = instrumentList.begin(); it != instrumentList.end(); it++) {
      (*it)->print(cout);
      /*
      if ((*it)->getCodeSpecified()) {
	cout <<(*it)->getCode();
  	strcpy(orig, "ppp\nkkk");
  	string mystr(orig);
  	cout<<mystr <<endl;
	cout <<orig <<endl;
      }
      */
    }
  }
#endif /* DEBUG */

}

/* using the list, write the additional statements */
void writeStatements(ofstream& ostr, const pdbRoutine *ro, 
         list< pair< int, list<string> > > & additional)
{

  /* search the list to locate the routine */
  list< pair<int, list<string> > >::iterator it ;

  for(it = additional.begin(); it != additional.end(); it++)
  {
    if ((*it).first == ro->id())
    { /* There is a match! Extract the pair */
#ifdef DEBUG 
      printf("There are additional declarations for routine %s\n", ro->fullName().c_str());
#endif /* DEBUG */
      list <string> l = (*it).second;
      list <string>::iterator lit;
      /* iterate over the list of strings */
      for (lit = l.begin(); lit != l.end(); lit++)
      {
#ifdef DEBUG
        cout <<"Adding: "<<(*lit)<<endl;
#endif /* DEBUG */
        ostr <<(*lit)<<endl;
      }
   
    }
  }
  return;  /* either it is not found, or all statements were written! */
}
/* After profiler and save statements are written, do we need to declare any other variables? */
void writeAdditionalFortranDeclarations(ofstream& ostr, const pdbRoutine *ro)
{
  if (isInstrumentListEmpty()) return; /* nothing specified */
#ifdef DEBUG 
  printf("writeAdditionalFortranDeclarations: Name %s, id=%d\n", ro->fullName().c_str(), ro->id());
#endif /* DEBUG */
  writeStatements(ostr, ro, additionalDeclarations);
}

/* After profiler and save statements are written, do we need 
   to call any other timer routines? */
void writeAdditionalFortranInvocations(ofstream& ostr, const pdbRoutine *ro)
{
  if (isInstrumentListEmpty()) return; /* nothing specified */
#ifdef DEBUG 
  printf("writeAdditionalFortranInvocations: Name %s, id=%d\n", ro->fullName().c_str(), ro->id());
#endif /* DEBUG */
  writeStatements(ostr, ro, additionalInvocations);
}

/* construct the timer name e.g., t_24; */
void getLoopTimerVariableName(string& varname, int line)
{
  char var[256];
  sprintf(var, "%d", line);
  varname = string("t_")+var;
  return;
}

/* Add request for instrumentation for Fortran loops */
void addFortranLoopInstrumentation(const pdbRoutine *ro, const pdbLoc& start, const pdbLoc& stop, vector<itemRef *>& itemvec)
{
  const pdbFile *f = start.file();
  char lines[256];

  /* first we construct a string with the line numbers */
  sprintf(lines, "{%d,%d}-{%d,%d}", 
	start.line(), start.col(), stop.line(), stop.col());
  /* we use the line numbers in building the name of the timer */

  const char *filename = f->name().c_str();
  while (strchr(filename,TAU_DIR_CHARACTER)) { // remove path
    filename = strchr(filename,TAU_DIR_CHARACTER)+1;
  }

  string timername (string("Loop: " + string(ro->fullName()) 
			   + " [{" + string(filename) + "} " + lines + "]"));

  /* we embed the line from_to in the name of the timer. e.g., t */
  string varname;
  getLoopTimerVariableName(varname, start.line());

  string declaration1(string("      integer ")+varname+"(2) / 0, 0 /");
  string declaration2(string("      save ")+varname);

  /* now we create the call to create the timer */
  string createtimer(string("      call TAU_PROFILE_TIMER(")+varname+", '"+timername+"')");

  if (createtimer.length() > 72) {
    /* We will always start the quote on the first line, then skip to the next
       inserting an & at column 73, then at column 6.  TAU_PROFILE_TIMER will
       clean up any mess made by -qfixed=132, etc */
    string s1 = string("      call TAU_PROFILE_TIMER(")+varname+", '";
    string s2 = "";
    int length = s1.length();
    for (int i=length; i < 72; i++) {
      s2 = s2 + " ";
    }

    createtimer = s1 + s2 + "&\n";

    // continue to break lines in the correct spot
    while (timername.length() > 64) {
      string first = timername.substr(0,64);
      timername.erase(0,64);
      createtimer = createtimer + "     &"+first+"&\n";
    }

    createtimer = createtimer + "     &"+timername+"')";
  }
  


#ifdef DEBUG
  printf("Adding instrumentation at %s, var: %s\n", timername.c_str(), varname.c_str());
  printf("%s\n", declaration1.c_str());
  printf("%s\n", declaration2.c_str());
  printf("%s\n", createtimer.c_str());
  printf("Routine id = %d, name = %s\n", ro->id(), ro->fullName().c_str());
#endif /* DEBUG */
  list<string> decls;
  decls.push_back(declaration1);
  decls.push_back(declaration2);

  additionalDeclarations.push_back(pair<int, list<string> >(ro->id(), decls)); /* assign the list of strings to the list */

  list<string> calls;
  calls.push_back(createtimer);
  /* now we create the list that has additional TAU calls for creating the timer */

  additionalInvocations.push_back(pair<int, list<string> >(ro->id(), calls)); /* assign the list of strings to the list */

  /* We have removed the check to see if the function should be instrumented here. It is now done in ProcessBlock */
  string startsnippet(string("      call TAU_PROFILE_START(")+varname+")");
  string stopsnippet (string("      call TAU_PROFILE_STOP(") +varname+")");
  itemvec.push_back( new itemRef((const pdbItem *)ro, START_DO_TIMER, start.line(), start.col(), varname, BEFORE));
  itemvec.push_back( new itemRef((const pdbItem *)ro, STOP_TIMER, stop.line(), stop.col()+1, varname, AFTER));
#ifdef DEBUG
  printf("instrumenting routine %s\n", ro->fullName().c_str());
#endif /* DEBUG */

#ifdef DEBUG 
  printf("routine: %s, line,col = <%d,%d> to <%d,%d>\n", 
	ro->fullName().c_str(), start.line(), start.col(), stop.line(), stop.col());
#endif /* DEBUG */
}

/* BUG: When do statement occurs with a label as in:
 * 30   do i=1,4
 *      ...
 *      end do
 *      we can't instrument it correctly. TAU inserts:
 *      call TAU_PROFILE_START(t_no)
 *      30    do i = 1, 4
 *      instead of 
 *      30   call TAU_PROFILE_START
 *           do i = 1, 4 
 *           end do
 *           */
 /* fixed: we use CPDB... (inbuf, "do") to find the correct DO column no. */

/* Does the label leave the statement boundary? */
int labelOutsideStatementBoundary(const pdbStmt *labelstmt, const pdbStmt *parentDO)
{
  int defaultreturn = 0; /* does not leave boundary */
  if (labelstmt == NULL) return defaultreturn; 
  if (parentDO == NULL) return defaultreturn; /* does not leave boundary */

  int labelline = labelstmt->stmtBegin().line();
  int labelcol  = labelstmt->stmtBegin().col();
  int parentbeginline = parentDO->stmtBegin().line();
  int parentbegincol = parentDO->stmtBegin().col();
  int parentendline = parentDO->stmtEnd().line();
  int parentendcol = parentDO->stmtEnd().col();
#ifdef DEBUG
  printf("OustideStmtBoundary: label line no. = %d, col = %d\n",
		  labelline, labelcol);
  printf("OutsideStmtBoundary: parentDO start = <%d,%d>, stop = <%d,%d>\n", 
    parentbeginline, parentbegincol, parentendline, parentendcol);
#endif /* DEBUG */
  /* is the label within the block defined by the parent do? */
  /* first examine the line numbers */
  if ((labelline > parentbeginline) && (labelline < parentendline))
  { /* sure! the label is indeed between these two boundaries */
    return 0; /* it does not leave the boundary */
  }
  else 
  {
    if (labelline < parentbeginline) return 1; /* yes, it is above */
    if ((labelline == parentbeginline) && (labelcol < parentbegincol)) return 1;
    if (labelline > parentendline) return 1; /* yes, it is outside */
    if ((labelline == parentendline) && (labelcol > parentendcol)) return 1; 
    /* yes it is outside */
  }
  return defaultreturn; /* by default, label is not outside parent? */
}

/* Add request for instrumentation for C/C++ loops */
void addRequestForLoopInstrumentation(const pdbRoutine *ro, const pdbLoc& start, const pdbLoc& stop, vector<itemRef *>& itemvec)
{
  const pdbFile *f = start.file();
  char lines[256];
  sprintf(lines, "{%d,%d}-{%d,%d}", 
	start.line(), start.col(), stop.line(), stop.col());

  const char *filename = f->name().c_str();
  while (strchr(filename,TAU_DIR_CHARACTER)) { // remove path
    filename = strchr(filename,TAU_DIR_CHARACTER)+1;
  }

  string *timername = new string(string("Loop: " + ro->fullName() 
					+ " [{"+string(filename)+ "} "+ lines + "]"));
#ifdef DEBUG
  printf("Adding instrumentation at %s\n", timername->c_str());
#endif /* DEBUG */
  /*string startsnippet(string("{ TAU_PROFILE_TIMER(lt, \"")+(*timername)+"\", \" \", TAU_USER); TAU_PROFILE_START(lt); ");
  string stopsnippet(string("TAU_PROFILE_STOP(lt); } "));
  itemvec.push_back( new itemRef((const pdbItem *)ro, INSTRUMENTATION_POINT, start.line(), start.col(), startsnippet, BEFORE));
  itemvec.push_back( new itemRef((const pdbItem *)ro, INSTRUMENTATION_POINT, stop.line(), stop.col()+1, stopsnippet, AFTER));
*/
  itemvec.push_back( new itemRef((const pdbItem *)ro, START_LOOP_TIMER, start.line(), start.col(), *timername, BEFORE));
  itemvec.push_back( new itemRef((const pdbItem *)ro, STOP_LOOP_TIMER, stop.line(), stop.col()+1, *timername, AFTER));
/*
  printf("Adding instrumentation at routine %s, file %s, start %d:%d, stop %d:%d\n",
    ro->fullName().c_str(), f->name(), start.line(), start.col(), stop.line(), stop.col());
*/
}

/* Process Block to examine the routine */
int processMemBlock(const pdbStmt *s, const pdbRoutine *ro, vector<itemRef *>& itemvec,
		int level, const pdbStmt *parentDO)
{
  pdbLoc start, stop; /* the location of start and stop timer statements */
#ifndef PDT_NOFSTMTS
/* statements are there */
  
#ifdef DEBUG
  printf("Inside processMemBlock()\n");
#endif /* DEBUG */
  if (!s) return 1;
  if (!ro) return 1; /* if null, do not instrument */

  if (!instrumentEntity(ro->fullName())) return 1;  /* we shouldn't instrument it? */
  
  pdbStmt::stmt_t k = s->kind();

#ifdef DEBUG
    if (parentDO)
    printf("Examining statement parentDo line=%d\n", parentDO->stmtEnd().line());
#endif /* DEBUG */
    /* NOTE: We currently do not support goto in C/C++ to close the timer.
     * This needs to be added at some point -- similar to Fortran */
    switch(k) {
#ifdef PDB_FORTRAN_EXTENDED_STATEMENTS_LEVEL_2
      case pdbStmt::ST_FEXIT:
      case pdbStmt::ST_FCYCLE:
      case pdbStmt::ST_FWHERE:
        if (s->downStmt())
          processMemBlock(s->downStmt(), ro, itemvec, level, parentDO);
        break; /* don't go into extraStmt for Fortran EXIT to avoid looping */
#endif /* PDB_FORTRAN_EXTENDED_STATEMENTS_LEVEL_1 */

      case pdbStmt::ST_FGOTO:
	break;
      case pdbStmt::ST_FALLOCATE:
#ifdef DEBUG
	printf("FALLOCATE statement line = %d\n", s->stmtBegin().line());
#endif /* DEBUG */
        itemvec.push_back( new itemRef((pdbItem *)NULL, ALLOCATE_STMT, s->stmtBegin().line(), s->stmtBegin().col(), s->stmtBegin().file()->name(), AFTER));

	break;
      case pdbStmt::ST_FDEALLOCATE:
#ifdef DEBUG
	printf("FDEALLOCAATE statement line = %d\n", s->stmtBegin().line());
#endif /* DEBUG */
        itemvec.push_back( new itemRef((pdbItem *)NULL, DEALLOCATE_STMT, s->stmtBegin().line(), s->stmtBegin().col(), s->stmtBegin().file()->name(), BEFORE));
	break;
      default:
        if (s->downStmt())
          processMemBlock(s->downStmt(), ro, itemvec, level, parentDO);
        if (s->extraStmt())
          processMemBlock(s->extraStmt(), ro, itemvec, level, parentDO);
	/* We only process down and extra statements for the default statements
	   that are not loops. When a loop is encountered, its down is not
	   processed. That way we retain outer loop level instrumentation */
#ifdef DEBUG
        printf("Other statement\n"); 
#endif /* DEBUG */
	break;
    }
  /* and then process the next statement */
  if (s->nextStmt())
    return processMemBlock(s->nextStmt(), ro, itemvec, level, parentDO);
  else
    return 1;

#else /* PDT_FNOSTMTS */
   return 0;
#endif /* PDT_NOFSTMTS */

}

/* construct the timer name e.g., tio_22; */
void getTauEntityName(char *prefix, string& varname, int line)
{ /* pass in tio as the prefix */
  char var[256];
  sprintf(var, "%d", line);
  varname = string(prefix)+var;
  return;
}

/* Add request for instrumentation for Fortran loops */
void addFortranIOInstrumentation(const pdbRoutine *ro, const pdbLoc& start, const pdbLoc &stop, vector<itemRef *>& itemvec)
{
  char lines[256];

  /* first we construct a string with the line numbers */
  sprintf(lines, "%d", start.line());

  /* we use the line numbers in building the name of the timer */
  const pdbFile *f = start.file();
  const char *filename = f->name().c_str();
  while (strchr(filename,TAU_DIR_CHARACTER)) { // remove path
    filename = strchr(filename,TAU_DIR_CHARACTER)+1;
  }

  string timername (string("IO <file=") + string(filename) + ", line=" + lines+">");

  /* we embed the line from_to in the name of the timer. e.g., t */
  string varname;
  getTauEntityName("tio_", varname, start.line());

  string declaration1(string("      real*8 ")+varname+"_sz");
  string declaration2(string("      integer ")+varname+"(2) / 0, 0 /");
  string declaration3(string("      save ")+varname);

  /* now we create the call to create the timer */
  string createtimer(string("      call TAU_REGISTER_CONTEXT_EVENT(")+varname+", '"+timername+"')");

  if (createtimer.length() > 72) {
    /* We will always start the quote on the first line, then skip to the next
       inserting an & at column 73, then at column 6.  TAU_PROFILE_TIMER will
       clean up any mess made by -qfixed=132, etc */
    string s1 = string("      call TAU_REGISTER_CONTEXT_EVENT(")+varname+", '";
    string s2 = "";
    int length = s1.length();
    for (int i=length; i < 72; i++) {
      s2 = s2 + " ";
    }

    createtimer = s1 + s2 + "&\n";

    // continue to break lines in the correct spot
    while (timername.length() > 64) {
      string first = timername.substr(0,64);
      timername.erase(0,64);
      createtimer = createtimer + "     &"+first+"&\n";
    }

    createtimer = createtimer + "     &"+timername+"')";
  }
  


#ifdef DEBUG
  printf("Adding instrumentation at %s, var: %s\n", timername.c_str(), varname.c_str());
  printf("%s\n", declaration1.c_str());
  printf("%s\n", declaration2.c_str());
  printf("%s\n", declaration3.c_str());
  printf("%s\n", createtimer.c_str());
  printf("Routine id = %d, name = %s\n", ro->id(), ro->fullName().c_str());
#endif /* DEBUG */
  list<string> decls;
  decls.push_back(declaration1);
  decls.push_back(declaration2);
  decls.push_back(declaration3);

  additionalDeclarations.push_back(pair<int, list<string> >(ro->id(), decls)); /* assign the list of strings to the list */

  list<string> calls;
  calls.push_back(createtimer);
  /* now we create the list that has additional TAU calls for creating the timer */

  additionalInvocations.push_back(pair<int, list<string> >(ro->id(), calls)); /* assign the list of strings to the list */

  itemvec.push_back( new itemRef((pdbItem *)NULL, IO_STMT, start, stop));
#ifdef DEBUG
  printf("instrumenting IO for routine %s\n", ro->fullName().c_str());
  /* we are losing the stop location for this statement -- needed to determine 
     continuation of line! */
#endif /* DEBUG */

#ifdef DEBUG 
  printf("IO in routine: %s, at line,col = <%d,%d> \n", 
	ro->fullName().c_str(), start.line(), start.col());
#endif /* DEBUG */
}

/* Process Block to examine the routine */
int processIOBlock(const pdbStmt *s, const pdbRoutine *ro, vector<itemRef *>& itemvec,
		int level, const pdbStmt *parentDO)
{
  pdbLoc start, stop; /* the location of start and stop timer statements */
#ifndef PDT_NOFSTMTS
/* statements are there */
  
#ifdef DEBUG
  printf("Inside processIOBlock()\n");
#endif /* DEBUG */
  if (!s) return 1;
  if (!ro) return 1; /* if null, do not instrument */

  if (!instrumentEntity(ro->fullName())) return 1;  /* we shouldn't instrument it? */
  
  pdbStmt::stmt_t k = s->kind();

#ifdef DEBUG
    if (parentDO)
    printf("Examining statement parentDo line=%d\n", parentDO->stmtEnd().line());
#endif /* DEBUG */
    /* NOTE: We currently do not support goto in C/C++ to close the timer.
     * This needs to be added at some point -- similar to Fortran */
    switch(k) {
#ifdef PDB_FORTRAN_EXTENDED_STATEMENTS_LEVEL_2
      case pdbStmt::ST_FEXIT:
      case pdbStmt::ST_FCYCLE:
      case pdbStmt::ST_FWHERE:
        if (s->downStmt())
          processMemBlock(s->downStmt(), ro, itemvec, level, parentDO);
        break; /* don't go into extraStmt for Fortran EXIT to avoid looping */
#endif /* PDB_FORTRAN_EXTENDED_STATEMENTS_LEVEL_1 */

      case pdbStmt::ST_FGOTO:
      case pdbStmt::ST_FALLOCATE:
      case pdbStmt::ST_FDEALLOCATE:
	break;
      case pdbStmt::ST_FIO:
#ifdef DEBUG
	printf("IO statement: %s <%d,%d>\n",
	s->stmtBegin().file()->name(), s->stmtBegin().line(), s->stmtBegin().col(), s->stmtBegin().file()->name());
#endif /* DEBUG */
	/* write the IO tracking statement at this location */
        addFortranIOInstrumentation(ro, s->stmtBegin(), s->stmtEnd(), itemvec);
	break;
      default:
        if (s->downStmt())
          processIOBlock(s->downStmt(), ro, itemvec, level, parentDO);
        if (s->extraStmt())
          processIOBlock(s->extraStmt(), ro, itemvec, level, parentDO);
	/* We only process down and extra statements for the default statements
	   that are not loops. When a loop is encountered, its down is not
	   processed. That way we retain outer loop level instrumentation */
#ifdef DEBUG
        printf("Other statement\n"); 
#endif /* DEBUG */
	break;
    }
  /* and then process the next statement */
  if (s->nextStmt())
    return processIOBlock(s->nextStmt(), ro, itemvec, level, parentDO);
  else
    return 1;

#else /* PDT_FNOSTMTS */
   return 0;
#endif /* PDT_NOFSTMTS */

}
/* Process Block to examine the routine */
int processBlock(const pdbStmt *s, const pdbRoutine *ro, vector<itemRef *>& itemvec,
		int level, const pdbStmt *parentDO)
{
  pdbLoc start, stop; /* the location of start and stop timer statements */
  
  if (!s) return 1;
  if (!ro) return 1; /* if null, do not instrument */

  if (!instrumentEntity(ro->fullName())) return 1;  /* we shouldn't instrument it? */
  
  pdbStmt::stmt_t k = s->kind();

#ifdef DEBUG
    if (parentDO)
    printf("Examining statement parentDo line=%d\n", parentDO->stmtEnd().line());
#endif /* DEBUG */
    /* NOTE: We currently do not support goto in C/C++ to close the timer.
     * This needs to be added at some point -- similar to Fortran */
    switch(k) {
      case pdbStmt::ST_FOR:
      case pdbStmt::ST_WHILE:
      case pdbStmt::ST_DO:
#ifndef PDT_NOFSTMTS 
/* PDT has Fortran statement level information. Use it! */
      case pdbStmt::ST_FDO:
#endif /* PDT_NOFSTMTS */
#ifdef DEBUG
        printf("loop statement:\n");
#endif /* DEBUG */
        start = s->stmtBegin();
        stop = s->stmtEnd();
#ifdef DEBUG
        printf("start=<%d:%d> - end=<%d:%d>\n",
          start.line(), start.col(), stop.line(), stop.col());
#endif /* DEBUG */
	if (level == 1)
	{
          /* C++/C or Fortran instrumentation? */
#ifndef PDT_NOFSTMTS
	  if (k == pdbStmt::ST_FDO)
            addFortranLoopInstrumentation(ro, start, stop, itemvec);
	  else 
#endif /* PDT_NOFSTMTS */
	    addRequestForLoopInstrumentation(ro, start, stop, itemvec); 
	}
	if (s->downStmt())
	{
	  if (level == 1)
  	    processBlock(s->downStmt(), ro, itemvec, level+1, s);
 	  else 
  	    processBlock(s->downStmt(), ro, itemvec, level+1, parentDO);
	/* NOTE: We are passing s as the parentDO argument for subsequent 
	 * processing of the DO loop. We also increment the level by 1 */
        }
        break;
      case pdbStmt::ST_GOTO:

#ifndef PDT_NOFSTMTS

#ifdef PDB_FORTRAN_EXTENDED_STATEMENTS_LEVEL_2
      case pdbStmt::ST_FEXIT:
      case pdbStmt::ST_FCYCLE:
      case pdbStmt::ST_FWHERE:
        if (s->downStmt())
          processBlock(s->downStmt(), ro, itemvec, level, parentDO);
        break; /* don't go into extraStmt for Fortran EXIT to avoid looping */
#endif /* PDB_FORTRAN_EXTENDED_STATEMENTS_LEVEL_1 */
//       case pdbStmt::ST_FCYCLE:
//       case pdbStmt::ST_FEXIT:
      case pdbStmt::ST_FGOTO:
#endif /* PDT_NOFSTMTS */
	if (s->extraStmt())
	{
#ifdef DEBUG
          printf("GOTO statement! label line no= %d \n", s->extraStmt()->stmtBegin().line());
#endif /* DEBUG */
	  /* if the GOTO leaves the boundary of the calling do loop, decrement
	   * the level by one. Else keep it as it is */
	  if (parentDO && labelOutsideStatementBoundary(s->extraStmt(), parentDO))
	  { /* if the goto is within the boundary of the parent and the label is outside the boundary, then stop the timer */
	    /* Uh oh! The go to is leaving the current do loop. We need to 
	     * stop the timer */
	    string timertoclose;
	    getLoopTimerVariableName(timertoclose, parentDO->stmtBegin().line());
	    //string stopsnippet (string("        call TAU_PROFILE_STOP(") +timertoclose+")");
	    itemvec.push_back( new itemRef((const pdbItem *)ro, GOTO_STOP_TIMER, s->stmtBegin().line(), s->stmtBegin().col(), timertoclose, BEFORE));
	    /* stop the timer right before writing the go statement */

#ifdef DEBUG
	    printf("LABEL IS OUTSIDE PARENT DO BOUNDARY! level - 1 \n");
	    printf("close timer: %s\n", timertoclose.c_str());
#endif /* DEBUG  */
	    
	  }
	}
	break;


      default:
        if (s->downStmt())
          processBlock(s->downStmt(), ro, itemvec, level, parentDO);
        if (s->extraStmt())
          processBlock(s->extraStmt(), ro, itemvec, level, parentDO);
	/* We only process down and extra statements for the default statements
	   that are not loops. When a loop is encountered, its down is not
	   processed. That way we retain outer loop level instrumentation */
#ifdef DEBUG
        printf("Other statement\n"); 
#endif /* DEBUG */
	break;
    }
  /* and then process the next statement */
  if (s->nextStmt())
    return processBlock(s->nextStmt(), ro, itemvec, level, parentDO);
  else
    return 1;

}

/* Process list of C routines */
bool processCRoutinesInstrumentation(PDB & p, vector<tauInstrument *>::iterator& it, vector<itemRef *>& itemvec, pdbFile *file) 
{
  /* compare the names of routines with our instrumentation request routine name */

  bool retval = true; 
  PDB::croutinevec::const_iterator rit;
  PDB::croutinevec croutines = p.getCRoutineVec();
  bool cmpResult1, cmpResult2, cmpFileResult; 

#ifdef DEBUG
  printf("Inside processCRoutinesInstrumentation!\n");
#endif /* DEBUG */
  pdbRoutine::locvec::iterator rlit;
  for(rit = croutines.begin(); rit != croutines.end(); ++rit)
  { /* iterate over all routines */
    /* the first argument contains wildcard, the second is the string */
    cmpResult1 = wildcardCompare((char *)((*it)->getRoutineName()).c_str(),
	(char *)(*rit)->name().c_str(), '#');
    /* the first argument contains wildcard, the second is the string */
    cmpResult2 = wildcardCompare((char *)((*it)->getRoutineName()).c_str(),
	(char *)(*rit)->fullName().c_str(), '#');
    if (cmpResult1 || cmpResult2)
    { /* there is a match */
      /* is this routine in the same file that we are instrumenting? */
      if ((*rit) && (*rit)->location().file() && file)
      {
        cmpFileResult = fuzzyMatch((*rit)->location().file()->name(), file->name()); 
      }
      else cmpFileResult = false;
      if (!cmpFileResult)
      { 
#ifdef DEBUG
        cout <<"File names do not match... continuing ..."<<endl;
#endif /* DEBUG */
        continue;
      }
      else
      {
#ifdef DEBUG
        cout <<"File names "<<(*rit)->location().file()->name()<<" and "
	     << file->name()<<" match!"<<endl;
#endif /* DEBUG */

      }

#ifdef DEBUG
      cout <<"Examining Routine "<<(*rit)->fullName()<<" and "<<(*it)->getRoutineName()<<endl;
#endif /* DEBUG */
      /* examine the type of request - entry/exit */
      if ((*it)->getKind() == TAU_ROUTINE_ENTRY)
      {
#ifdef DEBUG
        cout <<"Instrumenting entry of routine "<<(*rit)->fullName()<<endl;
        /* get routine entry line no. */
        cout <<"at line: "<<(*rit)->bodyBegin().line()<<", col"<< (*rit)->bodyBegin().col()<<"code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */
	itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col()+1, (*it)->getCode(), BEFORE));
	/* should the column be 1 greater than body begin's col, so 
	   instrumentation is placed after the beginning of the routine? */

      }
      if ((*it)->getKind() == TAU_ROUTINE_EXIT)
      {
#ifdef DEBUG
        cout <<"Instrumenting exit of routine "<<(*rit)->fullName()<<endl;
#endif /* DEBUG */
        /* get routine entry line no. */
        pdbRoutine::locvec retlocations = (*rit)->returnLocations();
        for (rlit = retlocations.begin(); rlit != retlocations.end(); ++rlit)
        {
#ifdef DEBUG
          cout <<"at line: "<<(*rlit)->line()<<", col"<< (*rlit)->col()<<" code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */
	
	  itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*rlit)->line(), (*rlit)->col(), (*it)->getCode(), BEFORE));
        }
      } /* end of routine exit */
      if ((*it)->getKind() == TAU_LOOPS)
      { /* we need to instrument all outer loops in this routine */
	processBlock((*rit)->body(), (*rit), itemvec, 1, NULL);
	/* level = 1 */
      }
      if ((*it)->getKind() == TAU_IO)
      { /* we need to instrument all io statements in this routine */
	printf("process I/O statements in C routine\n");
      }
      if ((*it)->getKind() == TAU_MEMORY)
      { /* we need to instrument all memory statements in this routine */
#ifdef DEBUG
	printf("process memory allocate/de-allocate statements in C routine\n");
#endif /* DEBUG */
      }
    }
  }

  return retval;
}
/* Process list of F routines */
bool processFRoutinesInstrumentation(PDB & p, vector<tauInstrument *>::iterator& it, vector<itemRef *>& itemvec, pdbFile *file) 
{
#ifdef DEBUG
  printf("INSIDE processFRoutinesInstrumentation\n");
#endif /* DEBUG */

  PDB::froutinevec::const_iterator rit;
  PDB::froutinevec froutines = p.getFRoutineVec();
  bool cmpResult, cmpFileResult; 
  pdbRoutine::locvec::iterator rlit;
  for(rit = froutines.begin(); rit != froutines.end(); ++rit)
  { /* iterate over all routines */  
    /* the first argument contains the wildcard, the second is the string */
    cmpResult = wildcardCompare((char *)((*it)->getRoutineName()).c_str(), (char *)(*rit)->name().c_str(), '#');
    if (cmpResult)
    { /* there is a match */
      /* is this routine in the same file that we are instrumenting? */
      if ((*rit) && (*rit)->location().file() && file)
      {
        cmpFileResult = fuzzyMatch((*rit)->location().file()->name(), file->name()); 
      }
      else cmpFileResult = false;
      if (!cmpFileResult)
      { 
#ifdef DEBUG
        cout <<"File names do not match... continuing ..."<<endl;
#endif /* DEBUG */
        continue;
      }
      else
      {
#ifdef DEBUG
        cout <<"File names "<<(*rit)->location().file()->name()<<" and "
	     << file->name()<<" match!"<<endl;
#endif /* DEBUG */

      }

#ifdef DEBUG
      cout <<"Examining Routine "<<(*rit)->fullName()<<" and "<<(*it)->getRoutineName()<<endl;
#endif /* DEBUG */
      /* examine the type of request - entry/exit */
      if ((*it)->getKind() == TAU_ROUTINE_ENTRY)
      {
#ifdef DEBUG
        cout <<"Instrumenting entry of routine "<<(*rit)->fullName()<<endl;
#endif /* DEBUG */
        /* We should treat snippet insertion code at routine entry like additional declarations
           that must be placed inside the routine */
        list<string> decls;
        decls.push_back((*it)->getCode());

        additionalDeclarations.push_back(pair<int, list<string> >((*rit)->id(), decls)); 
	/* assign the list of strings to the list */
      }
      if ((*it)->getKind() == TAU_ROUTINE_EXIT)
      {
#ifdef DEBUG
        cout <<"Instrumenting exit of routine "<<(*rit)->fullName()<<endl;
#endif /* DEBUG */
        /* get routine entry line no. */
        pdbRoutine::locvec retlocations = (*rit)->returnLocations();
        pdbRoutine::locvec stoplocations = (*rit)->stopLocations();
	
	/* we first start with the return locations */
        for (rlit = retlocations.begin(); rlit != retlocations.end(); ++rlit)
        {
#ifdef DEBUG
          cout <<"at line: "<<(*rlit)->line()<<", col"<< (*rlit)->col()<<" code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */
	
	  itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*rlit)->line(), (*rlit)->col(), (*it)->getCode(), BEFORE));
        }
	/* and then examine the stop locations */
        for (rlit = stoplocations.begin(); rlit != stoplocations.end(); ++rlit)
        {
#ifdef DEBUG
          cout <<"at line: "<<(*rlit)->line()<<", col"<< (*rlit)->col()<<" code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */
	
	  itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*rlit)->line(), (*rlit)->col(), (*it)->getCode(), BEFORE));
        }
      } /* end of routine exit */
      if ((*it)->getKind() == TAU_LOOPS)
      { /* we need to instrument all outer loops in this routine */
	processBlock((*rit)->body(), (*rit), itemvec, 1, NULL); /* level = 1 */
      }
      if ((*it)->getKind() == TAU_IO)
      { /* we need to instrument all io statements in this routine */
#ifdef DEBUG
	printf("process I/O statements in Fortran routine\n");
#endif /* DEBUG */
	processIOBlock((*rit)->body(), (*rit), itemvec, 1, NULL); /* level = 1 */
      }
      if ((*it)->getKind() == TAU_MEMORY )
      { /* we need to instrument all memory statements in this routine */
#ifdef DEBUG
	printf("processing memory allocate/de-allocate statements in Fortran routine...\n");
#endif /* DEBUG */
	processMemBlock((*rit)->body(), (*rit), itemvec, 1, NULL); /* level = 1 */
      }
    } /* end of match */
  } /* iterate over all routines */

  return true; /* everything is ok -- return true */
}
bool addFileInstrumentationRequests(PDB& p, pdbFile *file, vector<itemRef *>& itemvec)
{
  /* Let us iterate over the list of instrumentation requests and see if 
   * any requests match this file */
  vector<tauInstrument *>::iterator it;
  bool cmpResult; 
  int column ;
  bool retval = true;
  PDB::lang_t lang;
  PDB::croutinevec croutines;
  PDB::froutinevec froutines; 

#ifdef DEBUG
  printf("INSIDE addFileInstrumentationRequests empty Instrumentation List? %d \n", isInstrumentListEmpty());
#endif /* DEBUG */

  if (memory_flag && p.language() == PDB::LA_FORTRAN) 
  { /* instrumentation list is empty, but the user specified a -memory flag 
       for a fortran file. This is equivalent to instrumenting all memory 
       references */
#ifdef DEBUG
    printf("Instrumenting memory references for Fortran when selective instrumentation file was not specified. Using memory file=\"*\" routine = \"#\"\n");
#endif /* DEBUG */
    instrumentList.push_back(new tauInstrument(string("*"), string("#"), TAU_MEMORY));

  }
  for (it = instrumentList.begin(); it != instrumentList.end(); it++) 
  {
    if ((*it)->getFileSpecified())
    { /* a file is specified, does its name match? */
#ifdef DEBUG
      cout <<"Checking "<<file->name().c_str()<<" and "<<(*it)->getFileName().c_str()<<endl; 
#endif /* DEBUG */
      /* the first argument contains the wildcard, the second is the string */
      cmpResult = wildcardCompare((char *)(*it)->getFileName().c_str(), (char *)file->name().c_str(), '*');
    } /* is file specified?*/
    else 
    {
      cmpResult = false; 
      /* this file is either not specified or it doesn't match */
    }

    if (cmpResult)
    { /* check if the current file is to be instrumented */
#ifdef DEBUG
      cout <<"Matched the file names!"<<endl;
#endif /* DEBUG */
      /* Now we must add the lines for instrumentation if a line is specified! */
      if ((*it)->getLineSpecified())
      { /* Yes, a line number was specified */
#ifdef DEBUG
	cout << "Need to add line no. " <<(*it)->getLineNo()<<"column 1 to the list!"<<endl;
#endif /* DEBUG */
	/* We need to create a pdbLoc, pdbItem and add the pdbItem to the 
	 * itemvec. While creating the pdbLoc, we use the line no. of the 
	 * associated instrumentation point. We use the column no. of 1
	 * and 0 as the id of the pdbItem since it is not known. */
	
	/*
	pdbSimpleItem *item = new pdbSimpleItem((*it)->getFileName(), 0);
	item->location((const pdbLoc&) (*it)); 
	*/
	/* itemRef::itemRef(const pdbItem *i, itemKind_t k, int l, int c, string code)
	 */
	itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*it)->getLineNo(), 1, (*it)->getCode(), BEFORE));
			
      }
    }
    /* What else is specified with the instrumentation request? Are routines
       specified? Match routines with the instrumentation requests */
    /* Create a list of routines */
    if ((*it)->getRoutineSpecified())
    {
	/* is a file specified as well? If it is specified, does it match 
	   the file name properly? cmpResult answers that*/
	if ((*it)->getFileSpecified() && !cmpResult)
	{ /* a file was specified but it does not match! we do not need to 
	     instrument this routine */
#ifdef DEBUG
	  cout <<"File was specified and its name didn't match... not examining routines here"<<endl;
#endif /* DEBUG */
	  continue; /* we are done! (with this request) */
	} /* either a file was not specified or if it was, it matched and 
	     cmpResult was true, so we carry on... */

#ifdef DEBUG
      cout <<"A routine is specified! "<<endl;
#endif /* DEBUG */
      lang = p.language();
      switch (lang) { /* check the language first */
        case PDB::LA_C :
        case PDB::LA_CXX:
        case PDB::LA_C_or_CXX:
#ifdef DEBUG
	  cout <<"C routine!"<<endl; 
#endif /* DEBUG */
          retval = processCRoutinesInstrumentation(p, it, itemvec, file); 
	  /* Add file name to the routine instrumentation! */
          break; 
        case PDB::LA_FORTRAN:
#ifdef DEBUG
	  cout <<"F routine!"<<endl; 
#endif /* DEBUG */
          retval = processFRoutinesInstrumentation(p, it, itemvec, file); 
          break;
        default:
          break;
       }
    }

  }
  return retval;
}

bool addMoreInvocations(int routine_id, string& snippet)
{
  list<string> code; 
  code.push_back(snippet);
  additionalInvocations.push_back(pair<int, list<string> >(routine_id, code)); 

#ifdef DEBUG
  cout <<"Adding invocations routine id = "<<routine_id<<" snippet  = "<<snippet<<endl;
#endif /* DEBUG */
  /* assign the list of strings to the list */
  return true;
}

   

///////////////////////////////////////////////////////////////////////////
// Generate itemvec entries for instrumentation commands 
///////////////////////////////////////////////////////////////////////////


/* EOF */


/***************************************************************************
 * $RCSfile: tau_instrument.cpp,v $   $Author: sameer $
 * $Revision: 1.44 $   $Date: 2007/05/21 20:59:02 $
 * VERSION_ID: $Id: tau_instrument.cpp,v 1.44 2007/05/21 20:59:02 sameer Exp $
 ***************************************************************************/
