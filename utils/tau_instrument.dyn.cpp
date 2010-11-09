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
#include <sstream>
#include <iostream>
using namespace std;
#include "tau_instrument.dyn.h"
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
#include <string.h>
#ifndef TAU_DYNINST
#include "pdbAll.h"
#endif /* TAU_DYNINST */



extern bool wildcardCompare(char *wild, char *string, char kleenestar);
extern bool instrumentEntity(const string& function_name);
extern bool fuzzyMatch(const string& a, const string& b);
extern bool memory_flag;


void replaceAll(string& str, const string& search, const string& replace);
string intToString(int value);


/* Globals */
///////////////////////////////////////////////////////////////////////////
vector<tauInstrument *> instrumentList; 
list<pair<int, list<string> > > additionalDeclarations; 
list<pair<int, list<string> > > additionalInvocations; 
/* In this list resides a list of variable declarations that must be added before
   the first executable statement. It has a list of strings with the routine no. as 
   the first element of the pair (the second is the list of strings). */

/* These should *really* be in tau_instrumentor.cpp ... */
bool noinline_flag = false; /* instrument inlined functions by default */
bool use_spec = false;   /* by default, do not use code from specification file */

///////////////////////////////////////////////////////////////////////////

/* Constructors */
///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor which sets all the items
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, string r, int line, string c, 
			     instrumentKind_t k): filename(f), fileSpecified(true), routineName(r),
						  lineno(line), lineSpecified (true), code(c), codeSpecified(true), kind (k),
						  regionSpecified(false), qualifierSpecified(false)
{}


///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// file = "foo.cpp" line=245 code = "TAU_NODE(0);" lang = "c++"
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, int line, string c, instrumentKind_t k) : 
  filename(f), fileSpecified(true), lineno(line), lineSpecified(true), 
  routineSpecified(false), code(c), codeSpecified(true), kind(k), 
  regionSpecified(false), qualifierSpecified(false)
{}

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// init code="init();" lang = "c" */
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string c, bool cs, instrumentKind_t k) :
  fileSpecified(false), routineName("#"), routineSpecified(true),
  lineSpecified(false), code(c), codeSpecified(cs), kind(k),
  qualifierSpecified(false), regionSpecified(false)
{}

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// entry routine="foo" code="print *, 'Hi'; " lang = "fortran" */
//    /* FOR THIS TYPE, you must specify the codeSpecified argument */
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string r, string c, bool cs, instrumentKind_t k) :
  routineName(r), routineSpecified(true), code(c), codeSpecified(cs), 
  kind(k), fileSpecified(false), lineSpecified(false),
  regionSpecified(false), qualifierSpecified(false)
{}

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
//    /* entry/exit file = "foo.f90" routine = "foo" code = "printf" lang = "fortran" */
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, string r, string c, instrumentKind_t k) 
  : filename(f), fileSpecified(true), routineName(r), 
    routineSpecified(true), code (c), codeSpecified(true), 
    lineSpecified(false), kind (k), 
    regionSpecified(false), qualifierSpecified(false)
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// loops routine = "foo" 
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string r, instrumentKind_t k ) : 
  routineName(r), routineSpecified (true), kind (k), 
  lineSpecified(false), fileSpecified(false), codeSpecified(false),
  regionSpecified(false), qualifierSpecified(false)
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// loops file = "f1.cpp" routine = "foo" 
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, string r, instrumentKind_t k ) : 
  filename (f), fileSpecified(true), routineName(r), 
  routineSpecified (true), kind (k), lineSpecified(false), 
  codeSpecified(false), regionSpecified(false), qualifierSpecified(false)
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// [static/dynamic] [phase/timer] routine = "name"
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(itemQualifier_t q, instrumentKind_t k, string r) :
  qualifier(q), kind(k), routineName(r), routineSpecified(true), 
  codeSpecified(false), lineSpecified(false), fileSpecified(false), 
  regionSpecified(false), qualifierSpecified(true)
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// [static/dynamic] [phase/timer] name = "name" file= "fname" line=a to line=b
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(itemQualifier_t q, instrumentKind_t k, string n, 
			     string f, int linestart, int linestop) :
  qualifier(q), kind(k), code(n), codeSpecified(true), 
  filename(f), fileSpecified(true), regionStart(linestart), 
  regionStop(linestop), regionSpecified(true), qualifierSpecified(true),
  lineSpecified(false) /* region, not line */
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
  if (qualifierSpecified) 
    {
      switch(qualifier)
	{ 
	case STATIC:
	  ostr <<"static: ";
	  break;
	case DYNAMIC:
	  ostr <<"dynamic: ";
	  break;
	case NOT_SPECIFIED:
	  ostr <<"ERROR: qualifier (static/dynamic) not specified: ";
	  break;
	}
    }
       
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
    case TAU_ROUTINE_DECL:
      ostr<<"decl: ";
      break;
    case TAU_ROUTINE_ENTRY:
      ostr<<"entry: ";
      break;
    case TAU_ROUTINE_EXIT:
      ostr<<"exit: ";
      break;
    case TAU_ABORT:
      ostr<<"abort: ";
      break;
    case TAU_PHASE:
      ostr<<"phase: ";
      break;
    case TAU_TIMER:
      ostr<<"timer: ";
      break;
    case TAU_NOT_SPECIFIED:
      ostr<<"ERROR: NOT SPECIFIED KIND";
      break;
    case TAU_INIT:
      ostr<<"init: ";
      break;
    default:
      ostr<<"default: ???";
      break;
    }
  if (regionSpecified) ostr <<"line (start) = "<<regionStart << " to line (stop) = "<<regionStop<<endl;

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

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getRegionSpecified() accesses private data member
///////////////////////////////////////////////////////////////////////////
bool tauInstrument::getRegionSpecified(void) { return regionSpecified; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getRegionStart() accesses private data member
///////////////////////////////////////////////////////////////////////////
int tauInstrument::getRegionStart(void) { return regionStart; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getRegionStop() accesses private data member
///////////////////////////////////////////////////////////////////////////
int tauInstrument::getRegionStop(void) { return regionStop; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getQualifier() accesses private data member
///////////////////////////////////////////////////////////////////////////
itemQualifier_t tauInstrument::getQualifier(void) { return qualifier; }

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getQualifierSpecified() accesses private data member
///////////////////////////////////////////////////////////////////////////
bool tauInstrument::getQualifierSpecified(void) { return qualifierSpecified; }




/* Instrumentation section */

///////////////////////////////////////////////////////////////////////////
// parseError
// input: line number and column 
///////////////////////////////////////////////////////////////////////////
void parseError(const char *message, char *line, int lineno, int column)
{
  printf("ERROR: %s: parse error at selective instrumentation file line %d col %d\n",
	 message, lineno, column);
  printf("line=%s\n", line);
  exit(0);
}

#define WSPACE(line) while ( line[0] == ' ' || line[0] == '\t')		\
    {									\
      if (line[0] == '\0') parseError("EOL found", line, lineno, line - original); \
      line++;								\
    } 

#define TOKEN(k) if (line[0] != k || line[0] == '\0') parseError("token not found", line, lineno, (int ) (line - original)); \
  else line++; 

#define RETRIEVESTRING(pname, line) i = 0;				\
  while (line[0] != '"') {						\
    if (line [0] == '\0') parseError("EOL", line, lineno, line - original); \
    pname[i++] = line[0]; line++;					\
  }									\
  pname[i] = '\0';							\
  line++; /* found closing " */

#define RETRIEVECODE(pname, line) i = 0;				\
  while (line[0] != '"') {						\
    if (line [0] == '\0') parseError("EOL", line, lineno, line - original); \
    if (line[0] == '\\') {						\
      switch(line[1]) {							\
      case '\\':							\
      case '"':								\
	break;								\
      case 'n':								\
	line[1] = '\n';							\
	break;								\
      case 't':								\
	line[1] = '\t';							\
	break;								\
      default:								\
	parseError("Unknown escape sequence", line, lineno, line - original); \
	break;								\
      }									\
      line++;								\
    }									\
    pname[i++] = line[0]; line++;					\
  }									\
  pname[i] = '\0';							\
  line++; /* found closing " */

#define RETRIEVENUMBER(pname, line) i = 0;				\
  while (line[0] != ' ' && line[0] != '\t' ) {				\
    if (line [0] == '\0') parseError("EOL", line, lineno, line - original); \
    pname[i++] = line[0]; line++;					\
  }									\
  pname[i] = '\0';							\
  line++; /* found closing " */

#define RETRIEVENUMBERATEOL(pname, line) i = 0;				\
  while (line[0] != '\0' && line[0] != ' ' && line[0] != '\t' ) {	\
    pname[i++] = line[0]; line++;					\
  }									\
  pname[i] = '\0';							\
									\
  ///////////////////////////////////////////////////////////////////////////
// parseInstrumentationCommand
// input: line -  character string containing a line of text from the selective 
// instrumentation file 
// input: lineno - integer line no. (for reporting parse errors if any)
//
///////////////////////////////////////////////////////////////////////////
void parseDyninstInstrumentationCommand(char *line, int lineno, vector<tauInstrument *>& returnedList)
{
  char  *original;
  int i, ret, value; 
  bool filespecified = false; 
  bool phasespecified = false; 
  bool timerspecified = false; 
  itemQualifier_t qualifier = STATIC;
  bool staticspecified = false; 
  bool dynamicspecified = false; 
  char pname[INBUF_SIZE]; /* parsed name */
  char pfile[INBUF_SIZE]; /* parsed filename */
  char plineno[INBUF_SIZE]; /* parsed lineno */
  char pcode[INBUF_SIZE]; /* parsed code */
  char plang[INBUF_SIZE]; /* parsed language */

  int m1, m2, m3, m4; 
  int startlineno, stoplineno;
  startlineno = stoplineno = 0;
  instrumentKind_t kind = TAU_NOT_SPECIFIED;
 
  m1 = m2 = m3 = m4 = 1; /* does not match by default -- for matching loops/io/mem */

#ifdef DEBUG
  printf("Inside parseInstrumentationCommand: line %s lineno: %d\n",
	 line, lineno);
#endif /* DEBUG */

  original = line; 
  /* check the initial keyword */
  

  /* *************** Inserted by NickT *************  */
  if (strncmp(line, "dyninst loops", 13) == 0)
    {
#ifdef DEBUG
      printf("Found dyninst LOOPS!\n");
#endif /* DEBUG */
      /* parse: loops routine = "#" */
      line += 13;
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
	  printf("GOT file = %s\n", pfile);
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
#ifdef DEBUG
	  printf("GOT routine = %s\n", pname);
#endif /* DEBUG */
	}
      else
	{
	  strcpy(pname, "#");
	}
#ifdef DEBUG
      printf("loops routine = %s, file = %s\n", pname, pfile);
#endif /* DEBUG */
      if (filespecified)
	{
	  instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_LOOPS));
	}
      else 
	{
	  bool codespecified = true; 
	  instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_LOOPS));
	} /* file and routine are both specified for loops */
    } else 
    {
      /* *************** End inserted by NickT ************* */
 
      
      
      
      
      if (strncmp(line, "file", 4) == 0)
	{
#ifdef DEBUG
	  printf("Found LINE!\n");
#endif /* DEBUG */
	  /* parse: file = "foo.cc" line = 245 code = "TAU_NODE(0);" lang = "c++" */
	  line+=4; /* start checking from here */
	  /* WHITE SPACES */
	  WSPACE(line);
	  TOKEN('=');
	  WSPACE(line);
	  TOKEN('"');
	  RETRIEVESTRING(pfile, line);
	  filespecified = true; 
#ifdef DEBUG
	  printf("GOT name = %s\n", pfile);
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
	      printf("GOT line no = %d, line = %s\n", value, line);
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
	      printf("GOT code = %s\n", pcode);
#endif /* DEBUG */
	    }
	  else parseError("<code> token not found", line, lineno, line - original); 
	  WSPACE(line);
	  if (strncmp(line, "lang", 4) == 0)
	    {
	      line += 4; /* move 4 spaces */
	      /* check for = <WSPACE> " */
	      WSPACE(line);
	      TOKEN('=');
	      WSPACE(line);
	      TOKEN('"');
	      RETRIEVESTRING(plang, line);
#ifdef DEBUG
	      printf("GOT lang = %s\n", plang);
#endif /* DEBUG */

	    }
#ifdef DEBUG
	  printf("file = %s, code = %s, line no = %d = %d\n", pfile, pcode, value);
#endif /* DEBUG */
	  instrumentList.push_back(new tauInstrument(string(pfile), value, string(pcode), TAU_LINE));
	}
      else
	{ /* parse: entry routine = "foo()" code = "TAU_SET_NODE(0)" lang = "c" */
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
		  printf("GOT file = %s\n", pfile);
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
#ifdef DEBUG
		  printf("GOT routine = %s\n", pname);
#endif /* DEBUG */
		}
	      else
		{
		  strcpy(pname, "#");
		}
	      if (strncmp(line, "code", 4) == 0)
		{ 
		  line+= 4; /* move 4 spaces */
		  /* check for = <WSPACE> " */
		  WSPACE(line);
		  TOKEN('=');
		  WSPACE(line);
		  TOKEN('"');
		  RETRIEVECODE(pcode, line);
		  WSPACE(line);
#ifdef DEBUG
		  printf("GOT code = %s\n", pcode);
#endif /* DEBUG */
		}
	      else parseError("<code> token not found", line, lineno, line - original); 
	      if (strncmp(line, "lang", 4) == 0)
		{
		  line += 4; /* move 4 spaces */
		  /* check for = <WSPACE> " */
		  WSPACE(line);
		  TOKEN('=');
		  WSPACE(line);
		  TOKEN('"');
		  RETRIEVESTRING(plang, line);
		  WSPACE(line);
#ifdef DEBUG
		  printf("GOT lang = %s\n", plang);
#endif /* DEBUG */

		}
#ifdef DEBUG 
	      printf("entry routine = %s, code = %s, lang = %d\n", pname, pcode);
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
	    } /* end of entry token */
	  else 
	    { /* parse: exit routine = "foo()" code = "bar()" lang = "c" */
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
		      printf("GOT file = %s\n", pfile);
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
#ifdef DEBUG
		      printf("GOT routine = %s\n", pname);
#endif /* DEBUG */
		    }
		  else
		    {
		      strcpy(pname, "#");
		    }
		  if (strncmp(line, "code", 4) == 0)
		    { 
		      line+= 4; /* move 4 spaces */
		      /* check for = <WSPACE> " */
		      WSPACE(line);
		      TOKEN('=');
		      WSPACE(line);
		      TOKEN('"');
		      RETRIEVECODE(pcode, line);
		      WSPACE(line);
#ifdef DEBUG
		      printf("GOT code = %s\n", pcode);
#endif /* DEBUG */
		    }
		  else parseError("<code> token not found", line, lineno, line - original);
		  if (strncmp(line, "lang", 4) == 0)
		    {
		      line += 4; /* move 4 spaces */
		      /* check for = <WSPACE> " */
		      WSPACE(line);
		      TOKEN('=');
		      WSPACE(line);
		      TOKEN('"');
		      RETRIEVESTRING(plang, line);
		      WSPACE(line);
#ifdef DEBUG
		      printf("GOT lang = %s\n", plang);
#endif /* DEBUG */

		    }
#ifdef DEBUG
		  printf("exit routine = %s, code = %s, lang = %d\n", pname, pcode);
#endif /* DEBUG */
		  if (filespecified)
		    {
		      instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_ROUTINE_EXIT));
		    }
		  else 
		    {
		      bool codespecified = true; 
		      instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_ROUTINE_EXIT));
		    } /* file and routine are both specified for exit */
		} /* end of exit */
	      else
		{ /* parse: abort routine = "foo()" code = "bar()" lang = "c" */
		  if (strncmp(line, "abort", 5) == 0)
		    {
		      line+=5; 
#ifdef DEBUG
		      printf("Found ABORT!\n");
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
			  printf("GOT file = %s\n", pfile);
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
#ifdef DEBUG
			  printf("GOT routine = %s\n", pname);
#endif /* DEBUG */
			}
		      else
			{
			  strcpy(pname, "#");
			}
		      if (strncmp(line, "code", 4) == 0)
			{ 
			  line+= 4; /* move 4 spaces */
			  /* check for = <WSPACE> " */
			  WSPACE(line);
			  TOKEN('=');
			  WSPACE(line);
			  TOKEN('"');
			  RETRIEVECODE(pcode, line);
			  WSPACE(line);
#ifdef DEBUG
			  printf("GOT code = %s\n", pcode);
#endif /* DEBUG */
			}
		      else parseError("<code> token not found", line, lineno, line - original); 
		      if (strncmp(line, "lang", 4) == 0)
			{
			  line += 4; /* move 4 spaces */
			  /* check for = <WSPACE> " */
			  WSPACE(line);
			  TOKEN('=');
			  WSPACE(line);
			  TOKEN('"');
			  RETRIEVESTRING(plang, line);
			  WSPACE(line);
#ifdef DEBUG
			  printf("GOT lang = %s\n", plang);
#endif /* DEBUG */

			}
#ifdef DEBUG 
		      printf("entry routine = %s, code = %s, lang = %d\n", pname, pcode);
#endif /* DEBUG */
		      if (filespecified)
			{
			  instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_ABORT));
			}
		      else 
			{
			  bool codespecified = true; 
			  instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_ABORT));
			} /* file and routine are both specified for abort */
		    } /* end of abort token */
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
			    printf("GOT file = %s\n", pfile);
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
		      else 
			{ /* check for static/dynamic phase ... */
			  m1 = strncmp(line, "static", 6);
			  m2 = strncmp(line, "dynamic", 7);
			  m3 = strncmp(line, "phase", 5);
			  m4 = strncmp(line, "timer", 5);
			  if ((m1 == 0) || (m2 == 0) || (m3 == 0) || (m4 == 0)) {
			    if (m1 == 0) { /* static */
			      staticspecified = true;
			      qualifier = STATIC; 
			      line += 6; /* move the pointer 6 spaces (static) for next token */
#ifdef DEBUG
			      printf("GOT static lineno = %d\n", lineno);
#endif /* DEBUG */
			    }
			    else {
			      if (m2 == 0) { /* dynamic */
				dynamicspecified = true;
				qualifier = DYNAMIC;
				line += 7; /* move the pointer 7 spaces (dynamic) for next token */
#ifdef DEBUG
				printf("GOT dynamic lineno = %d\n", lineno);
#endif /* DEBUG */
			      }
			      else {
				if (m3 == 0) { /* phase */
				  phasespecified = true;
				  kind = TAU_PHASE;
				  line += 5; /* move the pointer 5 spaces (static) for next token */
#ifdef DEBUG
				  printf("GOT phase lineno = %d\n", lineno);
#endif /* DEBUG */
				}
				else {
				  if (m4 == 0) { /* timer */
				    timerspecified = true;
				    kind = TAU_TIMER;
				    line += 5;  /* move the pointer 5 spaces (timer) for next token */
#ifdef DEBUG
				    printf("GOT timer lineno = %d\n", lineno);
#endif /* DEBUG */
				  }
				}
			      } 
			    }
			    /* we have either static/dynamic phase/timer ... */
			    if (staticspecified || dynamicspecified) {
			      /* proceed to the next keyword */
			      WSPACE(line); 
			      /* go to phase/timer */
			      if (strncmp(line, "phase", 5) == 0)
				{ 
				  phasespecified = true;
				  kind = TAU_PHASE;
#ifdef DEBUG
				  printf("GOT phase command lineno = %d\n", lineno);
#endif /* DEBUG */
				} else {
				if (strncmp(line, "timer", 5) == 0)
				  { 
				    timerspecified = true;
				    kind = TAU_TIMER;
#ifdef DEBUG
				    printf("GOT timer command lineno = %d\n", lineno);
#endif /* DEBUG */
				  }
				else parseError("<phase/timer> token not found", line, lineno, line - original);      
			      } /* at this stage we have static/dynamic phase/timer definition */
			      line += 5;  /* move the pointer 5 spaces (timer) for next token */
			    } /* static || dynamic specified */

			    WSPACE(line); /* it can be routine or name */
			    if (strncmp(line, "routine", 7) == 0) 
			      { /* static/dynamic phase/timer routine = "..." */
				line+=7;
				/* found routine */ 
				WSPACE(line);
				TOKEN('=');
				WSPACE(line);
				TOKEN('"');
				RETRIEVESTRING(pname, line);
#ifdef DEBUG
				printf("s/d p/t got routine = %s\n", pname);
#endif /* DEBUG */
			      } else {
			      if (strncmp(line, "name", 4) == 0) 
				{ /* static/dynamic phase/timer name = "..." file=<name> line = <no> to line = <no> */
				  line+=4;
				  /* found name */ 
				  WSPACE(line);
				  TOKEN('=');
				  WSPACE(line);
				  TOKEN('"');
				  RETRIEVESTRING(pname, line);
				  WSPACE(line);
#ifdef DEBUG
				  printf("s/d p/t got name = %s\n", pname);
#endif /* DEBUG */
				}
			      else { /* name or routine not specified */
				parseError("<routine/name> token not found", line, lineno, line - original);      
			      }
			      /* name was parsed. Look for line = <no> to line = <no> next */
			      if (strncmp(line, "file", 4) == 0)
				{ /* got line token, get line no. */
				  line += 4; 
				  WSPACE(line);
				  TOKEN('=');
				  WSPACE(line);
				  TOKEN('"');
				  RETRIEVESTRING(pfile, line);
				  filespecified = true; 
				  WSPACE(line);
#ifdef DEBUG
				  printf("GOT file = %s\n", pfile);
#endif /* DEBUG */
				}
			      else {
				parseError("<file> token not found", line, lineno, line - original); 
			      }
			      if (strncmp(line, "line", 4) == 0)
				{ /* got line token, get line no. */
				  line += 4; 
				  WSPACE(line);
				  TOKEN('=');
				  WSPACE(line);
				  RETRIEVENUMBER(plineno, line);
				  ret = sscanf(plineno, "%d", &startlineno); 
#ifdef DEBUG
				  printf("GOT start line no = %d, line = %s\n", startlineno, line);
#endif /* DEBUG */
				  WSPACE(line);
				  if (strncmp (line, "to", 2) == 0)
				    {
				      line += 2; 
				      WSPACE(line); /* look for line=<no> next */
				      if (strncmp (line, "line", 4) == 0)
					{
					  line += 4; 
					  WSPACE(line); 
					  TOKEN('=');
					  WSPACE(line);
					  RETRIEVENUMBERATEOL(pcode, line);
					  ret = sscanf(pcode, "%d", &stoplineno); 
#ifdef DEBUG
					  printf("GOT stop line no = %d\n", stoplineno);
#endif /* DEBUG */
					} else { /* we got line=<no> to , but there was no line */
					parseError("<line> token not found in the stop declaration", line, lineno, line - original);
				      } /* parsed to clause */
				    }
				  else { /* hey, line = <no> is there, but there is no "to" */
				    parseError("<to> token not found", line, lineno, line - original);
				  } /* we have parsed all the tokens now. Let us see what was specified phase/timer routine = <name> or phase/timer name =<name> line = <no> to line = <no> */

				}  else { 
				parseError("<line> token not found in the start declaration", line, lineno, line - original); 
			      } /* line specified */ 
			    } /* end of routine/name processing */
			    /* create instrumentation requests here */
			    if (filespecified) 
			      { /* [static/dynamic] <phase/timer> name = "<name>" file="<name>" line=a to line=b */
				instrumentList.push_back(new tauInstrument(qualifier, kind, pname, pfile, startlineno, stoplineno));
			      }
			    else 
			      { /* [static/dynamic] <phase/timer> routine = "<name>" */
				instrumentList.push_back(new tauInstrument(qualifier, kind, pname));
			      }
			  } /* end of if static/dynamic/phase/timer */
			  else
			    {
			      /* parse: init code = "init();" lang = "c" */
			      if (strncmp(line, "init", 4) == 0)
				{
				  line += 4;
#ifdef DEBUG
				  printf("Found INIT!\n");
#endif /* DEBUG */
				  WSPACE(line);
				  if (strncmp(line, "code", 4) == 0)
				    {
				      line += 4;
				      WSPACE(line);
				      TOKEN('=');
				      WSPACE(line);
				      TOKEN('"');
				      RETRIEVECODE(pcode, line);
				      WSPACE(line);
#ifdef DEBUG
				      printf("GOT code = %s\n", pcode);
#endif /* DEBUG */
				    }
				  else parseError("<code> token not found", line, lineno, line - original); 
				  if (strncmp(line, "lang", 4) == 0)
				    {
				      line += 4; /* move 4 spaces */
				      /* check for = <WSPACE> " */
				      WSPACE(line);
				      TOKEN('=');
				      WSPACE(line);
				      TOKEN('"');
				      RETRIEVESTRING(plang, line);
				      WSPACE(line);
#ifdef DEBUG
				      printf("GOT lang = %s\n", plang);
#endif /* DEBUG */
				    }
#ifdef DEBUG
				  printf("code = %s\n", pcode);
#endif /* DEBUG */
				  instrumentList.push_back(new tauInstrument(string(pcode), true, TAU_INIT)); 
				} /* end of init directive */
			      else
				{
				  if (strncmp(line, "decl", 4) == 0)
				    {
				      line+=4;
#ifdef DEBUG
				      printf("Found DECL!\n");
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
					  printf("GOT file = %s\n", pfile);
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
#ifdef DEBUG
					  printf("GOT routine = %s\n", pname);
#endif /* DEBUG */
					}
				      else
					{
					  strcpy(pname, "#");
					}
				      if (strncmp(line, "code", 4) == 0)
					{ 
					  line+= 4; /* move 4 spaces */
					  /* check for = <WSPACE> " */
					  WSPACE(line);
					  TOKEN('=');
					  WSPACE(line);
					  TOKEN('"');
					  RETRIEVECODE(pcode, line);
					  WSPACE(line);
#ifdef DEBUG
					  printf("GOT code = %s\n", pcode);
#endif /* DEBUG */
					}
				      else parseError("<code> token not found", line, lineno, line - original); 
				      if (strncmp(line, "lang", 4) == 0)
					{
					  line += 4; /* move 4 spaces */
					  /* check for = <WSPACE> " */
					  WSPACE(line);
					  TOKEN('=');
					  WSPACE(line);
					  TOKEN('"');
					  RETRIEVESTRING(plang, line);
					  WSPACE(line);
#ifdef DEBUG
					  printf("GOT lang = %s\n", plang);
#endif /* DEBUG */

					}
#ifdef DEBUG 
				      printf("decl routine = %s, code = %s, lang = %d\n", pname, pcode);
#endif /* DEBUG */
				      if (filespecified)
					{
					  instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_ROUTINE_DECL));
					}
				      else 
					{
					  bool codespecified = true; 
					  instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_ROUTINE_DECL));
					} /* file and routine are both specified for entry */
				    } /* end of decl token */
				} /* end of init token */
			    }
			} /* check for phase/timer */
		    } /* end of loops directive */
		} /* abort */
	    } /* exit */
	} /* entry */
    }
  returnedList.swap(instrumentList);
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

