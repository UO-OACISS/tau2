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
bool isVoidRoutine(const pdbItem* r);
int parseLanguageString(const string& str);
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
        regionSpecified(false), qualifierSpecified(false), language(LA_ANY)
{}


///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// file = "foo.cpp" line=245 code = "TAU_NODE(0);" lang = "c++"
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, int line, string c, instrumentKind_t k, int lang) : 
      	filename(f), fileSpecified(true), lineno(line), lineSpecified(true), 
      	routineSpecified(false), code(c), codeSpecified(true), kind(k), 
        regionSpecified(false), qualifierSpecified(false), language(lang)
{}

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// init code="init();" lang = "c" */
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string c, bool cs, instrumentKind_t k, int lang) :
        fileSpecified(false), routineName("#"), routineSpecified(true),
        lineSpecified(false), code(c), codeSpecified(cs), kind(k),
        qualifierSpecified(false), regionSpecified(false), language(lang)
{}

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// entry routine="foo" code="print *, 'Hi'; " lang = "fortran" */
//    /* FOR THIS TYPE, you must specify the codeSpecified argument */
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string r, string c, bool cs, instrumentKind_t k, int lang) :
      	routineName(r), routineSpecified(true), code(c), codeSpecified(cs), 
  	kind(k), fileSpecified(false), lineSpecified(false),
        regionSpecified(false), qualifierSpecified(false), language(lang)
{}

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
//    /* entry/exit file = "foo.f90" routine = "foo" code = "printf" lang = "fortran" */
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, string r, string c, instrumentKind_t k, int lang) 
	: filename(f), fileSpecified(true), routineName(r), 
	routineSpecified(true), code (c), codeSpecified(true), 
	lineSpecified(false), kind (k), 
        regionSpecified(false), qualifierSpecified(false), language(lang)
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// loops routine = "foo" 
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string r, instrumentKind_t k ) : 
	routineName(r), routineSpecified (true), kind (k), 
	lineSpecified(false), fileSpecified(false), codeSpecified(false),
        regionSpecified(false), qualifierSpecified(false), language(LA_ANY)
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// loops file = "f1.cpp" routine = "foo" 
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(string f, string r, instrumentKind_t k ) : 
	filename (f), fileSpecified(true), routineName(r), 
	routineSpecified (true), kind (k), lineSpecified(false), 
	codeSpecified(false), regionSpecified(false), qualifierSpecified(false),
	language(LA_ANY)
{} 

///////////////////////////////////////////////////////////////////////////
// tauInstrument() ctor
// [static/dynamic] [phase/timer] routine = "name"
///////////////////////////////////////////////////////////////////////////
tauInstrument::tauInstrument(itemQualifier_t q, instrumentKind_t k, string r) :
	qualifier(q), kind(k), routineName(r), routineSpecified(true), 
	codeSpecified(false), lineSpecified(false), fileSpecified(false), 
        regionSpecified(false), qualifierSpecified(true), language(LA_ANY)
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
        lineSpecified(false) /* region, not line */, language(LA_ANY)
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
       ostr << "Language code: " << language << endl;
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

///////////////////////////////////////////////////////////////////////////
// tauInstrument::isActiveForLanguage() accesses private data member
///////////////////////////////////////////////////////////////////////////
bool tauInstrument::isActiveForLanguage(PDB::lang_t lang) const
{
  return (language & lang);
}

///////////////////////////////////////////////////////////////////////////
// tauInstrument::getCode() accesses private data member
///////////////////////////////////////////////////////////////////////////
string tauInstrument::getCode(const pdbLoc& loc, const pdbRoutine* r, bool isInit)
{
  string result = code;

  // Replace variables
  replaceAll(result, "@FILE@", loc.file()->name());
  replaceAll(result, "@LINE@", intToString(loc.line()));
  replaceAll(result, "@COL@", intToString(loc.col()));
  if (r)
  {
    replaceAll(result, "@ROUTINE@", r->fullName());
    replaceAll(result, "@BEGIN_LINE@", intToString(r->headBegin().line()));
    replaceAll(result, "@BEGIN_COL@", intToString(r->headBegin().col()));
    replaceAll(result, "@END_LINE@", intToString(r->bodyEnd().line()));
    replaceAll(result, "@END_COL@", intToString(r->bodyEnd().col()));
  }
  if (isInit)
  {
    replaceAll(result, "@ARGC@", "tau_argc");
    replaceAll(result, "@ARGV@", "tau_argv");
  }

  return result;
}


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
    if (line[0] == '\\') { \
      switch(line[1]) { \
        case '\\': \
        case '"': \
          break; \
        case 'n': \
          line[1] = '\n'; \
          break; \
        case 't': \
          line[1] = '\t'; \
          break; \
        default: \
          parseError("Unknown escape sequence", line, lineno, line - original); \
          break; \
      } \
      line++; \
    } \
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

#define RETRIEVENUMBERATEOL(pname, line) i = 0; \
  while (line[0] != '\0' && line[0] != ' ' && line[0] != '\t' ) { \
    pname[i++] = line[0]; line++; \
  } \
  pname[i] = '\0';  \

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
  int language = tauInstrument::LA_ANY;
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
      language = parseLanguageString(plang);
      if (language < 0)
        parseError("<lang> token invalid", line, lineno, line - original);
    }
#ifdef DEBUG
    printf("file = %s, code = %s, line no = %d, language = %d\n", pfile, pcode, value, language);
#endif /* DEBUG */
    instrumentList.push_back(new tauInstrument(string(pfile), value, string(pcode), TAU_LINE, language));
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
        language = parseLanguageString(plang);
        if (language < 0)
          parseError("<lang> token invalid", line, lineno, line - original);
      }
#ifdef DEBUG 
      printf("entry routine = %s, code = %s, lang = %d\n", pname, pcode, language);
#endif /* DEBUG */
      if (filespecified)
      {
        instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_ROUTINE_ENTRY, language));
      }
      else 
      {
	bool codespecified = true; 
        instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_ROUTINE_ENTRY, language));
      } /* file and routine are both specified for entry */
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
          language = parseLanguageString(plang);
          if (language < 0)
            parseError("<lang> token invalid", line, lineno, line - original);
        }
#ifdef DEBUG
	printf("exit routine = %s, code = %s, lang = %d\n", pname, pcode, language);
#endif /* DEBUG */
        if (filespecified)
	{
          instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_ROUTINE_EXIT, language));
	}
        else 
	{
	  bool codespecified = true; 
          instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_ROUTINE_EXIT, language));
	} /* file and routine are both specified for entry */
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
                language = parseLanguageString(plang);
                if (language < 0)
                  parseError("<lang> token invalid", line, lineno, line - original);
              }
#ifdef DEBUG
              printf("code = %s\n", pcode);
#endif /* DEBUG */
              instrumentList.push_back(new tauInstrument(string(pcode), true, TAU_INIT, language)); 
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
                  language = parseLanguageString(plang);
                  if (language < 0)
                    parseError("<lang> token invalid", line, lineno, line - original);
                }
#ifdef DEBUG 
	        printf("decl routine = %s, code = %s, lang = %d\n", pname, pcode, language);
#endif /* DEBUG */
                if (filespecified)
	        {
                  instrumentList.push_back(new tauInstrument(string(pfile), string(pname), string(pcode), TAU_ROUTINE_DECL, language));
	        }
                else 
	        {
	          bool codespecified = true; 
                  instrumentList.push_back(new tauInstrument(string(pname), string(pcode), codespecified, TAU_ROUTINE_DECL, language));
	        } /* file and routine are both specified for entry */
              } /* end of decl token */
            } /* end of init token */
          }
        } /* check for phase/timer */
      } /* end of loops directive */
    } /* exit */
  } /* entry */
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
void writeStatements(ostream& ostr, const pdbRoutine *ro, 
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
void writeAdditionalDeclarations(ostream& ostr, const pdbRoutine *ro)
{
  if (isInstrumentListEmpty()) return; /* nothing specified */
#ifdef DEBUG 
  printf("writeAdditionalDeclarations: Name %s, id=%d\n", ro->fullName().c_str(), ro->id());
#endif /* DEBUG */
  writeStatements(ostr, ro, additionalDeclarations);
}

/* After profiler and save statements are written, do we need 
   to call any other timer routines? */
void writeAdditionalFortranInvocations(ostream& ostr, const pdbRoutine *ro)
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

#ifdef TAU_ALT_FORTRAN_INSTRUMENTATION
  string declaration1(string("      integer, dimension(2) ::  ")+varname+"= (/ 0, 0 /)");
#else
  string declaration1(string("      integer ")+varname+"(2) / 0, 0 /");
#endif /*TAU_ALT_FORTRAN_INSTRUMENTATION*/
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
void getTauEntityName(const char *prefix, string& varname, int line)
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
	s->stmtBegin().file()->name().c_str(), s->stmtBegin().line(), s->stmtBegin().col());
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
  PDB::lang_t language = p.language();

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
      if ( (*rit)->location().file() == file &&
           !(*rit)->isCompilerGenerated() &&
           (*rit)->kind() != pdbItem::RO_EXT &&
           (*rit)->bodyBegin().line() != 0 &&
           (*rit)->bodyEnd().line() != 0 &&
	   instrumentEntity((*rit)->fullName()) &&
           (*it)->isActiveForLanguage(language) )
      {
#ifdef DEBUG
        cout <<"Examining Routine "<<(*rit)->fullName()<<" and "<<(*it)->getRoutineName()<<endl;
#endif /* DEBUG */
        /* Eventually skip inline functions */
        if ((*rit)->isInline() && noinline_flag)
          continue;

        /* examine the type of request - decl */
        if ((*it)->getKind() == TAU_ROUTINE_DECL)
        {
#ifdef DEBUG
          cout <<"Instrumenting declaration of routine "<<(*rit)->fullName()<<endl;
          /* get routine entry line no. */
          cout <<"at line: "<<(*rit)->bodyBegin().line()<<", col"<< (*rit)->bodyBegin().col()<<"code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */

          /* We treat snippet insertion code as additional declarations
             that must be placed inside the routine */
          list<string> decls;
          decls.push_back("\t" + (*it)->getCode((*rit)->bodyBegin(), *rit));

          additionalDeclarations.push_back(pair<int, list<string> >((*rit)->id(), decls)); 
	  /* assign the list of strings to the list */

          /* We need to create empty BODY_BEGIN & BODY_END request here to get the declaration */
          itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_BEGIN, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col(), "", BEFORE));
          itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_END, (*rit)->bodyEnd().line(), (*rit)->bodyEnd().col(), "", BEFORE));
        } /* end of routine decl */
        /* examine the type of request - entry */
        if ((*it)->getKind() == TAU_ROUTINE_ENTRY)
        {
#ifdef DEBUG
          cout <<"Instrumenting entry of routine "<<(*rit)->fullName()<<endl;
          /* get routine entry line no. */
          cout <<"at line: "<<(*rit)->bodyBegin().line()<<", col"<< (*rit)->bodyBegin().col()<<"code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */

          if (!use_spec &&
              (language == PDB::LA_CXX || language == PDB::LA_C_or_CXX))
          {
	    itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col()+1, (*it)->getCode(), BEFORE));
          }
          else
          {
	    itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_BEGIN, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col(), (*it)->getCode((*rit)->bodyBegin(), *rit), BEFORE));

            /* We need to create an empty BODY_END request here to close the '{' created by the BODY_BEGIN */
	    itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_END, (*rit)->bodyEnd().line(), (*rit)->bodyEnd().col(), "", BEFORE));
          }
        } /* end of routine entry */
        /* examine the type of request - exit */
        if ((*it)->getKind() == TAU_ROUTINE_EXIT)
        {
#ifdef DEBUG
            cout <<"Instrumenting exit of routine "<<(*rit)->fullName()<<endl;
#endif /* DEBUG */

          /* get routine exit line no. */
          pdbRoutine::locvec retlocations = (*rit)->returnLocations();
          for (rlit = retlocations.begin(); rlit != retlocations.end(); ++rlit)
          {
#ifdef DEBUG
            cout <<"at line: "<<(*rlit)->line()<<", col"<< (*rlit)->col()<<" code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */

            if (!use_spec &&
                (language == PDB::LA_CXX || language == PDB::LA_C_or_CXX))
            {
	      itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*rlit)->line(), (*rlit)->col(), (*it)->getCode(), BEFORE));
            }
            else
            {
              itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), RETURN, (*rlit)->line(), (*rlit)->col(), (*it)->getCode(**rlit, *rit), BEFORE));
            }
          }
#ifdef DEBUG
          cout <<"at line: "<<(*rit)->bodyEnd().line()<<", col"<< (*rit)->bodyEnd().col()<<"code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */
          if (isVoidRoutine(*rit))
          {
            if (!use_spec &&
                (language == PDB::LA_CXX || language == PDB::LA_C_or_CXX))
            {
              itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*rit)->bodyEnd().line(), (*rit)->bodyEnd().col(), (*it)->getCode(), BEFORE));
            }
            else
            {
              /* We need to create an empty BODY_BEGIN request here to open the '}' created by the BODY_END */
	      itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_BEGIN, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col(), "", BEFORE));
              itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_END, (*rit)->bodyEnd().line(), (*rit)->bodyEnd().col(), (*it)->getCode((*rit)->bodyEnd(), *rit), BEFORE));
            }
          } else if (use_spec) {
            /* We need to create an empty BODY_BEGIN to emit the 'tau_ret_val' declaration, */
            /* however, this also requires an empty BODY_END to have matching braces. */
	    itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_BEGIN, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col(), "", BEFORE));
            itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_END, (*rit)->bodyEnd().line(), (*rit)->bodyEnd().col(), "", BEFORE));
          }
        } /* end of routine exit */
        /* examine the type of request - init */
        if ((*it)->getKind() == TAU_INIT)
        {
          if (!use_spec &&
              (language == PDB::LA_CXX || language == PDB::LA_C_or_CXX))
          {
	    itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col()+1, (*it)->getCode(), BEFORE));
          }
          else
          {
            if ((*rit)->name().compare("main") == 0)
            {
#ifdef DEBUG
              cout << "Instrumenting init" << endl;
#endif /* DEBUG */

              static bool needsSetup = true;
              if (needsSetup)
              {
                list<string> decls;
                string       setup;

                /* Generate argc/argv temporaries */
                decls.push_back("\tstatic int    tau_argc;");
                decls.push_back("\tstatic char **tau_argv;");

                pdbType::argvec args = (*rit)->signature()->arguments();
                if (2 == args.size() &&
                    0 != args[0].name().compare("-") &&
                    0 != args[1].name().compare("-"))
                {
                  setup = "tau_argc = " + args[0].name() + ";\n\t"
                          "tau_argv = " + args[1].name() + ";\n";
                }
                else
                {
                  decls.push_back("\tstatic char * tau_unknown = \"unknown\";");

                  setup = "tau_argc = 1;\n\t"
                          "tau_argv = &tau_unknown;\n";
                }
                additionalDeclarations.push_back(pair<int, list<string> >((*rit)->id(), decls)); 
                itemvec.push_back( new itemRef(static_cast<pdbItem*>(*rit), BODY_BEGIN, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col(), setup, BEFORE));
                needsSetup = false;
              }

              itemvec.push_back( new itemRef((pdbItem *)NULL, BODY_BEGIN, (*rit)->bodyBegin().line(), (*rit)->bodyBegin().col(), (*it)->getCode((*rit)->bodyBegin(), *rit, true), BEFORE));

              /* We need to create an empty BODY_END request here to close the '{' created by the BODY_BEGIN */
	      itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_END, (*rit)->bodyEnd().line(), (*rit)->bodyEnd().col(), "", BEFORE));
            }
          }
        } /* end of init */
      }
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
      instrumentKind_t isPhaseOrTimer = (*it)->getKind(); 
      if (isPhaseOrTimer == TAU_PHASE || isPhaseOrTimer == TAU_TIMER)
      { /* we need to instrument this routine as a phase/timer */
        /* We need to identify the itemRef record associated with this
          routine and mark it as a phase/timer over there. 
          Iterate over the list.
        */
        for(vector<itemRef *>::iterator iter = itemvec.begin();
            iter != itemvec.end(); iter++)
        {
          if ((*iter)->item)
          { /* item's pdbItem entry is not null */
#ifdef DEBUG_PROF
            printf("examining %s. id = %d. Current routine id = %d\n",
            (*iter)->item->name(), (*iter)->item->id(), (*rit)->id());     
#endif /* DEBUG_PROF */
            if ((*iter)->item->id() == (*rit)->id()) 
            { /* found it! We need to annotate this as a phase */
              if (isPhaseOrTimer == TAU_PHASE) 
              {
#ifdef DEBUG
	        printf("Routine %s is a PHASE \n", (*rit)->fullName().c_str());
#endif /* DEBUG */
                (*iter)->isPhase = true;
              }
              else 
              {
#ifdef DEBUG
	        printf("Routine %s is a TIMER \n", (*rit)->fullName().c_str());
#endif /* DEBUG */
              }
              if ((*it)->getQualifier() == DYNAMIC)               
                (*iter)->isDynamic = true;
            }
          }
        } 
      }
    }
  }

  return retval;
}

/* Process list of F routines */
bool processFRoutinesInstrumentation(PDB & p, vector<tauInstrument *>::iterator& it, vector<itemRef *>& itemvec, pdbFile *file) 
{
  PDB::lang_t language = p.language();

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
      if ( (*rit)->location().file() == file &&  
           (*rit)->kind() != pdbItem::RO_FSTFN &&  
	   (*rit)->firstExecStmtLocation().file() && 
	   instrumentEntity((*rit)->fullName()) &&
           (*it)->isActiveForLanguage(language) )
      {
       /* examine the type of request - decl */
       if ((*it)->getKind() == TAU_ROUTINE_DECL)
       {
#ifdef DEBUG
         cout <<"Instrumenting declaration of routine "<<(*rit)->fullName()<<endl;
         /* get routine entry line no. */
         cout <<"at line: "<<(*rit)->bodyBegin().line()<<", col"<< (*rit)->bodyBegin().col()<<"code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */

         /* We treat snippet insertion code as additional declarations
            that must be placed inside the routine */
         list<string> decls;
         decls.push_back("\t" + (*it)->getCode((*rit)->bodyBegin(), *rit));

         additionalDeclarations.push_back(pair<int, list<string> >((*rit)->id(), decls)); 
	 /* assign the list of strings to the list */

        /* We need to create empty BODY_BEGIN request here to get the declaration */
         itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_BEGIN, (*rit)->firstExecStmtLocation().line(), (*rit)->firstExecStmtLocation().col(), "", BEFORE));
       } /* end of routine decl */
       /* examine the type of request - entry */
       if ((*it)->getKind() == TAU_ROUTINE_ENTRY)
       {
#ifdef DEBUG
         cout <<"Instrumenting entry of routine "<<(*rit)->fullName()<<endl;
#endif /* DEBUG */
         itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_BEGIN, (*rit)->firstExecStmtLocation().line(), (*rit)->firstExecStmtLocation().col(), (*it)->getCode((*rit)->firstExecStmtLocation(), *rit), BEFORE));
       }
       /* examine the type of request - exit */
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
	
	    itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), RETURN, (*rlit)->line(), (*rlit)->col(), (*it)->getCode(**rlit, *rit), BEFORE));
          }
	  /* and then examine the stop locations */
          for (rlit = stoplocations.begin(); rlit != stoplocations.end(); ++rlit)
          {
#ifdef DEBUG
            cout <<"at line: "<<(*rlit)->line()<<", col"<< (*rlit)->col()<<" code = "<<(*it)->getCode()<<endl;
#endif /* DEBUG */
	
	    itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), EXIT, (*rlit)->line(), (*rlit)->col(), (*it)->getCode(**rlit, *rit), BEFORE));
          }
        } /* end of routine exit */
        if ((*it)->getKind() == TAU_INIT)
        {
          if ((*rit)->kind() == pdbItem::RO_FPROG)
          {
#ifdef DEBUG
            cout << "Instrumenting init" << endl;
#endif /* DEBUG */
    
            itemvec.push_back( new itemRef(static_cast<pdbItem *>(*rit), BODY_BEGIN, (*rit)->firstExecStmtLocation().line(), (*rit)->firstExecStmtLocation().col(), (*it)->getCode((*rit)->firstExecStmtLocation(), *rit), BEFORE));
          }
        }
      }
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
      instrumentKind_t isPhaseOrTimer = (*it)->getKind(); 
      if (isPhaseOrTimer == TAU_PHASE || isPhaseOrTimer == TAU_TIMER)
      { /* we need to instrument this routine as a phase/timer */
       /* We need to identify the itemRef record associated with this
          routine and mark it as a phase over there. Iterate over the list.
       */
        for(vector<itemRef *>::iterator iter = itemvec.begin();
            iter != itemvec.end(); iter++)
        {
          if ((*iter)->item)
          { /* item's pdbItem entry is not null */
#ifdef DEBUG
            printf("examining %s. id = %d. Current routine id = %d, kind = %d\n",
            (*iter)->item->name().c_str(), (*iter)->item->id(), (*rit)->id(), (*iter)->kind);     
 
#endif /* DEBUG */
            if ((*iter)->item->id() == (*rit)->id()) 
            { /* found it! We need to annotate this as a phase */
              if (isPhaseOrTimer == TAU_PHASE) 
              {
#ifdef DEBUG
	        printf("Routine %s is a PHASE \n", (*rit)->fullName().c_str());
#endif /* DEBUG */
                (*iter)->isPhase = true;
              }
              if ((*it)->getQualifier() == DYNAMIC && (*iter)->kind == BODY_BEGIN)
              {
                (*iter)->isDynamic = true;
                /* Add a tau_iter declaration to this routine. */
                list<string> dynamicDecls;
                dynamicDecls.push_back(string("      integer tau_iter / 0 /"));
                dynamicDecls.push_back(string("      save tau_iter"));

                additionalDeclarations.push_back(pair<int, list<string> >((*rit)->id(),
                dynamicDecls));

              }
            }
          }
        }
      } /* dynamic timers are supported now */
    } /* end of match */
  } /* iterate over all routines */

  return true; /* everything is ok -- return true */
}

/* Process all routines from the given file and extract routine relevant to 
   line*/
pdbRoutine * getFRoutineFromFileAndLine(PDB& p, int line)
{
  PDB::froutinevec::const_iterator rit;
  PDB::froutinevec froutines = p.getFRoutineVec();
  pdbRoutine * result; 

#ifdef DEBUG
  printf("Inside getFRoutineFromFileAndLine!\n");
#endif /* DEBUG */
  pdbRoutine::locvec::iterator rlit;
  for(rit = froutines.begin(); rit != froutines.end(); ++rit)
  { /* iterate over all routines */
#ifdef DEBUG
    printf("Iterating... routine = %s, first stmt = %d, looking for line = %d\n", 

       (*rit)->fullName().c_str(), 
       (*rit)->firstExecStmtLocation().line(), line);
#endif /* DEBUG */
    if ((*rit)->firstExecStmtLocation().line() <= line) result = *rit; 
    else break;
  }
  
  return result;
}

/* Add file and routine based instrumentation requests to the itemRef vector
   for C/C++ and Fortran */
bool addFileInstrumentationRequests(PDB& p, pdbFile *file, vector<itemRef *>& itemvec)
{
  /* Let us iterate over the list of instrumentation requests and see if 
   * any requests match this file */
  vector<tauInstrument *>::iterator it;
  bool cmpResult; 
  int column ;
  bool retval = true;
  PDB::lang_t language = p.language();
  PDB::croutinevec croutines;
  PDB::froutinevec froutines; 

#ifdef DEBUG
  printf("INSIDE addFileInstrumentationRequests empty Instrumentation List? %d \n", isInstrumentListEmpty());
#endif /* DEBUG */

  if (memory_flag && language == PDB::LA_FORTRAN) 
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
      /* process file = <name> line=<no> code=<code> request */
      /* phases can also be specified in this manner, we need to distinguish 
the two. With phases, line is not specified, rather a region is specified. So, 
it will not enter here. */
      if ((*it)->getLineSpecified() && (*it)->isActiveForLanguage(language))
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
	itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*it)->getLineNo(), 1, (*it)->getCode(pdbLoc(file, (*it)->getLineNo(), 0)), BEFORE));
      }
      /* is a region specified for phase/timer based instrumentation? */
      if ((*it)->getRegionSpecified())
      { 
#ifdef DEBUG
        cout <<"Region based instrumentation: start: "
             <<(*it)->getRegionStart()<<" stop: "
             <<(*it)->getRegionStop()<< " name: "
             <<(*it)->getCode()<<endl;
  #endif /* DEBUG */
        string startRegionCode;
        string stopRegionCode;
        string regionKind, regionQualifier;
        if ((*it)->getKind() == TAU_PHASE) regionKind=string("PHASE");
        else regionKind=string("TIMER");
        if ((*it)->getQualifier() == DYNAMIC) regionQualifier=string("DYNAMIC");
          else regionQualifier=string("STATIC");
        /* Fortran has a slightly different syntax and requirements 
           compared to C/C++ */
        if (language != PDB::LA_FORTRAN)
        { /* ASSUMPTION: If it is not Fortran it is C or C++ */
          startRegionCode = string("  TAU_")+regionQualifier+string("_")+regionKind+"_START(\""+(*it)->getCode()+"\");";
          stopRegionCode = string("  TAU_")+regionQualifier+string("_")+regionKind+"_STOP(\""+(*it)->getCode()+"\");";
          
        }
        else 
        { /* Fortran region based instrumentation */
          if ((*it)->getQualifier() == STATIC) 
          { /* great! it is easy to instrument static phases in Fortran */
            startRegionCode = string("       call TAU_")+regionQualifier+string("_")+regionKind+"_START(\""+(*it)->getCode()+"\");"; 
            stopRegionCode = string("       call TAU_")+regionQualifier+string("_")+regionKind+"_STOP(\""+(*it)->getCode()+"\");"; 
          }
          else
          { /* To instrument dynamic phases, we need to determine what routine
               the given region belongs to. */
            pdbRoutine *r= getFRoutineFromFileAndLine(p, (*it)->getRegionStart());
#ifdef DEBUG
            cout <<"Instrumenting for dynamic phases at entry of routine "
                 <<r->fullName()<<endl;
            /* get routine entry line no. */
            cout <<"at line: "<<r->bodyBegin().line()<<", col"<< r->bodyBegin().col()<<"code = "<<endl;
#endif /* DEBUG */
            list<string> dynamicDecls;
            dynamicDecls.push_back(string("      integer tau_iteration(2) / 0, 0 /"));
            dynamicDecls.push_back(string("      save tau_iteration"));

            additionalDeclarations.push_back(pair<int, list<string> >(r->id(), 
                dynamicDecls)); 
            /* this takes care of the entry based declarations. Now we need to
               take care of the start/stop code */
//             startRegionCode = string("      tau_iteration = tau_iteration + 1");
//   	    itemvec.push_back( new itemRef((pdbItem *)NULL, 
//               INSTRUMENTATION_POINT, (*it)->getRegionStart(), 1, 
//               startRegionCode, BEFORE));

            startRegionCode = string("       call TAU_")+regionQualifier+string("_")+regionKind+"_START(tau_iteration,\""+(*it)->getCode()+"\");"; 
            stopRegionCode = string("       call TAU_")+regionQualifier+string("_")+regionKind+"_STOP(tau_iteration,\""+(*it)->getCode()+"\");"; 

          }
        }
        /* the region start/stop code goes in here as an instrumentation 
           point for either language. Stop region is one line after the end 
           of the given region.*/
  	itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*it)->getRegionStart(), 1, startRegionCode, BEFORE));
  	itemvec.push_back( new itemRef((pdbItem *)NULL, INSTRUMENTATION_POINT, (*it)->getRegionStop()+1, 1, stopRegionCode, BEFORE));
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
      switch (language) { /* check the language first */
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
#ifdef DEBUG
  printInstrumentList();
#endif /* DEBUG */
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


/* implementation of struct itemRef */
itemRef::itemRef(const pdbItem *i, bool isT) : item(i), isTarget(isT) {
    line = i->location().line();
    col  = i->location().col();
    kind = ROUTINE; /* for C++, only routines are listed */ 
    attribute = NOT_APPLICABLE;
    isPhase = false; /* timer by default */ 
    isDynamic = false; /* static by default */ 
  }
itemRef::itemRef(const pdbItem *i, itemKind_t k, int l, int c) : 
	line (l), col(c), item(i), kind(k) {
#ifdef DEBUG
    cout <<"Added: "<<i->name() <<" line " << l << " col "<< c <<" kind " 
	 << k <<endl;
#endif /* DEBUG */
    isTarget = true; 
    attribute = NOT_APPLICABLE;
    isPhase = false; /* timer by default */ 
    isDynamic = false; /* static by default */ 
  }
itemRef::itemRef(const pdbItem *i, itemKind_t k, int l, int c, string code, itemAttr_t a) : 
	line (l), col(c), item(i), kind(k), snippet(code), attribute(a) {
#ifdef DEBUG
    if (i)
      cout <<"Added: "<<i->name() <<" line " << l << " col "<< c <<" kind " 
	 << k << " snippet " << snippet << endl;
    if (a == BEFORE) cout <<"BEFORE"<<endl;
#endif /* DEBUG */
    isTarget = true; 
    isPhase = false; /* timer by default */ 
    isDynamic = false; /* static by default */ 
  }
itemRef::itemRef(const pdbItem *i, bool isT, int l, int c)
         : item(i), isTarget(isT), line(l), col(c) {
    kind = ROUTINE; 
    attribute = NOT_APPLICABLE;
    isPhase = false; /* timer by default */ 
    isDynamic = false; /* static by default */ 
  }
itemRef::itemRef(const pdbItem *i, itemKind_t k, pdbLoc start, pdbLoc stop)
   : item(i), kind(k), begin(start), end(stop) {
    attribute = NOT_APPLICABLE;  
    line = begin.line();
    col = begin.col();
    isPhase = false; /* timer by default */ 
    isDynamic = false; /* static by default */ 
  }

/* -------------------------------------------------------------------------- */
/* -- Returns true is string is void else returns false --------------------- */
/* -------------------------------------------------------------------------- */
bool isVoidRoutine(const pdbItem * i)
{
  string return_string;
  const pdbType *t = ((const pdbRoutine*)i)->signature()->returnType();

  if ( const pdbGroup* gr = t->isGroup() )
    return_string = gr->name();
  else
    return_string = t->name();
  /* old code
  string return_string = ((pdbRoutine *)(i->item))->signature()->returnType()->name() ;
  */
  if (return_string.compare("void") == 0)
        return true;
  else
        return false;
}


/* -------------------------------------------------------------------------- */
/* -- Returns an integer encoding the languages specified ------------------- */
/* -------------------------------------------------------------------------- */
int parseLanguageString(const string& str)
{
  int language = 0;

  // Split given string at commas
  string::size_type pos = 0;
  string::size_type end = 0;
  do {
    end = str.find(",", pos);

    // Chop off whitespaces
    string::size_type lpos = str.find_first_not_of(" \f\n\r\t\v", pos);
    string::size_type lend = str.find_last_not_of(", \f\n\r\t\v", end);
    if (string::npos == lpos || string::npos == lend)
      return -1;

    // Enable language
    string lang = str.substr(lpos, lend - lpos + 1);
    if (0 == lang.compare("c"))
      language |= PDB::LA_C;
    else if (0 == lang.compare("c++"))
      language |= PDB::LA_CXX;
    else if (0 == lang.compare("fortran"))
      language |= PDB::LA_FORTRAN;
    else
      return -1;

    pos = end + 1;
  } while (string::npos != end);

#ifdef DEBUG
  printf("Language code = %x\n", language);
#endif

  return language;  
}


/* -------------------------------------------------------------------------- */
/* -- Replaces all occurrences of <search> in <str> by <replace> ------------ */
/* -------------------------------------------------------------------------- */
void replaceAll(string& str, const string& search, const string& replace)
{
  string::size_type pos = str.find(search);
  while (string::npos != pos)
  {
    str.replace(pos, search.length(), replace);
    pos = str.find(search, pos + replace.length());
  }
}


/* -------------------------------------------------------------------------- */
/* -- Returns the given integer converted to a std::string ------------------ */
/* -------------------------------------------------------------------------- */
string intToString(int value)
{
  ostringstream str;
  str << value;
  return str.str();
}


/* EOF */


/***************************************************************************
 * $RCSfile: tau_instrument.cpp,v $   $Author: geimer $
 * $Revision: 1.68 $   $Date: 2008/12/04 12:24:17 $
 * VERSION_ID: $Id: tau_instrument.cpp,v 1.68 2008/12/04 12:24:17 geimer Exp $
 ***************************************************************************/
