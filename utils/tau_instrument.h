/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2005  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: tau_instrument.h				  **
**	Description 	: Provides selective instrumentation support in   **
**                        TAU.                                            **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu                  	  **
**	Documentation	:                                                 **
***************************************************************************/

#ifndef _TAU_INSTRUMENT_H_
#define _TAU_INSTRUMENT_H_

#include <string> 
#include <ostream>
#include <vector>
using std::string;
using std::vector;
using std::ostream;

#include "tau_datatypes.h"

/* defines */
#ifdef TAU_WINDOWS
#define TAU_DIR_CHARACTER '\\'
#else
#define TAU_DIR_CHARACTER '/'
#endif /* TAU_WINDOWS */


/* This class contains entries that are read in from the BEGIN_INSTRUMENT_SECTION ... END_INSTRUMENT_SECTION */
#ifndef TAU_DYNINST
class tauInstrument
{
  public:
    enum {
      LA_ANY = PDB::LA_C | PDB::LA_CXX | PDB::LA_FORTRAN
    };

    /* specify everything */
    tauInstrument(string f, string r, int line, string c, instrumentKind_t k);
    tauInstrument(string f, int line, string c, instrumentKind_t k, int lang = LA_ANY);

    /* init code = "init();" lang = "c" */
    tauInstrument(string c, bool cs, instrumentKind_t k, int lang = LA_ANY);

    /* FOR THIS TYPE, you must specify the codeSpecified argument */
    tauInstrument(string r, string c, bool cs, instrumentKind_t k, int lang = LA_ANY);

    /* entry/exit file = "foo.f90" routine = "foo" code = "printf" lang = "fortran" */
    tauInstrument(string f, string r, string c, instrumentKind_t k, int lang = LA_ANY);

    /* loops routine = "foo" */
    tauInstrument(string r, instrumentKind_t k ) ;

    /* loops file = "f1.cpp" routine = "foo" */
    tauInstrument(string f, string r, instrumentKind_t k ) ;

    /* [static/dynamic] [phase/timer] routine="name" */
    tauInstrument(itemQualifier_t q, instrumentKind_t k, string r);

    /* [static/dynamic] [phase/timer] file="fname" line=start to line=stop*/
    tauInstrument(itemQualifier_t q, instrumentKind_t k, string n, string f, int linestart, int linestop );
    /* Destructor */
    ~tauInstrument() ;
    
    /* print all members of this class to the ostream */
    ostream& print(ostream& ostr) const ;
    
    /* routines to access information from the private data members */
    bool getFileSpecified(void);
    string& getFileName(void) ;
    bool getRoutineSpecified(void);
    string& getRoutineName(void); 
    bool getLineSpecified(void) ;
    int getLineNo(void);
    bool getCodeSpecified(void) ;
    string& getCode(void) ;
    string getCode(const pdbLoc& loc, const pdbRoutine* r=NULL, bool isInit=false);
    instrumentKind_t getKind(void) ;
    bool getRegionSpecified(void);
    int getRegionStart(void);
    int getRegionStop(void);
    itemQualifier_t getQualifier(void);
    bool getQualifierSpecified(void);
    bool isActiveForLanguage(PDB::lang_t lang) const;

    /* private data members */
  private:
    string filename;
    bool fileSpecified;
    string routineName;
    bool routineSpecified;
    int lineno; 
    bool lineSpecified; 
    string code; 
    bool codeSpecified;
    instrumentKind_t kind; 
    itemQualifier_t qualifier; 
    bool qualifierSpecified; 
    int regionStart;
    int regionStop;
    bool regionSpecified;
    int language;
};

extern vector<tauInstrument *> instrumentList; 
#endif //TAU_DYNINST

/*
extern int addFileInstrumentationRequests(pdbFile *file, vector<itemRef *>& itemvec);
extern bool isInstrumentListEmpty(void);
*/

/* MACROS */
#define INBUF_SIZE 2048

#endif /* _TAU_INSTRUMENT_H_ */
/***************************************************************************
 * $RCSfile: tau_instrument.h,v $   $Author: scottb $
 * $Revision: 1.7 $   $Date: 2008/10/13 23:00:39 $
 * VERSION_ID: $Id: tau_instrument.h,v 1.7 2008/10/13 23:00:39 scottb Exp $
 ***************************************************************************/
