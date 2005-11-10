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


/* This class contains entries that are read in from the BEGIN_INSTRUMENT_SECTION ... END_INSTRUMENT_SECTION */

class tauInstrument
{
  public:
    /* specify everything */
    tauInstrument(string f, string r, int l, string c, instrumentKind_t k);
    tauInstrument(string f, int l, string c, instrumentKind_t k);
    /* FOR THIS TYPE, you must specify the codeSpecified argument */
    tauInstrument(string r, string c, bool cs, instrumentKind_t k);

    /* entry/exit file = "foo.f90" routine = "foo" code = "printf" */
    tauInstrument(string f, string r, string c, instrumentKind_t k) ;

    /* loops routine = "foo" */
    tauInstrument(string r, instrumentKind_t k ) ;

    /* loops file = "f1.cpp" routine = "foo" */
    tauInstrument(string f, string r, instrumentKind_t k ) ;

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
    instrumentKind_t getKind(void) ;

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
};

extern vector<tauInstrument *> instrumentList; 

/*
extern int addFileInstrumentationRequests(pdbFile *file, vector<itemRef *>& itemvec);
extern bool isInstrumentListEmpty(void);
*/

/* MACROS */
#define INBUF_SIZE 2048

#endif /* _TAU_INSTRUMENT_H_ */
/***************************************************************************
 * $RCSfile: tau_instrument.h,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2005/11/10 02:00:58 $
 * VERSION_ID: $Id: tau_instrument.h,v 1.1 2005/11/10 02:00:58 sameer Exp $
 ***************************************************************************/
