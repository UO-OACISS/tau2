/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************/

#include <pdbAll.h>
#include <string>
using std::string;

enum instrumentKind_t { TAU_LOOPS, TAU_LINE, TAU_ROUTINE_ENTRY, TAU_ROUTINE_EXIT, TAU_NOT_SPECIFIED};

/* For C instrumentation */
enum itemKind_t { ROUTINE, BODY_BEGIN, FIRST_EXECSTMT, BODY_END, RETURN, EXIT, INSTRUMENTATION_POINT};
enum tau_language_t { tau_c, tau_cplusplus, tau_fortran };

struct itemRef {
  itemRef(const pdbItem *i, bool isT);
  itemRef(const pdbItem *i, itemKind_t k, int l, int c);
  itemRef(const pdbItem *i, itemKind_t k, int l, int c, string code);
  itemRef(const pdbItem *i, bool isT, int l, int c);
  const pdbItem *item;
  itemKind_t kind; /* For C instrumentation */ 
  bool     isTarget;
  int      line;
  int      col;
  string   snippet;
};

extern bool fuzzyMatch(const string& a, const string& b);

/***************************************************************************
 * $RCSfile: tau_datatypes.h,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2005/11/10 02:24:41 $
 * VERSION_ID: $Id: tau_datatypes.h,v 1.1 2005/11/10 02:24:41 sameer Exp $
 ***************************************************************************/
