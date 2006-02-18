/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************/

#include <string>
using std::string;

enum instrumentKind_t { TAU_LOOPS, TAU_LINE, TAU_ROUTINE_ENTRY, TAU_ROUTINE_EXIT, TAU_NOT_SPECIFIED};

/* For C instrumentation */
enum itemKind_t { ROUTINE, BODY_BEGIN, FIRST_EXECSTMT, BODY_END, RETURN, EXIT, INSTRUMENTATION_POINT};
enum itemAttr_t { BEFORE, AFTER, NOT_APPLICABLE};
enum tau_language_t { tau_c, tau_cplusplus, tau_fortran };

#ifndef TAU_DYNINST
#include <pdbAll.h>
struct itemRef {
  itemRef(const pdbItem *i, bool isT);
  itemRef(const pdbItem *i, itemKind_t k, int l, int c);
  itemRef(const pdbItem *i, itemKind_t k, int l, int c, string code, itemAttr_t);
  itemRef(const pdbItem *i, bool isT, int l, int c);
  const pdbItem *item;
  itemKind_t kind; /* For C instrumentation */ 
  bool     isTarget;
  int      line;
  int      col;
  string   snippet;
  itemAttr_t attribute;
};
#endif /* TAU_DYNINST */

extern bool fuzzyMatch(const string& a, const string& b);

/***************************************************************************
 * $RCSfile: tau_datatypes.h,v $   $Author: sameer $
 * $Revision: 1.3 $   $Date: 2006/02/18 04:18:41 $
 * VERSION_ID: $Id: tau_datatypes.h,v 1.3 2006/02/18 04:18:41 sameer Exp $
 ***************************************************************************/
