/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************/

#include <string>

enum instrumentKind_t { TAU_LOOPS, TAU_LINE, TAU_ROUTINE_DECL, TAU_ROUTINE_ENTRY, TAU_ROUTINE_EXIT, TAU_ABORT, TAU_NOT_SPECIFIED, TAU_IO, TAU_MEMORY, TAU_TIMER, TAU_PHASE, TAU_INIT, TAU_FORALL, TAU_BARRIER, TAU_FENCE, TAU_NOTIFY };

/* For C instrumentation */
enum itemKind_t { ROUTINE, BODY_BEGIN, FIRST_EXECSTMT, BODY_END, RETURN, EXIT, INSTRUMENTATION_POINT, START_TIMER, STOP_TIMER, START_DO_TIMER, GOTO_STOP_TIMER, START_LOOP_TIMER, STOP_LOOP_TIMER, ALLOCATE_STMT, DEALLOCATE_STMT, IO_STMT };
enum itemAttr_t { BEFORE, AFTER, NOT_APPLICABLE};
enum itemQualifier_t { STATIC, DYNAMIC, NOT_SPECIFIED};
enum tau_language_t { tau_c, tau_cplusplus, tau_fortran, tau_upc };

#ifndef TAU_DYNINST
#include <pdbAll.h>
struct itemRef {
  itemRef(const pdbItem *i, bool isT);
  itemRef(const pdbItem *i, itemKind_t k, int l, int c);
  itemRef(const pdbItem *i, itemKind_t k, int l, int c, std::string code, itemAttr_t);
  //itemRef(const pdbItem *i, itemKind_t k, itemQualifier_t q, std::string name);
  itemRef(const pdbItem *i, bool isT, int l, int c);
  itemRef(const pdbItem *i, itemKind_t k, pdbLoc start, pdbLoc stop);
  const pdbItem *item;
  itemKind_t kind; /* For C instrumentation */ 
  bool     isTarget;
  bool     isDynamic;
  bool     isPhase;
  int      line;
  int      col;
  pdbLoc   begin;
  pdbLoc   end;
  std::string   snippet;
  itemAttr_t attribute;
};
#endif /* TAU_DYNINST */

/***************************************************************************
 * $RCSfile: tau_datatypes.h,v $   $Author: geimer $
 * $Revision: 1.13 $   $Date: 2008/12/11 16:02:22 $
 * VERSION_ID: $Id: tau_datatypes.h,v 1.13 2008/12/11 16:02:22 geimer Exp $
 ***************************************************************************/
