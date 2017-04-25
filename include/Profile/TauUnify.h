/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File            : TauUnify.h                                       **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
**	Documentation	: See http://tau.uoregon.edu                       **
**                                                                         **
**      Description     : Event unification                                **
**                                                                         **
****************************************************************************/

#ifndef _TAU_UNIFY_H_
#define _TAU_UNIFY_H_


/** Unification object containing the local -> global mapping table */
typedef struct {
  /** the number of local items */
  int localNumItems;

  /** the number of global items */
  int globalNumItems;

  /** the global identifiers, valid only on rank 0 */
  char **globalStrings;

  /** the local sort map */
  int *sortMap;

  /** the local to global mapping table */
  int *mapping;
} Tau_unify_object_t;


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* external "C" definitions */
int Tau_unify_unifyDefinitions_MPI();
int Tau_unify_unifyDefinitions_SHMEM();
Tau_unify_object_t *Tau_unify_getFunctionUnifier();
Tau_unify_object_t *Tau_unify_getAtomicUnifier();

#ifdef __cplusplus
}
#endif /* __cplusplus */


#ifdef __cplusplus

#include <sstream>

// Putting "using namespace" statements in header files can create ambiguity
// between user-defined symbols and std symbols, creating unparsable code
// or even changing the behavior of user codes.  This is also widely considered
// to be bad practice.  Here's a code PDT can't parse because of this line:
//   EX: #include <complex>
//   EX: typedef double real;
//
//using namespace std;

/** EventLister interface class */
class EventLister {

private:
  /** duration of merge operation */
  double duration;

public:

  virtual ~EventLister(void)
  { }

  /** retuns the number of events */
  virtual int getNumEvents() = 0;

  /** returns an event identifier */
  virtual const char *getEvent(int id) = 0;

  /** gets the duration of the merge operation */
  double getDuration() {
    return duration;
  }

  /** sets the duration of the merge operation */
  void setDuration(double duration) {
    this->duration = duration;
  }
};


/** Adapter class for the interval event database */
class FunctionEventLister : public EventLister {
  int getNumEvents() {
    return TheFunctionDB().size();
  }
  const char *getEvent(int id) {
    return TheFunctionDB()[id]->GetFullName();
  }
};

/** Adapter class for the atomic event database */
class AtomicEventLister : public EventLister {
  int getNumEvents() {
    return tau::TheEventDB().size();
  }
  const char *getEvent(int id) {
    return tau::TheEventDB()[id]->GetName().c_str();
  }
};


/** Using MPI, unify events for a given EventLister */
Tau_unify_object_t *Tau_unify_unifyEvents_MPI(EventLister *eventLister);

/** Using SHMEM, unify events for a given EventLister */
Tau_unify_object_t *Tau_unify_unifyEvents_SHMEM(EventLister *eventLister);

#endif /* __cplusplus */


#endif /* _TAU_UNIFY_H_ */
