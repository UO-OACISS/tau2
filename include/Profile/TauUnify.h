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

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int Tau_unify_unifyDefinitions();

#ifdef __cplusplus
}
#endif /* __cplusplus */


#ifdef __cplusplus

class EventLister {
private:
  double duration;


public:
  virtual int getNumEvents() = 0;
  virtual const char *getEvent(int id) = 0;
  double getDuration() {
    return duration;
  }
  void setDuration(double duration) {
    this->duration = duration;
  }
};




class FunctionEventLister : public EventLister {
  int getNumEvents() {
    return TheFunctionDB().size();
  }
  const char *getEvent(int id) {
    return TheFunctionDB()[id]->GetName();
  }
};


class AtomicEventLister : public EventLister {
  int getNumEvents() {
    return TheEventDB().size();
  }
  const char *getEvent(int id) {
    return TheEventDB()[id]->GetEventName();
  }
};


int Tau_unify_unifyEvents(EventLister *eventLister);

#endif /* __cplusplus */
