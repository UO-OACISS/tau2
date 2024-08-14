// -*- C++ -*-
//
//-----------------------------------------------------------------------------
//
//                       TAU Development Team
//                       University of Oregon, Los Alamos National Laboratory,
//                       FZJ Germany
//                       (C) 2003 All Rights Reserved
//
// <LicenseText>
//
//-----------------------------------------------------------------------------
//
// $Log: PyTimer.cpp,v $
// Revision 1.11  2009/03/26 19:15:39  sameer
// Updated the start/stop calls with the additional tid argument. Before:
//
// Tau_start_timer(void *timer, int phase)
// {
//   int tid = RtsLayer::myThread();
//
//   p->Start();  // Start internally calls tid = RtsLayer::myThread()
// }
//
// After:
//
// Tau_start_timer(void *timer, int phase, int tid)
// {
//   // int tid  = RtsLayer::myThread();
//   p->Start(tid);
// }
//
// This is used extensively in the new TASK API. See examples/profilercreate/taskc++.
//
// Revision 1.10  2009/02/24 21:30:23  amorris
// Getting rid of Profiler::CurrentProfiler, it doesn't make sense to maintain
// this linked list if we have an explicit data structure for the callstack.  It's
// replaced with a routine: TauInternal_CurrentProfiler(tid)
//
// Revision 1.9  2009/01/15 02:47:23  amorris
// Changes for C++ measurement API
//
// Revision 1.8  2008/10/16 22:58:24  amorris
//
// Revision 1.7  2007/03/02 02:36:24  amorris
// Made explicit the phase calls, true and false.
//
// Revision 1.6  2007/03/01 22:17:28  amorris
// Added phase API for python
//
// Revision 1.5  2007/03/01 02:45:39  amorris
// The map for reusing timers was not taking the 'type' into account, so when
// different routines with the same name occurred, they were both assigned to the
// same timer with the file/line information of whichever occurred first.  We now
// hash on the combined name+type.
//
// Revision 1.4  2003/03/20 18:41:05  sameer
// Added TAU_HPUX guards for <limit> header.
//
// Revision 1.3  2003/03/15 01:49:13  sameer
// Python bindings and wish fix.
//
// Revision 1.2  2003/03/15 01:39:11  sameer
// Added <limits> [HP-UX] and moved funcDB in PyTimer.cpp to inside the profileTimer routine.
//
// Revision 1.1  2003/02/28 23:26:52  sameer
// Added Python Bindings to TAU [Julian Cummings, Brian Miller].
//
//
// 

#include <stdio.h>
#include <Python.h>
#include <string>
#include <map>

#include <Profile/Profiler.h>

#ifdef TAU_HPUX
#include <limits>
#endif /* TAU_HPUX */

using namespace std;

extern "C" int Tau_is_shutdown(void);

// tells whether a FunctionInfo object is a phase or not
struct PhaseMap : public TAU_HASH_MAP<int, bool>
{
  virtual ~PhaseMap() {
    Tau_destructor_trigger();
  }
};

struct PyFunctionDB : public TAU_HASH_MAP<string, int>
{
  virtual ~PyFunctionDB() {
    Tau_destructor_trigger();
  }
};

struct UserEventNameMap : public TAU_HASH_MAP<string, int>
{
  virtual ~UserEventNameMap() {
    Tau_destructor_trigger();
  }
};

struct UserEventDB : public TAU_HASH_MAP<int, tau::TauUserEvent *>
{
  virtual ~UserEventDB() {
    Tau_destructor_trigger();
  }
};

PhaseMap & ThePhaseMap()
{
  static PhaseMap map;
  return map;
}

PyFunctionDB & ThePyFunctionDB()
{
  static PyFunctionDB db;
  return db;
}

UserEventNameMap & TheUserEventNameMap() {
  static UserEventNameMap uenm;
  return uenm;
}

UserEventDB & TheUserEventDB() {
  static UserEventDB uedb;
  return uedb;
}


///////////////////////////////////////////////////////////////////////////////
// Extract name, type and group strings and return the id of the routine
///////////////////////////////////////////////////////////////////////////////
static PyObject * createTimer(PyObject * self, PyObject * args, PyObject * kwargs, bool phase)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  static char * argnames[] = { "name", "type", "group", NULL };

  int tauid = 0;
  char * name = "None";
  char * type = "";
  char * group = "TAU_PYTHON";

  // Get Python arguments 
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sss", argnames, &name, &type, &group)) {
    return NULL;
  }

  // Format function name
  const auto len = strlen(name) + strlen(type) + 5;
  char * buff = new char[len];
  snprintf(buff, len,  "%s %s", name, type);
  string functionName(buff);
  delete[] buff;

  PyFunctionDB & funcDB = ThePyFunctionDB();
  PyFunctionDB::iterator it = funcDB.find(functionName);

  if (it != funcDB.end()) {
    tauid = it->second;
  } else {
    if (phase) {
      // Add TAU_PHASE to the group
      group = Tau_phase_enable(group);
    }

    TauGroup_t groupid = RtsLayer::getProfileGroup(group);
    FunctionInfo * f = new FunctionInfo(functionName, "", groupid, group, true);
    tauid = TheFunctionDB().size() - 1;
    funcDB[functionName] = tauid;
    ThePhaseMap()[tauid] = phase;
  }

  return Py_BuildValue("i", tauid);
}

static PyObject * createUserEvent(PyObject * self, PyObject * args) {
  TauInternalFunctionGuard protects_this_function;

  char * name = "None";
  int tauid = 0;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    printf("createUserEvent: Couldn't Parse the tuple!\n");
    return NULL;
  }

  UserEventNameMap & nameMap = TheUserEventNameMap();
  UserEventNameMap::iterator it = nameMap.find(name);
  if(it != nameMap.end()) {
    tauid = it->second;
  } else {
    tauid = nameMap.size() - 1;
    nameMap[name] = tauid;
    tau::TauUserEvent * ue = new tau::TauUserEvent(name);
    TheUserEventDB()[tauid] = ue;
  }

  return Py_BuildValue("i", tauid);
}

static int tau_check_and_set_nodeid(void)
{
  if (RtsLayer::myNode() == -1) {
#ifndef TAU_MPI
    TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI */
  }
  return 0;
}


///////////////////////////////////////////////////////////////////////////////
// create non-phase timer
///////////////////////////////////////////////////////////////////////////////
char pytau_profileTimer__name__[] = "profileTimer";
char pytau_profileTimer__doc__[] = "access or create a TAU timer";
extern "C" PyObject * pytau_profileTimer(PyObject *self, PyObject *args, PyObject *kwargs)
{
  return createTimer(self, args, kwargs, false);
}

///////////////////////////////////////////////////////////////////////////////
// create phase timer
///////////////////////////////////////////////////////////////////////////////
char pytau_phase__name__[] = "phase";
char pytau_phase__doc__[] = "access or create a TAU phase";
extern "C" PyObject * pytau_phase(PyObject *self, PyObject *args, PyObject *kwargs)
{
  return createTimer(self, args, kwargs, true);
}

///////////////////////////////////////////////////////////////////////////////
// start timer
///////////////////////////////////////////////////////////////////////////////
char pytau_start__name__[] = "start";
char pytau_start__doc__[] = "start a TAU timer";
extern "C" PyObject * pytau_start(PyObject *self, PyObject *args)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  int id;
  if (!PyArg_ParseTuple(args, "i", &id)) {
    printf("Couldn't Parse the tuple!\n");
    return NULL;
  }
  FunctionInfo * f = TheFunctionDB()[id];

  int phase = ThePhaseMap()[id] ? 1 : 0;
  Tau_start_timer(f, phase, RtsLayer::myThread());

  Py_INCREF (Py_None);
  return Py_None;
}

///////////////////////////////////////////////////////////////////////////////
// stop timer
///////////////////////////////////////////////////////////////////////////////
char pytau_stop__name__[] = "stop";
char pytau_stop__doc__[] = "stop a TAU timer";
extern "C" PyObject * pytau_stop(PyObject *self, PyObject *args)
{
  if (!Tau_is_shutdown()) {
    int tid = RtsLayer::myThread();
    Profiler * p = TauInternal_CurrentProfiler(tid);
    if (!p) {
      fprintf(stderr, "TAU: pytau_stop: Stack error: profiler is NULL!\n");
      return NULL;
    }

    Tau_stop_timer(p->ThisFunction, tid);
    Py_INCREF (Py_None);
    return Py_None;
  }
  return NULL;
}


///////////////////////////////////////////////////////////////////////////////
// register user event
///////////////////////////////////////////////////////////////////////////////
char pytau_registerEvent__name__[] = "registerEvent";
char pytau_registerEvent__doc__[]  = "register a new TAU user event";
extern "C" PyObject * pytau_registerEvent(PyObject * self, PyObject * args) {
  return createUserEvent(self, args);
}


///////////////////////////////////////////////////////////////////////////////
// trigger user event
///////////////////////////////////////////////////////////////////////////////
char pytau_event__name__[] = "event";
char pytau_event__doc__[]  = "trigger an existing TAU user event";
extern "C" PyObject * pytau_event(PyObject * self, PyObject * args) {
  TauInternalFunctionGuard protects_this_function;
  
  int tid = RtsLayer::myThread();
  int id;
  double data;
  if(!PyArg_ParseTuple(args, "id", &id, &data)) {
    printf("pytau_event: Couldn't Parse the tuple!\n");
    return NULL;
  }

  tau::TauUserEvent * ue = TheUserEventDB()[id];
  ue->TriggerEvent(data, tid);
  Py_INCREF(Py_None);
  return Py_None;
}

// version
// $Id: PyTimer.cpp,v 1.11 2009/03/26 19:15:39 sameer Exp $

// End of file

