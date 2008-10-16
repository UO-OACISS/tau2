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
// Revision 1.8  2008/10/16 22:58:24  amorris
// The current Profiler object was not being deleted in stop, causing a big memory leak.  Try GPAW now Sameer!
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

#ifdef TAU_HPUX
#include <limits>
#endif /* TAU_HPUX */
#include <Python.h>

// Tau includes

#include "Profile/Profiler.h"
#include <stdio.h>
#include <map>
using namespace std;



// Utility routines
struct ltstr{
	  bool operator()(const char* s1, const char* s2) const{
		      return strcmp(s1, s2) < 0;
		        }//operator()
};


// tells whether a FunctionInfo object is a phase or not
static map<int, bool> phaseMap;

static PyObject *createTimer(PyObject *self, PyObject *args, PyObject *kwargs, bool phase) {
  // Extract name, type and group strings and return the id of the routine
  int tauid = 0; 
  char *name = "None";
  char *type = "";
  char *group = "TAU_PYTHON"; 
  static char *argnames[] = { "name", "type", "group", NULL}; 
  /* GLOBAL database of function names */
  static map<const char*, int, ltstr> funcDB;
  map<const char *, int, ltstr>::iterator it;
  
  // Get Python arguments 
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sss", argnames, 
				   &name, &type, &group)) {
    return NULL;
  }
#ifdef DEBUG
  printf("Got Name = %s, Type = %s, Group = %s, tauid = %d\n", name, type, group, tauid);
#endif /* DEBUG */
  
  char * functionName = new char[strlen(name) + strlen(type) +5]; // create a new storage - STL req.
  sprintf (functionName,"%s %s",name,type);
  if (( it = funcDB.find((const char *)functionName)) != funcDB.end()) {
#ifdef DEBUG
    printf("Found the name %s\n", functionName); 
#endif /* DEBUG */
    
    tauid = (*it).second;
    delete functionName; // don't need this if its already there.
  } else {

    if (phase) {
      // Add TAU_PHASE to the group
      group = Tau_phase_enable(group);
    }
    TauGroup_t groupid = RtsLayer::getProfileGroup(group);
    FunctionInfo *f = new FunctionInfo(functionName, "", groupid, group, true); 
    tauid = TheFunctionDB().size() - 1;
    // These two need to be an atomic operation if threads are involved. LockDB happens
    // inside FunctionInfoInit()
    
    // Store the id in our map
    funcDB[(const char *)functionName] = tauid; 
    // Do not delete functionName, STL requirement!

    if (phase) {
      phaseMap[tauid] = true;
    } else {
      phaseMap[tauid] = false;
    }
  }

  return Py_BuildValue("i", tauid);
}


char pytau_profileTimer__name__[] = "profileTimer";
char pytau_profileTimer__doc__[] = "access or create a TAU timer";
extern "C"
PyObject * pytau_profileTimer(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // create non-phase timer
  return createTimer(self, args, kwargs, false);
}

char pytau_phase__name__[] = "phase";
char pytau_phase__doc__[] = "access or create a TAU phase";
extern "C"
PyObject * pytau_phase(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // create phase timer
  return createTimer(self, args, kwargs, true);
}

char pytau_start__name__[] = "start";
char pytau_start__doc__[] = "start a TAU timer";
extern "C"
PyObject * pytau_start(PyObject *self, PyObject *args)
{
    int id;
    if (!PyArg_ParseTuple(args, "i", &id)) {
      printf("Couldn't Parse the tuple!\n"); 
      return NULL;
    }
    
    FunctionInfo *f = TheFunctionDB()[id];

#ifdef DEBUG
    printf("Received timer Named %s, id = %d\n", f->GetName(), id);
#endif /* DEBUG */
    int tid = RtsLayer::myThread();
    /* Get the FunctionInfo object */
    Profiler *p = new Profiler(f,  f != (FunctionInfo *) 0 ? f->GetProfileGroup() : 
      TAU_DEFAULT, true, tid);
    if (p == (Profiler *) NULL) { 
      printf("ERROR: Out of Memory in pytau_start! new returns NULL!\n");
      return NULL;
    }

#ifdef TAU_PROFILEPHASE
    bool isPhase = phaseMap[id];
    if (isPhase) {
      p->SetPhase(true);
    } else {
      p->SetPhase(false);
    }
#endif

    p->Start(tid);
    Py_INCREF(Py_None);
    return Py_None;

}

int tau_check_and_set_nodeid(void)
{
    if (RtsLayer::myNode() == -1)
    {
#ifndef TAU_MPI
      TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI */
    }
    return 0;
}

char pytau_stop__name__[] = "stop";
char pytau_stop__doc__[] = "stop a TAU timer";
extern "C"
PyObject * pytau_stop(PyObject *self, PyObject *args)
{
    int tid = RtsLayer::myThread();
    static int taunode = tau_check_and_set_nodeid();

    Profiler *p = Profiler::CurrentProfiler[tid];
    if (p != (Profiler *) NULL)
    {
#ifdef DEBUG
      printf("Looking at function %s\n", p->ThisFunction->GetName());
#endif /* DEBUG */
      p->Stop();
      // It was stopped properly
      delete p;
      Py_INCREF(Py_None);
      return Py_None;
    }
    else
    {
      printf("pytau_stop: Stack error. Profiler is NULL!");
      return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

// version
// $Id: PyTimer.cpp,v 1.8 2008/10/16 22:58:24 amorris Exp $

// End of file
  
