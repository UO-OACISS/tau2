// -*- C++ -*-
//
//-----------------------------------------------------------------------------
//
//                       VTF Development Team
//                       California Institute of Technology
//                       (C) 2002 All Rights Reserved
//
// <LicenseText>
//
//-----------------------------------------------------------------------------
//
// $Log: PyDatabase.cpp,v $
// Revision 1.10  2010/05/12 19:42:17  amorris
// Only call TAU_DB_MERGED_DUMP() if TAU_MPI is defined.
//
// Revision 1.9  2010/03/18 17:36:46  amorris
// Refactoring to implement TAU_DB_MERGED_DUMP and python dbMergeDump call that
// allow the user to control when the merged profile is written.
//
// Revision 1.8  2008/10/24 22:48:18  sameer
// Added a pytau.exit("message") binding for TAU_PROFILE_EXIT(msg).
//
// Revision 1.7  2008/09/15 23:25:49  sameer
// Added pytau_setNode(<nodeid>) number for the Python API.
//
// Revision 1.6  2007/03/02 02:36:51  amorris
// Added snapshot API for python.
//
// Revision 1.5  2006/02/03 03:03:08  amorris
// Fixed error with PyArg_ParseTuple:
//
// Exception exceptions.TypeError: 'function takes exactly 1 argument (0 given)' in 'garbage collection' ignored
// Fatal Python error: unexpected exception during garbage collection
// Aborted (core dumped)
//
// Revision 1.4  2003/07/18 18:48:19  sameer
// Added support for TAU_DB_DUMP(prefix). In python you can optionally specify
// an argument:
// tau.dbDump() --> Calls TAU_DB_DUMP();
// or
// tau.dbDump("prefix") --> Calls TAU_DB_DUMP_PREFIX("prefix") (Kathleen wanted
// prefix to be profile so it'd dump performance data for an application that
// didn't terminate right and she'd use jracy directly). [LLNL]
//
// Revision 1.3  2003/03/20 18:41:05  sameer
// Added TAU_HPUX guards for <limit> header.
//
// Revision 1.2  2003/03/15 01:39:10  sameer
// Added <limits> [HP-UX] and moved funcDB in PyTimer.cpp to inside the profileTimer routine.
//
// Revision 1.1  2003/02/28 23:26:51  sameer
// Added Python Bindings to TAU [Julian Cummings, Brian Miller].
//
// Revision 1.2  2002/11/14 02:28:50  cummings
// Added bindings for some new Tau functions that let you access the
// profiling statistics database at run time.
//
// Revision 1.1  2002/01/16 02:05:07  cummings
// Original source and build procedure files for Python bindings of
// TAU runtime API.  These bindings allow you to do some rudimentary
// things from the Python script, such as enable/disable all Tau
// instrumentation, enable/disable a particular Tau profile group,
// and dump or purge the current Tau statistics.  Still to come are
// bindings for creating and using Tau global timers and user events.
//
// 

#ifdef TAU_HPUX
#include <limits>
#endif /* TAU_HPUX */
#include <Python.h>

#include "Profile/PyDatabase.h"

// Tau includes

#include "Profile/Profiler.h"


char pytau_snapshot__name__[] = "snapshot";
char pytau_snapshot__doc__[] = "take a snapshot of the current profile";
PyObject * pytau_snapshot(PyObject *self, PyObject *args) { 

  char *name = NULL;
  int number = -1;

    if (PyArg_ParseTuple(args, "s|i", &name, &number)) {
      if (number == -1) {
	TAU_PROFILE_SNAPSHOT(name);
      } else {
	TAU_PROFILE_SNAPSHOT_1L(name,number);
      }
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_dbMergeDump__name__[] = "dbMergeDump";
char pytau_dbMergeDump__doc__[] = "dump the Tau Profiler statistics using MPI to merge";
PyObject * pytau_dbMergeDump(PyObject *self, PyObject *args)
{ 
    char *prefix = "dump";
    int len = 4;

    // Check to see if a prefix is specified
    if (PyArg_ParseTuple(args, "|s", &prefix, &len))
    {
      // extracted the prefix, call dump routine
#ifdef DEBUG
      printf("dbMergeDump: extracted prefix = %s, len = %d\n", prefix, len);
#endif /* DEBUG */
#ifdef TAU_MPI
      TAU_DB_MERGED_DUMP();
#endif
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_dbDump__name__[] = "dbDump";
char pytau_dbDump__doc__[] = "dump the Tau Profiler statistics";
PyObject * pytau_dbDump(PyObject *self, PyObject *args)
{ 
    char *prefix = "dump";
    int len = 4;

    // Check to see if a prefix is specified
    if (PyArg_ParseTuple(args, "|s", &prefix, &len))
    {
      // extracted the prefix, call dump routine
#ifdef DEBUG
      printf("dbDump: extracted prefix = %s, len = %d\n", prefix, len);
#endif /* DEBUG */
      TAU_DB_DUMP_PREFIX(prefix);
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_dbDumpIncr__name__[] = "dbDumpIncr";
char pytau_dbDumpIncr__doc__[] = "incremental dump of the Tau Profiler statistics";
PyObject * pytau_dbDumpIncr(PyObject *, PyObject *)
{
    // call Tau function to dump statistics incrementally
    TAU_DB_DUMP_INCR();

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_dbPurge__name__[] = "dbPurge";
char pytau_dbPurge__doc__[] = "purge the Tau Profiler statistics";
PyObject * pytau_dbPurge(PyObject *, PyObject *)
{
    // call Tau function to purge statistics
    TAU_DB_PURGE();

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_exit__name__[] = "exit";
char pytau_exit__doc__[] = "close the callstack and flush the Tau statistics";
PyObject * pytau_exit(PyObject *self, PyObject *args)
{
    char *message = "pythonexit";
    int len = 10;

    // Check to see if a prefix is specified
    if (PyArg_ParseTuple(args, "|s", &message, &len))
    {
      // extracted the prefix, call dump routine
#ifdef DEBUG
      printf("dbDump: extracted message = %s, len = %d\n", message, len);
#endif /* DEBUG */
    }
    TAU_PROFILE_EXIT(message);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_getFuncNames__name__[] = "getFuncNames";
char pytau_getFuncNames__doc__[] = "get list of profiled functions";
PyObject * pytau_getFuncNames(PyObject *, PyObject *)
{
    PyObject * pyFuncList;
    const char** functionList;
    int numOfFunctions;

    // call Tau function to get function names
    TAU_GET_FUNC_NAMES(functionList, numOfFunctions);

    // build Python sequence of function names
    pyFuncList = PyTuple_New(numOfFunctions);
    for (int i=0; i<numOfFunctions; ++i)
        PyTuple_SET_ITEM(pyFuncList,i,PyString_FromString(functionList[i]));

    // return
    return pyFuncList;
}

char pytau_dumpFuncNames__name__[] = "dumpFuncNames";
char pytau_dumpFuncNames__doc__[] = "dump the list of profiled functions to a file";
PyObject * pytau_dumpFuncNames(PyObject *, PyObject *)
{
    // call Tau function to dump function names
    TAU_DUMP_FUNC_NAMES();

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_getCounterNames__name__[] = "getCounterNames";
char pytau_getCounterNames__doc__[] = "get list of monitored counters";
PyObject * pytau_getCounterNames(PyObject *, PyObject *)
{
    PyObject * pyCtrList;
    const char** counterList;
    int numOfCounters;

    // call Tau function to get counter names
    TAU_GET_COUNTER_NAMES(counterList, numOfCounters);

    // build Python sequence of counter names
    pyCtrList = PyTuple_New(numOfCounters);
    for (int i=0; i<numOfCounters; ++i)
        PyTuple_SET_ITEM(pyCtrList,i,PyString_FromString(counterList[i]));

    // return
    return pyCtrList;
}

char pytau_getFuncVals__name__[] = "getFuncVals";
char pytau_getFuncVals__doc__[] = "get statistics for the given functions";
PyObject * pytau_getFuncVals(PyObject *, PyObject * args)
{
    PyObject * pyFuncNames;
    PyObject * pyCounterNames;
    PyObject * pyExcVals;
    PyObject * pyIncVals;
    PyObject * pyNumCalls;
    PyObject * pyNumSubroutines;
    const char** inFuncs;
    int numOfFunctions;
    double** counterExclusiveValues;
    double** counterInclusiveValues;
    int* numOfCalls;
    int* numOfSubroutines;
    const char** counterNames;
    int numOfCounters;

    // extract function list from Python args
    int ok = PyArg_ParseTuple(args, "O:getFuncVals", pyFuncNames);
    if (!ok) {
        return 0;
    }

    // check that we received a sequence
    if (!PySequence_Check(pyFuncNames)) {
        PyErr_SetString(PyExc_TypeError, "Function names list argument must be a sequence");
        return 0;
    }

    // convert to C types
    numOfFunctions = PySequence_Length(pyFuncNames);
    inFuncs = new const char*[numOfFunctions];
    for (int i=0; i<numOfFunctions; ++i)
        inFuncs[i] = PyString_AsString(PySequence_GetItem(pyFuncNames,i));

    // call Tau function to get function statistics
    TAU_GET_FUNC_VALS(inFuncs, numOfFunctions,
        counterExclusiveValues, counterInclusiveValues, numOfCalls,
        numOfSubroutines, counterNames, numOfCounters);

    // build Python sequence of results
    pyExcVals = PyTuple_New(numOfFunctions);
    pyIncVals = PyTuple_New(numOfFunctions);
    pyNumCalls = PyTuple_New(numOfFunctions);
    pyNumSubroutines = PyTuple_New(numOfFunctions);
    for (int i=0; i<numOfFunctions; ++i) {
        PyObject* pyExcSeq = PyTuple_New(numOfCounters);
        PyObject* pyIncSeq = PyTuple_New(numOfCounters);
        for (int j=0; j<numOfCounters; ++j) {
            PyTuple_SET_ITEM(pyExcSeq,j,
                             PyFloat_FromDouble(counterExclusiveValues[i][j]));
            PyTuple_SET_ITEM(pyIncSeq,j,
                             PyFloat_FromDouble(counterInclusiveValues[i][j]));
        }
        PyTuple_SET_ITEM(pyExcVals,i,pyExcSeq);
        PyTuple_SET_ITEM(pyIncVals,i,pyIncSeq);
        PyTuple_SET_ITEM(pyNumCalls,i,PyInt_FromLong(numOfCalls[i]));
        PyTuple_SET_ITEM(pyNumSubroutines,i,
                         PyInt_FromLong(numOfSubroutines[i]));
    }
    pyCounterNames = PyTuple_New(numOfCounters);
    for (int j=0; j<numOfCounters; ++j)
        PyTuple_SET_ITEM(pyCounterNames,j,
                         PyString_FromString(counterNames[j]));

    // return
    delete [] inFuncs;
    return Py_BuildValue("OOOOO",
        pyExcVals,pyIncVals,pyNumCalls,pyNumSubroutines,pyCounterNames);
}


char pytau_dumpFuncVals__name__[] = "dumpFuncVals";
char pytau_dumpFuncVals__doc__[] = "dump statistics for the given functions to a file";
PyObject * pytau_dumpFuncVals(PyObject *, PyObject * args)
{
    PyObject * pyFuncNames;
    const char** inFuncs;
    int numOfFunctions;

    // extract function list from Python args
    int ok = PyArg_ParseTuple(args, "O:dumpFuncVals", pyFuncNames);
    if (!ok) {
        return 0;
    }

    // check that we received a sequence
    if (!PySequence_Check(pyFuncNames)) {
        PyErr_SetString(PyExc_TypeError, "Function names list argument must be a sequence");
        return 0;
    }

    // convert to C types
    numOfFunctions = PySequence_Length(pyFuncNames);
    inFuncs = new const char*[numOfFunctions];
    for (int i=0; i<numOfFunctions; ++i)
        inFuncs[i] = PyString_AsString(PySequence_GetItem(pyFuncNames,i));

    // call Tau function to dump function statistics
    TAU_DUMP_FUNC_VALS(inFuncs, numOfFunctions);

    // return
    delete [] inFuncs;
    Py_INCREF(Py_None);
    return Py_None;
}


char pytau_dumpFuncValsIncr__name__[] = "dumpFuncValsIncr";
char pytau_dumpFuncValsIncr__doc__[] = "dump incremental statistics for the given functions to a file";
PyObject * pytau_dumpFuncValsIncr(PyObject *, PyObject * args)
{
    PyObject * pyFuncNames;
    const char** inFuncs;
    int numOfFunctions;

    // extract function list from Python args
    int ok = PyArg_ParseTuple(args, "O:dumpFuncValsIncr", pyFuncNames);
    if (!ok) {
        return 0;
    }

    // check that we received a sequence
    if (!PySequence_Check(pyFuncNames)) {
        PyErr_SetString(PyExc_TypeError, "Function names list argument must be a sequence");
        return 0;
    }

    // convert to C types
    numOfFunctions = PySequence_Length(pyFuncNames);
    inFuncs = new const char*[numOfFunctions];
    for (int i=0; i<numOfFunctions; ++i)
        inFuncs[i] = PyString_AsString(PySequence_GetItem(pyFuncNames,i));

    // call Tau function to dump function incremental statistics
    TAU_DUMP_FUNC_VALS_INCR(inFuncs, numOfFunctions);

    // return
    delete [] inFuncs;
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_setNode__name__[] = "setNode";
char pytau_setNode__doc__[] = "set node number for genrating profile.<node>.0.0";
PyObject * pytau_setNode(PyObject *self, PyObject *args)
{
    int node_number;

    node_number = 0;

    // extract function list from Python args
    int ok = PyArg_ParseTuple(args, "i:nodeNumber", &node_number);
    if (!ok) {
        return 0;
    }

#ifdef DEBUG_PROF
    printf("pytau_setnode: %d\n", node_number);
#endif /* DEBUG_PROF */
    TAU_PROFILE_SET_NODE(node_number);


    Py_INCREF(Py_None);
    return Py_None;
}

// version
// $Id: PyDatabase.cpp,v 1.10 2010/05/12 19:42:17 amorris Exp $

// End of file
  
