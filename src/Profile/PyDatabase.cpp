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

#include <limits>
#include <Python.h>

#include "Profile/PyDatabase.h"

// Tau includes

#include "Profile/Profiler.h"


char pytau_dbDump__name__[] = "dbDump";
char pytau_dbDump__doc__[] = "dump the Tau Profiler statistics";
PyObject * pytau_dbDump(PyObject *, PyObject *)
{
    // call Tau function to dump statistics
    TAU_DB_DUMP();

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


// version
// $Id: PyDatabase.cpp,v 1.2 2003/03/15 01:39:10 sameer Exp $

// End of file
  
