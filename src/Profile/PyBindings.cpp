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
// $Log: PyBindings.cpp,v $
// Revision 1.4  2008/09/15 23:25:49  sameer
// Added pytau_setNode(<nodeid>) number for the Python API.
//
// Revision 1.3  2007/03/02 02:36:51  amorris
// Added snapshot API for python.
//
// Revision 1.2  2007/03/01 22:17:28  amorris
// Added phase API for python
//
// Revision 1.1  2003/02/28 23:26:50  sameer
// Added Python Bindings to TAU [Julian Cummings, Brian Miller].
//
// Revision 1.3  2002/11/14 02:28:50  cummings
// Added bindings for some new Tau functions that let you access the
// profiling statistics database at run time.
//
// Revision 1.2  2002/01/23 02:47:38  cummings
// Added Python wrappers for new Tau functions enableAllGroups() and
// disableAllGroups(), which will enable or disable profiling for all
// existing profile groups with one function call.  The only exception
// is the group TAU_DEFAULT, which includes main() and cannot be disabled.
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

#include <Python.h>

#include "Profile/PyBindings.h"
#include "Profile/PyDatabase.h"
#include "Profile/PyGroups.h"
#include "Profile/PyTimer.h"


// method table

struct PyMethodDef pytau_methods[] = {

// database

    {pytau_snapshot__name__, pytau_snapshot, METH_VARARGS, pytau_snapshot__doc__},
    {pytau_dbDump__name__, pytau_dbDump, METH_VARARGS, pytau_dbDump__doc__},
    {pytau_dbDumpIncr__name__, pytau_dbDumpIncr, METH_VARARGS, pytau_dbDumpIncr__doc__},
    {pytau_dbPurge__name__, pytau_dbPurge, METH_VARARGS, pytau_dbPurge__doc__},
    {pytau_getFuncNames__name__, pytau_getFuncNames, METH_VARARGS, pytau_getFuncNames__doc__},
    {pytau_dumpFuncNames__name__, pytau_dumpFuncNames, METH_VARARGS, pytau_dumpFuncNames__doc__},
    {pytau_getCounterNames__name__, pytau_getCounterNames, METH_VARARGS, pytau_getCounterNames__doc__},
    {pytau_getFuncVals__name__, pytau_getFuncVals, METH_VARARGS, pytau_getFuncVals__doc__},
    {pytau_dumpFuncVals__name__, pytau_dumpFuncVals, METH_VARARGS, pytau_dumpFuncVals__doc__},
    {pytau_dumpFuncValsIncr__name__, pytau_dumpFuncValsIncr, METH_VARARGS, pytau_dumpFuncValsIncr__doc__},

// groups

    {pytau_getProfileGroup__name__, pytau_getProfileGroup, METH_VARARGS, pytau_getProfileGroup__doc__},
    {pytau_enableGroup__name__, pytau_enableGroup, METH_VARARGS, pytau_enableGroup__doc__},
    {pytau_disableGroup__name__, pytau_disableGroup, METH_VARARGS, pytau_disableGroup__doc__},
    {pytau_enableGroupName__name__, pytau_enableGroupName, METH_VARARGS, pytau_enableGroupName__doc__},
    {pytau_disableGroupName__name__, pytau_disableGroupName, METH_VARARGS, pytau_disableGroupName__doc__},
    {pytau_enableAllGroups__name__, pytau_enableAllGroups, METH_VARARGS, pytau_enableAllGroups__doc__},
    {pytau_disableAllGroups__name__, pytau_disableAllGroups, METH_VARARGS, pytau_disableAllGroups__doc__},
    {pytau_enableInstrumentation__name__, pytau_enableInstrumentation, METH_VARARGS, pytau_enableInstrumentation__doc__},
    {pytau_disableInstrumentation__name__, pytau_disableInstrumentation, METH_VARARGS, pytau_disableInstrumentation__doc__},
// timers
    {pytau_phase__name__, pytau_phase, METH_VARARGS | METH_KEYWORDS, pytau_phase__doc__}, 
    {pytau_profileTimer__name__, pytau_profileTimer, METH_VARARGS | METH_KEYWORDS, pytau_profileTimer__doc__}, 
    {pytau_start__name__, pytau_start, METH_VARARGS, pytau_start__doc__}, 
    {pytau_stop__name__, pytau_stop, METH_VARARGS, pytau_stop__doc__}, 
// runtime system
    {pytau_setNode__name__, pytau_setNode, METH_VARARGS, pytau_setNode__doc__}, 

// Sentinel
    {0, 0}
};

// version
// $Id: PyBindings.cpp,v 1.4 2008/09/15 23:25:49 sameer Exp $

// End of file
