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
// $Log: PyGroups.cpp,v $
// Revision 1.1  2003/02/28 23:26:51  sameer
// Added Python Bindings to TAU [Julian Cummings, Brian Miller].
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

#include "Profile/PyGroups.h"

// Tau includes

#include "Profile/Profiler.h"


char pytau_getProfileGroup__name__[] = "getProfileGroup";
char pytau_getProfileGroup__doc__[] = "retrieve a Tau Profiler group";
PyObject * pytau_getProfileGroup(PyObject *, PyObject * args)
{
    char * name;
    TauGroup_t group;

    // extract group name from Python args
    int ok = PyArg_ParseTuple(args, "s:getProfileGroup", &name);
    if (!ok) {
        return 0;
    }

    // call Tau function to retrieve group
    group = TAU_GET_PROFILE_GROUP(name);

    // return Tau profiler group
    return Py_BuildValue("l", group);
}
   
char pytau_enableGroup__name__[] = "enableGroup";
char pytau_enableGroup__doc__[] = "enable a Tau Profiler group";
PyObject * pytau_enableGroup(PyObject *, PyObject * args)
{
    TauGroup_t group;

    // extract group from Python args
    int ok = PyArg_ParseTuple(args, "l:enableGroup", &group);
    if (!ok) {
        return 0;
    }

    // call Tau function to enable group
    TAU_ENABLE_GROUP(group);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
   
char pytau_disableGroup__name__[] = "disableGroup";
char pytau_disableGroup__doc__[] = "disable a Tau Profiler group";
PyObject * pytau_disableGroup(PyObject *, PyObject * args)
{
    TauGroup_t group;

    // extract group from Python args
    int ok = PyArg_ParseTuple(args, "l:disableGroup", &group);
    if (!ok) {
        return 0;
    }

    // call Tau function to disable group
    TAU_DISABLE_GROUP(group);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
   
char pytau_enableGroupName__name__[] = "enableGroupName";
char pytau_enableGroupName__doc__[] = "enable a Tau Profiler group by name";
PyObject * pytau_enableGroupName(PyObject *, PyObject * args)
{
    char * name;

    // extract group name from Python args
    int ok = PyArg_ParseTuple(args, "s:enableGroupName", &name);
    if (!ok) {
        return 0;
    }

    // call Tau function to enable group by name
    TAU_ENABLE_GROUP_NAME(name);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
   
char pytau_disableGroupName__name__[] = "disableGroupName";
char pytau_disableGroupName__doc__[] = "disable a Tau Profiler group by name";
PyObject * pytau_disableGroupName(PyObject *, PyObject * args)
{
    char * name;

    // extract group name from Python args
    int ok = PyArg_ParseTuple(args, "s:disableGroupName", &name);
    if (!ok) {
        return 0;
    }

    // call Tau function to disable group by name
    TAU_DISABLE_GROUP_NAME(name);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
   
char pytau_enableAllGroups__name__[] = "enableAllGroups";
char pytau_enableAllGroups__doc__[] = "enable all Tau Profiler groups";
PyObject * pytau_enableAllGroups(PyObject *, PyObject *)
{
    // call Tau function to enable all groups
    TAU_ENABLE_ALL_GROUPS();

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
   
char pytau_disableAllGroups__name__[] = "disableAllGroups";
char pytau_disableAllGroups__doc__[] = "disable all Tau Profiler groups";
PyObject * pytau_disableAllGroups(PyObject *, PyObject *)
{
    // call Tau function to disable all groups
    TAU_DISABLE_ALL_GROUPS();

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
   
char pytau_enableInstrumentation__name__[] = "enableInstrumentation";
char pytau_enableInstrumentation__doc__[] = "enable all Tau Profiler instrumentation";
PyObject * pytau_enableInstrumentation(PyObject *, PyObject *)
{
    // call Tau function to enable instrumentation
    TAU_ENABLE_INSTRUMENTATION();

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
   
char pytau_disableInstrumentation__name__[] = "disableInstrumentation";
char pytau_disableInstrumentation__doc__[] = "disable all Tau Profiler instrumentation";
PyObject * pytau_disableInstrumentation(PyObject *, PyObject *)
{
    // call Tau function to disable instrumentation
    TAU_DISABLE_INSTRUMENTATION();

    // return
    Py_INCREF(Py_None);
    return Py_None;
}
 
// version
// $Id: PyGroups.cpp,v 1.1 2003/02/28 23:26:51 sameer Exp $

// End of file
  
