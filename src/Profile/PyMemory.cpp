// -*- C++ -*-
//
//-----------------------------------------------------------------------------
//
//                       TAU Development Team
//                       University of Oregon, Los Alamos National Laboratory,
//                       FZJ Germany
//                       (C) 2008 All Rights Reserved
//
// <LicenseText>
//
//-----------------------------------------------------------------------------
//

#ifdef TAU_HPUX
#include <limits>
#endif /* TAU_HPUX */
#include <Python.h>
// Tau includes

#include "Profile/Profiler.h"
#include <stdio.h>

char pytau_trackMemory__name__[] = "trackMemory";
char pytau_trackMemory__doc__[] = "track heap memory utilization";
extern "C"
PyObject * pytau_trackMemory(PyObject *self, PyObject *args)
{
  
    TAU_TRACK_MEMORY();
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_trackMemoryHeadroom__name__[] = "trackMemoryHeadroom";
char pytau_trackMemoryHeadroom__doc__[] = "track memory headroom available to grow";
extern "C"
PyObject * pytau_trackMemoryHeadroom(PyObject *self, PyObject *args)
{
  
    TAU_TRACK_MEMORY_HEADROOM();
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_trackMemoryHeadroomHere__name__[] = "trackMemoryHeadroomHere";
char pytau_trackMemoryHeadroomHere__doc__[] = "track memory headroom available to grow at a given location in the source code";
extern "C"
PyObject * pytau_trackMemoryHeadroomHere(PyObject *self, PyObject *args)
{
  
    TAU_TRACK_MEMORY_HEADROOM_HERE();
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_trackMemoryHere__name__[] = "trackMemoryHere";
char pytau_trackMemoryHere__doc__[] = "track memory available at a given location in the source code";
extern "C"
PyObject * pytau_trackMemoryHere(PyObject *self, PyObject *args)
{
  
    TAU_TRACK_MEMORY_HERE();
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_enableTrackingMemory__name__[] = "enableTrackingMemory";
char pytau_enableTrackingMemory__doc__[] = "enable tracking memory";
extern "C"
PyObject * pytau_enableTrackingMemory(PyObject *self, PyObject *args)
{
  
    TAU_ENABLE_TRACKING_MEMORY();
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_disableTrackingMemory__name__[] = "disableTrackingMemory";
char pytau_disableTrackingMemory__doc__[] = "track memory headroom available to grow at a given location in the source code";
extern "C"
PyObject * pytau_disableTrackingMemory(PyObject *self, PyObject *args)
{
  
    TAU_DISABLE_TRACKING_MEMORY();
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_enableTrackingMemoryHeadroom__name__[] = "enableTrackingMemoryHeadroom";
char pytau_enableTrackingMemoryHeadroom__doc__[] = "enable tracking memory headroom";
extern "C"
PyObject * pytau_enableTrackingMemoryHeadroom(PyObject *self, PyObject *args)
{
  
    TAU_ENABLE_TRACKING_MEMORY_HEADROOM();
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_disableTrackingMemoryHeadroom__name__[] = "disableTrackingMemoryHeadroom";
char pytau_disableTrackingMemoryHeadroom__doc__[] = "disable tracking memory headroom";
extern "C"
PyObject * pytau_disableTrackingMemoryHeadroom(PyObject *self, PyObject *args)
{

    TAU_DISABLE_TRACKING_MEMORY_HEADROOM();
    Py_INCREF(Py_None);
    return Py_None;
}

char pytau_setInterruptInterval__name__[] = "setInterruptInterval";
char pytau_setInterruptInterval__doc__[] = "set interrupt interval";
extern "C"
PyObject * pytau_setInterruptInterval(PyObject *self, PyObject *args)
{
    int interval;

    interval = 1;

    // extract function list from Python args
    int ok = PyArg_ParseTuple(args, "i:interval", &interval);
    if (!ok) {
        return 0;
    }

#ifdef DEBUG_PROF
    printf("pytau_setInterruptInterval: %d\n", interval);
#endif /* DEBUG_PROF */
    TAU_SET_INTERRUPT_INTERVAL(interval);


    Py_INCREF(Py_None);
    return Py_None;
}

// version
// $Id: PyMemory.cpp,v 1.1 2008/09/18 23:40:17 sameer Exp $

// End of file
  
