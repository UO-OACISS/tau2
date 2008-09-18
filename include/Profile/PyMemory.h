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

#if !defined(pytau_memory_h)
#define pytau_memory_h

extern char pytau_setInterruptInterval__name__[];
extern char pytau_setInterruptInterval__doc__[];
extern "C"
PyObject * pytau_setInterruptInterval(PyObject *, PyObject *);

extern char pytau_disableTrackingMemoryHeadroom__name__[];
extern char pytau_disableTrackingMemoryHeadroom__doc__[];
extern "C"
PyObject * pytau_disableTrackingMemoryHeadroom(PyObject *, PyObject *);

extern char pytau_enableTrackingMemoryHeadroom__name__[];
extern char pytau_enableTrackingMemoryHeadroom__doc__[];
extern "C"
PyObject * pytau_enableTrackingMemoryHeadroom(PyObject *, PyObject *);

extern char pytau_disableTrackingMemory__name__[];
extern char pytau_disableTrackingMemory__doc__[];
extern "C"
PyObject * pytau_disableTrackingMemory(PyObject *, PyObject *);

extern char pytau_enableTrackingMemory__name__[];
extern char pytau_enableTrackingMemory__doc__[];
extern "C"
PyObject * pytau_enableTrackingMemory(PyObject *, PyObject *);

extern char pytau_trackMemoryHeadroomHere__name__[];
extern char pytau_trackMemoryHeadroomHere__doc__[];
extern "C"
PyObject * pytau_trackMemoryHeadroomHere(PyObject *, PyObject *);

extern char pytau_trackMemoryHere__name__[];
extern char pytau_trackMemoryHere__doc__[];
extern "C"
PyObject * pytau_trackMemoryHere(PyObject *, PyObject *);

extern char pytau_trackMemoryHeadroom__name__[];
extern char pytau_trackMemoryHeadroom__doc__[];
extern "C"
PyObject * pytau_trackMemoryHeadroom(PyObject *, PyObject *);

extern char pytau_trackMemory__name__[];
extern char pytau_trackMemory__doc__[];
extern "C"
PyObject * pytau_trackMemory(PyObject *, PyObject *);
#endif // pytau_memory_h

// version
// $Id: PyMemory.h,v 1.1 2008/09/18 23:39:48 sameer Exp $

// End of file
