// -*- C++ -*-
//
//-----------------------------------------------------------------------------
//
//                              VTF Development Team
//                       California Institute of Technology
//                          (C) 2002 All Rights Reserved
//
// <LicenseText>
//
//-----------------------------------------------------------------------------
//
// $Log: PyExceptions.cpp,v $
// Revision 1.1  2003/02/28 23:26:51  sameer
// Added Python Bindings to TAU [Julian Cummings, Brian Miller].
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

// Local
#include "Profile/PyExceptions.h"

PyObject *pytau_badArgument = 0;

// $Id: PyExceptions.cpp,v 1.1 2003/02/28 23:26:51 sameer Exp $

// End of file
