// -*- C++ -*-
//
//-----------------------------------------------------------------------------
//
//                              VTF Development Team
//                       California Institute of Technology
//                       (C) 1998-2002  All Rights Reserved
//
// <LicenseText>
//
//-----------------------------------------------------------------------------
//
// $Log: PyTau.cpp,v $
// Revision 1.1  2003/02/28 23:26:52  sameer
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

#include "Profile/PyBindings.h"
#include "Profile/PyExceptions.h"


// Module documentation string
char pytau_module__doc__[] = "Tau extensions module";

// Initialization function for the module (*must* be called initpytau)
extern "C"
void
initpytau()
{
// create the module and add the functions
    PyObject * m =
        Py_InitModule4("pytau", pytau_methods, pytau_module__doc__, 0, PYTHON_API_VERSION);

// get its dictionary
    PyObject * d = PyModule_GetDict(m);

// check for errors
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module pytau");
    }

// Register the exceptions of the module
    pytau_badArgument = PyErr_NewException("pytau.BadArgument", 0, 0);
    PyDict_SetItemString(d, "BadArgument", pytau_badArgument);

// Finished
    return;
}

// version
// $Id: PyTau.cpp,v 1.1 2003/02/28 23:26:52 sameer Exp $

// End of file
