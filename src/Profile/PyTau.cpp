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

// Python 3 uses a different module initialization procedure
// than Python 2
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef pytau_moduledef = {
        PyModuleDef_HEAD_INIT,
        "pytau",             /* m_name */
        pytau_module__doc__, /* m_doc */
        -1,                  /* m_size */
        pytau_methods,       /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

// Initialization function for the module (*must* be called initpytau
// in Python 2 and PyInit_pytau in Python 3)
extern "C"
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit_pytau()
#else
void
initpytau()
#endif
{
// create the module and add the functions
#if PY_MAJOR_VERSION >= 3
    PyObject * m = PyModule_Create(&pytau_moduledef);
#else
    PyObject * m =
        Py_InitModule4("pytau", pytau_methods, pytau_module__doc__, 0, PYTHON_API_VERSION);
#endif

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
#if PY_MAJOR_VERSION >= 3
    // In Python 3, we have to return the created module object
    return m;
#else
    return;
#endif
}

// version
// $Id: PyTau.cpp,v 1.1 2003/02/28 23:26:52 sameer Exp $

// End of file
