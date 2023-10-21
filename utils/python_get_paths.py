#!/usr/bin/env python3

# This prints three lines.
# Line 1: Python INCLUDE path
# Line 2: Python library path
# Line 3: Name of the Python shared library

# Python 2.7 compatibility
from __future__ import print_function

import sys

print_include = False
print_libdir = False
print_libname = False

if len(sys.argv) > 1:
    if sys.argv[1] == "inc":
        print_include = True
    elif sys.argv[1] == "libdir":
        print_libdir = True
    elif sys.argv[1] == "libname":
        print_libname = True
else:
    print_include = True
    print_libdir = True
    print_libname = True


try:
    import sysconfig
    if print_include:
        print(sysconfig.get_config_var("INCLUDEPY"))
    if print_libdir:
        print(sysconfig.get_config_var("LIBDIR"))
    if print_libname:
        suffix = sysconfig.get_config_var("SHLIB_SUFFIX")
        if suffix is None:
            suffix = sysconfig.get_config_var("SHLIB_EXT").strip('"')
        libname = sysconfig.get_config_var("LDLIBRARY")
        if suffix is not None:
            libname = libname.replace(".a", suffix)
        print(libname)
    sys.exit(0)
except Exception as e:
    sys.exit(1)
