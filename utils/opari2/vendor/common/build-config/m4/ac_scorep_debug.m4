## -*- mode: autoconf -*-

## 
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2011, 
##    RWTH Aachen University, Germany
##    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##    Technische Universitaet Dresden, Germany
##    University of Oregon, Eugene, USA
##    Forschungszentrum Juelich GmbH, Germany
##    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##    Technische Universitaet Muenchen, Germany
##
## See the COPYING file in the package base directory for details.
##


## file       ac_scorep_debug.m4
## maintainer Christian Roessel <c.roessel@fz-juelich.de>


AC_DEFUN([AC_SCOREP_DEBUG_OPTION],
[
AC_ARG_ENABLE([debug],
              [AS_HELP_STRING([--enable-debug], [activate internal debug output [no]])],
              [],
              [enable_debug=no])

ac_scorep_enable_debug="no"
if test "x${enable_debug}" = "xyes"; then
    ac_scorep_enable_debug="yes"
fi
])


AC_DEFUN([AC_SCOREP_ON_DEBUG_OPTION],
[
if test  "x${ac_scorep_enable_debug}" = "xyes"; then
    AC_DEFINE([HAVE_SCOREP_DEBUG], [1], [activate internal debug output])
elif test  "x${ac_scorep_enable_debug}" != "xno"; then
    AC_MSG_ERROR([ac_scorep_enable_debug neither "yes" nor "no" but "$ac_scorep_enable_debug", that's weird.])
fi
])
