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

## file       ac_scorep_summary.m4
## maintainer Bert Wesarg <bert.wesarg@tu-dresden.de>

AC_DEFUN([AC_SCOREP_SUMMARY_INIT], [
    AS_ECHO(["$1:"]) >config.summary
])

AC_DEFUN([AC_SCOREP_SUMMARY_SECTION], [
    AS_IF([test ! -f config.summary], [
        AC_MSG_WARN([SCOREP_SUMMARY_SECTION used without calling SCOREP_SUMMARY_INIT.])
    ])
    AS_ECHO([" $1"]) >>config.summary
])

AC_DEFUN([AC_SCOREP_SUMMARY], [
    AS_IF([test ! -f config.summary], [
        AC_MSG_WARN([SCOREP_SUMMARY used without calling SCOREP_SUMMARY_INIT.])
    ])
    AS_ECHO(["  $1: $2"]) >>config.summary
])

# additional output if ./configure is called with --verbose
AC_DEFUN([AC_SCOREP_SUMMARY_VERBOSE], [
    AS_IF([test ! -f config.summary], [
        AC_MSG_WARN([SCOREP_SUMMARY_VERBOSE used without calling SCOREP_SUMMARY_INIT.])
    ])
    AS_IF([test "x${verbose}" = "xyes"], [
        AS_ECHO(["   $1: $2"]) >>config.summary
    ])
])

# should be called after AC_OUTPUT
AC_DEFUN([AC_SCOREP_SUMMARY_COLLECT], [
    AS_IF([test -f config.summary], [
        cat config.summary
    ])
    find */ -depth -name config.summary | xargs cat
])
