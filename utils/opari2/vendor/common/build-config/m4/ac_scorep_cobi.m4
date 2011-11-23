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


AC_DEFUN([AC_SCOREP_COBI], 
[

AC_ARG_WITH([cobi], 
            [AS_HELP_STRING([--with-cobi=(yes|no|<path-to-cobi-binary>)], 
                            [Whether to build with cobi binary instrumentation support. "yes" assumes it is in PATH [no].])],
            # action-if-given
            [AS_CASE([$withval],
                     ["yes"], [scorep_with_cobi="yes"],
                     ["no"],  [scorep_with_cobi="no"],
                     [scorep_with_cobi="$withval"])],
            # action-if-not-given
            [scorep_with_cobi="no"])


if test "x${scorep_with_cobi}" != "xno"; then

    if test "x${enable_shared}" = "xno"; then
        AC_MSG_CHECKING([for cobi])
        AC_MSG_RESULT([prerequisites missing])
        AC_MSG_ERROR([Cobi support requires building shared libraries. Use --enable-shared to build shared libraries or omit the --with-cobi option.])
    fi

    if test "x${scorep_with_cobi}" = "xyes"; then
        AC_CHECK_PROG([COBI], [cobi], [`which cobi`], ["no"])
    else
        AC_CHECK_PROG([COBI], [cobi], [${scorep_with_cobi}/cobi], ["no"], [${scorep_with_cobi}])
    fi

fi


# The output of this macro
AM_CONDITIONAL([HAVE_COBI], [test "x${COBI}" != "xno"])

if test "x${COBI}" != "xno"; then
    AC_DEFINE([HAVE_COBI], [1], [Defined if configured to use Cobi.])
else
    AC_DEFINE([HAVE_COBI], [0], [Defined if configured to use Cobi.])
fi
AC_DEFINE_UNQUOTED([SCOREP_COBI_PATH], ["${COBI}"], [Defines path to Cobi.])

])
