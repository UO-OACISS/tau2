## -*- mode: autoconf -*-

## 
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2011, 
##    RWTH Aachen, Germany
##    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##    Technische Universitaet Dresden, Germany
##    University of Oregon, Eugene, USA
##    Forschungszentrum Juelich GmbH, Germany
##    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##    Technische Universitaet Muenchen, Germany
##
## See the COPYING file in the package base directory for details.
##

## file       ac_scorep_metrics.m4
## maintainer Christian Roessel <c.roessel@fz-juelich.de>


AC_DEFUN([AC_SCOREP_LIBPAPI], [

dnl Don't check for PAPI on the frontend.
AS_IF([test "x$ac_scorep_backend" = xno], [AC_MSG_ERROR([cannot check for PAPI on frontend.])])

# advertise the $PAPI_INC and $PAPI_LIB variables in the --help output
AC_ARG_VAR([PAPI_INC], [Include path to the papi.h header.])
AC_ARG_VAR([PAPI_LIB], [Library path to the papi library.])

dnl checking for the header
AC_ARG_WITH([papi-header],
            [AS_HELP_STRING([--with-papi-header=<path-to-papi.h>], 
                            [If papi.h is not installed in the default location, specify the dirname where it can be found.])],
            [scorep_papi_inc_dir="${withval}"],  # action-if-given.
            [scorep_papi_inc_dir="${PAPI_INC-}"] # action-if-not-given
)
AS_IF([test "x$scorep_papi_inc_dir" != "x"], [
    # The -DC99 is a necessary gcc workaround for a
    # bug in papi 4.1.2.1. It might be compiler dependent.
    scorep_papi_cppflags=-"I$scorep_papi_inc_dir -DC99"
], [
    scorep_papi_cppflags=""
])

AC_LANG_PUSH([C])
cppflags_save="$CPPFLAGS"

CPPFLAGS="$scorep_papi_cppflags $CPPFLAGS"
AC_CHECK_HEADER([papi.h],
                [scorep_papi_header="yes"],
                [scorep_papi_header="no"])
CPPFLAGS="$cppflags_save"
AC_LANG_POP([C])


dnl checking for the library
AC_ARG_WITH([papi-lib],
            [AS_HELP_STRING([--with-papi-lib=<path-to-libpapi.*>], 
                            [If libpapi.* is not installed in the default location, specify the dirname where it can be found.])],
            [scorep_papi_lib_dir="${withval}"],  # action-if-given
            [scorep_papi_lib_dir="${PAPI_LIB-}"] # action-if-not-given
)
AC_LANG_PUSH([C])
ldflags_save="$LDFLAGS"
LDFLAGS="-L$scorep_papi_lib_dir $LDFLAGS"
scorep_papi_lib_name="papi"
AC_CHECK_LIB([$scorep_papi_lib_name], [PAPI_library_init],
             [scorep_papi_library="yes"], # action-if-found
             [scorep_papi_library="no"]   # action-if-not-found
) 
LDFLAGS="$ldflags_save"
AC_LANG_POP([C])


dnl generating results/output/summary
scorep_have_papi="no"
if test "x${scorep_papi_header}" = "xyes" && test "x${scorep_papi_library}" = "xyes"; then
    scorep_have_papi="yes"
fi
AC_MSG_CHECKING([for papi support])
AC_MSG_RESULT([$scorep_have_papi])
if test "x${scorep_have_papi}" = "xyes"; then
    AC_DEFINE([HAVE_PAPI], [1],     [Defined if libpapi is available.])
    AC_SUBST([SCOREP_PAPI_LDFLAGS], [-L${scorep_papi_lib_dir}])
    AC_SUBST([SCOREP_PAPI_LIBS],    [-l${scorep_papi_lib_name}])
else
    AC_SUBST([SCOREP_PAPI_LDFLAGS], [""])
    AC_SUBST([SCOREP_PAPI_LIBS],    [""])
fi
AM_CONDITIONAL([HAVE_PAPI],         [test "x${scorep_have_papi}" = "xyes"])
AC_SUBST([SCOREP_PAPI_CPPFLAGS],    [$scorep_papi_cppflags])
AC_SUBST([SCOREP_PAPI_LIBDIR],      [$scorep_papi_lib_dir])
AC_SCOREP_SUMMARY([PAPI support],   [${scorep_have_papi}])
AS_IF([test "x${scorep_have_papi}" = "xyes"], [
    AC_SCOREP_SUMMARY_VERBOSE([PAPI include directory], [$scorep_papi_inc_dir])
    AC_SCOREP_SUMMARY_VERBOSE([PAPI library directory], [$scorep_papi_lib_dir])
])
])



AC_DEFUN([AC_SCOREP_RUSAGE], [

dnl Check for headers
dnl Header availability is checked implicitly by AC_CHECK_DECL checks, see below.
dnl AC_CHECK_HEADERS([sys/time.h sys/resource.h], [scorep_rusage_header="yes"], [scorep_rusage_header="no"])


dnl Check for getrusage function
AC_LANG_PUSH([C])
AC_CHECK_DECL([getrusage], [scorep_getrusage="yes"], [scorep_getrusage="no"], [[#include <sys/time.h>
#include <sys/resource.h>]])
AC_LANG_POP([C])


dnl Check for availability of RUSAGE_THREAD
AC_LANG_PUSH([C])
scorep_rusage_cppflags=""
AC_CHECK_DECL([RUSAGE_THREAD], [scorep_rusage_thread="yes"], [scorep_rusage_thread="no"], [[#include <sys/time.h>
#include <sys/resource.h>]])
AS_IF([test "x$scorep_rusage_thread" = "xno"], 
      [unset ac_cv_have_decl_RUSAGE_THREAD
       cppflags_save="$CPPFLAGS"
       dnl For the affects of _GNU_SOURCE see /usr/include/features.h. Without
       dnl -D_GNU_SOURCE it seems that we don't get rusage per thread (RUSAGE_THREAD)
       dnl but per process only.
       scorep_rusage_cppflags="-D_GNU_SOURCE"
       CPPFLAGS="${scorep_rusage_cppflags} $CPPFLAGS"
       AC_CHECK_DECL([RUSAGE_THREAD], [scorep_rusage_thread="yes"], [scorep_rusage_thread="no"], [[#include <sys/time.h>
       #include <sys/resource.h>]])
       CPPFLAGS="$cppflags_save"
])
AC_LANG_POP([C])


dnl generating results/output/summary
AS_IF([test "x${scorep_getrusage}" = "xyes"],
      [AC_DEFINE([HAVE_GETRUSAGE], [1], [Defined if getrusage() is available.])])
AS_IF([test "x${scorep_rusage_thread}" = "xyes"],
      [AC_DEFINE([HAVE_RUSAGE_THREAD], [1], [Defined if RUSAGE_THREAD is available.])
       AC_DEFINE([SCOREP_RUSAGE_SCOPE], [RUSAGE_THREAD], [Defined to RUSAGE_THREAD, if it is available, else to RUSAGE_SELF.])],
      [AC_DEFINE([SCOREP_RUSAGE_SCOPE], [RUSAGE_SELF],   [Defined to RUSAGE_THREAD, if it is available, else to RUSAGE_SELF.])])
AM_CONDITIONAL([HAVE_GETRUSAGE], [test "x${scorep_getrusage}" = "xyes"])
AC_SUBST([SCOREP_RUSAGE_CPPFLAGS], [$scorep_rusage_cppflags])
AC_SCOREP_SUMMARY([getrusage support], [${scorep_getrusage}])
AS_IF([test "x"${scorep_rusage_cppflags} = "x"],
      [AC_SCOREP_SUMMARY([RUSAGE_THREAD support], [${scorep_rusage_thread}])],
      [AC_SCOREP_SUMMARY([RUSAGE_THREAD support], [${scorep_rusage_thread}, using ${scorep_rusage_cppflags}])])
])
