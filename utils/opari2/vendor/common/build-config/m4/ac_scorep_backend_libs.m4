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


## file       ac_scorep_backend_libs.m4
## maintainer Christian Roessel <c.roessel@fz-juelich.de>


dnl The intention of the following macros is to provide a consistent means of
dnl checking backend libraries. E.g. we can't just use AC_CHECK_LIB([z]) to check
dnl for libz because this will usually find the frontend version installed in
dnl /usr/lib or /user/local/lib. As there are no standard paths for backend
dnl libraries and includes we need to specify them explicitly. For this purpose,
dnl use AC_SCOREP_BACKEND_LIB_OPTIONS. It will provide the configure options
dnl --with-backend-<libname>-includes and --with-backend-<libname>-lib.

dnl The macro AC_SCOREP_BACKEND_LIB_REQUIRE_ARGUMENTS will determine if the user
dnl needs to specify the paths to the lib and the includes (on cross compile
dnl systems)

dnl The macro AC_SCOREP_BACKEND_LIB_HEADER_USABLE will check if the requested header
dnl files are usable.

dnl Following variables are of interest so far ($1 is the name of the lib):
dnl $ac_scorep_with_backend_$1_includes
dnl $ac_scorep_with_backend_$1_lib
dnl $ac_scorep_check_for_backend_$1
dnl $ac_scorep_has_backend_$1_includes
dnl m4_join([_], [LIB], m4_toupper([$1]), [CPPFLAGS])
dnl m4_join([_], [HAVE_LIB], m4_toupper([$1])

dnl There will be a macro AC_SCOREP_BACKEND_LIB_USABLE that will try to link a small
dnl test program. Needs to be implemented.

dnl A user can then build an own macro using:
dnl AC_SCOREP_BACKEND_LIB_OPTIONS([cubew4])
dnl AC_SCOREP_BACKEND_LIB_REQUIRE_ARGUMENTS([cubew4])
dnl AC_SCOREP_BACKEND_LIB_HEADER_USABLE([cubew4], [cubew.h])
dnl AC_SCOREP_BACKEND_LIB_USABL([cubew4], ...)


AC_DEFUN([AC_SCOREP_BACKEND_LIB_OPTIONS], [
# arg1: the name of the library, e.g. cubew4 for libcubew4.(so|a)
AC_ARG_WITH([backend-$1-includes],
            [AS_HELP_STRING([--with-backend-$1-includes=path], 
                            [Path to the lib$1 includes (a backend version on cross-compile systems)])],
            [ac_scorep_with_backend_$1_includes=$withval],
            [ac_scorep_with_backend_$1_includes=""])

AC_ARG_WITH([backend-$1-lib],
            [AS_HELP_STRING([--with-backend-$1-lib=path], 
                            [Path to lib$1 (a backend version on cross-compile systems)])],
            [ac_scorep_with_backend_$1_lib=$withval],
            [ac_scorep_with_backend_$1_lib=""])

])

#------------------------------------------------------------------------------

AC_DEFUN([AC_SCOREP_BACKEND_LIB_HEADER_USABLE], [
# arg1: lib name as in AC_SCOREP_BACKEND_LIB_OPTIONS, e.g. cubew4
# arg2-n: name of includes, e.g. cubew.h

if test "x${ac_scorep_check_for_backend_$1}" = "yes"; then

    AC_LANG_PUSH([C])
    cppflags_save="$CPPFLAGS"

    user_provided_cppflags=""
    if test "x${ac_scorep_with_backend_$1_includes}" != "x"; then
        user_provided_cppflags="-I${ac_scorep_with_backend_$1_includes}"
        CPPFLAGS="${user_provided_cppflags} $CPPFLAGS"
    fi

    ifs_old="$IFS"
    IFS=","
    headers="$@"
    for header in $headers; do
        IFS="$ifs_old"    
        if test "x$header" = "x$1"; then
            continue
        fi
        AC_CHECK_HEADER([$header], [has_header="yes"], [has_header="no"], [])
        if test "x$has_header" = "xno"; then
            break
        fi
        IFS=","
    done
    IFS="$ifs_old"

    ac_scorep_has_backend_$1_includes="no"
    if  test "x$has_header" = "xyes"; then
        ac_scorep_has_backend_$1_includes="yes"
    fi

    CPPFLAGS="$cppflags_save"
    AC_LANG_POP([C])

    AC_SUBST(m4_join([_], [LIB], m4_toupper([$1]), [CPPFLAGS]), [${user_provided_cppflags}])
    AC_MSG_CHECKING([for lib$1 includes])
    AC_MSG_RESULT([$ac_scorep_has_backend_$1_includes])

fi 
])

#------------------------------------------------------------------------------

AC_DEFUN([AC_SCOREP_BACKEND_LIB_REQUIRE_ARGUMENTS], [
# arg1: lib name as in AC_SCOREP_BACKEND_LIB_OPTIONS, e.g. cubew4
ac_scorep_check_for_backend_$1="yes"
if test "x${ac_scorep_cross_compiling}" = "xyes"; then
    if test "x${ac_scorep_with_backend_$1_includes}" = "x" -o "x${ac_scorep_with_backend_$1_lib}" = "x"; then
        AC_MSG_WARN([Can't reliably determine backend lib$1 in cross compiling mode. Please use the configure options --with-backend-$1-includes and --with-backend-$1-lib. Disabling $1 support.])
        AM_CONDITIONAL(m4_join([_], [HAVE_LIB], m4_toupper([$1])), [test 1 -eq 1])
        ac_scorep_check_for_backend_$1="no"
    fi
fi
# needs to be set in AC_SCOREP_BACKEND_LIB_USABLE. It is here just to please autoreconf
AM_CONDITIONAL(m4_join([_], [HAVE_LIB], m4_toupper([$1])), [test 1 -eq 1])
])

#------------------------------------------------------------------------------

dnl AC_DEFUN([AC_SCOREP_BACKEND_LIB_USABLE], [
dnl # arg1: lib name as in AC_SCOREP_BACKEND_LIB_OPTIONS, e.g. cubew4
dnl # arg2: test program to be used in AC_LANG_PROGRAM
dnl # arg3: additional libraries, -Lpath -ladditional_lib combinations

dnl AC_LANG_PUSH([C])
dnl AC_MSG_CHECKING([for lib$1])
dnl libs_save="$LIBS"


dnl LIBS="$libs_save"
dnl AC_MSG_RESULT([])
dnl AC_LANG_POP([C])

dnl ])
