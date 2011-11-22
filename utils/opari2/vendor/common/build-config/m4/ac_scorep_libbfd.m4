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


AC_DEFUN([SCOREP_LIBBFD_LINK_TEST], [
AC_LINK_IFELSE([AC_LANG_PROGRAM([[char bfd_init ();
char bfd_openr ();
char bfd_check_format ();
char bfd_close ();]],
                                [[bfd_init ();
bfd_openr ();
bfd_check_format ();
bfd_close ();]])],
               [scorep_have_libbfd="yes"], 
               [scorep_have_libbfd="no"])
])


AC_DEFUN([AC_SCOREP_LIBBFD], [
AC_REQUIRE([AC_SCOREP_COMPILER_CHECKS])
# In scalasca we use bfd only for the gnu compiler instrumentation.

# What shall we do if there is a frontend and a backend libbfd? We need the
# backend version, but usually the frontend version will be found. One
# approach is to require user input on dual-architecture machines. But for now
# we will just abort (until this problem shows up in reality).

if test "x${scorep_compiler_gnu}" = "xyes" -o "x${scorep_compiler_intel}" = "xyes"; then

    if test "x${ac_scorep_cross_compiling}" = "xyes"; then
        AC_MSG_ERROR([Can't reliably determine backend libbfd in cross compiling mode.])
        AC_MSG_ERROR([Can't reliably determine backend nm in cross compiling mode.])
    fi

    AC_LANG_PUSH([C])
    AC_CHECK_HEADER([bfd.h])
    
    AC_MSG_CHECKING([for libbfd])    
    scorep_libbfd_save_LIBS="$LIBS"

    LIBS="-lbfd"
    SCOREP_LIBBFD_LINK_TEST
    AS_IF([test "x${scorep_have_libbfd}" = "xno"],
          [LIBS="-lbfd -liberty"; 
           SCOREP_LIBBFD_LINK_TEST
           AS_IF([test "x${scorep_have_libbfd}" = "xno"],
                 [LIBS="-lbfd -liberty -lz";
                  SCOREP_LIBBFD_LINK_TEST
                  AS_IF([test "x${scorep_have_libbfd}" = "xno"],
                        [AC_MSG_RESULT([no])],
                        [AC_MSG_RESULT([$LIBS])])],
                 [AC_MSG_RESULT([$LIBS])])],
          [AC_MSG_RESULT([$LIBS])])
    
    AC_CHECK_HEADER([demangle.h])
    AC_MSG_CHECKING([for cplus_demangle])    
    AC_LINK_IFELSE([AC_LANG_PROGRAM([[char* cplus_demangle( const char* mangled, int options );]],
                                    [[cplus_demangle("test", 27)]])],
                   [scorep_have_demangle="yes"], 
                   [scorep_have_demangle="no"])
    AC_MSG_RESULT([$scorep_have_demangle])

    scorep_bfd_libs="$LIBS"
    LIBS="$scorep_libbfd_save_LIBS"

    AM_CONDITIONAL([HAVE_LIBBFD], [test "x${ac_cv_header_bfd_h}" = "xyes" && test "x${scorep_have_libbfd}" = "xyes"])
    if test "x${ac_cv_header_bfd_h}" = "xno" || test "x${scorep_have_libbfd}" = "xno"; then
        AC_MSG_WARN([libbfd not available. Trying compiler instrumentation via nm.])
        AC_SUBST([LIBBFD], [""])

        # ok, bfd not available, search for nm
        AC_CHECK_PROG([scorep_have_nm], [nm], ["yes"], ["no"])
        AM_CONDITIONAL([HAVE_NM_AS_BFD_REPLACEMENT], [test "x${scorep_have_nm}" = "xyes"])
        if test "x${scorep_have_nm}" = "xno"; then
            AC_MSG_WARN([Neither libbfd nor nm are available. Compiler instrumentation will not work.])
        fi
    else
        AC_SUBST([LIBBFD], ["$scorep_bfd_libs"])
        AM_CONDITIONAL([HAVE_NM_AS_BFD_REPLACEMENT], [test 1 -ne 1])
    fi
    AC_LANG_POP([C])
else
    AM_CONDITIONAL([HAVE_LIBBFD], [test 1 -ne 1])
    AC_SUBST([LIBBFD], [""])
    AM_CONDITIONAL([HAVE_NM_AS_BFD_REPLACEMENT], [test 1 -ne 1])
fi

AM_CONDITIONAL([HAVE_DEMANGLE], [test "x${scorep_have_demangle}" = "xyes"])
])
