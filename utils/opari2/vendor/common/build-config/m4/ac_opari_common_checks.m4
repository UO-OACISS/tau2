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


AC_DEFUN([AC_OPARI_COMMON_CHECKS],
[
## Determine a C compiler to use. If CC is not already set in the environment,
## check for gcc and cc, then for other C compilers. Set output variable CC to
## the name of the compiler found.
## 
## This macro may, however, be invoked with an optional first argument which,
## if specified, must be a blank-separated list of C compilers to search
## for. This just gives the user an opportunity to specify an alternative
## search list for the C compiler. For example, if you didn't like the default
## order, then you could invoke AC_PROG_CC like this: AC_PROG_CC([gcc cl cc])
AC_REQUIRE([AC_PROG_CC])
AC_SCOREP_COMPILER_CHECKS

## If the C compiler is not in C99 mode by default, try to add an option to
## output variable CC to make it so. This macro tries various options that
## select C99 on some system or another. It considers the compiler to be in
## C99 mode if it handles _Bool, // comments, flexible array members, inline,
## signed and unsigned long long int, mixed code and declarations, named
## initialization of structs, restrict, va_copy, varargs macros, variable
## declarations in for loops, and variable length arrays.  After calling this
## macro you can check whether the C compiler has been set to accept C99; if
## not, the shell variable ac_cv_prog_cc_c99 is set to `no'.
AC_REQUIRE([SCOREP_PROG_CC_C99])

## Determine a C++ compiler to use. Check whether the environment variable CXX 
## or CCC (in that order) is set; if so, then set output variable CXX to its 
## value.
## 
## Otherwise, if the macro is invoked without an argument, then search for a
## C++ compiler under the likely names (first g++ and c++ then other
## names). If none of those checks succeed, then as a last resort set CXX to
## g++.
## 
## This macro may, however, be invoked with an optional first argument which,
## if specified, must be a blank-separated list of C++ compilers to search
## for.  This just gives the user an opportunity to specify an alternative
## search list for the C++ compiler. For example, if you didn't like the
## default order, then you could invoke AC_PROG_CXX like this:
## AC_PROG_CXX([gcc cl KCC CC cxx cc++ xlC aCC c++ g++])
AC_REQUIRE([AC_PROG_CXX])

## Enable using per-target flags or subdir-objects with C sources
AC_REQUIRE([AM_PROG_CC_C_O])


## Determine a Fortran 77 compiler to use. If F77 is not already set in the
## environment, then check for g77 and f77, and then some other names. Set the
## output variable F77 to the name of the compiler found.
##
## This macro may, however, be invoked with an optional first argument which,
## if specified, must be a blank-separated list of Fortran 77 compilers to
## search for. This just gives the user an opportunity to specify an
## alternative search list for the Fortran 77 compiler. For example, if you
## didn't like the default order, then you could invoke AC_PROG_F77 like this:
##
##          AC_PROG_F77([fl32 f77 fort77 xlf g77 f90 xlf90])
##
## If using g77 (the GNU Fortran 77 compiler), then set the shell variable G77
## to ‘yes’. If the output variable FFLAGS was not already set in the
## environment, then set it to -g -02 for g77 (or -O2 where g77 does not
## accept -g). Otherwise, set FFLAGS to -g for all other Fortran 77 compilers.
AC_REQUIRE([AC_PROG_F77])
AC_SCOREP_HAVE_F77

## Determine a Fortran compiler to use. If FC is not already set in the
## environment, then dialect is a hint to indicate what Fortran dialect to
## search for; the default is to search for the newest available dialect. Set
## the output variable FC to the name of the compiler found.
##
## By default, newer dialects are preferred over older dialects, but if
## dialect is specified then older dialects are preferred starting with the
## specified dialect. dialect can currently be one of Fortran 77, Fortran 90,
## or Fortran 95. However, this is only a hint of which compiler name to
## prefer (e.g., f90 or f95), and no attempt is made to guarantee that a
## particular language standard is actually supported. Thus, it is preferable
## that you avoid the dialect option, and use AC_PROG_FC only for code
## compatible with the latest Fortran standard.
##
## This macro may, alternatively, be invoked with an optional first argument
## which, if specified, must be a blank-separated list of Fortran compilers to
## search for, just as in AC_PROG_F77.
##
## If the output variable FCFLAGS was not already set in the environment, then
## set it to -g -02 for GNU g77 (or -O2 where g77 does not accept
## -g). Otherwise, set FCFLAGS to -g for all other Fortran compilers.
AC_REQUIRE([AC_PROG_FC])
AC_SCOREP_HAVE_FC

#AC_CXX_NAMESPACES
#AC_CXX_HAVE_SSTREAM
#AC_CXX_HAVE_STRSTREAM

AC_LANG_PUSH([C])
AC_OPENMP
AC_LANG_POP([C])

AM_CONDITIONAL([OPENMP_SUPPORTED], 
               [test "x${ac_cv_prog_c_openmp}" != "xunsupported" && test "x${enable_openmp}" != "xno"])

if test "x${ac_cv_prog_c_openmp}" = "xunsupported"; then
    AC_MSG_WARN([no suitbale OpenMP compilers found. POMP2 dummy lib will not be build.])
else
    AC_LANG_PUSH([C++])
    AC_OPENMP
    AC_LANG_POP([C++])

    AC_LANG_PUSH([Fortran 77])
    AC_OPENMP
    AC_LANG_POP([Fortran 77])

    AC_LANG_PUSH([Fortran])
    AC_OPENMP
    AC_LANG_POP([Fortran])
fi

AC_SCOREP_FORTRAN_SUPPORT_ALLOCATABLE
SCOREP_ATTRIBUTE_ALIGNMENT
AC_PROG_RANLIB
AC_SCOREP_DEFINE_REVISIONS
])
