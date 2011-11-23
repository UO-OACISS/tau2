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


# Add checks here that are common for frontend- and backend-
# build-configurations.

AC_DEFUN([AC_SCOREP_COMMON_CHECKS],
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
AC_SCOREP_SUMMARY([C99 compiler used], [$CC])

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

## Macro: AC_C_BIGENDIAN ([action-if-true], [action-if-false],
## [action-if-unknown], [action-if-universal])
## The default for action-if-true is to define `WORDS_BIGENDIAN'. The default
## for action-if-false is to do nothing. The default for action-if-unknown is
## to abort configure and tell the installer how to bypass this test. And
## finally, the default for action-if-universal is to ensure that
## `WORDS_BIGENDIAN' is defined if and only if a universal build is detected
## and the current code is big-endian
AC_REQUIRE([AC_C_BIGENDIAN])

## Search for a library defining function if it's not already available. This
## equates to calling ‘AC_LINK_IFELSE([AC_LANG_CALL([], [function])])’ first
## with no libraries, then for each library listed in search-libs.
##
## Add -llibrary to LIBS for the first library found to contain function, and
## run action-if-found. If the function is not found, run action-if-not-found.
##
## If linking with library results in unresolved symbols that would be
## resolved by linking with additional libraries, give those libraries as the
## other-libraries argument, separated by spaces: e.g., -lXt -lX11. Otherwise,
## this macro fails to detect that function is present, because linking the
## test program always fails with unresolved symbols.
#AC_SEARCH_LIBS([trunc], [m])

AC_CXX_NAMESPACES
AC_CXX_HAVE_SSTREAM
AC_CXX_HAVE_STRSTREAM

AC_REQUIRE([AC_PROG_RANLIB])

AC_SCOREP_DEBUG_OPTION
AC_SCOREP_ON_DEBUG_OPTION
AC_CUTEST_COLOR_TESTS
])
