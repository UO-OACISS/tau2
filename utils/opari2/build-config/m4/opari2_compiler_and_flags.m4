dnl -*- mode: autoconf -*-

dnl 
dnl This file is part of the Score-P software (http://www.score-p.org)
dnl
dnl Copyright (c) 2013
dnl Forschungszentrum Juelich GmbH, Germany
dnl
dnl This software may be modified and distributed under the terms of
dnl a BSD-style license.  See the COPYING file in the package base
dnl directory for details.
dnl

dnl file build-config/m4/opari2_compiler_and_flags.m4


# On cross-compile system we might get provided with the *_FOR_BUILD
# compilers and flags and need to map them to CC, CFLAGS etc. The
# *_FOR_BUILD parameters take precedence.
AC_DEFUN([AC_SCOREP_OPARI2_FOR_BUILD_ARGS_TAKES_PRECEDENCE],
[
    opari2_cross_build_args=""
    _OPARI2_CONVERT_FOR_BUILD_ARG([CC])
    _OPARI2_CONVERT_FOR_BUILD_ARG([CXX])
    _OPARI2_CONVERT_FOR_BUILD_ARG([F77])
    _OPARI2_CONVERT_FOR_BUILD_ARG([FC])
    _OPARI2_CONVERT_FOR_BUILD_ARG([CPPFLAGS])
    _OPARI2_CONVERT_FOR_BUILD_ARG([CFLAGS])
    _OPARI2_CONVERT_FOR_BUILD_ARG([CXXFLAGS])
    _OPARI2_CONVERT_FOR_BUILD_ARG([FFLAGS])
    _OPARI2_CONVERT_FOR_BUILD_ARG([FCFLAGS])
    _OPARI2_CONVERT_FOR_BUILD_ARG([LDFLAGS])
    _OPARI2_CONVERT_FOR_BUILD_ARG([LIBS])
])


AC_DEFUN([_OPARI2_CONVERT_FOR_BUILD_ARG],
[
AS_IF([test "x${ac_cv_env_[$1]_FOR_BUILD_set}" = "xset"],
    [[$1]=$ac_cv_env_[$1]_FOR_BUILD_value])dnl
])dnl


dnl dont' use together with AC_SCOREP_WITH_COMPILER_SUITE, intended to be used by OPARI only
AC_DEFUN([AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE],
[
ac_scorep_compilers_frontend="platform-frontend-${ac_scorep_platform}"
ac_scorep_compilers_backend="platform-backend-${ac_scorep_platform}"

m4_pattern_allow([AC_SCOREP_WITH_COMPILER_SUITE])
m4_pattern_allow([AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE])
if test "x${ac_scorep_compiler_suite_called}" != "x"; then
    # We need m4 quoting magic here ...
    AC_MSG_ERROR([cannot use [AC_SCOREP_WITH_COMPILER_SUITE] and [AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE] in one configure.ac.])
else
    ac_scorep_compiler_suite_called="yes"
fi

AC_ARG_WITH([compiler-suite],
            [AS_HELP_STRING([--with-compiler-suite=(gcc|ibm|intel|pgi|studio)], 
                            [The compiler suite used to build this package. Needs to be in $PATH [gcc].])],
            [AS_CASE([$withval],
                     ["gcc"],       [ac_scorep_compilers_frontend="compiler-nocross-gcc"],
                     ["ibm"],       [ac_scorep_compilers_frontend="compiler-nocross-ibm"],
                     ["intel"],     [ac_scorep_compilers_frontend="compiler-nocross-intel"],
                     ["pgi"],       [ac_scorep_compilers_frontend="compiler-nocross-pgi"],
                     ["studio"],    [ac_scorep_compilers_frontend="compiler-nocross-studio"],
                     ["no"],        [AC_MSG_ERROR([option --without-compiler-suite makes no sense.])],
                     [AC_MSG_ERROR([compiler suite "${withval}" not supported by --with-compiler-suite.])])])
AS_IF([test -f "AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_frontend}"],
      [ac_scorep_compilers_frontend="AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_frontend}"],
      [ac_scorep_compilers_frontend="AFS_COMPILER_FILES_COMMON/${ac_scorep_compilers_frontend}"])
AS_IF([test -f "AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_backend}"],
      [ac_scorep_compilers_backend="AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_backend}"],
      [ac_scorep_compilers_backend="AFS_COMPILER_FILES_COMMON/${ac_scorep_compilers_backend}"])
])#AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE
