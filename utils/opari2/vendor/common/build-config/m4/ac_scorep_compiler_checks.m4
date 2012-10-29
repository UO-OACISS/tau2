## -*- mode: autoconf -*-

## 
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2012,
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


AC_DEFUN([AC_SCOREP_COMPILER_INTEL],[
AC_MSG_CHECKING([for intel compiler])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
[[#if defined(__INTEL_COMPILER) || defined(__ICC)
#else
# error "Not an Intel compiler."
#endif
]])],
                  [ac_scorep_compiler_intel="yes"; ac_scorep_compiler_unknown="no"],
                  [ac_scorep_compiler_intel="no"])
AC_MSG_RESULT([$ac_scorep_compiler_intel])
AS_IF([test "x${ac_scorep_compiler_intel}" = "xyes"],
      [ac_scorep_compiler_instrumentation_cppflags="-tcollect"]
       AC_DEFINE([FORTRAN_MANGLED(var)], [var ## _], 
                 [Name of var after mangled by the Fortran compiler.]))
])

##

AC_DEFUN([AC_SCOREP_COMPILER_SUN],[
AC_MSG_CHECKING([for sun compiler])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
[[#if defined(__SUNPRO_C)
#else
# error "Not a Sun compiler."
#endif
]])],
                  [ac_scorep_compiler_sun="yes"; ac_scorep_compiler_unknown="no"],
                  [ac_scorep_compiler_sun="no"])
AC_MSG_RESULT([$ac_scorep_compiler_sun])
AS_IF([test "x${ac_scorep_compiler_sun}" = "xyes"],
      [ac_scorep_compiler_instrumentation_cppflags="-O -Qoption f90comp -phat"]
       AC_DEFINE([FORTRAN_MANGLED(var)], [var ## _], 
                 [Name of var after mangled by the Fortran compiler.]))
])

##

AC_DEFUN([AC_SCOREP_COMPILER_IBM],[
AC_MSG_CHECKING([for ibm compiler])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
[[#if defined(__IBMC__)
# if __IBMC__ >= 800
# else
#  error "Not an IBM XL compiler."
# endif
#else
# error "Not an IBM compiler."
#endif
]])],
                  [ac_scorep_compiler_ibm="yes"; ac_scorep_compiler_unknown="no"],
                  [ac_scorep_compiler_ibm="no"])
AC_MSG_RESULT([$ac_scorep_compiler_ibm])
AS_IF([test "x${ac_scorep_compiler_ibm}" = "xyes"],
      [ac_scorep_compiler_instrumentation_cppflags="-qdebug=function_trace"]
       AC_DEFINE([FORTRAN_MANGLED(var)], [var], 
                 [Name of var after mangled by the Fortran compiler.]))
])

##

AC_DEFUN([AC_SCOREP_COMPILER_PGI],[
AC_MSG_CHECKING([for pgi compiler])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
[[#if defined(__PGI)
#else
# error "Not a PGI compiler."
#endif
]])],
                  [ac_scorep_compiler_pgi="yes"; ac_scorep_compiler_unknown="no"],
                  [ac_scorep_compiler_pgi="no"])
AC_MSG_RESULT([$ac_scorep_compiler_pgi])
AS_IF([test "x${ac_scorep_compiler_pgi}" = "xyes"],
      [ac_scorep_compiler_instrumentation_cppflags="-Mprof=func"]
       AC_DEFINE([FORTRAN_MANGLED(var)], [var ## _], 
                 [Name of var after mangled by the Fortran compiler.]))
])

##

AC_DEFUN([AC_SCOREP_COMPILER_GNU],[
AC_MSG_CHECKING([for gnu compiler])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
[[#if defined(__GNUC__)
#if defined(__INTEL_COMPILER) || defined(__ICC)
# error "Not a GNU compiler."
#endif
#else
# error "Not a GNU compiler."
#endif
]])],
                  [ac_scorep_compiler_gnu="yes"; ac_scorep_compiler_unknown="no"],
                  [ac_scorep_compiler_gnu="no"])
AC_MSG_RESULT([$ac_scorep_compiler_gnu])
AS_IF([test "x${ac_scorep_compiler_gnu}" = "xyes"],
      [ac_scorep_compiler_instrumentation_cppflags="-finstrument-functions"
       AC_DEFINE([FORTRAN_MANGLED(var)], [var ## _], 
                 [Name of var after mangled by the Fortran compiler.])])
])

##

dnl AC_DEFUN([AC_SCOREP_COMPILER_HP],[
dnl AC_MSG_CHECKING([for hp compiler])
dnl AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
dnl [[#if defined(__HP_cc) || defined(__hpux) || defined(__hpua)
dnl #else
dnl # error "Not a HP compiler."
dnl #endif
dnl ]])],
dnl                   [ac_scorep_compiler_hp="yes"; ac_scorep_compiler_unknown="no"],
dnl                   [ac_scorep_compiler_hp="no"])
dnl AC_MSG_RESULT([$ac_scorep_compiler_hp])
dnl AS_IF([test "x${ac_scorep_compiler_hp}" = "xyes"],
dnl       [ac_scorep_compiler_instrumentation_cppflags=""]
dnl        AC_DEFINE([FORTRAN_MANGLED(var)], [hp compiler's Fortran mangling not implemented yet, see ac_scorep_compiler_checks.m4], 
dnl                  [Name of var after mangled by the Fortran compiler.]))
dnl ])

## 

AC_DEFUN([AC_SCOREP_COMPILER_SX],[
AC_MSG_CHECKING([for sx compiler])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
[[#if defined(__SX_cc) || defined(__hpux) || defined(__hpua)
#else
# error "Not a SX compiler."
#endif
]])],
                  [ac_scorep_compiler_sx="yes"; ac_scorep_compiler_unknown="no"],
                  [ac_scorep_compiler_sx="no"])
AC_MSG_RESULT([$ac_scorep_compiler_sx])
AS_IF([test "x${ac_scorep_compiler_sx}" = "xyes"],
      [ac_scorep_compiler_instrumentation_cppflags=""]
       AC_DEFINE([FORTRAN_MANGLED(var)], [sx compiler's Fortran mangling not implemented yet, see ac_scorep_compiler_checks.m4], 
                 [Name of var after mangled by the Fortran compiler.]))
])

## 

AC_DEFUN([AC_SCOREP_COMPILER_CRAY],[
AC_MSG_CHECKING([for cray compiler])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
[[#if defined(_CRAYC)
#else
# error "Not a Cray compiler."
#endif
]])],
                  [ac_scorep_compiler_cray="yes"; ac_scorep_compiler_unknown="no"],
                  [ac_scorep_compiler_cray="no"])
AC_MSG_RESULT([$ac_scorep_compiler_cray])
AS_IF([test "x${ac_scorep_compiler_cray}" = "xyes"],
      [ac_scorep_compiler_instrumentation_cppflags="-hfunc_trace"]
      [CC="${CC} -hnoomp -O2"]
      [CXX="${CXX} -hnoomp -O2"]
      [F77="${F77} -hnoomp -O2"]
      [FC="${FC} -hnoomp -O2"]
       AC_DEFINE([FORTRAN_MANGLED(var)], [var ## _], 
                 [Name of var after mangled by the Fortran compiler.]))
])

##

AC_DEFUN([AC_SCOREP_ATTRIBUTE_ALIGNMENT],[
AC_LANG_PUSH([C])
AC_MSG_CHECKING([for alignment attribute])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],
[[int __attribute__((aligned (16))) tpd;]])],
                [ac_scorep_has_alignment_attribute="yes"],
                [ac_scorep_has_alignment_attribute="no"]
)

if test "x${ac_scorep_has_alignment_attribute}" = "xyes"; then
  AC_DEFINE([FORTRAN_ALIGNED],[__attribute__((aligned (16)))],[Makes C variable alignment consistent with Fortran])
else
  AC_DEFINE([FORTRAN_ALIGNED],[],[Alignment attribute not supported])
fi

AC_MSG_RESULT([$ac_scorep_has_alignment_attribute])
AC_LANG_POP([C])
])

##

AC_DEFUN([AC_SCOREP_COMPILER_CHECKS],[
ac_scorep_compiler_unknown="yes"

ac_scorep_compiler_intel="no"
ac_scorep_compiler_sun="no"
ac_scorep_compiler_ibm="no"
ac_scorep_compiler_pgi="no"
ac_scorep_compiler_gnu="no"
dnl ac_scorep_compiler_hp="no"
ac_scorep_compiler_sx="no"
ac_scorep_compiler_cray="no"

ac_scorep_compiler_instrumentation_cppflags=""

# I (croessel) don't think that more than one test can possibly succeed,
# so I skip extra testing here.
AC_LANG_PUSH([C])
AC_SCOREP_COMPILER_INTEL
AC_SCOREP_COMPILER_SUN
AC_SCOREP_COMPILER_IBM
AC_SCOREP_COMPILER_PGI
AC_SCOREP_COMPILER_GNU
dnl AC_SCOREP_COMPILER_HP
AC_SCOREP_COMPILER_SX
AC_SCOREP_COMPILER_CRAY
AC_LANG_POP([C])

if test "x${ac_scorep_compiler_unknown}" = "xyes"; then
    AC_MSG_WARN([Could not determine compiler vendor. Compiler instrumentation may not work.])
fi

AM_CONDITIONAL([SCOREP_COMPILER_INTEL], [test "x${ac_scorep_compiler_intel}" = "xyes"])
AM_CONDITIONAL([SCOREP_COMPILER_SUN],   [test "x${ac_scorep_compiler_sun}"   = "xyes"])
AM_CONDITIONAL([SCOREP_COMPILER_IBM],   [test "x${ac_scorep_compiler_ibm}"   = "xyes"])
AM_CONDITIONAL([SCOREP_COMPILER_PGI],   [test "x${ac_scorep_compiler_pgi}"   = "xyes"])
AM_CONDITIONAL([SCOREP_COMPILER_GNU],   [test "x${ac_scorep_compiler_gnu}"   = "xyes"])
dnl AM_CONDITIONAL([SCOREP_COMPILER_HP],    [test "x${ac_scorep_compiler_hp}"    = "xyes"])
AM_CONDITIONAL([SCOREP_COMPILER_SX],    [test "x${ac_scorep_compiler_sx}"    = "xyes"])
AM_CONDITIONAL([SCOREP_COMPILER_CRAY],  [test "x${ac_scorep_compiler_cray}"  = "xyes"])
])

##

AC_DEFUN([AC_SCOREP_COMPILER_INSTRUMENTATION_FLAGS],[
AC_REQUIRE([AC_SCOREP_COMPILER_CHECKS])dnl
AC_ARG_WITH([extra-instrumentation-flags],
            [AS_HELP_STRING([--with-extra-instrumentation-flags=flags],
                            [Add additional instrumentation flags.])],
            [ac_scorep_with_extra_instrumentation_cppflags=$withval],
            [ac_scorep_with_extra_instrumentation_cppflags=""])

AC_SUBST([COMPILER_INSTRUMENTATION_CPPFLAGS], ["${ac_scorep_compiler_instrumentation_cppflags} ${ac_scorep_with_extra_instrumentation_cppflags}"])

AC_MSG_NOTICE([using instrumentation flags: ${ac_scorep_compiler_instrumentation_cppflags} ${ac_scorep_with_extra_instrumentation_cppflags}])
AC_SCOREP_SUMMARY_VERBOSE([instrumentation flags: ${ac_scorep_compiler_instrumentation_cppflags} ${ac_scorep_with_extra_instrumentation_cppflags}])
])
