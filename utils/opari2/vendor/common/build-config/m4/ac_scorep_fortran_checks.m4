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


## file       ac_scorep_fortran_checks.m4

AC_DEFUN([AC_SCOREP_FORTRAN_SUPPORT_ALLOCATABLE],[
AC_LANG_PUSH(Fortran)
AC_MSG_CHECKING([whether double precision, allocatable arrays are supported])
AC_COMPILE_IFELSE([
       PROGRAM test
       TYPE mydata
       double precision, allocatable :: afF(:,:)
       END TYPE mydata
       END PROGRAM test
], [scorep_support_allocatable="yes"], [scorep_support_allocatable="no"]
) #AC_COMPILE_IFELSE
AC_LANG_POP(Fortran)
AC_MSG_RESULT($scorep_support_allocatable)
AM_CONDITIONAL(FORTRAN_SUPPORT_ALLOCATABLE, test "x$scorep_support_allocatable" = "xyes")
]) #AC_DEFUN
