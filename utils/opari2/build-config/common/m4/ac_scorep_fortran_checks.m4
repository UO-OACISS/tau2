## -*- mode: autoconf -*-

##
## This file is part of the Score-P software ecosystem (http://www.score-p.org)
##
## Copyright (c) 2009-2011,
## RWTH Aachen University, Germany
##
## Copyright (c) 2009-2011,
## Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##
## Copyright (c) 2009-2011,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2011,
## Forschungszentrum Juelich GmbH, Germany
##
## Copyright (c) 2009-2011,
## German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##
## Copyright (c) 2009-2011,
## Technische Universitaet Muenchen, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##


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
