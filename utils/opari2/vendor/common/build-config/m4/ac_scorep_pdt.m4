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


AC_DEFUN([AC_SCOREP_PDT], [

## Evalute parameters
AC_ARG_WITH(pdt, [AS_HELP_STRING([--with-pdt=path_to_binaries], [Specifies the path where the binaries of the program database toolkit (PDT) are located])],[
    pdt_path=$withval
    have_pdt=yes
],[have_pdt=no])

AC_SUBST(PDT_PATH,"$pdt_path")
AM_CONDITIONAL(HAVE_PDT,[test x$have_pdt = xyes])

])
