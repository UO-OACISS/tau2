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


## file  build-config/m4/opari2_fortran.m4

AC_DEFUN([AC_SCOREP_HAVE_FC],[
    scorep_have_fc="no"
    AC_REQUIRE([AC_PROG_FC])
    AC_MSG_CHECKING([whether the fortran compiler ${FC} works])
    AC_LANG_PUSH([Fortran])
    AC_LINK_IFELSE([
      program main
      implicit none
      integer tid
      end
                   ],
                   [scorep_have_fc="yes"],
                   [])
    AC_LANG_POP([Fortran])
    AM_CONDITIONAL([SCOREP_HAVE_FC], [test "x${scorep_have_fc}" == "xyes"])
    AC_MSG_RESULT([${scorep_have_fc}])
])


AC_DEFUN([AC_SCOREP_HAVE_F77],[
    scorep_have_f77="no"
    AC_REQUIRE([AC_PROG_F77])
    AC_MSG_CHECKING([whether the fortran 77 compiler ${F77} works])
    AC_LANG_PUSH([Fortran 77])
    AC_LINK_IFELSE([
      program main
      implicit none
      integer tid
      end
                   ],
                   [scorep_have_f77="yes"],
                   [])
    AC_LANG_POP([Fortran 77])
    AM_CONDITIONAL([SCOREP_HAVE_F77], [test "x${scorep_have_f77}" == "xyes"])
    AC_MSG_RESULT([${scorep_have_f77}])
])
