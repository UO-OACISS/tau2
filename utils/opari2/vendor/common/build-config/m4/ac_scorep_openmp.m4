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


## file       ac_scorep_openmp.m4
## maintainer Christian Roessel <c.roessel@fz-juelich.de>

AC_DEFUN([AC_SCOREP_OPENMP],
[
# set *FLAGS temporarily to "" because the cray compiler ignores
# the OpenMP flags if -g is set (which is done per default by configure)

AC_LANG_PUSH([C])
scorep_cflags_save=${CFLAGS}
CFLAGS=""
AC_OPENMP
CFLAGS=${scorep_cflags_save}
AC_LANG_POP([C])
AM_CONDITIONAL([OPENMP_SUPPORTED], 
               [test "x${ac_cv_prog_c_openmp}" != "xunsupported"])

if test "x${ac_cv_prog_c_openmp}" = "xunsupported"; then
  AC_MSG_WARN([no suitbale OpenMP compilers found. SCOREP OpenMP and hybrid libraries will not be build.])
fi
AC_SCOREP_SUMMARY([OpenMP support], [${ac_cv_prog_c_openmp}])

AC_LANG_PUSH([C++])
scorep_cxxflags_save=${CXXFLAGS}
CXXFLAGS=""
AC_OPENMP
CXXFLAGS=${scorep_cxxflags_save}
AC_LANG_POP([C++])

AC_LANG_PUSH([Fortran 77])
scorep_fflags_save=${FFLAGS}
FFLAGS=""
AC_OPENMP
FFLAGS=${scorep_fflags_save}
AC_LANG_POP([Fortran 77])

AC_LANG_PUSH([Fortran])
scorep_fcflags_save=${FCFLAGS}
FCFLAGS=""
AC_OPENMP
FCFLAGS=${scorep_fflags_save}
AC_LANG_POP([Fortran])
])
