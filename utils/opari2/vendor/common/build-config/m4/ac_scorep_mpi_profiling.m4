dnl -*- mode: autoconf -*-

dnl 
dnl This file is part of the Score-P software (http://www.score-p.org)
dnl
dnl Copyright (c) 2009-2011, 
dnl    RWTH Aachen, Germany
dnl    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
dnl    Technische Universitaet Dresden, Germany
dnl    University of Oregon, Eugene, USA
dnl    Forschungszentrum Juelich GmbH, Germany
dnl    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
dnl    Technische Universitaet Muenchen, Germany
dnl
dnl See the COPYING file in the package base directory for details.
dnl

dnl file       build-config/m4/ac_scorep_mpi_profiling.m4
dnl maintainer Christian Roessel <c.roessel@fz-juelich.de>

AC_DEFUN([AC_SCOREP_MPI_PROFILING_HEADERS],
[
ac_scorep_have_mpi_profiling_headers="yes"
AC_CHECK_HEADER([string.h], [], [ac_scorep_have_mpi_profiling_headers="no"])

AM_CONDITIONAL([HAVE_MPI_PROFILING_HEADERS], [test "x${ac_scorep_have_mpi_profiling_headers}" = "xyes"])

AC_MSG_CHECKING([for MPI profiling headers])
AC_MSG_RESULT([${ac_scorep_have_mpi_profiling_headers}])
]) 
