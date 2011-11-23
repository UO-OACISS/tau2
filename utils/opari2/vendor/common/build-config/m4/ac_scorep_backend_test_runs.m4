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


## file       ac_scorep_backend_test_runs.m4
## maintainer Christian Roessel <c.roessel@fz-juelich.de>

AC_DEFUN([AC_SCOREP_BACKEND_TEST_RUNS], [
AC_ARG_ENABLE([backend-test-runs],
              [AS_HELP_STRING([--enable-backend-test-runs], 
                              [Run tests at make check [no]. If disabled, tests are still build at make check. Additionally, scripts (scorep_*tests.sh) containing the tests are generated in <builddir>/build-backend.])],
              [ac_scorep_enable_backend_test_runs=$enableval],
              [ac_scorep_enable_backend_test_runs="no"])

AM_CONDITIONAL([BACKEND_TEST_RUNS], [test "x$ac_scorep_enable_backend_test_runs" = "xyes"])
])
