## -*- mode: autoconf -*-

##
## This file is part of the Score-P software (http://www.score-p.org)
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
## Copyright (c) 2009-2011, 2015,
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

AC_DEFUN([AC_SCOREP_SVN_CONTROLLED],
[
ac_scorep_svn_controlled="no"
svn info ${srcdir} > /dev/null 2>&1
AS_IF([test $? -eq 0],
      [ac_scorep_svn_controlled="yes"
       AC_DEFINE([SCOREP_IN_DEVELOPEMENT], [], [Defined if we are working from svn.])],
      [AC_DEFINE([SCOREP_IN_PRODUCTION], [], [Defined if we are working from a make dist generated tarball.])])
AM_CONDITIONAL([SVN_CONTROLLED], [test "x${ac_scorep_svn_controlled}" = xyes])
])
