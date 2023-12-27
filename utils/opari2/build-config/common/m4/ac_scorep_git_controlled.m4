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
## Copyright (c) 2009-2011, 2019-2020,
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


AC_DEFUN([_AC_SCOREP_GIT_CONTROLLED],
[
AC_MSG_CHECKING([Git controlled])
ac_scorep_git_controlled="no"

# test if ${afs_srcdir} is a git top-level, not any parent directory:
# * if ${afs_srcdir} is a top-level, then the prefix is empty (e.g., we are git controlled)
# * if ${afs_srcdir} is below a top-level, then it wont be empty (e.g., we operate in a
#   tarball, which was extracted below a top-level)
# * if git could not find any top-level, it prints an error to stderr and stop,
#   we catch this error, which makes the test also fail (e.g., we operate in a
#   tarball which is *not* below any top-level)
AS_IF([test -z "$(
    unset $(git rev-parse --local-env-vars 2>/dev/null) &&
    cd ${afs_srcdir} &&
    git rev-parse --show-prefix 2>&1)"],
      [ac_scorep_git_controlled="yes"
       AC_DEFINE([SCOREP_IN_DEVELOPEMENT], [], [Defined if we are working from git.])],
      [AC_DEFINE([SCOREP_IN_PRODUCTION], [], [Defined if we are working from a make dist generated tarball.])])
AC_MSG_RESULT([$ac_scorep_git_controlled])
])

AC_DEFUN([AC_SCOREP_GIT_CONTROLLED],
[
AC_REQUIRE([_AC_SCOREP_GIT_CONTROLLED])
AM_CONDITIONAL([GIT_CONTROLLED], [test "x${ac_scorep_git_controlled}" = xyes])
AFS_SUMMARY_VERBOSE([Git controlled], [$ac_scorep_git_controlled])
])
