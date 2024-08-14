## -*- mode: autoconf -*-

##
## This file is part of the Score-P software ecosystem (http://www.score-p.org)
##
## Copyright (c) 2018,
## Forschungszentrum Juelich GmbH, Germany
##
## Copyright (c) 2019,
## Technische Universitaet Dresden, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##


# AFS_MAINTAINER_MODE
# -------------------
# Enable maintainer mode only when building from repository sources.
# We want automatic regeneration of files when our build environemnt
# is available, i.e., when correct and patched versions of autotools
# are available. Otherwise we want to prevent automatic regeneration
# of files.
AC_DEFUN([AFS_MAINTAINER_MODE],
[AC_REQUIRE([AC_SCOREP_GIT_CONTROLLED])
AS_IF([test "x${ac_scorep_git_controlled}" = xyes],
    [AM_MAINTAINER_MODE([enable])],
    [AM_MAINTAINER_MODE([])])dnl
]) # AFS_MAINTAINER_MODE
