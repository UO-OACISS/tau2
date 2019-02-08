## -*- mode: autoconf -*-

##
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2012,
## RWTH Aachen University, Germany
##
## Copyright (c) 2009-2012,
## Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##
## Copyright (c) 2009-2014,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2012,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2012,
## Forschungszentrum Juelich GmbH, Germany
##
## Copyright (c) 2009-2012,
## German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##
## Copyright (c) 2009-2012,
## Technische Universitaet Muenchen, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##

## file       ac_common_package.m4

# AFS_PACKAGE_INIT
# ----------------
# Common AC_DEFINE's and AC_SUBST's for the package based on its name.
# Call this only in the top-level configure.ac, sub-builds should call
# `AFS_PACKAGE_BUILD_INIT`.
#
# List of defined autoconf macros:
#  `AFS_PACKAGE_name`::       The tarname of the package in lower case
#  `AFS_PACKAGE_NAME`::       The tarname of the package in upper case
#  `AFS_PACKAGE_TO_TOP`::     The relative path to the top-level configure
#                             including a trailing slash (empty for the
#                             top-level configure itself)
# List of provided automake substitutions:
#  `AFS_PACKAGE_name`::       The value of AFS_PACKAGE_name
#  `AFS_PACKAGE_NAME`::       The value of AFS_PACKAGE_NAME
#
AC_DEFUN_ONCE([AFS_PACKAGE_INIT], [

m4_case([$#],
    [0], [],
    [1], [m4_ifnblank($1, [m4_fatal([$0: too many arguments: $@])])],
    [m4_fatal([$0: too many arguments: $@])])dnl

m4_pushdef([_afs_package_tmp], m4_tolower(AC_PACKAGE_TARNAME))dnl
AC_SUBST([AFS_PACKAGE_name], _afs_package_tmp)
m4_define([AFS_PACKAGE_name], _afs_package_tmp)dnl
m4_popdef([_afs_package_tmp])dnl

m4_pushdef([_afs_package_tmp], m4_toupper(AC_PACKAGE_TARNAME))dnl
AC_SUBST([AFS_PACKAGE_NAME], _afs_package_tmp)
m4_define([AFS_PACKAGE_NAME], _afs_package_tmp)dnl
m4_popdef([_afs_package_tmp])dnl

dnl May be redefined by AFS_PACKAGE_BUILD_INIT
m4_define([AFS_PACKAGE_TO_TOP], [])dnl
])

# AFS_PACKAGE_BUILD_INIT(BUILD-NAME, [TO-TOP])
# --------------------------------------------
# Initializes an sub-build configure. It is sufficient to call
# `AFS_PACKAGE_BUILD_INIT`, `AFS_PACKAGE_INIT` will be called automatically.
#
# The relative path to the top-level configure to this configure
# can be specified with TO-TOP, defaulting to `../`.
#
# List of defined autoconf macros:
#  `AFS_PACKAGE_BUILD`::      The normalized name of the build (e.g., 'backend',
#                             'MPI backend')
#  `AFS_PACKAGE_BUILD_name`:: The build name usable as a symbol in lower case
#                             (e.g., backend, mpi_backend)
#  `AFS_PACKAGE_BUILD_NAME`:: The build name usable as a symbol in upper case
#                             (e.g., BACKEND, MPI_BACKEND)
# List of provided automake substitutions:
#  'AFS_PACKAGE_BUILD_name'   The value of AFS_PACKAGE_BUILD_name
#  'AFS_PACKAGE_BUILD_NAME'   The value of AFS_PACKAGE_BUILD_NAME
#  `AFS_PACKAGE_TO_TOP`::     The value of AFS_PACKAGE_TO_TOP
#  'afs_srcdir'               The relative path to the source directory (i.e.,
#                             where the top-level configure resides)
# List of provided config header defines:
#  `AFS_PACKAGE_BUILD`::      The value of AFS_PACKAGE_BUILD as a string
#                             constant
#  `AFS_PACKAGE_BUILD_name`:: The value of AFS_PACKAGE_BUILD_name
#  `AFS_PACKAGE_BUILD_NAME`:: The value of AFS_PACKAGE_BUILD_NAME
#  `AFS_PACKAGE_SRCDIR`::     The relative path to the source directory as
#                             string constant including a trailing slash
#  `AFS_PACKAGE_name`::       The value of AFS_PACKAGE_name
#  `AFS_PACKAGE_NAME`::       The value of AFS_PACKAGE_NAME
#
AC_DEFUN_ONCE([AFS_PACKAGE_BUILD_INIT], [
AC_REQUIRE([AFS_PACKAGE_INIT])dnl

m4_case([$#],
    [0], [m4_fatal([$0: missing arguments])],
    [1], [m4_ifblank(m4_normalize($1), [m4_fatal([$0: empty BUILD-NAME argument])])],
    [2], [m4_ifblank(m4_normalize($1), [m4_fatal([$0: empty BUILD-NAME argument])])],
    [m4_fatal([$0: too many arguments: $@])])dnl

m4_pushdef([_afs_package_tmp], m4_normalize($1))dnl
AC_DEFINE_UNQUOTED([AFS_PACKAGE_BUILD], "_afs_package_tmp",
    [Name of the sub-build.])
m4_define([AFS_PACKAGE_BUILD], _afs_package_tmp)dnl
m4_popdef([_afs_package_tmp])dnl

dnl Overwrites AFS_PACKAGE_TO_TOP defined in AFS_PACKAGE_INIT.
m4_pushdef([_afs_package_tmp], m4_default([$2], [../]))dnl
dnl undefine previously defined by AFS_PACKAGE_INIT
m4_undefine([AFS_PACKAGE_TO_TOP])
AC_SUBST([AFS_PACKAGE_TO_TOP], _afs_package_tmp)
m4_define([AFS_PACKAGE_TO_TOP], _afs_package_tmp)dnl
m4_if(m4_substr(AFS_PACKAGE_TO_TOP, decr(len(AFS_PACKAGE_TO_TOP))), [/],
    [], [m4_fatal([$0: no trailing slash in TO-TOP argument: ]AFS_PACKAGE_TO_TOP)])
m4_popdef([_afs_package_tmp])dnl

# when building inplace, $srcdir equals ., ignore $srcdir than
AS_CASE([$srcdir],
    [.], [afs_srcdir="]AFS_PACKAGE_TO_TOP["],
    [afs_srcdir="${srcdir}/]AFS_PACKAGE_TO_TOP["])dnl
AC_DEFINE_UNQUOTED([[AFS_PACKAGE_SRCDIR]],
    ["${afs_srcdir}"], [Relative path to the top-level source directory.])
AC_SUBST([afs_srcdir])

m4_pushdef([_afs_package_tmp],
    m4_bpatsubst(m4_tolower(m4_normalize($1)), [[^a-z0-9]+], [_]))dnl
AC_DEFINE_UNQUOTED([AFS_PACKAGE_BUILD_name], _afs_package_tmp,
    [Symbol name of the sub-build in lower case.])
AC_SUBST([AFS_PACKAGE_BUILD_name], _afs_package_tmp)
m4_define([AFS_PACKAGE_BUILD_name], _afs_package_tmp)dnl
m4_popdef([_afs_package_tmp])dnl

m4_pushdef([_afs_package_tmp],
    m4_toupper(AFS_PACKAGE_BUILD_name))dnl
AC_DEFINE_UNQUOTED([AFS_PACKAGE_BUILD_NAME], _afs_package_tmp,
    [Symbol name of the sub-build in upper case.])
AC_SUBST([AFS_PACKAGE_BUILD_NAME], _afs_package_tmp)
m4_define([AFS_PACKAGE_BUILD_NAME], _afs_package_tmp)dnl
m4_popdef([_afs_package_tmp])dnl

AC_DEFINE_UNQUOTED([[AFS_PACKAGE_name]], AFS_PACKAGE_name,
    [The package name usable as a symbol in lower case.])

AC_DEFINE_UNQUOTED([[AFS_PACKAGE_NAME]], AFS_PACKAGE_NAME,
    [The package name usable as a symbol in upper case.])
])

# AC_SCOREP_DEFINE_HAVE(VARIABLE, VALUE[, DESCRIPTION])
# ------------------------------------------------------
# Like AC_DEFINE, but prepends the HAVE_ prefix and also defines the
# HAVE_'PACKAGE_BUILD'_ variant, if in a sub configure by utilizing the
# AFS_PACKAGE_BUILD_NAME macro.
#
AC_DEFUN([AC_SCOREP_DEFINE_HAVE], [
AC_DEFINE([HAVE_]$1, [$2], [$3])
m4_ifdef([AFS_PACKAGE_BUILD], [
    AC_DEFINE([HAVE_]AFS_PACKAGE_BUILD_NAME[_]$1, [$2], [$3])
])
])

# AC_SCOREP_COND_HAVE(VARIABLE, CONDITION[, DESCRIPTION[, COND_TRUE[, COND_FALSE]]])
# --------------------------------------------------
# Convenience macro to define a AM_CONDITIONAL and always a
# AC_SCOREP_DEFINE_HAVE at once. VARIABLE will be prefixed with HAVE_
#
AC_DEFUN([AC_SCOREP_COND_HAVE], [
AM_CONDITIONAL(HAVE_[]$1, [$2])
AM_COND_IF(HAVE_[]$1,
           [AC_SCOREP_DEFINE_HAVE([$1], [1], [$3])
            $4],
           [AC_SCOREP_DEFINE_HAVE([$1], [0], [$3])
            $5])
])
