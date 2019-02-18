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
## Copyright (c) 2009-2015,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2013,
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

## file afs_summary.m4

# AFS_SUMMARY_INIT
# ----------------
# Initializes the summary system and adds the package header (possibly
# including the sub-build name) to it. It removes config.summary files
# from previous configure runs recursively, therefore you need to call
# AFS_SUMMARY_INIT before any sub-configures.
# The sub-build name is used from the `AFS_PACKAGE_BUILD` variable
# set by the AFS_PACKAGE_INIT macro.
AC_DEFUN([AFS_SUMMARY_INIT], [
rm -f AC_PACKAGE_TARNAME.summary
LC_ALL=C find . -name config.summary -exec rm -f '{}' \;
m4_define([_AFS_SUMMARY_INDENT], [m4_ifndef([AFS_PACKAGE_BUILD], [], [  ])])dnl
m4_define([_AFS_SUMMARY_FILE], [config.summary])
cat >_AFS_SUMMARY_FILE <<_ACEOF
AS_HELP_STRING(_AFS_SUMMARY_INDENT[]AC_PACKAGE_NAME[ ]m4_ifndef([AFS_PACKAGE_BUILD], AC_PACKAGE_VERSION, [(]AFS_PACKAGE_BUILD[)])[:], [], 32, 128)
_ACEOF
])


# AFS_SUMMARY_SECTION_BEGIN( [DESCR, [VALUE]] )
# ---------------------------------------------
# Starts a new section, optionally with the given description.
# All summary lines after this call will be indented by 2 spaces.
# Close the section with 'AFS_SUMMARY_SECTION_END'.
AC_DEFUN([AFS_SUMMARY_SECTION_BEGIN], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

m4_ifnblank($1, AFS_SUMMARY([$1], [$2]))
m4_pushdef([_AFS_SUMMARY_INDENT], _AFS_SUMMARY_INDENT[  ])dnl
])


# AFS_SUMMARY_SECTION_END
# -----------------------
# Close a previously opened section with 'AFS_SUMMARY_SECTION_BEGIN'.
AC_DEFUN([AFS_SUMMARY_SECTION_END], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

m4_popdef([_AFS_SUMMARY_INDENT])dnl
])


# AFS_SUMMARY_PUSH
# ----------------
# Starts a new section (see 'AFS_SUMMARY_SECTION_BEGIN'), but without a
# section heading and it collects all subsequent summaries and sections in
# a hold space.
# All summary lines after this call will be indented by 2 spaces.
# Output the hold space with 'AFS_SUMMARY_POP'.
AC_DEFUN([AFS_SUMMARY_PUSH], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

AFS_SUMMARY_SECTION_BEGIN
m4_pushdef([_AFS_SUMMARY_FILE], _AFS_SUMMARY_FILE[.x])dnl
: >_AFS_SUMMARY_FILE
])


# AFS_SUMMARY_POP( DESCR, VALUE )
# -------------------------------
# Close a previously opened section with 'AFS_SUMMARY_PUSH'. Outputs the
# section header with DESCR and VALUE, and than outputs the summary from the
# hold space.
AC_DEFUN([AFS_SUMMARY_POP], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

m4_define([_afs_summary_tmp], _AFS_SUMMARY_FILE)dnl
m4_popdef([_AFS_SUMMARY_FILE])dnl
AFS_SUMMARY_SECTION_END

AFS_SUMMARY([$1], [$2])
cat _afs_summary_tmp >>_AFS_SUMMARY_FILE
rm _afs_summary_tmp
])


# AFS_SUMMARY( DESCR, VALUE )
# ---------------------------
# Generates a summary line with the given description and value.
AC_DEFUN([AFS_SUMMARY], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

cat >>_AFS_SUMMARY_FILE <<_ACEOF
AS_HELP_STRING(_AFS_SUMMARY_INDENT[  $1:], [$2], 32, 128)
_ACEOF
])


# AFS_SUMMARY_VERBOSE( DESCR, VALUE )
# -----------------------------------
# Generates a summary line with the given description and value, but only
# if ./configure was called with --verbose
AC_DEFUN([AFS_SUMMARY_VERBOSE], [

AS_IF([test "x${verbose}" = "xyes"], [
    AFS_SUMMARY([$1], [$2])
])
])


# internal
AC_DEFUN([_AFS_SUMMARY_SHOW], [

AS_ECHO([""])
cat AC_PACKAGE_TARNAME.summary
])

# AFS_SUMMARY_COLLECT( [SHOW-COND] )
# --------------------------------
# Collectes the summary of all configures recursively into the file
# $PACKAGE.summary. If SHOW-COND is not given, or the expression
# evaluates to true the summary is also printed to stdout.
# Should be called after AC_OUTPUT.
AC_DEFUN([AFS_SUMMARY_COLLECT], [
    (
    AS_ECHO(["Configure command:"])
    prefix="  $as_myself "
    printf "%-32s" "$prefix"
    padding="                                "
    AS_IF([test ${#prefix} -gt 32], [
        sep="\\$as_nl$padding"
    ], [
        sep=""
    ])

    eval "set x $ac_configure_args"
    shift
    AS_FOR([ARG], [arg], [], [
        AS_CASE([$arg],
        [*\'*], [arg="`$as_echo "$arg" | sed "s/'/'\\\\\\\\''/g"`"])
        AS_ECHO_N(["$sep'$arg'"])
        sep=" \\$as_nl$padding"
    ])
    AS_ECHO([""])

    AS_IF([test x"${MODULESHOME:+set}" = x"set"], [
        AS_ECHO([""])
        AS_ECHO(["Loaded modules:"])
        sep=""
        AS_IF([test x"${LOADEDMODULES:+set}" = x"set"], [
            prefix="  module load "
            printf "%-32s" "$prefix"
            IFS=': ' eval 'set x $LOADEDMODULES'
            shift
            AS_FOR([MODULE], [module], [], [
                AS_ECHO_N(["$sep$module"])
                sep=" \\$as_nl$padding"
            ])
            AS_ECHO([""])
        ], [
            AS_ECHO(["  No modules loaded"])
        ])
    ])

    AS_ECHO([""])
    sep="Configuration summary:"
    LC_ALL=C find . -name config.summary |
        LC_ALL=C $AWK -F "config.summary" '{print $[]1}' |
        LC_ALL=C sort |
        LC_ALL=C $AWK '{print $[]0 "config.summary"}' |
        while read summary
    do
        AS_ECHO(["$sep"])
        cat $summary
        sep=""
    done
    ) >AC_PACKAGE_TARNAME.summary
    m4_ifblank($1,
              [_AFS_SUMMARY_SHOW],
              [AS_IF([$1], [_AFS_SUMMARY_SHOW])])
])
