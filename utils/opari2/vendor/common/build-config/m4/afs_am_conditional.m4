dnl -*- mode: autoconf -*-

dnl
dnl This file is part of the Score-P software (http://www.score-p.org)
dnl
dnl Copyright (c) 2013
dnl Forschungszentrum Juelich GmbH, Germany
dnl
dnl This software may be modified and distributed under the terms of
dnl a BSD-style license.  See the COPYING file in the package base
dnl directory for details.
dnl

dnl file afs_am_conditional.m4


# AFS_AM_CONDITIONAL(NAME, SHELL-CONDITION, BOOLEAN-VALUE)
# -------------------------------------
# Define an automake conditional as in AM_CONDITIONAL, but let it
# default to BOOLEAN-VALUE (true|false) if this macro is not
# encountered during configure. This allows you to define automake
# conditionals conditionally.
AC_DEFUN([AFS_AM_CONDITIONAL],
[m4_case([$3], [true], [],
               [false], [],
               [m4_fatal([AFS_AM_CONDITIONAL requires the third parameter to be either 'true' or 'false'.])])dnl
m4_divert_text([DEFAULTS],
[m4_if([$3], [true], [dnl
$1_TRUE=
$1_FALSE='#'],[dnl
$1_TRUE='#'
$1_FALSE=])
])dnl
AM_CONDITIONAL($1, [$2])dnl
])dnl
