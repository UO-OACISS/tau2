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

dnl file       build-config/m4/ac_scorep_online_access.m4
dnl maintainer Christian Roessel <c.roessel@fz-juelich.de>

AC_DEFUN([AC_SCOREP_ONLINE_ACCESS_HEADERS],
[
ac_scorep_have_online_access="no"
ac_scorep_have_online_access_flex="no"
ac_scorep_have_online_access_headers="no"

AC_CHECK_HEADERS([stdio.h strings.h ctype.h netdb.h sys/types.h sys/socket.h netinet/in.h unistd.h string.h], [ac_scorep_have_online_access_headers="yes"], [])

AM_PROG_LEX

AC_MSG_CHECKING([for a suitable version of flex])
[flex_version=`${LEX} -V | sed 's/[a-z,\.]//g'`]
if test "x${LEX}" != "x:" -a ${flex_version} -gt 254; then
ac_scorep_have_online_access_flex="yes"
fi
AC_MSG_RESULT([${ac_scorep_have_online_access_flex}])

AC_PROG_YACC

if test "x${ac_scorep_have_online_access_headers}" = "xyes" -a "x${ac_scorep_have_online_access_flex}" = "xyes"; then
ac_scorep_have_online_access="yes"
fi

if test "x${ac_scorep_platform}" = "xbgp" -o "x${ac_scorep_platform}" = "xbgl"; then
ac_scorep_have_online_access="no"
fi

AM_CONDITIONAL([HAVE_ONLINE_ACCESS_HEADERS], [test "x${ac_scorep_have_online_access}" = "xyes" ])

AC_MSG_CHECKING([for online access possible])
AC_MSG_RESULT([${ac_scorep_have_online_access}])
]) 
