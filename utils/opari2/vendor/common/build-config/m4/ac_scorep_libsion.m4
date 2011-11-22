## -*- mode: autoconf -*-

##
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2011,
##    RWTH Aachen, Germany
##    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##    Technische Universitaet Dresden, Germany
##    University of Oregon, Eugene, USA
##    Forschungszentrum Juelich GmbH, Germany
##    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##    Technische Universitaet Muenchen, Germany
##
## See the COPYING file in the package base directory for details.
##

## file       ac_scorep_libsion.m4
## maintainer Christian Roessel <c.roessel@fz-juelich.de>

# output of sionconfig on jugene:
# sionconfig --be --ser --cflags
# -I/usr/local/sionlib/v1.2p2/include -DBGP -D_SION_BGP
# sionconfig --be --ser --libs
# -L/usr/local/sionlib/v1.2p2/lib -lsionser_32
# sionconfig --be --ser --path
# /usr/local/sionlib/v1.2p2
# sionconfig --be --mpi --cflags
# -I/usr/local/sionlib/v1.2p2/include -DBGP -DSION_MPI -D_SION_BGP
# sionconfig --be --mpi --libs
# -L/usr/local/sionlib/v1.2p2/lib -lsion_32 -lsionser_32
# sionconfig --be --mpi --path
# /usr/local/sionlib/v1.2p2
# sionconfig --fe --ser --cflags
# -I/usr/local/sionlib/v1.2p2/include -DBGP -D_SION_BGP
# sionconfig --fe --ser --libs
# -L/usr/local/sionlib/v1.2p2/lib -lsionserfe_32
# sionconfig --fe --ser --path
# /usr/local/sionlib/v1.2p2
# sionconfig --fe --mpi --cflags
# -I/usr/local/sionlib/v1.2p2/include -DBGP -DSION_MPI -D_SION_BGP
# sionconfig --fe --mpi --libs
# -L/usr/local/sionlib/v1.2p2/lib -lsionfe_32 -lsionserfe_32
# sionconfig --fe --mpi --path
# /usr/local/sionlib/v1.2p2


# AC_SCOREP_LIBSION(SERIAL|OMP|MPI|MPI_OMP)
AC_DEFUN([AC_SCOREP_LIBSION],
[
m4_case([$1], [SERIAL], [], [OMP], [], [MPI], [], [MPI_OMP], [], [m4_fatal([parameter must be either SERIAL, OMP, MPI or MPI_OMP])])

# make SIONCONFIG precious as we use it in AC_CHECK_PROG
AC_ARG_VAR([SIONCONFIG], [Absolute path to sionconfig, including "sionconfig".])

AC_ARG_WITH([sionconfig],
            [AS_HELP_STRING([--with-sionconfig=(yes|no|<path-to-sionconfig>)],
                            [Whether to use sionconfig and where to find it. "yes" assumes it is in PATH [no].])],
            # action-if-given
            [AS_CASE([$withval],
                     ["yes"], [scorep_with_sionconfig="yes"],
                     ["no"],  [scorep_with_sionconfig="no"],
                     [scorep_with_sionconfig="$withval"])],
            # action-if-not-given
            [scorep_with_sionconfig="no"])

# macro-internal variables
scorep_sion_cppflags=""
scorep_sion_ldflags=""
scorep_sion_libs=""
scorep_have_sion="no"

if test "x${scorep_with_sionconfig}" != "xno"; then
    if test "x${scorep_with_sionconfig}" = "xyes"; then
        AC_CHECK_PROG([SIONCONFIG], [sionconfig], [`which sionconfig`], ["no"])
    else
        AC_CHECK_PROG([SIONCONFIG], [sionconfig], [${scorep_with_sionconfig}/sionconfig], ["no"], [${scorep_with_sionconfig}])
    fi

    if test "x${SIONCONFIG}" != "xno"; then
        AC_LANG_PUSH([C])
        cppflags_save=$CPPFLAGS
        ldflags_save=$LDFLAGS
        libs_save=$LIBS

        scorep_have_sion="yes"

        sionconfig_febe_flag=""
        if test "x${ac_scorep_backend}" = "xyes"; then
            sionconfig_febe_flag="--be"
        elif test "x${ac_scorep_frontend}" = "xyes"; then
            sionconfig_febe_flag="--fe"
        fi

        AS_CASE([${build_cpu}],
                [i?86],   [sionconfig_architecture_flags="--32"],
                [x86_64], [sionconfig_architecture_flags="--64"],
                [sionconfig_architecture_flags=""])

        m4_case([$1], [SERIAL],  [sionconfig_paradigm_flag="--ser"],
                      [OMP],     [sionconfig_paradigm_flag="--ser"],
                      [MPI],     [sionconfig_paradigm_flag="--mpi"],
                      [MPI_OMP], [sionconfig_paradigm_flag="--mpi"])

        scorep_sion_cppflags=`$SIONCONFIG $sionconfig_febe_flag $sionconfig_paradigm_flag --cflags`
        CPPFLAGS="$scorep_sion_cppflags $CPPFLAGS"
        AC_CHECK_HEADER([sion.h], [], [scorep_have_sion="no"; scorep_sion_cppflags=""])

        if test "x${scorep_have_sion}" = "xyes"; then
            scorep_sion_ldflags=`$SIONCONFIG ${sionconfig_febe_flag} ${sionconfig_paradigm_flag} ${sionconfig_architecture_flags} --libs | \
                                 awk '{for (i=1; i<=NF; i++) {if ([index]($i, "-L") == 1){ldflags = ldflags " " $i}}}END{print ldflags}'`
            scorep_sion_libs=`$SIONCONFIG ${sionconfig_febe_flag} ${sionconfig_paradigm_flag} ${sionconfig_architecture_flags} --libs | \
                              awk '{for (i=1; i<=NF; i++) {if ([index]($i, "-l") == 1){libs = libs " " $i}}}END{print libs}'`

            AC_MSG_CHECKING([for libsion $1])
            LDFLAGS="$scorep_sion_ldflags $LDFLAGS"
            LIBS="$scorep_sion_libs $LIBS"

            # commom libsion checks. for the paradigm specific ones, see below.
            AC_LINK_IFELSE([AC_LANG_PROGRAM([[
#include <stdio.h>
#include <sion.h>
#include <stddef.h>
]],[[
/* common sion functions */
sion_ensure_free_space(42, 84);
sion_feof(42);
sion_bytes_avail_in_block(42);
sion_seek(42,42,42,42);
sion_seek_fp(42,42,42,42, NULL);
sion_fwrite(NULL,42,42,42);
sion_fwrite(NULL,42,42,42);
]])],
                           [],
                           [scorep_have_sion="no"; scorep_sion_ldflags=""; scorep_sion_libs=""])


            # paradigm specific libsion checks
            m4_case([$1],
[SERIAL],
[AC_LINK_IFELSE([AC_LANG_PROGRAM([[
#include <stdio.h>
#include <sion.h>
#include <stddef.h>
]],[[
/* serial sion functions */
sion_open(NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
sion_open_rank(NULL,NULL,NULL,NULL,NULL,NULL);
sion_close(42);
sion_get_locations(42,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
]])],
                                  [],
                                  [scorep_have_sion="no"; scorep_sion_ldflags=""; scorep_sion_libs=""])],

[OMP],
[scorep_have_sion="no"; scorep_sion_ldflags=""; scorep_sion_libs=""],

[MPI],
[AC_LINK_IFELSE([AC_LANG_PROGRAM([[
#include <stdio.h>
#include <sion.h>
#include <stddef.h>
#include <mpi.h>
]],[[
/* mpi sion functions */
MPI_Comm foo = MPI_COMM_WORLD;
sion_paropen_mpi(NULL,NULL,NULL,foo,&foo,NULL,NULL,NULL,NULL,NULL);
sion_parclose_mpi(42);
]])],
                                  [],
                                  [scorep_have_sion="no"; scorep_sion_ldflags=""; scorep_sion_libs=""])],

[MPI_OMP],
[scorep_have_sion="no"; scorep_sion_ldflags=""; scorep_sion_libs=""])


            AC_MSG_RESULT([$scorep_have_sion])
        fi

        CPPFLAGS=$cppflags_save
        LDFLAGS=$ldflags_save
        LIBS=$libs_save
        AC_LANG_POP([C])
    fi
fi

#echo "debug: scorep_sion_cppflags=$scorep_sion_cppflags"
#echo "debug: scorep_sion_ldflags=$scorep_sion_ldflags"
#echo "debug: scorep_sion_libs=$scorep_sion_libs"

# The output of this macro
AC_SUBST([SCOREP_SION_$1_CPPFLAGS], [$scorep_sion_cppflags])
AC_SUBST([SCOREP_SION_$1_LDFLAGS],  [$scorep_sion_ldflags])
AC_SUBST([SCOREP_SION_$1_LIBS],     [$scorep_sion_libs])
AS_IF([test "x${scorep_have_sion}" = "xyes"],
    [AC_DEFINE([HAVE_SION_$1], [1], [Defined if libsion $1 is available.])])
AM_CONDITIONAL([HAVE_SION_$1], [test "x${scorep_have_sion}" = "xyes"])
AC_SCOREP_SUMMARY([SION $1 support], [${scorep_have_sion}])
])


dnl add omp and hybrid tests when sionconfig supports it
dnl # omp
dnl #sion_paropen_omp(NULL,NULL,NULL,NULL,NULL,NULL,NULL);
dnl #sion_parclose_omp(42);

dnl # mpi_omp
dnl #sion_paropen_ompi(NULL,NULL,NULL,MPI_COMM_WORLD,&MPI_COMM_WOLRD,NULL,NULL,NULL,NULL,NULL)
dnl #sion_parclose_ompi(42);
