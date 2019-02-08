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
## Copyright (c) 2009-2013,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2017,
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

## file ac_scorep_sys_detection.m4

# The purpose of platform detection is to provide reasonable default
# compilers, MPI implementations, etc. The user always has the
# possibility to override the defaults by setting environment
# variables, see section "Some influential environment variables" in
# configure --help.  On some systems there may be no reasonable
# defaults for the MPI implementation, so specify them using
# --with-mpi=... Also, on some systems there are different
# compiler-suites available which can be choosen via
# --with-compiler=(gcc|intel|sun|ibm|...)


# intended to be called by toplevel configure
AC_DEFUN_ONCE([AC_SCOREP_DETECT_PLATFORMS], [
AC_REQUIRE([AC_CANONICAL_BUILD])dnl

# Notes about platform detection on Cray systems:
# First, we check for x86-64 CPU type and an existing /opt/cray/pmi/default link.
# This test should succeed for all supported Cray systems (Cray XT, XE, XK, XC).
# In the second step we will classify the systems depending on their network.
# Therefore, we use the link /opt/cray/pmi/default. We determine the link target
# and select the network. For example, /opt/cray/pmi/default points to
# /opt/cray/pmi/4.0.1-1.0000.9421.73.3.gem. 'gem' signifies the Gemini network.
# As a result we can classify the following systems:
# ss  (SeaStar)  Cray XT
# gem (Gemini)   Cray XE, XK
# ari (Aries)    Cray XC
# To distinguish Cray XE and XK systems we determine whether the system uses GPU
# accelerators (Cray XK) or not (Cray XE).
#
# Newer software stacks do not encode the network name in their files, e.g.,
# /opt/cray/pmi/5.0.11
# Therefore, we need a fall-back solution. We will test for several directories
# including the network name.
#
# Even newer software stacks on Cray XC systems (Cori, Piz Daint)
# don't provide the /opt/cray/pmi/default link but instead
# /opt/cray/pe/pmi/default. They still provide the network via
# /opt/cray/ari/modulefiles.
#
AC_MSG_CHECKING([for platform])
AC_ARG_ENABLE([platform-mic],
    [AS_HELP_STRING([--enable-platform-mic],
                    [Force build for Intel Xeon Phi co-processors [no].
                     This option is only needed for Xeon Phi
                     co-processors, like the Knights Corner (KNC). It
                     is not needed for self-hosted Xeon Phis, like the
                     Knights Landing (KNL); for these chips no special
                     treatment is required.])],

    [AS_CASE([${enableval}],
        [yes],
            [ac_scorep_platform=mic],
        [no],
            [],
        [AC_MSG_ERROR([value '$enableval' unsupported for '--enable-platform-mic'])])
    ])

AS_IF([test "x${ac_scorep_platform}" = "x"],
    [AS_CASE([${build_os}],
         [linux*],
             [AS_IF([test "x${build_cpu}" = "xia64" && test -f /etc/sgi-release],
                      [ac_scorep_platform="altix"],
                  [test "x${build_cpu}" = "xpowerpc64" && test -d /bgl/BlueLight],
                      [ac_scorep_platform="bgl"],
                  [test "x${build_cpu}" = "xpowerpc64" && test -d /bgsys/drivers/ppcfloor/hwi],
                      [ac_scorep_platform="bgq"],
                  [test "x${build_cpu}" = "xpowerpc64" && test -d /bgsys],
                      [ac_scorep_platform="bgp"],
                  [test "x${build_cpu}" = "xx86_64" && test -d /opt/cray],
                      [AS_IF([test -L /opt/cray/pmi/default],
                           [AS_IF([test "x`readlink -f /opt/cray/pmi/default | grep -o --regexp=[[a-z]]*$ | grep -q ss && echo TRUE`" = "xTRUE"],
                                   [ac_scorep_platform="crayxt"],
                               [test "x`readlink -f /opt/cray/pmi/default | grep -o --regexp=[[a-z]]*$ | grep -q gem && echo TRUE`" = "xTRUE" && test "x`apstat -G | grep \"(none)\" | wc -l`" = "x1"],
                                   [ac_scorep_platform="crayxe"],
                               [test "x`readlink -f /opt/cray/pmi/default | grep -o --regexp=[[a-z]]*$ | grep -q gem && echo TRUE`" = "xTRUE" && test "x`apstat -G | grep \"(none)\" | wc -l`" = "x0"],
                                   [ac_scorep_platform="crayxk"],
                               [test "x`readlink -f /opt/cray/pmi/default | grep -o --regexp=[[a-z]]*$ | grep -q ari && echo TRUE`" = "xTRUE"],
                                   [ac_scorep_platform="crayxc"],
                               [test -d /opt/cray/ari/modulefiles],
                                   [ac_scorep_platform="crayxc"],
                               [test -d /opt/cray/gem/modulefiles && test "x`apstat -G | grep \"(none)\" | wc -l`" = "x0"],
                                   [ac_scorep_platform="crayxk"],
                               [test -d /opt/cray/gem/modulefiles && test "x`apstat -G | grep \"(none)\" | wc -l`" = "x1"],
                                   [ac_scorep_platform="crayxe"])],
                      [test -L /opt/cray/pe/pmi/default],
                           [AS_IF([test -d /opt/cray/ari/modulefiles],
                               [ac_scorep_platform="crayxc"])])
                       AS_IF([test "x${ac_scorep_platform}" = "x"],
                           [AC_MSG_ERROR([Unknown Cray platform.])])],
                  [test "x${build_cpu}" = "xarmv7l" || test "x${build_cpu}" = "xarmv7hl" || test "x${build_cpu}" = "xaarch64" ],
                      [ac_scorep_platform="arm"],
                  [test "x${build_cpu}" = "xx86_64" && test -d /opt/FJSVtclang],
                      [ac_scorep_platform="k"],
                  [test "x${build_cpu}" = "xx86_64" && test -d /opt/FJSVfxlang],
                      [ac_scorep_platform="fx10"],
                  [test "x${build_cpu}" = "xx86_64" && test -d /opt/FJSVmxlang],
                      [ac_scorep_platform="fx100"],
                  [ac_scorep_platform=linux])],
         [sunos* | solaris*],
              [ac_scorep_platform="solaris"],
         [darwin*],
              [ac_scorep_platform="mac"],
         [aix*],
              [ac_scorep_platform="aix"],
         [superux*],
              [ac_scorep_platform="necsx"],
         [mingw*],
              [ac_scorep_platform="mingw"],
         [ac_scorep_platform="unknown"])

     AS_IF([test "x${ac_scorep_platform}" = "xunknown"],
         [AC_MSG_RESULT([$ac_scorep_platform, please contact <AC_PACKAGE_BUGREPORT> if you encounter any problems.])],
         [AC_MSG_RESULT([$ac_scorep_platform (auto detected)])
          AFS_SUMMARY([Platform], [$ac_scorep_platform (auto detected)])])
    ],
    [AC_MSG_RESULT([$ac_scorep_platform (provided)])
     AFS_SUMMARY([Platform], [$ac_scorep_platform (provided)])])
])# AC_SCOREP_DETECT_PLATFORMS


# intended to be called by toplevel configure
AC_DEFUN_ONCE([AFS_CROSSCOMPILING],
[
AC_REQUIRE([AC_SCOREP_DETECT_PLATFORMS])dnl

AS_IF([test "x${ac_scorep_cross_compiling}" = "x"],
    [AS_CASE([${ac_scorep_platform}],
         [altix],   [ac_scorep_cross_compiling="no"],
         [bgl],     [ac_scorep_cross_compiling="yes"],
         [bgp],     [ac_scorep_cross_compiling="yes"],
         [bgq],     [ac_scorep_cross_compiling="yes"],
         [crayxt],  [ac_scorep_cross_compiling="yes"],
         [crayxe],  [ac_scorep_cross_compiling="yes"],
         [crayxk],  [ac_scorep_cross_compiling="yes"],
         [crayxc],  [ac_scorep_cross_compiling="yes"],
         [arm],     [ac_scorep_cross_compiling="no"],
         [k],       [ac_scorep_cross_compiling="yes"],
         [fx10],    [ac_scorep_cross_compiling="yes"],
         [fx100],   [ac_scorep_cross_compiling="yes"],
         [linux],   [ac_scorep_cross_compiling="no"],
         [solaris], [ac_scorep_cross_compiling="no"],
         [mac],     [ac_scorep_cross_compiling="no"],
         [mic],     [ac_scorep_cross_compiling="yes"],
         [mingw],   [ac_scorep_cross_compiling="no"],
         [aix],     [ac_scorep_cross_compiling="no"],
         [necsx],   [ac_scorep_cross_compiling="yes"],
         [unknown], [ac_scorep_cross_compiling="no"],
         [AC_MSG_ERROR([provided platform '${ac_scorep_platform}' unknown.])])
     AFS_SUMMARY([Cross compiling], [$ac_scorep_cross_compiling (auto detected)])
    ],
    [# honor ac_scorep_cross_compiling from the commandline
     AS_IF([test ${ac_scorep_cross_compiling} != "yes" && \
            test ${ac_scorep_cross_compiling} != "no" ],
         [AC_MSG_ERROR([invalid value '${ac_scorep_cross_compiling}' for provided 'ac_scorep_cross_compiling'])])
     AFS_SUMMARY([Cross compiling], [$ac_scorep_cross_compiling (provided)])
    ])

AC_MSG_CHECKING([for cross compilation])
AC_MSG_RESULT([$ac_scorep_cross_compiling])
])# AFS_CROSSCOMPILING


# This macro is called by the build-backend/frontend/mpi configures only.
AC_DEFUN([AC_SCOREP_PLATFORM_SETTINGS],
[
    AC_REQUIRE([AC_CANONICAL_BUILD])

    AM_CONDITIONAL([PLATFORM_ALTIX],   [test "x${ac_scorep_platform}" = "xaltix"])
    AM_CONDITIONAL([PLATFORM_AIX],     [test "x${ac_scorep_platform}" = "xaix" && test "x${build_cpu}" = "xpowerpc"])
    AM_CONDITIONAL([PLATFORM_BGL],     [test "x${ac_scorep_platform}" = "xbgl"])
    AM_CONDITIONAL([PLATFORM_BGP],     [test "x${ac_scorep_platform}" = "xbgp"])
    AM_CONDITIONAL([PLATFORM_BGQ],     [test "x${ac_scorep_platform}" = "xbgq"])
    AM_CONDITIONAL([PLATFORM_CRAYXT],  [test "x${ac_scorep_platform}" = "xcrayxt"])
    AM_CONDITIONAL([PLATFORM_CRAYXE],  [test "x${ac_scorep_platform}" = "xcrayxe"])
    AM_CONDITIONAL([PLATFORM_CRAYXK],  [test "x${ac_scorep_platform}" = "xcrayxk"])
    AM_CONDITIONAL([PLATFORM_CRAYXC],  [test "x${ac_scorep_platform}" = "xcrayxc"])
    AM_CONDITIONAL([PLATFORM_LINUX],   [test "x${ac_scorep_platform}" = "xlinux"])
    AM_CONDITIONAL([PLATFORM_SOLARIS], [test "x${ac_scorep_platform}" = "xsolaris"])
    AM_CONDITIONAL([PLATFORM_MAC],     [test "x${ac_scorep_platform}" = "xmac"])
    AM_CONDITIONAL([PLATFORM_MIC],     [test "x${ac_scorep_platform}" = "xmic"])
    AM_CONDITIONAL([PLATFORM_NECSX],   [test "x${ac_scorep_platform}" = "xnecsx"])
    AM_CONDITIONAL([PLATFORM_ARM],     [test "x${ac_scorep_platform}" = "xarm"])
    AM_CONDITIONAL([PLATFORM_MINGW],   [test "x${ac_scorep_platform}" = "xmingw"])
    AM_CONDITIONAL([PLATFORM_K],       [test "x${ac_scorep_platform}" = "xk"])
    AM_CONDITIONAL([PLATFORM_FX10],    [test "x${ac_scorep_platform}" = "xfx10"])
    AM_CONDITIONAL([PLATFORM_FX100],   [test "x${ac_scorep_platform}" = "xfx100"])

    AS_CASE([${ac_scorep_platform}],
            [crayx*],     [afs_platform_cray="yes"],
            [afs_platform_cray="no"])
    AM_CONDITIONAL([PLATFORM_CRAY],[ test "x${afs_platform_cray}" = "xyes" ])

    AM_COND_IF([PLATFORM_ALTIX],
        [AC_DEFINE([HAVE_PLATFORM_ALTIX], [1], [Set if we are building for the ALTIX platform])])
    AM_COND_IF([PLATFORM_AIX],
        [AC_DEFINE([HAVE_PLATFORM_AIX], [1], [Set if we are building for the AIX platform])])
    AM_COND_IF([PLATFORM_BGL],
        [AC_DEFINE([HAVE_PLATFORM_BGL], [1], [Set if we are building for the BG/L platform])])
    AM_COND_IF([PLATFORM_BGP],
        [AC_DEFINE([HAVE_PLATFORM_BGP], [1], [Set if we are building for the BG/P platform])])
    AM_COND_IF([PLATFORM_BGQ],
        [AC_DEFINE([HAVE_PLATFORM_BGQ], [1], [Set if we are building for the BG/Q platform])])
    AM_COND_IF([PLATFORM_CRAY],
        [AC_DEFINE([HAVE_PLATFORM_CRAY], [1], [Set if we are building for the Cray platform])])
    AM_COND_IF([PLATFORM_CRAYXT],
        [AC_DEFINE([HAVE_PLATFORM_CRAYXT], [1], [Set if we are building for the Cray XT platform])])
    AM_COND_IF([PLATFORM_CRAYXE],
        [AC_DEFINE([HAVE_PLATFORM_CRAYXE], [1], [Set if we are building for the Cray XE platform])])
    AM_COND_IF([PLATFORM_CRAYXK],
        [AC_DEFINE([HAVE_PLATFORM_CRAYXK], [1], [Set if we are building for the Cray XK platform])])
    AM_COND_IF([PLATFORM_CRAYXC],
        [AC_DEFINE([HAVE_PLATFORM_CRAYXC], [1], [Set if we are building for the Cray XC platform])])
    AM_COND_IF([PLATFORM_LINUX],
        [AC_DEFINE([HAVE_PLATFORM_LINUX], [1], [Set if we are building for the Linux platform])])
    AM_COND_IF([PLATFORM_SOLARIS],
        [AC_DEFINE([HAVE_PLATFORM_SOLARIS], [1], [Set if we are building for the Solaris platform])])
    AM_COND_IF([PLATFORM_MAC],
        [AC_DEFINE([HAVE_PLATFORM_MAC], [1], [Set if we are building for the Mac platform])])
    AM_COND_IF([PLATFORM_MIC],
        [AC_DEFINE([HAVE_PLATFORM_MIC], [1], [Set if we are building for the Intel MIC platform])])
    AM_COND_IF([PLATFORM_NECSX],
        [AC_DEFINE([HAVE_PLATFORM_NECSX], [1], [Set if we are building for the NEC SX platform])])
    AM_COND_IF([PLATFORM_ARM],
        [AC_DEFINE([HAVE_PLATFORM_ARM], [1], [Set if we are building for the ARM platform])])
    AM_COND_IF([PLATFORM_MINGW],
        [AC_DEFINE([HAVE_PLATFORM_MINGW], [1], [Set if we are building for the MinGW platform])])
    AM_COND_IF([PLATFORM_K],
        [AC_DEFINE([HAVE_PLATFORM_K], [1], [Set if we are building for the K platform])])
    AM_COND_IF([PLATFORM_FX10],
        [AC_DEFINE([HAVE_PLATFORM_FX10], [1], [Set if we are building for the FX10 platform])])
    AM_COND_IF([PLATFORM_FX100],
        [AC_DEFINE([HAVE_PLATFORM_FX100], [1], [Set if we are building for the FX100 platform])])
])
