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


AC_DEFUN([_AC_SCOREP_DETECT_LINUX_PLATFORMS],
[
    if test "x${ac_scorep_platform}" = "xunknown"; then
        case ${build_os} in
            linux*)
                AS_IF([test "x${build_cpu}" = "xia64"      && test -f /etc/sgi-release], 
                          [ac_scorep_platform="altix";    ac_scorep_cross_compiling="no"],
                      [test "x${build_cpu}" = "xpowerpc64" && test -d /bgl/BlueLight],   
                          [ac_scorep_platform="bgl";      ac_scorep_cross_compiling="yes"],
                      [test "x${build_cpu}" = "xpowerpc64" && test -d /bgsys/drivers/ppcfloor/hwi],           
                          [ac_scorep_platform="bgq";      ac_scorep_cross_compiling="yes"],
                      [test "x${build_cpu}" = "xpowerpc64" && test -d /bgsys],           
                          [ac_scorep_platform="bgp";      ac_scorep_cross_compiling="yes"],
                      [test "x${build_cpu}" = "xx86_64"    && test -d /opt/cray/xt-asyncpe],     
                          [ac_scorep_platform="crayxt";   ac_scorep_cross_compiling="yes"],
                      [test "x${build_cpu}" = "xarmv7l"],
                          [ac_scorep_platform="arm"; ac_scorep_cross_compiling="no"],
                      [ac_scorep_platform=linux]
                )
            ;;
        esac
    fi
])


AC_DEFUN([_AC_SCOREP_DETECT_NON_LINUX_PLATFORMS],
[
    if test "x${ac_scorep_platform}" = "xunknown"; then
        case ${build_os} in
            sunos* | solaris*)
                ac_scorep_platform="solaris"
                ac_scorep_cross_compiling="no"
                ;;
            darwin*)
                ac_scorep_platform="mac"
                ac_scorep_cross_compiling="no"
                ;;
            aix*)
                ac_scorep_platform="aix"
                ac_scorep_cross_compiling="no"
                ;;
            superux*)
                ac_scorep_platform="necsx"
                ac_scorep_cross_compiling="yes"
                ;;
        esac
    fi
])

# The purpose of platform detection is to provide reasonable default
# compilers, MPI implementations, OpenMP flags, etc.  The user always has the
# possibility to override the defaults by setting environment variables, see
# section "Some influential environment variables" in configure --help.  On
# some systems there may be no reasonable defaults for the MPI implementation,
# so specify them using --with-mpi=... I think we need to specify one or more
# paths too. Also, on some systems there are different compiler-suites available
# which can be choosen via --with-compiler=(gnu?|intel|sun|ibm|...)
# Have to think this over...

AC_DEFUN([AC_SCOREP_DETECT_PLATFORMS],
[
    AC_REQUIRE([AC_CANONICAL_BUILD])
    ac_scorep_platform="unknown"
    ac_scorep_cross_compiling="no"
    ac_scorep_platform_detection=""
    ac_scorep_platform_detection_given=""

    ac_scorep_compilers_frontend=""
    ac_scorep_compilers_backend=""
    ac_scorep_compilers_mpi=""

    path_to_compiler_files="$srcdir/vendor/common/build-config/platforms/"

    if test "x${host_alias}" != "x"; then
        AC_CANONICAL_HOST
        if test "x${build}" != "x${host}"; then
            ac_scorep_cross_compiling="yes"
        fi
    fi

    AC_ARG_WITH([platform],
                [AS_HELP_STRING([--with-platform=(auto,disabled,<platform>)],
                                [autodetect platform [auto], disabled or select one from:
                                 altix, aix, arm, bgl, bgp, bgq, crayxt, linux, solaris, mac, necsx.])],
                [AS_CASE([$withval],
                     ["auto"     | "yes"], [ac_scorep_platform_detection_given="yes"],
                     ["disabled" | "no"],  [ac_scorep_platform_detection_given="no"],
                     [ac_scorep_platform_detection_given="no"
                      AS_IF([! test -e "${path_to_compiler_files}platform-frontend-${withval}"],
                            [AC_MSG_ERROR([Unknown platform ${withval}.])])
                      ac_scorep_platform="$withval"])],
                [AS_IF([test "x${build_alias}" = "x" && test "x${host_alias}" = "x"],
                       [ac_scorep_platform_detection="yes"],
                       [ac_scorep_platform_detection="no"])])

    if test "x${ac_scorep_platform_detection_given}" = "xyes"; then
        if test "x${build_alias}" != "x" || test "x${host_alias}" != "x"; then
            AC_MSG_ERROR([it makes no sense to request for platform detection while providing --host and/or --build.])
        fi
    fi
    if test "x${ac_scorep_platform_detection_given}" != "x"; then
        ac_scorep_platform_detection="$ac_scorep_platform_detection_given"
    fi

    if test "x${ac_scorep_platform_detection}" = "xyes"; then
        _AC_SCOREP_DETECT_LINUX_PLATFORMS
        _AC_SCOREP_DETECT_NON_LINUX_PLATFORMS        
        AC_MSG_CHECKING([for platform])
        if test "x${ac_scorep_platform}" = "xunknown"; then
            AC_MSG_RESULT([$ac_scorep_platform, please contact <AC_PACKAGE_BUGREPORT> if you encounter any problems.])
        else
            AC_MSG_RESULT([$ac_scorep_platform (auto detected)])
        fi
    elif test "x${ac_scorep_platform_detection}" = "xno"; then
        AC_MSG_CHECKING([for platform])
        AC_MSG_RESULT([$ac_scorep_platform (user selected)])
        if test "x${build_alias}" != "x" || test "x${host_alias}" != "x"; then
            if test "x$$ac_scorep_platform" = "xunknown"; then
                AC_MSG_ERROR([providing --host and/or --build without --with-platform is erroneous.])
            fi
        fi
    else
        AC_MSG_ERROR([unknown value for ac_scorep_platform_detection: $ac_scorep_platform_detection])
    fi
    AC_MSG_CHECKING([for cross compilation])
    AC_MSG_RESULT([$ac_scorep_cross_compiling])
    ac_scorep_compilers_frontend="${path_to_compiler_files}platform-frontend-${ac_scorep_platform}"
    ac_scorep_compilers_backend="${path_to_compiler_files}platform-backend-${ac_scorep_platform}"
    ac_scorep_compilers_mpi="${path_to_compiler_files}platform-mpi-${ac_scorep_platform}"
])

# This macro is called also by the build-backend/frondend/mpi configures
AC_DEFUN([AC_SCOREP_PLATFORM_SETTINGS],
[
    AC_REQUIRE([AC_CANONICAL_BUILD])

    AM_CONDITIONAL([PLATFORM_ALTIX],   [test "x${ac_scorep_platform}" = "xaltix"])
    AM_CONDITIONAL([PLATFORM_AIX],     [test "x${ac_scorep_platform}" = "xaix" && test "x${build_cpu}" = "xpowerpc"])
    AM_CONDITIONAL([PLATFORM_BGL],     [test "x${ac_scorep_platform}" = "xbgl"])
    AM_CONDITIONAL([PLATFORM_BGP],     [test "x${ac_scorep_platform}" = "xbgp"])
    AM_CONDITIONAL([PLATFORM_BGQ],     [test "x${ac_scorep_platform}" = "xbgq"])
    AM_CONDITIONAL([PLATFORM_CRAYXT],  [test "x${ac_scorep_platform}" = "xcrayxt"])
    AM_CONDITIONAL([PLATFORM_LINUX],   [test "x${ac_scorep_platform}" = "xlinux"])
    AM_CONDITIONAL([PLATFORM_SOLARIS], [test "x${ac_scorep_platform}" = "xsolaris"])
    AM_CONDITIONAL([PLATFORM_MAC],     [test "x${ac_scorep_platform}" = "xmac"])
    AM_CONDITIONAL([PLATFORM_NECSX],   [test "x${ac_scorep_platform}" = "xnecsx"])
    AM_CONDITIONAL([PLATFORM_ARM],     [test "x${ac_scorep_platform}" = "xarm"])

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
    AM_COND_IF([PLATFORM_CRAYXT],
               [AC_DEFINE([HAVE_PLATFORM_CRAYXT], [1], [Set if we are building for the Cray XT platform])])
    AM_COND_IF([PLATFORM_LINUX],
               [AC_DEFINE([HAVE_PLATFORM_LINUX], [1], [Set if we are building for the Linux platform])])
    AM_COND_IF([PLATFORM_SOLARIS],
               [AC_DEFINE([HAVE_PLATFORM_SOLARIS], [1], [Set if we are building for the Solaris platform])])
    AM_COND_IF([PLATFORM_MAC],
               [AC_DEFINE([HAVE_PLATFORM_MAC], [1], [Set if we are building for the Mac platform])])
    AM_COND_IF([PLATFORM_NECSX],
               [AC_DEFINE([HAVE_PLATFORM_NECSX], [1], [Set if we are building for the NEC SX platform])])
    AM_COND_IF([PLATFORM_ARM],
               [AC_DEFINE([HAVE_PLATFORM_ARM], [1], [Set if we are building for the ARM platform])])
])
