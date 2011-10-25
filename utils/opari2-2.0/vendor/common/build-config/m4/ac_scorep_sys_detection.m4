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
                AS_IF([test "x${build_cpu}" = "xia64"      -a -f /etc/sgi-release], 
                          [ac_scorep_platform="altix";    ac_scorep_cross_compiling="no"],
                      [test "x${build_cpu}" = "xpowerpc64" -a -d /bgl/BlueLight],   
                          [ac_scorep_platform="bgl";      ac_scorep_cross_compiling="yes"],
                      [test "x${build_cpu}" = "xpowerpc64" -a -d /bgsys],           
                          [ac_scorep_platform="bgp";      ac_scorep_cross_compiling="yes"],
                      [test "x${build_cpu}" = "xx86_64"    -a -d /opt/cray/xt-asyncpe],     
                          [ac_scorep_platform="crayxt";   ac_scorep_cross_compiling="yes"],
                      [test "x${build_cpu}" = "xmips64"    -a -d /opt/sicortex],    
                          [ac_scorep_platform="sicortex"; ac_scorep_cross_compiling="yes"],
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
                ac_scorep_platform="sun"
                ac_scorep_cross_compiling="no"
                ;;
            darwin*)
                ac_scorep_platform="mac"
                ac_scorep_cross_compiling="no"
                ;;
            aix*)
                ac_scorep_platform="ibm"
                ac_scorep_cross_compiling="no"
                ;;
            unicosmp*)
                ac_scorep_platform="crayx1"
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
# compilers, mpi-implementations, OpenMP flags etc.  The user always has the
# possibility to override the defaults by setting environment variables, see
# section "Some influential environment variables" in configure --help.  On
# some systems there may be no reasonable defaults for the mpi-implementation,
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

    AC_ARG_ENABLE([platform-detection],
                  [AS_HELP_STRING([--enable-platform-detection],
                                  [autodetect platform [yes]])],
                  [ac_scorep_platform_detection_given="$enableval"],
                  [AS_IF([test "x${build_alias}" = "x" -a "x${host_alias}" = "x"],
                         [ac_scorep_platform_detection="yes"],
                         [ac_scorep_platform_detection="no"])])

    if test "x${ac_scorep_platform_detection_given}" = "xyes"; then
        if test "x${build_alias}" != "x" -o "x${host_alias}" != "x"; then
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
            AC_MSG_RESULT([$ac_scorep_platform])
            ac_scorep_compilers_frontend="${path_to_compiler_files}platform-frontend-${ac_scorep_platform}"
            ac_scorep_compilers_backend="${path_to_compiler_files}platform-backend-${ac_scorep_platform}"
            ac_scorep_compilers_mpi="${path_to_compiler_files}platform-mpi-${ac_scorep_platform}"  
        fi
        AC_MSG_CHECKING([for cross compilation])
        AC_MSG_RESULT([$ac_scorep_cross_compiling])
    elif test "x${ac_scorep_platform_detection}" = "xno"; then
        AC_MSG_NOTICE([platform detection disabled.])
        AC_MSG_CHECKING([for cross compilation])
        AC_MSG_RESULT([$ac_scorep_cross_compiling]) 
        ac_scorep_compilers_frontend="${path_to_compiler_files}platform-frontend-user-provided"
        ac_scorep_compilers_backend="${path_to_compiler_files}platform-backend-user-provided"
        ac_scorep_compilers_mpi="${path_to_compiler_files}platform-mpi-user-provided"
    else
        AC_MSG_ERROR([unknown value for ac_scorep_platform_detection: $ac_scorep_platform_detection])
    fi
])


AC_DEFUN([AC_SCOREP_PLATFORM_SETTINGS],
[
    AC_REQUIRE([AC_SCOREP_DETECT_PLATFORMS])

    AM_CONDITIONAL([PLATFORM_ALTIX],    [test "x${ac_scorep_platform}" = "xaltix"])
    AM_CONDITIONAL([PLATFORM_POWER6],   [test "x${ac_scorep_platform}" = "xibm" -a "x${build_cpu}" = "xpowerpc"])
    AM_CONDITIONAL([PLATFORM_BGL],      [test "x${ac_scorep_platform}" = "xbgl"])
    AM_CONDITIONAL([PLATFORM_BGP],      [test "x${ac_scorep_platform}" = "xbgp"])
    AM_CONDITIONAL([PLATFORM_CRAYXT],   [test "x${ac_scorep_platform}" = "xcrayxt"])
    AM_CONDITIONAL([PLATFORM_SICORTEX], [test "x${ac_scorep_platform}" = "xsicortex"])
    AM_CONDITIONAL([PLATFORM_LINUX],    [test "x${ac_scorep_platform}" = "xlinux"])
    AM_CONDITIONAL([PLATFORM_SUN],      [test "x${ac_scorep_platform}" = "xsun"])
    AM_CONDITIONAL([PLATFORM_MAC],      [test "x${ac_scorep_platform}" = "xmac"])
    AM_CONDITIONAL([PLATFORM_CRAYX1],   [test "x${ac_scorep_platform}" = "xcrayx1"])
    AM_CONDITIONAL([PLATFORM_NECSX],    [test "x${ac_scorep_platform}" = "xnecsx"])
])
