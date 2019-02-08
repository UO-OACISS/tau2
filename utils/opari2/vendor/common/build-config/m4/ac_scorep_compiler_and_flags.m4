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
## Copyright (c) 2009-2011,2014
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2014, 2017,
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

## file ac_scorep_compiler_and_flags.m4


AC_DEFUN([AC_SCOREP_CONVERT_FOR_BUILD_FLAGS],
[
if test "x${ac_cv_env_[$1]_FOR_BUILD_set}" != "xset"; then
   # don't use the default flags if nothing is specified for the frontend
   unset [$1]
else
   # use the FOR_BUILD flags
   [$1]="$ac_cv_env_[$1]_FOR_BUILD_value"
fi
])

AC_DEFUN([AC_SCOREP_CONVERT_MPI_FLAGS],
[
if test "x${ac_cv_env_MPI_[$1]_set}" != "xset"; then
   # don't use the default flags if nothing is specified for MPI
   unset [$1]
else
   # use the MPI flags
   [$1]="$ac_cv_env_MPI_[$1]_value"
fi
])

AC_DEFUN([AC_SCOREP_CHECK_COMPILER_VAR_SET],
[
if test "x${ac_cv_env_[$1]_set}" != "xset"; then
    AC_MSG_ERROR([argument $1 not provided in configure call.], [1])
fi
])


AC_DEFUN([AC_SCOREP_CONVERT_FOR_BUILD_COMPILER],
[
if test "x${ac_cv_env_[$1]_FOR_BUILD_set}" != "xset"; then
    # don't use the default compiler if nothing is specified for the frontend
    unset [$1]
else
    [$1]="$ac_cv_env_[$1]_FOR_BUILD_value"
fi
])

AC_DEFUN([AC_SCOREP_CONVERT_MPI_COMPILER],
[
if test "x${ac_cv_env_MPI[$1]_set}" != "xset"; then
    # don't use the default compiler if nothing is specified for MPI
    unset [$1]
else
    [$1]="$ac_cv_env_MPI[$1]_value"
fi
])


dnl do not use together with AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE
AC_DEFUN([AC_SCOREP_WITH_COMPILER_SUITE],
[
AC_REQUIRE([AC_SCOREP_DETECT_PLATFORMS])

ac_scorep_compilers_frontend="platform-frontend-${ac_scorep_platform}"
ac_scorep_compilers_backend="platform-backend-${ac_scorep_platform}"
# ac_scorep_compilers_mpi set in AC_SCOREP_WITH_MPI_COMPILER_SUITE

m4_pattern_allow([AC_SCOREP_WITH_COMPILER_SUITE])
m4_pattern_allow([AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE])
AS_IF([test "x${ac_scorep_compiler_suite_called}" != "x"],
    [AC_MSG_ERROR([cannot use [AC_SCOREP_WITH_COMPILER_SUITE] and [AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE] in one configure.ac.])],
    [ac_scorep_compiler_suite_called="yes"])

AC_ARG_WITH([nocross-compiler-suite],
            [AS_HELP_STRING([--with-nocross-compiler-suite=(gcc|ibm|intel|pgi|studio)],
                            [The compiler suite used to build this package in non cross-compiling environments. Needs to be in $PATH [gcc].])],
            [AS_IF([test "x${ac_scorep_cross_compiling}" = "xno"],
                   [ac_scorep_compilers_backend="compiler-nocross-gcc" # default
                    AS_CASE([$withval],
                            ["gcc"],       [ac_scorep_compilers_backend="compiler-nocross-gcc"],
                            ["ibm"],       [ac_scorep_compilers_backend="compiler-nocross-ibm"],
                            ["intel"],     [ac_scorep_compilers_backend="compiler-nocross-intel"],
                            ["pgi"],       [ac_scorep_compilers_backend="compiler-nocross-pgi"],
                            ["studio"],    [ac_scorep_compilers_backend="compiler-nocross-studio"],
                            ["no"],        [AC_MSG_ERROR([option --without-nocross-compiler-suite makes no sense.])],
                            [AC_MSG_ERROR([compiler suite "${withval}" not supported by --with-nocross-compiler-suite.])])],
                   [AC_MSG_WARN([option --with-nocross-compiler-suite ignored in cross-compiling mode. You may use --with-frontend-compiler-suite to customize the frontend compiler.])])])
AS_IF([test -f "AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_backend}"],
      [ac_scorep_compilers_backend="AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_backend}"],
      [ac_scorep_compilers_backend="AFS_COMPILER_FILES_COMMON/${ac_scorep_compilers_backend}"])


AC_ARG_WITH([frontend-compiler-suite],
            [AS_HELP_STRING([--with-frontend-compiler-suite=(gcc|ibm|intel|pgi|studio)],
                            [The compiler suite used to build the frontend parts of this package in cross-compiling environments. Needs to be in $PATH [gcc].])],
            [AS_IF([test "x${ac_scorep_cross_compiling}" = "xyes"],
                   [ac_scorep_compilers_frontend="compiler-frontend-gcc"
                    AS_CASE([$withval],
                            ["gcc"],       [ac_scorep_compilers_frontend="compiler-frontend-gcc"],
                            ["ibm"],       [ac_scorep_compilers_frontend="compiler-frontend-ibm"],
                            ["intel"],     [ac_scorep_compilers_frontend="compiler-frontend-intel"],
                            ["pgi"],       [ac_scorep_compilers_frontend="compiler-frontend-pgi"],
                            ["studio"],    [ac_scorep_compilers_frontend="compiler-frontend-studio"],
                            ["no"],        [AC_MSG_ERROR([option --without-frontend-compiler-suite makes no sense.])],
                            [AC_MSG_ERROR([compiler suite "${withval}" not supported by --with-frontend-compiler-suite.])])],
                   [AC_MSG_ERROR([Option --with-frontend-compiler-suite not supported in non cross-compiling mode. Please use --with-nocross-compiler-suite instead.])])])
AS_IF([test -f "AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_frontend}"],
      [ac_scorep_compilers_frontend="AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_frontend}"],
      [ac_scorep_compilers_frontend="AFS_COMPILER_FILES_COMMON/${ac_scorep_compilers_frontend}"])
])# AC_SCOREP_WITH_COMPILER_SUITE


AC_DEFUN([AC_SCOREP_WITH_MPI_COMPILER_SUITE],
[
AC_REQUIRE([AC_SCOREP_DETECT_PLATFORMS])

scorep_mpi_user_disabled="no"
AC_ARG_WITH([mpi],
    [AS_HELP_STRING([--with-mpi=(bullxmpi|hp|ibmpoe|intel|intel2|intel3|intelpoe|lam|mpibull2|mpich|mpich2|mpich3|openmpi|platform|scali|sgimpt|sgimptwrapper|sun)],
         [The MPI compiler suite to build this package in non cross-compiling mode. Usually autodetected. Needs to be in $PATH.])],
    [AS_IF([test "x${withval}" = xno],
         [scorep_mpi_user_disabled=yes
          ac_scorep_compilers_mpi="compiler-mpi-without"
         ],
         [AS_IF([test "x${ac_scorep_cross_compiling}" = "xno" && test "x${ac_scorep_platform}" != "xaix"],
              [AS_CASE([$withval],
                   ["bullxmpi"], [ac_scorep_compilers_mpi="compiler-mpi-bullxmpi"],
                   ["hp"], [ac_scorep_compilers_mpi="compiler-mpi-hp"],
                   ["ibmpoe"], [ac_scorep_compilers_mpi="compiler-mpi-ibmpoe"],
                   ["intel"], [ac_scorep_compilers_mpi="compiler-mpi-intel"],
                   ["intel2"], [ac_scorep_compilers_mpi="compiler-mpi-intel2"],
                   ["intel3"], [ac_scorep_compilers_mpi="compiler-mpi-intel3"],
                   ["impi"], [AC_MSG_WARN([option 'impi' to --with-mpi deprecated, use 'intel2' instead.])
                              ac_scorep_compilers_mpi="compiler-mpi-intel2"],
                   ["intelpoe"], [ac_scorep_compilers_mpi="compiler-mpi-intelpoe"],
                   ["lam"], [ac_scorep_compilers_mpi="compiler-mpi-lam"],
                   ["mpibull2"], [ac_scorep_compilers_mpi="compiler-mpi-mpibull2"],
                   ["mpich"], [ac_scorep_compilers_mpi="compiler-mpi-mpich"],
                   ["mpich2"], [ac_scorep_compilers_mpi="compiler-mpi-mpich2"],
                   ["mpich3"], [ac_scorep_compilers_mpi="compiler-mpi-mpich3"],
                   ["openmpi"], [ac_scorep_compilers_mpi="compiler-mpi-openmpi"],
                   ["platform"], [ac_scorep_compilers_mpi="compiler-mpi-platform"],
                   ["scali"], [ac_scorep_compilers_mpi="compiler-mpi-scali"],
                   ["sgimpt"], [ac_scorep_compilers_mpi="compiler-mpi-sgimpt"],
                   ["sgimptwrapper"], [ac_scorep_compilers_mpi="compiler-mpi-sgimptwrapper"],
                   ["sun"], [ac_scorep_compilers_mpi="compiler-mpi-sun"],
                   [AC_MSG_ERROR([MPI compiler suite "${withval}" not supported by --with-mpi.])])
              ])
         ])
     # omit check "if in PATH" for now. Will fail in build-mpi configure.
    ],
    [AS_IF([test "x${ac_scorep_cross_compiling}" = "xno" && test "x${ac_scorep_platform}" != "xaix"],
         [AFS_COMPILER_MPI
          ac_scorep_compilers_mpi="compiler-mpi-${afs_compiler_mpi}"])
    ])

AS_IF([test "x${scorep_mpi_user_disabled}" = xno],
    [AS_IF([test "x${ac_scorep_cross_compiling}" = "xyes" || test "x${ac_scorep_platform}" = "xaix"],
         [ac_scorep_compilers_mpi="platform-mpi-${ac_scorep_platform}"])])

AS_IF([test -f "AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_mpi}"],
      [ac_scorep_compilers_mpi="AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_mpi}"],
      [ac_scorep_compilers_mpi="AFS_COMPILER_FILES_COMMON/${ac_scorep_compilers_mpi}"])
# sanity checks missing
])# AC_SCOREP_WITH_MPI_COMPILER_SUITE


AC_DEFUN([AC_SCOREP_PRECIOUS_VARS_MPI],
[
AC_ARG_VAR(MPICC,[MPI C compiler command])
AC_ARG_VAR(MPICXX,[MPI C++ compiler command])
AC_ARG_VAR(MPIF77,[MPI Fortran 77 compiler command])
AC_ARG_VAR(MPIFC,[MPI Fortran compiler command])
AC_ARG_VAR(MPI_CPPFLAGS, [MPI (Objective) C/C++ preprocessor flags, e.g. -I<include dir> if you have headers in a nonstandard directory <include dir>])
AC_ARG_VAR(MPI_CFLAGS, [MPI C compiler flags])
AC_ARG_VAR(MPI_CXXFLAGS, [MPI C++ compiler flags])
AC_ARG_VAR(MPI_FFLAGS, [MPI Fortran 77 compiler flags])
AC_ARG_VAR(MPI_FCFLAGS, [MPI Fortran compiler flags])
AC_ARG_VAR(MPI_LDFLAGS, [MPI linker flags, e.g. -L<lib dir> if you have libraries in a nonstandard directory <lib dir>])
AC_ARG_VAR(MPI_LIBS, [MPI libraries to pass to the linker, e.g. -l<library>])
])

AC_DEFUN([AC_SCOREP_PRECIOUS_VARS_FOR_BUILD],
[
AC_ARG_VAR(CC_FOR_BUILD, [C compiler command for the frontend build])
AC_ARG_VAR(CXX_FOR_BUILD, [C++ compiler command for the frontend build])
AC_ARG_VAR(F77_FOR_BUILD, [Fortran 77 compiler command for the frontend build])
AC_ARG_VAR(FC_FOR_BUILD, [Fortran compiler command for the frontend build])
AC_ARG_VAR(CPPFLAGS_FOR_BUILD, [(Objective) C/C++ preprocessor flags for the frontend build, e.g. -I<include dir> if you have headers in a nonstandard directory <include dir>])
AC_ARG_VAR(CFLAGS_FOR_BUILD, [C compiler flags for the frontend build])
AC_ARG_VAR(CXXFLAGS_FOR_BUILD, [C++ compiler flags for the frontend build])
AC_ARG_VAR(FFLAGS_FOR_BUILD, [Fortran 77 compiler flags for the frontend build])
AC_ARG_VAR(FCFLAGS_FOR_BUILD, [Fortran compiler flags for the frontend build])
AC_ARG_VAR(LDFLAGS_FOR_BUILD, [linker flags for the frontend build, e.g. -L<lib dir> if you have libraries in a nonstandard directory <lib dir>])
AC_ARG_VAR(LIBS_FOR_BUILD, [libraries to pass to the linker for the frontend build, e.g. -l<library>])
])


dnl --------------------------------------------------------------------


AC_DEFUN([AFS_WITH_SHMEM_COMPILER_SUITE],
[
AC_REQUIRE([AC_SCOREP_DETECT_PLATFORMS])

scorep_shmem_user_disabled="no"
AC_ARG_WITH([shmem],
    [AS_HELP_STRING([--with-shmem=(openshmem|openmpi|sgimpt|sgimptwrapper)],
         [The SHMEM compiler suite to build this package in non cross-compiling mode. Usually autodetected. Needs to be in $PATH.])],
    [AS_IF([test "x${withval}" = xno],
         [scorep_shmem_user_disabled=yes
          ac_scorep_compilers_shmem="compiler-shmem-without"
         ],
         [AS_IF([test "x${ac_scorep_cross_compiling}" = "xno" && test "x${ac_scorep_platform}" != "xaix"],
              [AS_CASE([$withval],
                   ["openshmem"], [ac_scorep_compilers_shmem="compiler-shmem-openshmem"],
                   ["openmpi"], [ac_scorep_compilers_shmem="compiler-shmem-openmpi"],
                   ["sgimpt"], [ac_scorep_compilers_shmem="compiler-shmem-sgimpt"],
                   ["sgimptwrapper"], [ac_scorep_compilers_shmem="compiler-shmem-sgimptwrapper"],
                   [AC_MSG_ERROR([SHMEM compiler suite "${withval}" not supported by --with-shmem.])])
              ])
         ])
     # omit check "if in PATH" for now.
    ],
    [AS_IF([test "x${ac_scorep_cross_compiling}" = "xno" && test "x${ac_scorep_platform}" != "xaix"],
         [AFS_COMPILER_SHMEM
          ac_scorep_compilers_shmem="compiler-shmem-${afs_compiler_shmem}"])
    ])

AS_IF([test "x${scorep_shmem_user_disabled}" = xno],
      [AS_IF([test "x${ac_scorep_cross_compiling}" = "xyes"],
             [AS_CASE([${ac_scorep_platform}],
                [cray*],
                    [ac_scorep_compilers_shmem="platform-shmem-${ac_scorep_platform}"],
                [ac_scorep_compilers_shmem=""])])])

AS_IF([test "x${ac_scorep_compilers_shmem}" != "x" && test -f "AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_shmem}"],
      [ac_scorep_compilers_shmem="AFS_COMPILER_FILES_PACKAGE/${ac_scorep_compilers_shmem}"],
      [ac_scorep_compilers_shmem="AFS_COMPILER_FILES_COMMON/${ac_scorep_compilers_shmem}"])
# sanity checks missing
])# AFS_WITH_SHMEM_COMPILER_SUITE


AC_DEFUN([AFS_PRECIOUS_VARS_SHMEM],
[
AC_ARG_VAR(SHMEMCC,[SHMEM C compiler command])
AC_ARG_VAR(SHMEMCXX,[SHMEM C++ compiler command])
AC_ARG_VAR(SHMEMF77,[SHMEM Fortran 77 compiler command])
AC_ARG_VAR(SHMEMFC,[SHMEM Fortran compiler command])
AC_ARG_VAR(SHMEM_CPPFLAGS, [SHMEM (Objective) C/C++ preprocessor flags, e.g. -I<include dir> if you have headers in a nonstandard directory <include dir>])
AC_ARG_VAR(SHMEM_CFLAGS, [SHMEM C compiler flags])
AC_ARG_VAR(SHMEM_CXXFLAGS, [SHMEM C++ compiler flags])
AC_ARG_VAR(SHMEM_FFLAGS, [SHMEM Fortran 77 compiler flags])
AC_ARG_VAR(SHMEM_FCFLAGS, [SHMEM Fortran compiler flags])
AC_ARG_VAR(SHMEM_LDFLAGS, [SHMEM linker flags, e.g. -L<lib dir> if you have libraries in a nonstandard directory <lib dir>])
AC_ARG_VAR(SHMEM_LIBS, [SHMEM libraries to pass to the linker, e.g. -l<library>])
AC_ARG_VAR(SHMEM_LIB_NAME, [name of the SHMEM library])
AC_ARG_VAR(SHMEM_NAME, [name of the implemented SHMEM specification])
])

AC_DEFUN([AFS_CONVERT_SHMEM_COMPILER],
[
if test "x${ac_cv_env_SHMEM[$1]_set}" != "xset"; then
    # don't use the default compiler if nothing is specified for SHMEM
    unset [$1]
else
    [$1]="$ac_cv_env_SHMEM[$1]_value"
fi
])

AC_DEFUN([AFS_CONVERT_SHMEM_FLAGS],
[
if test "x${ac_cv_env_SHMEM_[$1]_set}" != "xset"; then
   # don't use the default flags if nothing is specified for SHMEM
   unset [$1]
else
   # use the SHMEM flags
   [$1]="$ac_cv_env_SHMEM_[$1]_value"
fi
])
