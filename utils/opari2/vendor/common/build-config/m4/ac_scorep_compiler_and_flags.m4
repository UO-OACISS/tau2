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


## file       ac_scorep_compiler_and_flags.m4
## maintainer Christian Roessel <c.roessel@fz-juelich.de>

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


# On cross-compile system we might get provided with the *_FOR_BUILD compilers and flags
# and need to map them to CC, CFLAGS etc. The *_FOR_BUILD parameters take precedence.
AC_DEFUN([AC_SCOREP_OPARI2_FOR_BUILD_ARGS_TAKES_PRECEDENCE],
[
    opari2_cross_build_args=""
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([CC])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([CXX])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([F77])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([FC])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([CPPFLAGS])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([CFLAGS])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([CXXFLAGS])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([FFLAGS])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([FCFLAGS])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([LDFLAGS])
    AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG([LIBS])
])

AC_DEFUN([AC_SCOREP_OPARI2_CONVERT_FOR_BUILD_ARG],
[
    if test "x${ac_cv_env_[$1]_FOR_BUILD_set}" == "xset"; then
       [$1]=$ac_cv_env_[$1]_FOR_BUILD_value
    fi
])


dnl dont' use together with AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE
AC_DEFUN([AC_SCOREP_WITH_COMPILER_SUITE],
[
m4_pattern_allow([AC_SCOREP_WITH_COMPILER_SUITE])
m4_pattern_allow([AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE])
if test "x${ac_scorep_compiler_suite_called}" != "x"; then
    # We need m4 quoting magic here ...
    AC_MSG_ERROR([cannot use [AC_SCOREP_WITH_COMPILER_SUITE] and [AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE] in one configure.ac.])
else
    ac_scorep_compiler_suite_called="yes"
fi

path_to_compiler_files="$srcdir/vendor/common/build-config/platforms/"

AC_ARG_WITH([nocross-compiler-suite],
            [AS_HELP_STRING([--with-nocross-compiler-suite=(gcc|ibm|intel|pathscale|pgi|studio)], 
                            [The compiler suite to build this package in non cross-compiling environments with. Needs to be in $PATH [gcc].])],
            [AS_IF([test "x${ac_scorep_cross_compiling}" = "xno"], 
                   [AS_CASE([$withval],
                            ["gcc"],       [ac_scorep_compilers_backend="${path_to_compiler_files}compiler-nocross-gcc"],
                            ["ibm"],       [ac_scorep_compilers_backend="${path_to_compiler_files}compiler-nocross-ibm"],
                            ["intel"],     [ac_scorep_compilers_backend="${path_to_compiler_files}compiler-nocross-intel"],
                            ["pathscale"], [ac_scorep_compilers_backend="${path_to_compiler_files}compiler-nocross-pathscale"],
                            ["pgi"],       [ac_scorep_compilers_backend="${path_to_compiler_files}compiler-nocross-pgi"],
                            ["studio"],    [ac_scorep_compilers_backend="${path_to_compiler_files}compiler-nocross-studio"],
                            [AC_MSG_WARN([Compiler suite "${withval}" not supported by --with-nocross-compiler-suite, ignoring.])])],
                   [AC_MSG_ERROR([Option --with-nocross-compiler-suite not supported in cross-compiling mode. Please use --with-backend-compiler-suite and --with-frontend-compiler-suite instead.])])],
            [])

dnl backend-compiler-suite is not very useful. if we are on a cross-compiling
dnl platform, we usually want to use the vendor tools that should be detected
dnl automatically. otherwise, use platform-*-user-provided
dnl AC_ARG_WITH([backend-compiler-suite],
dnl             [AS_HELP_STRING([--with-backend-compiler-suite=(ibm|sx)], 
dnl                             [The compiler suite to build the backend parts of this package in cross-compiling environments with. Needs to be in $PATH [gcc].])],
dnl             [AS_IF([test "x${ac_scorep_cross_compiling}" = "xyes"], 
dnl                    [AS_CASE([$withval],
dnl                             ["ibm"],       [ac_scorep_compilers_backend="${path_to_compiler_files}compiler-backend-ibm"],
dnl                             ["sx"],        [ac_scorep_compilers_backend="${path_to_compiler_files}compiler-backend-sx"],
dnl                             [AC_MSG_WARN([Compiler suite "${withval}" not supported by --with-backend-compiler-suite, ignoring.])])], 
dnl                    [AC_MSG_ERROR([Option --with-backend-compiler-suite not supported in non cross-compiling mode. Please use --with-nocross-compiler-suite instead.])])],
dnl             [])


AC_ARG_WITH([frontend-compiler-suite],
            [AS_HELP_STRING([--with-frontend-compiler-suite=(gcc|ibm|intel|pathscale|pgi|studio)], 
                            [The compiler suite to build the frontend parts of this package in cross-compiling environments with. Needs to be in $PATH [gcc].])],
            [AS_IF([test "x${ac_scorep_cross_compiling}" = "xyes"], 
                   [AS_CASE([$withval],
                            ["gcc"],       [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-frontend-gcc"],
                            ["ibm"],       [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-frontend-ibm"],
                            ["intel"],     [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-frontend-intel"],
                            ["pathscale"], [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-frontend-pathscale"],
                            ["pgi"],       [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-frontend-pgi"],
                            ["studio"],    [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-frontend-studio"],
                            [AC_MSG_WARN([Compiler suite "${withval}" not supported by --with-frontend-compiler-suite, ignoring.])])],
                   [AC_MSG_ERROR([Option --with-frontend-compiler-suite not supported in non cross-compiling mode. Please use --with-nocross-compiler-suite instead.])])],
            [])
])


dnl dont' use together with AC_SCOREP_WITH_COMPILER_SUITE, intended to be used by OPARI only
AC_DEFUN([AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE],
[
m4_pattern_allow([AC_SCOREP_WITH_COMPILER_SUITE])
m4_pattern_allow([AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE])
if test "x${ac_scorep_compiler_suite_called}" != "x"; then
    # We need m4 quoting magic here ...
    AC_MSG_ERROR([cannot use [AC_SCOREP_WITH_COMPILER_SUITE] and [AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE] in one configure.ac.])
else
    ac_scorep_compiler_suite_called="yes"
fi

path_to_compiler_files="$srcdir/vendor/common/build-config/platforms/"

AC_ARG_WITH([compiler-suite],
            [AS_HELP_STRING([--with-compiler-suite=(gcc|ibm|intel|pathscale|pgi|studio)], 
                            [The compiler suite to build this package with. Needs to be in $PATH [gcc].])],
            [AS_CASE([$withval],
                     ["gcc"],       [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-nocross-gcc"],
                     ["ibm"],       [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-nocross-ibm"],
                     ["intel"],     [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-nocross-intel"],
                     ["pathscale"], [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-nocross-pathscale"],
                     ["pgi"],       [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-nocross-pgi"],
                     ["studio"],    [ac_scorep_compilers_frontend="${path_to_compiler_files}compiler-nocross-studio"],
                     [AC_MSG_WARN([Compiler suite "${withval}" not supported by --with-compiler-suite, ignoring.])])],
            [])
])



AC_DEFUN([AC_SCOREP_WITH_MPI_COMPILER_SUITE],
[
path_to_compiler_files="$srcdir/vendor/common/build-config/platforms/"

AC_ARG_WITH([mpi],
            [AS_HELP_STRING([--with-mpi=(mpich2|impi|openmpi)], 
                            [The mpi compiler suite to build this package with. Needs to be in $PATH [mpich2].])],
            [AS_CASE([$withval],
                     ["mpich2"],      [ac_scorep_compilers_mpi="${path_to_compiler_files}compiler-mpi-mpich2"],
                     ["impi"],        [ac_scorep_compilers_mpi="${path_to_compiler_files}compiler-mpi-impi"],
                     ["openmpi"],     [ac_scorep_compilers_mpi="${path_to_compiler_files}compiler-mpi-openmpi"],
                     [AC_MSG_WARN([MPI compiler suite "${withval}" not supported by --with-mpi, ignoring.])])],
            [])

# use mpi_compiler_suite as input for process_arguments.awk
cat ${ac_scorep_compilers_mpi} > mpi_compiler_suite

# find suitable defaults if not already set by platform detection or
# configure arguments. Note that we can't source
# ${ac_scorep_compilers_mpi} directly as it may contain lines like
# 'MPIF77=mpiifort -fc=${F77}' which are not valid shell code. Adding
# quotes like in 'MPIF77="mpiifort -fc=${F77}"' would solve the
# problem here but cause headaches using AC_CONFIG_SUBDIR_CUSTOM
$AWK '{print $[]1}' ${ac_scorep_compilers_mpi} | grep MPI > mpi_compiler_suite_to_source
. ./mpi_compiler_suite_to_source

if test "x${MPICC}" = "x"; then
    AC_CHECK_PROGS(MPICC, mpicc hcc mpxlc_r mpxlc mpcc cmpicc mpiicc, $CC)
    echo "MPICC=${MPICC}" >> mpi_compiler_suite
fi

if test "x${MPICXX}" = "x"; then
    AC_CHECK_PROGS(MPICXX, mpic++ mpicxx mpiCC hcp mpxlC_r mpxlC mpCC cmpic++ mpiicpc, $CXX)
    echo "MPICXX=${MPICXX}" >> mpi_compiler_suite
fi

if test "x${MPIF77}" = "x"; then
    AC_CHECK_PROGS(MPIF77, mpif77 hf77 mpxlf_r mpxlf mpf77 cmpifc mpiifort, $F77)
    echo "MPIF77=${MPIF77}" >> mpi_compiler_suite
fi

if test "x${MPIFC}" = "x"; then
    AC_CHECK_PROGS(MPIFC, mpif90 mpxlf95_r mpxlf90_r mpxlf95 mpxlf90 mpf90 cmpif90c mpiifort, $FC)
    echo "MPIFC=${MPIFC}"   >> mpi_compiler_suite
fi

])


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
