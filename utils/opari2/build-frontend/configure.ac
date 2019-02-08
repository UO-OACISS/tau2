AC_PREREQ([2.69])
AC_INIT([OPARI2], m4_esyscmd([../vendor/common/build-config/generate-package-version.sh ../build-config/VERSION]), [support@score-p.org], [opari2])

AFS_PACKAGE_BUILD_INIT([frontend])

AC_SUBST([LIBRARY_INTERFACE_VERSION], m4_esyscmd([../vendor/common/build-config/generate-library-version.sh ../build-config/VERSION]))

AC_CONFIG_AUX_DIR([../build-config])
## See also ACLOCAL_AMFLAGS in Makefile.am.
AC_CONFIG_MACRO_DIRS([../build-config/m4 ../vendor/common/build-config/m4])

AC_CONFIG_HEADERS([../src/config-frontend.h])

AC_DEFUN([AC_SCOREP_FRONTEND], [])

ac_scorep_frontend="yes"
ac_scorep_backend="no"

AC_SCOREP_PRECIOUS_VARS_FOR_BUILD

AC_SCOREP_OPARI2_FOR_BUILD_ARGS_TAKES_PRECEDENCE

AC_SCOREP_CHECK_COMPILER_VAR_SET([CC])
AC_SCOREP_CHECK_COMPILER_VAR_SET([CXX])
AC_SCOREP_CHECK_COMPILER_VAR_SET([F77])
AC_SCOREP_CHECK_COMPILER_VAR_SET([FC])

#AC_SCOREP_CONVERT_FLAGS([CFLAGS])
#AC_SCOREP_CONVERT_FLAGS([CXXFLAGS])
#AC_SCOREP_CONVERT_FLAGS([CPPFLAGS])

#AC_SCOREP_CONVERT_COMPILER([CC])
#AC_SCOREP_CONVERT_COMPILER([CXX])

adl_RECURSIVE_EVAL([${includedir}], [INCLUDEDIR])
AC_SUBST([INCLUDEDIR])
adl_RECURSIVE_EVAL([${bindir}], [BINDIR])
AC_SUBST([BINDIR])

AC_SCOREP_SVN_CONTROLLED

AM_INIT_AUTOMAKE([foreign color-tests 1.13.4 -Wall tar-pax])
AM_SILENT_RULES([yes])
AM_PROG_AR
AFS_MAINTAINER_MODE

adl_RECURSIVE_EVAL([${includedir}], [INCLUDEDIR])
AC_SUBST([INCLUDEDIR])

AM_CONDITIONAL([BUILD_TESTS], [test 1 -eq 1])
AC_SCOREP_PLATFORM_SETTINGS

# <AC_OPARI_COMMON_CHECKS>
AC_PROG_CC
AC_SCOREP_COMPILER_CHECKS

AC_SCOREP_PROG_CC_C99([], [AC_MSG_ERROR([No ISO C99 support in C compiler.])])

AC_PROG_CXX

AM_PROG_CC_C_O

AC_PROG_F77
AC_SCOREP_HAVE_F77

AC_PROG_FC
AC_SCOREP_HAVE_FC

AC_LANG_PUSH([C])
AC_OPENMP
AC_LANG_POP([C])

AM_CONDITIONAL([HAVE_OPENMP_SUPPORT],
               [test "x${ac_cv_prog_c_openmp}" != "xunsupported" && test "x${enable_openmp}" != "xno"])

if test "x${ac_cv_prog_c_openmp}" = "xunsupported"; then
    AC_MSG_WARN([no suitable OpenMP compilers found. POMP2 dummy lib will not be build.])
else
    AC_LANG_PUSH([C++])
    AC_OPENMP
    AC_LANG_POP([C++])

    AC_LANG_PUSH([Fortran 77])
    AC_OPENMP
    AC_LANG_POP([Fortran 77])

    AC_LANG_PUSH([Fortran])
    AC_OPENMP
    AC_LANG_POP([Fortran])
fi

AC_SCOREP_FORTRAN_SUPPORT_ALLOCATABLE

# Special alignement needed with --tpd option. 16bit alignement used
# by Fortran, so we need to use it in C/C++ as well.
AC_DEFINE([FORTRAN_ALIGNED],
    [__attribute__((aligned (16)))],
    [Makes C variable alignment consistent with Fortran])

AC_SCOREP_DEFINE_REVISIONS
# </AC_OPARI_COMMON_CHECKS>
AC_SCOREP_PACKAGE_AND_LIBRARY_VERSION

LT_PREREQ([2.4.6])
LT_INIT([disable-shared])
AC_SUBST([LIBTOOL_DEPS])

adl_RECURSIVE_EVAL([${libexecdir}], [LIBEXECDIR])
AC_SUBST([LIBEXECDIR], [${LIBEXECDIR}])

AC_MSG_CHECKING([for readlink])
AC_LINK_IFELSE([AC_LANG_SOURCE([
#include <unistd.h>
#include <stdio.h>
int main()
{
    char buffer[[BUFSIZ]];
    readlink("/proc/self/exe", buffer, BUFSIZ);
    printf("%s\n", buffer);
    return 0;
}
])],
    [AC_MSG_RESULT(yes);
     AC_DEFINE(HAVE_READLINK, 1, [Can link a readlink function])],
    [AC_MSG_RESULT(no)])

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([../test/jacobi_c_test.sh], [chmod +x ../test/jacobi_c_test.sh])
AC_CONFIG_FILES([../test/jacobi_c++_test.sh], [chmod +x ../test/jacobi_c++_test.sh])
AC_CONFIG_FILES([../test/jacobi_f90_test.sh], [chmod +x ../test/jacobi_f90_test.sh])
AC_CONFIG_FILES([../test/jacobi_f77_test.sh], [chmod +x ../test/jacobi_f77_test.sh])
AC_CONFIG_FILES([../doc/example/Makefile])
AC_CONFIG_FILES([../doc/example_user_instrumentation/Makefile])
AC_CONFIG_FILES([../src/opari2_config_tool_frontend.h:../src/opari/opari2_config_tool_frontend.h.in])
AC_CONFIG_FILES([../include/opari2/pomp2_lib.h:../include/opari2/pomp2_lib.h])
AC_CONFIG_FILES([../include/opari2/pomp2_user_lib.h:../include/opari2/pomp2_user_lib.h])

AC_OUTPUT
