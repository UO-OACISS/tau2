AC_PREREQ([2.68])
AC_INIT([opari2], m4_esyscmd([../vendor/common/build-config/generate-package-version.sh ../build-config/VERSION]), [scorep-bugs@groups.tu-dresden.de])

AC_SUBST([LIBRARY_INTERFACE_VERSION], m4_esyscmd([../vendor/common/build-config/generate-library-version.sh ../build-config/VERSION]))

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

AC_SCOREP_SVN_CONTROLLED
AH_SCOREP_CONFIG_HEADER_CUSTOM
AC_CONFIG_HEADERS([../src/config-frontend.h])




AC_CONFIG_AUX_DIR([../build-config])
AC_CONFIG_MACRO_DIR([../vendor/common/build-config/m4])

AM_INIT_AUTOMAKE([foreign color-tests 1.11.1 -Wall tar-pax])
AM_SILENT_RULES([yes])

AM_CONDITIONAL([BUILD_TESTS], [test 1 -eq 1])
AC_SCOREP_PLATFORM_SETTINGS

AC_OPARI_COMMON_CHECKS

LT_PREREQ([2.4])
LT_INIT([disable-shared])
AC_SUBST([LIBTOOL_DEPS])



AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([../test/jacobi_c_test.sh], [chmod +x ../test/jacobi_c_test.sh])
AC_CONFIG_FILES([../test/jacobi_c++_test.sh], [chmod +x ../test/jacobi_c++_test.sh])
AC_CONFIG_FILES([../test/jacobi_f90_test.sh], [chmod +x ../test/jacobi_f90_test.sh])
AC_CONFIG_FILES([../test/jacobi_f77_test.sh], [chmod +x ../test/jacobi_f77_test.sh])
AC_CONFIG_FILES([../doc/example/Makefile]) 
AC_CONFIG_FILES([../doc/example_user_instrumentation/Makefile]) 

AC_OUTPUT
