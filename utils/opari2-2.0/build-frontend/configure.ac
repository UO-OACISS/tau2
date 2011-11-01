AC_PREREQ([2.68])
AC_INIT([opari2],[2.0],[zih-silc-dev@groups.tu-dresden.de])

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

AM_INIT_AUTOMAKE([foreign silent-rules color-tests 1.11.1 -Wall tar-ustar])

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
AC_CONFIG_FILES([pomp2_parse_init_regions.awk:../src/pomp-lib-dummy/pomp2_parse_init_regions.awk.in])
AC_CONFIG_FILES([../doc/example/Makefile]) 

AC_OUTPUT
