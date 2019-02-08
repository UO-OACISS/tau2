AC_PREREQ([2.69])
AC_INIT([OPARI2], m4_esyscmd([vendor/common/build-config/generate-package-version.sh build-config/VERSION]), [support@score-p.org], [opari2])

AFS_PACKAGE_INIT

AC_SCOREP_REVISION

AC_SCOREP_TOPLEVEL_ARGS

## Packages that do manual configuration or use the install program might need
## to tell configure where to find some other shell scripts by calling
## AC_CONFIG_AUX_DIR, though the default places it looks are correct for most
## cases.
## Need to be called early on configure.ac because other macros search tools
## in this directory.
AC_CONFIG_AUX_DIR([build-config])
## See also ACLOCAL_AMFLAGS in Makefile.am.
AC_CONFIG_MACRO_DIRS([build-config/m4 vendor/common/build-config/m4])

## Set the default installation prefix to /opt/opari2 instead of /usr/local
AC_PREFIX_DEFAULT([/opt/opari2])

AC_SCOREP_SVN_CONTROLLED

## Call not before AC_CONFIG_MACRO_DIR
AM_INIT_AUTOMAKE([foreign color-tests 1.13.4 -Wall tar-pax])
AM_SILENT_RULES([yes])
AFS_MAINTAINER_MODE

AC_SCOREP_DETECT_PLATFORMS
AFS_CROSSCOMPILING

# don't detect and load defaults in nested configures (e.g. otf2)
AS_IF([test "x$ac_scorep_platform_data_provided" = "x" ||
       test "${ac_scorep_platform_data_provided}" = "no"],
    [AFS_CUSTOM_COMPILERS
     AS_IF([test "x${afs_custom_compilers_given}" = "xno"],
        [AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE])
     # determine arguments for subdir configures
     args="`$AWK -f $srcdir/vendor/common/build-config/process_arguments.awk \
            $ac_scorep_compilers_backend                                     \
            $ac_scorep_compilers_frontend                                    \
            user_provided_configure_args`"
     AS_IF([test $? -ne 0], [AC_MSG_ERROR([cannot process provided and/or autodetected arguments. Please contact <AC_PACKAGE_BUGREPORT> and provide the above output. Thanks.])])
    ],
    [AC_MSG_NOTICE([platform data provided by toplevel configure.])
     # determine arguments for subdir configures
     args="`$AWK -f $srcdir/vendor/common/build-config/process_arguments.awk \
            user_provided_configure_args`"
     AS_IF([test $? -ne 0], [AC_MSG_ERROR([cannot process provided arguments. Please contact <AC_PACKAGE_BUGREPORT> and provide the above output. Thanks.])])
    ])

AC_SCOREP_UNCRUSTIFY

AC_SCOREP_DOXYGEN
AS_IF([test "x${ac_scorep_svn_controlled}" = xyes],
    [AC_CONFIG_FILES([doc/doxygen-user.cfg:doc/doxygen-user/doxygen-user.cfg.in])
     AC_CONFIG_FILES([doc/doxygen-dev.cfg:doc/doxygen-dev/doxygen-dev.cfg.in])])

## Autoconf supports changing the names of programs when installing them. In
## order to use these transformations, configure.ac must call the macro
## AC_ARG_PROGRAM.
# implicitly called by ?
#AC_ARG_PROGRAM

# used for EXTRA_DIST in Makefile.am
AM_CONDITIONAL([CROSS_BUILD], [test "x${ac_scorep_cross_compiling}" = "xyes"])

# handle own package's subdir configures
AX_CONFIG_SUBDIR([build-frontend], [${args} ac_scorep_platform=${ac_scorep_platform}])

#sed is needed for check target
AC_PROG_SED
AC_SUBST([OPARI2_AWK], [`which ${AWK}`])

AC_CONFIG_FILES([
    Makefile
    src/config-common.h:vendor/common/build-config/config-common.h
])
AC_CONFIG_FILES([test/c_test.sh], [chmod +x test/c_test.sh])
AC_CONFIG_FILES([test/f90_test.sh], [chmod +x test/f90_test.sh])
AC_CONFIG_FILES([test/f77_test.sh], [chmod +x test/f77_test.sh])
AC_CONFIG_FILES([test/awk_script_test.sh], [chmod +x test/awk_script_test.sh])
AC_CONFIG_FILES([test/awk_script_test.output:test/data/awk_script_test.out.in])
AC_CONFIG_FILES([build-frontend/pomp2-parse-init-regions.awk:src/opari-lib-dummy/pomp2_parse_init_regions.awk.in], \
                [chmod +x build-frontend/pomp2-parse-init-regions.awk])
AC_OUTPUT
