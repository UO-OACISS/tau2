AC_PREREQ([2.68])
AC_INIT([opari2],[2.0],[zih-silc-dev@groups.tu-dresden.de])

#echo "cross_compiling = $cross_compiling"

AC_SCOREP_VERSION_AND_REVISION

AC_SCOREP_TOPLEVEL_ARGS

## Packages that do manual configuration or use the install program might need
## to tell configure where to find some other shell scripts by calling
## AC_CONFIG_AUX_DIR, though the default places it looks are correct for most
## cases.
## Need to be called early on configure.ac because other macros search tools
## in this directory.
AC_CONFIG_AUX_DIR([build-config])

## Packages that use aclocal should declare where local macros can be found
## using AC_CONFIG_MACRO_DIR.
AC_CONFIG_MACRO_DIR([vendor/common/build-config/m4])

## Set the default installation prefix to /opt/scorep instead of /usr/local
AC_PREFIX_DEFAULT([/opt/scorep])

## Call not before AC_CONFIG_MACRO_DIR
AM_INIT_AUTOMAKE([foreign silent-rules color-tests 1.11.1 -Wall tar-ustar])



# don't detect and load defaults in nested configures (e.g. otf2)
if test ! -n "$ac_scorep_platform_data_provided" -o "x${ac_scorep_platform_data_provided}" = "xno"; then
    # $ac_scorep_cross_compiling, $ac_scorep_platform are set here
    AC_SCOREP_DETECT_PLATFORMS
    AC_SCOREP_WITH_NOCROSS_COMPILER_SUITE
    # determine arguments for subdir configures
    args="`$AWK -f $srcdir/vendor/common/build-config/process_arguments.awk \
           $ac_scorep_compilers_backend                                     \
           $ac_scorep_compilers_frontend                                    \
           user_provided_configure_args`"
else
    AC_MSG_NOTICE([platform data provided by toplevel configure.])
    # determine arguments for subdir configures
    args="`$AWK -f $srcdir/vendor/common/build-config/process_arguments.awk \
           user_provided_configure_args`"
fi

AC_SCOREP_SVN_CONTROLLED

AC_SCOREP_UNCRUSTIFY

AC_SCOREP_DOXYGEN

## Autoconf supports changing the names of programs when installing them. In
## order to use these transformations, configure.ac must call the macro
## AC_ARG_PROGRAM.
# implicitly called by ?
#AC_ARG_PROGRAM

# used for EXTRA_DIST in Makefile.am
AM_CONDITIONAL([CROSS_BUILD], [test "x${ac_scorep_cross_compiling}" = "xyes"])

# handle external package's subdir configures
#AC_CONFIG_SUBDIR_CUSTOM([vendor/otf2], [${args} ac_scorep_platform=${ac_scorep_platform} ac_scorep_platform_data_provided=yes ac_scorep_cross_compiling=${ac_scorep_cross_compiling}])

# handle own package's subdir configures
AC_CONFIG_SUBDIR_CUSTOM([build-frontend], [${args} ac_scorep_platform=${ac_scorep_platform}])

#sed is needed for check target
AC_PROG_SED

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([test/c_test.sh], [chmod +x test/c_test.sh]) 
AC_CONFIG_FILES([test/f90_test.sh], [chmod +x test/f90_test.sh]) 
AC_CONFIG_FILES([test/f77_test.sh], [chmod +x test/f77_test.sh]) 
AC_OUTPUT
