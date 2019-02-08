## -*- mode: autoconf -*-

##
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2013
## Forschungszentrum Juelich GmbH, Germany
##
## Copyright (c) 2014
## Technische Universitaet Dresden, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##

## file build-config/m4/afs_compiler_backend.m4


dnl AFS_COMPILER_BACKEND($1 [,$2])
dnl $1: comma separated list of required compilers (out of CC, CXX, F77, FC)
dnl $2: comma separated list of optional compilers (out of CC, CXX, F77, FC)
dnl Creates output argument string 'afs_backend_compiler_vars' to be passed to build-backend configure
AC_DEFUN([AFS_COMPILER_BACKEND],
[
# parameter consistency checks
m4_ifblank([$1], [m4_fatal([Macro requires at least one argument])])
m4_foreach([afs_tmp_var],
    [$1],
    [m4_case(afs_tmp_var, [CC], [], [CXX], [], [F77], [], [FC], [],
         [m4_fatal([first parameter must be a list out of CC, CXX, F77, FC])])
])
m4_ifnblank([$2],
    m4_foreach([afs_tmp_var],
        [$2],
        [m4_case(afs_tmp_var, [CC], [], [CXX], [], [F77], [], [FC], [],
             [m4_fatal([second parameter must be a list out of CC, CXX, F77, FC or empty])])
]))

AC_REQUIRE([AC_SCOREP_TOPLEVEL_ARGS])
AC_REQUIRE([AC_SCOREP_DETECT_PLATFORMS])
AC_REQUIRE([_LT_PROG_ECHO_BACKSLASH])
AC_REQUIRE([AC_PROG_SED])
AC_REQUIRE([AC_PROG_GREP])

# Create set of requested compilers, and a list of relevant variables
m4_set_add_all([afs_requested_compilers_set], $1, $2)
m4_set_foreach([afs_requested_compilers_set],
    [afs_tmp_var],
    [m4_set_add([afs_relevant_build_variables_set], afs_tmp_var)
     m4_case(afs_tmp_var,
         [CC], [m4_set_add([afs_relevant_build_variables_set], [CFLAGS])],
         [CXX], [m4_set_add([afs_relevant_build_variables_set], [CXXFLAGS])],
         [F77], [m4_set_add([afs_relevant_build_variables_set], [FFLAGS])],
         [FC], [m4_set_add([afs_relevant_build_variables_set], [FCFLAGS])])])
m4_set_add([afs_relevant_build_variables_set], [CPPFLAGS])
m4_set_add([afs_relevant_build_variables_set], [LDFLAGS])
m4_set_add([afs_relevant_build_variables_set], [LIBS])

afs_compiler_files="$srcdir/vendor/common/build-config/platforms/backend-"

# clear all variables
m4_foreach([afs_tmp_var],
    [afs_all_build_variables_list],
    [AS_UNSET([afs_tmp_var])
    ])

# read relevant variables from platform file, provides at least CC, CXX, etc.
m4_set_foreach([afs_relevant_build_variables_set],
    [afs_tmp_var],
    [_AFS_READ_VAR_FROM_FILE([${afs_compiler_files}${ac_scorep_platform}], [platform])
     afs_tmp_var="${afs_[]afs_tmp_var[]_platform}"
    ])

# handle --with-compiler-suite= configure option, overrides platform defaults
AC_ARG_WITH([compiler-suite],
    [AS_HELP_STRING([--with-compiler-suite=(gcc|ibm|intel|pgi|studio)],
         [The compiler suite used to build this package. Applies currently only to non-cross-compile systems. Compilers need to be in $PATH [gcc].])],
    [# action if given
     AS_IF([test "x${ac_scorep_cross_compiling}" = "xno"],
         [# non-cross-compile systems, read variables from vendor file
          # user provided $withval valid?
          afs_compiler_suite="${withval}"
          AS_CASE([${withval}],
              ["no"],     [AC_MSG_ERROR([option --without-compiler-suite makes no sense.])],
              ["gcc"],    [],
              ["gnu"],    [afs_compiler_suite="gcc"],
              ["ibm"],    [],
              ["xl"],     [afs_compiler_suite="ibm"],
              ["intel"],  [],
              ["pgi"],    [],
              ["studio"], [],
              [AC_MSG_ERROR([compiler suite "${withval}" not supported by --with-compiler-suite.])])

          # read relevant variables from compiler vendor file, provides at least CC, CXX, etc.
          # override current setting with vendor settings
          AC_MSG_NOTICE([overriding platform compilers with '--with-compiler-suite=${afs_compiler_suite}'])
          m4_set_foreach([afs_relevant_build_variables_set],
              [afs_tmp_var],
              [_AFS_READ_VAR_FROM_FILE([${afs_compiler_files}${afs_compiler_suite}], [vendor])
               AS_IF([test "${afs_[]afs_tmp_var[]_vendor}" != "${afs_[]afs_tmp_var[]_platform}"],
                   [AC_MSG_NOTICE([afs_tmp_var: overriding '${afs_[]afs_tmp_var[]_platform}' with '${afs_[]afs_tmp_var[]_vendor}'])
                    afs_tmp_var="${afs_[]afs_tmp_var[]_vendor}"])
              ])
          ],
          [# cross-compile systems
           AC_MSG_WARN([--with-compiler-suite currently not supported on cross-compile systems as backend compilers are chosen by system type. To specify frontend compilers, use --with-frontend-compiler-suite.])])
    ] dnl ,[ # action if not given ]
)

# handle user-provided/configure cmd-line CC, CXX, etc.
# read relevant variables from file ./user_provided_configure_args
# override current setting with user settings, if provided
m4_set_foreach([afs_relevant_build_variables_set],
    [afs_tmp_var],
    [_AFS_READ_VAR_FROM_FILE([./user_provided_configure_args], [user])
     AS_IF([test -n "${afs_[]afs_tmp_var[]_user}"],
         [AC_MSG_NOTICE([afs_tmp_var: overriding '${afs_tmp_var}' with user provided '${afs_[]afs_tmp_var[]_user}'])
          afs_tmp_var="${afs_[]afs_tmp_var[]_user}"])
    ])

# remove all build variables and compiler options from file
# ./user_provided_configure_args to prevent duplicate processing
afs_filter_user_provided_configure_args_cmd="cat ./user_provided_configure_args | ${GREP} -v \"^--with-compiler-suite\""
m4_foreach([afs_tmp_var],
    [afs_all_build_variables_list],
    [afs_filter_user_provided_configure_args_cmd="${afs_filter_user_provided_configure_args_cmd} | ${GREP} -v \"^[]afs_tmp_var[]=\""
    ])
afs_filter_user_provided_configure_args_cmd="${afs_filter_user_provided_configure_args_cmd} > ./afs_user_provided_configure_args_tmp"
eval ${afs_filter_user_provided_configure_args_cmd}
mv ./afs_user_provided_configure_args_tmp ./user_provided_configure_args

# get the compiler's basename, ignore options like -std=c99.
m4_set_foreach([afs_requested_compilers_set],
    [afs_tmp_var],
    [_LT_CC_BASENAME(["$[]afs_tmp_var"])
     afs_tmp_var[]_basename=`$ECHO "${cc_basename}" | $AWK '{print $[]1}' | $SED "s%'%%g" | $SED "s%\"%%g"`
    ])

# check if required compilers in PATH
m4_foreach([afs_tmp_var],
    [$1],
    [afs_in_path=`which ${afs_tmp_var[]_basename} 2> /dev/null`
     AS_IF([test -z "${afs_in_path}"],
         [AC_MSG_ERROR([required compiler 'afs_tmp_var[]=${afs_tmp_var[]_basename}' not in PATH.])])
    ])

# check if optional compilers in PATH
m4_foreach([afs_tmp_var],
    [$2],
    [afs_in_path=`which ${afs_tmp_var[]_basename} 2> /dev/null`
     AS_IF([test -z "${afs_in_path}"],
         [AC_MSG_WARN([optional compiler 'afs_tmp_var[]=${afs_tmp_var[]_basename}' not in PATH.])
          AS_UNSET([afs_tmp_var])
          m4_case(afs_tmp_var,
              [CC], [AS_UNSET([CFLAGS])],
              [CXX], [AS_UNSET([CXXFLAGS])],
              [F77], [AS_UNSET([FFLAGS])],
              [FC], [AS_UNSET([FCFLAGS])])
         ])
    ])

# check if compilers are from same vendor (3)
# 1. set <>_vendor
m4_set_foreach([afs_requested_compilers_set],
    [afs_tmp_var],
    [AS_CASE([${afs_tmp_var[]_basename}],
         [gcc* | g++* | gfortran*],        [afs_tmp_var[]_vendor="gnu"],
         [icc* | icpc* | ifort*],          [afs_tmp_var[]_vendor="intel"],
         [pgcc* | pgCC* | pgcpp* | pgf* ], [afs_tmp_var[]_vendor="pgi"],
         [xlc* | xlC* | xlf*],             [afs_tmp_var[]_vendor="ibm"],
         [AS_CASE([`${afs_tmp_var[]_basename} -V 2>&1`],
              [*Sun\ C* | *Sun\ F*],       [afs_tmp_var[]_vendor="studio"],
              [AC_MSG_ERROR([unsupported compiler 'afs_tmp_var[]=${afs_tmp_var[]_basename}'. Please contact <AC_PACKAGE_BUGREPORT> or use --with-custom-compilers.])])])
    ])
# 2. check if required compilers are from same vendor
AS_UNSET([afs_common_vendor])
m4_foreach([afs_tmp_var],
    [$1],
    [AS_IF([test -z "${afs_common_vendor}"],
         [afs_common_vendor=${afs_tmp_var[]_vendor}],
         [AS_IF([test "x${afs_common_vendor}" != "x${afs_tmp_var[]_vendor}"],
              [AC_MSG_ERROR([required compilers not from a single vendor ('${afs_common_vendor}' and '${afs_tmp_var[]_vendor}'). Please use --with-custom-compilers for experimental configurations.])])])
    ])
# 3. check if optional compilers are from same vendor as required compilers
m4_foreach([afs_tmp_var],
    [$2],
    [AS_IF([test "x${afs_common_vendor}" != "x${afs_tmp_var[]_vendor}"],
        [AC_MSG_WARN([optional compiler 'afs_tmp_var=${afs_tmp_var}' not from vendor '${afs_common_vendor}'. Ignoring optional compiler. You may use --with-custom-compilers for experimental configurations.])
         AS_UNSET([afs_tmp_var])
         m4_case(afs_tmp_var,
             [CC], [AS_UNSET([CFLAGS])],
             [CXX], [AS_UNSET([CXXFLAGS])],
             [F77], [AS_UNSET([FFLAGS])],
             [FC], [AS_UNSET([FCFLAGS])])
        ])
    ])

dnl # print all variables and values
dnl m4_foreach([afs_tmp_var],
dnl     [afs_all_build_variables_list], [AS_IF([test -n "${afs_tmp_var}"], [echo afs_tmp_var[]: ${afs_tmp_var}
dnl ])
dnl ])

# create output argument string 'afs_backend_compiler_vars' to be passed to build-backend configure
AS_UNSET([afs_backend_compiler_vars])
m4_set_foreach([afs_relevant_build_variables_set],
    [afs_tmp_var],
    [AS_IF([test -n "${afs_tmp_var}"],
         [AS_CASE(["${afs_tmp_var}"],
              [*\ *],
                  [# contains spaces, already quoted?
                   ( $ECHO "${afs_tmp_var}" | $GREP ^\' | $GREP \'$ >/dev/null 2>&1 )
                   afs_single_quoted=$?
                   ( $ECHO "${afs_tmp_var}" | $GREP ^\" | $GREP \"$ >/dev/null 2>&1 )
                   afs_double_quoted=$?
                   AS_IF([test ${afs_single_quoted} -ne 0 && test ${afs_double_quoted} -ne 0 ],
                       [# needs quoting
                        afs_backend_compiler_vars="${afs_backend_compiler_vars} afs_tmp_var='${afs_tmp_var}'"],
                       [# already quoted
                        afs_backend_compiler_vars="${afs_backend_compiler_vars} afs_tmp_var=${afs_tmp_var}"])],
              [*],
                  [# contains no spaces, no quoting needed
                   afs_backend_compiler_vars="${afs_backend_compiler_vars} afs_tmp_var=${afs_tmp_var}"])
         ])
    ])

dnl echo afs_backend_compiler_vars $afs_backend_compiler_vars

m4_set_delete(afs_requested_compilers_set)
m4_set_delete(afs_relevant_build_variables_set)
])# AFS_COMPILER_BACKEND


m4_define([afs_all_build_variables_list],
    [CC, CXX, F77, FC, CPPFLAGS, CFLAGS, CXXFLAGS, FFLAGS, FCFLAGS, LDFLAGS, LIBS])

dnl _AFS_READ_VAR_FROM_FILE($1, $2)
dnl $1: file to read variables from
dnl $2: variable name suffix
m4_define([_AFS_READ_VAR_FROM_FILE],
    [afs_[]afs_tmp_var[]_[]$2[]="`${GREP} "^[]afs_tmp_var[]=" $1 | ${AWK} -F "=" '{print $[]2}'`"])


dnl ----------------------------------------------------------------------------

# AFS_CUSTOM_COMPILERS
# --------------------
# Provide --with-custom-compiler configure option; if given sets
#   ac_scorep_compilers_backend
#   ac_scorep_compilers_frontend
#   ac_scorep_compilers_mpi
# to AFS_COMPILER_FILES_COMMON/platform-*-user-provided files.
# Provides afs_custom_compilers_given=(yes|no) to be used by other
# macros setting ac_scorep_compilers_*.
AC_DEFUN([AFS_CUSTOM_COMPILERS],
[
    AC_REQUIRE([AC_SCOREP_DETECT_PLATFORMS])

    AC_ARG_WITH([custom-compilers],
        [AS_HELP_STRING([--with-custom-compilers],
             [Customize compiler settings by 1. copying the files <srcdir>/vendor/common/build-config/platforms/platform-*-user-provided to the directory where you run configure <builddir>, 2. editing those files to your needs, and 3. running configure. Alternatively, edit the files under <srcdir> directly. Files in <builddir> take precedence. You are entering unsupported terrain. Namaste, and good luck!])
        ],
        [afs_custom_compilers_given="yes"
         AS_CASE([${withval}],
             ["yes"], [AS_IF([test -f ./platform-backend-user-provided && \
                              test -f ./platform-frontend-user-provided && \
                              test -f ./platform-mpi-user-provided && \
                              test -f ./platform-shmem-user-provided],
                           [AC_MSG_NOTICE([Using compiler specification from ./platform-*-user-provided files.])
                            ac_scorep_compilers_backend="./platform-backend-user-provided"
                            ac_scorep_compilers_frontend="./platform-frontend-user-provided"
                            ac_scorep_compilers_mpi="./platform-mpi-user-provided"
                            ac_scorep_compilers_shmem="./platform-shmem-user-provided"],
                           [AC_MSG_NOTICE([Using compiler specification from AFS_COMPILER_FILES_COMMON/platform-*-user-provided files.])
                            ac_scorep_compilers_backend="AFS_COMPILER_FILES_COMMON/platform-backend-user-provided"
                            ac_scorep_compilers_frontend="AFS_COMPILER_FILES_COMMON/platform-frontend-user-provided"
                            ac_scorep_compilers_mpi="AFS_COMPILER_FILES_COMMON/platform-mpi-user-provided"
                            ac_scorep_compilers_shmem="AFS_COMPILER_FILES_COMMON/platform-shmem-user-provided"])
                      ],
             [AC_MSG_ERROR(['${withval}' not supported by --with-custom-compilers.])])
        ],
        [afs_custom_compilers_given="no"
        ])
])

dnl ----------------------------------------------------------------------------

# AFS_COMPILER_FILES_(COMMON|PACKAGE)
# -----------------------------------
# Use AFS_COMPILER_FILES_* as alias for $srcdir/[vendor/common/]build-config/platforms
# for setting paths to compiler files.
m4_define([AFS_COMPILER_FILES_COMMON], [$srcdir/vendor/common/build-config/platforms])
m4_define([AFS_COMPILER_FILES_PACKAGE], [$srcdir/build-config/platforms])

dnl ----------------------------------------------------------------------------

# Following two macros copied from
# <autotools-prefix>/share/aclocal/libtool.m4. I don't wont to call
# LT_INIT from the toplevel configure but want to use
# _LT_CC_BASENAME(CC) as in libtool.
# Should be moved into a separate file with license=GPL

# _LT_CC_BASENAME(CC)
# -------------------
# Calculate cc_basename.  Skip known compiler wrappers and cross-prefix.
m4_defun([_LT_CC_BASENAME],
[for cc_temp in $1""; do
  case $cc_temp in
    compile | *[[\\/]]compile | ccache | *[[\\/]]ccache ) ;;
    distcc | *[[\\/]]distcc | purify | *[[\\/]]purify ) ;;
    \-*) ;;
    *) break;;
  esac
done
cc_basename=`$ECHO "$cc_temp" | $SED "s%.*/%%; s%^$host_alias-%%"`
])


# _LT_PROG_ECHO_BACKSLASH
# -----------------------
# Find how we can fake an echo command that does not interpret backslash.
# In particular, with Autoconf 2.60 or later we add some code to the start
# of the generated configure script which will find a shell with a builtin
# printf (which we can use as an echo command).
m4_defun([_LT_PROG_ECHO_BACKSLASH],
[ECHO='\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\'
ECHO=$ECHO$ECHO$ECHO$ECHO$ECHO
ECHO=$ECHO$ECHO$ECHO$ECHO$ECHO$ECHO

AC_MSG_CHECKING([how to print strings])
# Test print first, because it will be a builtin if present.
if test "X`( print -r -- -n ) 2>/dev/null`" = X-n && \
   test "X`print -r -- $ECHO 2>/dev/null`" = "X$ECHO"; then
  ECHO='print -r --'
elif test "X`printf %s $ECHO 2>/dev/null`" = "X$ECHO"; then
  ECHO='printf %s\n'
else
  # Use this function as a fallback that always works.
  func_fallback_echo ()
  {
    eval 'cat <<_LTECHO_EOF
$[]1
_LTECHO_EOF'
  }
  ECHO='func_fallback_echo'
fi

# func_echo_all arg...
# Invoke $ECHO with all args, space-separated.
func_echo_all ()
{
    $ECHO "$*"
}

case "$ECHO" in
  printf*) AC_MSG_RESULT([printf]) ;;
  print*) AC_MSG_RESULT([print -r]) ;;
  *) AC_MSG_RESULT([cat]) ;;
esac

m4_ifdef([_AS_DETECT_SUGGESTED],
[_AS_DETECT_SUGGESTED([
  test -n "${ZSH_VERSION+set}${BASH_VERSION+set}" || (
    ECHO='\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\'
    ECHO=$ECHO$ECHO$ECHO$ECHO$ECHO
    ECHO=$ECHO$ECHO$ECHO$ECHO$ECHO$ECHO
    PATH=/empty FPATH=/empty; export PATH FPATH
    test "X`printf %s $ECHO`" = "X$ECHO" \
      || test "X`print -r -- $ECHO`" = "X$ECHO" )])])

dnl Don't declare libtool variables, we are just interested in ECHO here.
dnl _LT_DECL([], [SHELL], [1], [Shell to use when invoking shell scripts])
dnl _LT_DECL([], [ECHO], [1], [An echo program that protects backslashes])
])# _LT_PROG_ECHO_BACKSLASH
