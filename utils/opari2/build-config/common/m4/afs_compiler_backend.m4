## -*- mode: autoconf -*-

##
## This file is part of the Score-P software ecosystem (http://www.score-p.org)
##
## Copyright (c) 2013,
## Forschungszentrum Juelich GmbH, Germany
##
## Copyright (c) 2014, 2019,
## Technische Universitaet Dresden, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##


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
             [Customize compiler settings by 1. copying the files <srcdir>/build-config/common/platforms/platform-*-user-provided to the directory where you run configure <builddir>, 2. editing those files to your needs, and 3. running configure. Alternatively, edit the files under <srcdir> directly. Files in <builddir> take precedence. You are entering unsupported terrain. Namaste, and good luck!])
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
m4_define([AFS_COMPILER_FILES_COMMON], [$srcdir/build-config/common/platforms])
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
