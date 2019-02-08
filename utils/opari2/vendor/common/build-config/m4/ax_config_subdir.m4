#
# SYNOPSIS
#
#   AX_CONFIG_SUBDIR(DIR, ARGUMENTS [, DISABLE_HELP_RECURSIVE])
#   AX_CONFIG_SUBDIR_IMMEDIATE(DIR, ARGUMENTS [, DISABLE_HELP_RECURSIVE])
#
#
# DESCRIPTION
#
#    These macros allow to configure packages in subdirectories using
#    custom arguments. They are intended as a replacement of
#    AC_CUSTOM_SUBDIRS if custom arguments need to be provided to the
#    sub-configures. They build on AC_CUSTOM_SUBDIRS code that was
#    copy-pasted from autoconf's status.m4 and slightly modified. In
#    contrast to AC_CUSTOM_SUBDIRS you call this macros with a single
#    directory DIR-ARGUMENTS tuple at a time. Ensures that DIR is
#    unique. As in AC_CUSTOM_SUBDIRS, ARGUMENTS are adjusted (relative
#    name for the cache file, relative name for the source directory),
#    the current value of $prefix is propagated, including if it was
#    defaulted, and --disable-option-checking is prepended to silence
#    warnings, since different subdirs can have different --enable and
#    --with options. Use this macros inside a conditional to control
#    the execution of the nested configure at runtime. A list of
#    runtime directories ax_config_subdirs is provided via AC_SUBST.
#    DIR should be a literal to allow for proper execution of
#    'configure --help=recursive'. By default, 'configure
#    --help=recursive' will recurse into DIR. To prevent this, pass
#    DISABLE_HELP_RECURSIVE as third parameter.
#
#    AX_CONFIG_SUBDIR emits code that might invoke a nested configure
#    at AC_CONFIG_COMMANDS_POST time in directory DIR (if existent)
#    with arguments ARGUMENTS.
#
#    AX_CONFIG_SUBDIR_IMMEDIATE emits code that might invoke a nested
#    configure immediatly in directory DIR (if existent) with
#    arguments ARGUMENTS.
#
#
# LICENSE
#
#   Copyright (C) 1992-1996, 1998-2012 Free Software Foundation, Inc.
#   Copyright (c) 2016-2017, Christian Feld <c.feld@fz-juelich.de>
#
#   This file is based on parts of autoconf's (version 2.69)
#   status.m4. This program is free software; you can redistribute it
#   and/or modify it under the terms of the GNU General Public License
#   as published by the Free Software Foundation, either version 3 of
#   the License, or (at your option) any later version.
#


# AX_CONFIG_SUBDIR(DIR, ARGUMENTS [, DISABLE_HELP_RECURSIVE])
# -----------------------------------------------------------
#
AC_DEFUN([AX_CONFIG_SUBDIR],
dnl
_AX_CONFIG_SUBDIR_COMMON($1, $2, $3)dnl
dnl
dnl increment macro-invocation counter
[m4_pushdef([_AX_CONFIG_SUBDIR_COUNT], m4_incr(_AX_CONFIG_SUBDIR_COUNT))]dnl
dnl
dnl Add empty ax_config_subdir_dir/args_<i> variables to DEFAULTS section
[m4_divert_text([DEFAULTS],
[ax_config_subdir_dir_[]_AX_CONFIG_SUBDIR_COUNT[]=
ax_config_subdir_args_[]_AX_CONFIG_SUBDIR_COUNT[]=])]dnl
dnl
dnl Assign runtime values to ax_config_subdir_dir/args_<i> variables.
[ax_config_subdir_dir_[]_AX_CONFIG_SUBDIR_COUNT[]="$1"]
[ax_config_subdir_args_[]_AX_CONFIG_SUBDIR_COUNT[]="$2"]
dnl
dnl Append code to _AX_CONFIG_SUBDIRS_RUN to execute configure script
dnl in $1 if AX_CONFIG_SUBDIR was called at runtime.
[m4_append([_AX_CONFIG_SUBDIRS_RUN],
AS_IF([test "x${no_recursion}" != xyes &&
       test "x${ax_config_subdir_dir_[]_AX_CONFIG_SUBDIR_COUNT[]}" != x &&
       test -d "${srcdir}/${ax_config_subdir_dir_[]_AX_CONFIG_SUBDIR_COUNT[]}"],
    [ax_fn_config_subdir "${ax_config_subdir_dir_[]_AX_CONFIG_SUBDIR_COUNT[]}" "${ax_config_subdir_args_[]_AX_CONFIG_SUBDIR_COUNT[]}"])
)]dnl
dnl
dnl Provide list of runtime-subdirs to Makefile.
[AC_SUBST([ax_config_subdirs], ["${ax_config_subdirs} m4_normalize([$1])"])]dnl
)# AX_CONFIG_SUBDIR


# AX_CONFIG_SUBDIR_IMMEDIATE(DIR, ARGUMENTS [, DISABLE_HELP_RECURSIVE])
# ---------------------------------------------------------------------
#
AC_DEFUN([AX_CONFIG_SUBDIR_IMMEDIATE],
dnl
_AX_CONFIG_SUBDIR_COMMON($1, $2, $3)dnl
dnl
AS_IF([test "x${no_recursion}" != xyes && test -d "${srcdir}/$1"],
    [ax_fn_config_subdir "$1" "$2"])
dnl
dnl Provide list of runtime-subdirs to Makefile.
[AC_SUBST([ax_config_subdirs], ["${ax_config_subdirs} m4_normalize([$1])"])]dnl
)# AX_CONFIG_SUBDIR_IMMEDIATE


# _AX_CONFIG_SUBDIR_COMMON
# ------------------------
# Functionality common to AX_CONFIG_SUBDIR and AX_CONFIG_SUBDIR_IMMEDIATE.
#
m4_define([_AX_CONFIG_SUBDIR_COMMON],
[AC_REQUIRE([AC_CONFIG_AUX_DIR_DEFAULT])]dnl
[AC_REQUIRE([AC_DISABLE_OPTION_CHECKING])]dnl
[AC_REQUIRE([_AX_CONFIG_SUBDIR_INIT])]dnl
dnl
dnl Two argument required
[m4_ifblank([$1], [m4_fatal([Macro requires at least two arguments])])]dnl
[m4_ifblank([$2], [m4_fatal([Macro requires at least two arguments])])]dnl
dnl
[AS_LITERAL_IF([$1], [],
    [AC_DIAGNOSE([syntax], [$0: you should use literals])])]dnl
dnl
dnl Prevent multiple registration of DIR.
[_AX_CONFIG_SUBDIR_UNIQUE(_AC_CONFIG_COMPUTE_DEST($1))]dnl
dnl
dnl Append DIR to _AC_LIST_SUBDIRS which defines ac_subdirs_all. This
dnl list is used for recursive help output only. Note that with a
dnl separate AX list and using the HELP_END diversion it is not
dnl possible to generate our own recursive help output as configure
dnl exits after ac_subdirs_all is processed. I (CF) did not find a
dnl way to move my code before the generic processing. Prevent
dnl recursive help if $3 is given.
[m4_ifblank([$3], [m4_append([_AC_LIST_SUBDIRS], [$1], [
])])]dnl
)# _AX_CONFIG_SUBDIR_COMMON


# _AX_CONFIG_SUBDIR_INIT
# -------------------------
#
AC_DEFUN_ONCE([_AX_CONFIG_SUBDIR_INIT],
dnl initialize macro-invocation counter
[m4_pushdef([_AX_CONFIG_SUBDIR_COUNT], 0)]dnl
[m4_append([_AX_CONFIG_SUBDIRS_RUN],
# Execute subdir configures defined via AX_CONFIG_SUBDIR.
#
# Do not complain if ax_config_subdir_dir is missing, so a configure
# script can configure whichever parts of a large source tree are
# present.
)]dnl
dnl Run configures after config.status creation
AC_CONFIG_COMMANDS_POST([_AX_CONFIG_SUBDIRS_RUN])dnl
_AX_CONFIG_SUBDIR_DEFINE_FN
)# _AX_CONFIG_SUBDIR_INIT


# _AX_CONFIG_SUBDIR_UNIQUE(DIR)
# -----------------------------
# See also _AC_CONFIG_UNIQUE in status.m4. Prevent multiple usage of
# DIR in AC_CONFIG_SUBDIRS or AX_CONFIG_SUBDIR. Note that if DIR was
# defined in AX_CONFIG_SUBDIR first, using DIR in AC_CONFIG_SUBDIRS
# will issue a wrong error message ('... is already registered with
# AC_CONFIG_SUBDIRS.').
#
m4_define([_AX_CONFIG_SUBDIR_UNIQUE],
dnl Abort if DIR was already defined with AX_CONFIG_SUBDIR
[m4_ifdef([_AX_CONFIG_SUBDIR_SEEN_TAG($1)],
   [m4_fatal([`$1' is already registered with AX_CONFIG_SUBDIR])])]dnl
dnl Abort if DIR was already defined with AC_CONFIG_SUBDIRS
[m4_ifdef([_AC_SEEN_TAG($1)],
   [m4_fatal([`$1' is already registered with AC_CONFIG_]m4_defn(
     [_AC_SEEN_TAG($1)]).)])]dnl
dnl Make DIR unique
[m4_define([_AC_SEEN_TAG($1)], [SUBDIRS])]dnl
[m4_define([_AX_CONFIG_SUBDIR_SEEN_TAG($1)])]dnl
)# _AX_CONFIG_SUBDIR_UNIQUE(DIR)


# _AX_CONFIG_SUBDIR_DEFINE_FN
# ---------------------------
# Define a shell function to call configures in subdirectories with
# custom arguments.
#
m4_define([_AX_CONFIG_SUBDIR_DEFINE_FN],
[AS_REQUIRE_SHELL_FN([ax_fn_config_subdir],
    [AS_FUNCTION_DESCRIBE([ax_fn_config_subdir], [],
        [trigger configure in ax_subdir with arguments ax_subdir_args. Adjusts
parameters by updating --cache-file, --srcdir, --prefix, and adding
--disable-option-checking as in AC_CONFIG_SUBIRS.])],
[dnl body to expand
ax_subdir="$[]1"
ax_subdir_args="$[]2"

# BEGIN: slightly modified snippet from status.m4
# Remove --cache-file, --srcdir, --prefix, and
# --disable-option-checking arguments so they do not pile up.
# Input: ax_subdir_args; output: ax_sub_configure_args
ax_sub_configure_args=
ac_prev=
eval "set x $ax_subdir_args"
shift
for ac_arg
do
  if test -n "$ac_prev"; then
    ac_prev=
    continue
  fi
  case $ac_arg in
  -cache-file | --cache-file | --cache-fil | --cache-fi \
  | --cache-f | --cache- | --cache | --cach | --cac | --ca | --c)
    ac_prev=cache_file ;;
  -cache-file=* | --cache-file=* | --cache-fil=* | --cache-fi=* \
  | --cache-f=* | --cache-=* | --cache=* | --cach=* | --cac=* | --ca=* \
  | --c=*)
    ;;
  --config-cache | -C)
    ;;
  -srcdir | --srcdir | --srcdi | --srcd | --src | --sr)
    ac_prev=srcdir ;;
  -srcdir=* | --srcdir=* | --srcdi=* | --srcd=* | --src=* | --sr=*)
    ;;
  -prefix | --prefix | --prefi | --pref | --pre | --pr | --p)
    ac_prev=prefix ;;
  -prefix=* | --prefix=* | --prefi=* | --pref=* | --pre=* | --pr=* | --p=*)
    ;;
  --disable-option-checking)
    ;;
  *)
    case $ac_arg in
    *\'*) ac_arg=`AS_ECHO(["$ac_arg"]) | sed "s/'/'\\\\\\\\''/g"` ;;
    esac
    AS_VAR_APPEND([ax_sub_configure_args], [" '$ac_arg'"]) ;;
  esac
done

# Always prepend --prefix to ensure using the same prefix
# in subdir configurations.
ac_arg="--prefix=$prefix"
case $ac_arg in
*\'*) ac_arg=`AS_ECHO(["$ac_arg"]) | sed "s/'/'\\\\\\\\''/g"` ;;
esac
ax_sub_configure_args="'$ac_arg' $ax_sub_configure_args"

# Pass --silent
if test "$silent" = yes; then
  ax_sub_configure_args="--silent $ax_sub_configure_args"
fi

# Always prepend --disable-option-checking to silence warnings, since
# different subdirs can have different --enable and --with options.
ax_sub_configure_args="--disable-option-checking $ax_sub_configure_args"

ax_popdir=`pwd`

ac_msg="=== configuring in $ax_subdir (`pwd`/$ax_subdir)"
_AS_ECHO_LOG([$ac_msg])
_AS_ECHO([$ac_msg])
AS_MKDIR_P(["$ax_subdir"])
_AC_SRCDIRS(["$ax_subdir"])

cd "$ax_subdir"

# Check for guested configure; otherwise get Cygnus style configure.
if test -f "$ac_srcdir/configure.gnu"; then
  ax_sub_configure=$ac_srcdir/configure.gnu
elif test -f "$ac_srcdir/configure"; then
  ax_sub_configure=$ac_srcdir/configure
elif test -f "$ac_srcdir/configure.in"; then
  # This should be Cygnus configure.
  ax_sub_configure=$ac_aux_dir/configure
else
  AC_MSG_WARN([no configuration information is in $ax_subdir])
  ax_sub_configure=
fi

# The recursion is here.
if test -n "$ax_sub_configure"; then
  # Make the cache file name correct relative to the subdirectory.
  case $cache_file in
  [[\\/]]* | ?:[[\\/]]* ) ac_sub_cache_file=$cache_file ;;
  *) # Relative name.
  ac_sub_cache_file=$ac_top_build_prefix$cache_file ;;
  esac

  AC_MSG_NOTICE([running $SHELL $ax_sub_configure $ax_sub_configure_args --cache-file=$ac_sub_cache_file --srcdir=$ac_srcdir])
  # The eval makes quoting arguments work.
  eval "\$SHELL \"\$ax_sub_configure\" $ax_sub_configure_args \
     --cache-file=\"\$ac_sub_cache_file\" --srcdir=\"\$ac_srcdir\"" ||
     AC_MSG_ERROR([$ax_sub_configure failed for $ax_subdir])
fi

cd "$ax_popdir"
# END: slightly modified snippet from status.m4])]dnl
)# _AX_CONFIG_SUBDIR_DEFINE_FN
