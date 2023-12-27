## -*- mode: autoconf -*-

##
## This file is part of the Score-P software ecosystem (http://www.score-p.org)
##
## Copyright (c) 2021,
## Forschungszentrum Juelich GmbH, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##


# AFS_PROG_CC([VERSION], [ext|noext], [mandatory|optional])
# ---------------------------------------------------------
# (Arguments follow the `AX_C_COMPILE_STDC` macro, i.e., the documentation
# should be kept in sync!)
#
# Determines a C compiler to use, its vendor and version, and configures
# common features.
#
# The first argument, if specified, enforces a check for baseline language
# coverage in the compiler for the specified version of the C standard,
# which adds switches to CC and CPP to enable support.  VERSION may be '99'
# (for the C99 standard) or '11' (for the C11 standard).
#
# The second argument is only significant if a specific language VERSION
# is requested.  If specified, indicates whether you insist on an extended
# mode (e.g., '-std=gnu99') or a strict conformance mode (e.g., '-std=c99').
# If neither option is specified, you get whatever works, with preference
# for no added switch, then for an extended mode.
#
# The third argument is also only significant if a specific VERSION is
# requested.  If set to 'mandatory' or left unspecified, indicates that
# baseline support for the given C standard is required and that the macro
# should error out if no mode with that support is found.  If set to
# 'optional', then configuration proceeds regardless, after defining
# 'HAVE_C${VERSION}' if and only if a supporting mode is found.
#
# NOTE:
#   The third argument only affects the C standard conformance test.
#   A working C compiler is always required!
#
# List of configure variables set:
#   `afs_cv_prog_cc_works`::  'yes' if C compiler works, error out otherwise
#   `HAVE_C<VERSION>`::       'yes' if requested language version is supported,
#                             'no' otherwise (only if VERSION is specified)
#
AC_DEFUN_ONCE([AFS_PROG_CC], [
    AC_REQUIRE([_AFS_PROG_CC_COMPILER])
    AC_BEFORE([$0], [AFS_PROG_CC_SUMMARY])

    dnl Store C language standard, used by AFS_PROG_CC_SUMMARY
    _afs_prog_c_std="$1"

    dnl All compilers we care about support '-c -o', i.e., use of the 'compile'
    dnl wrapper script is not necessary.  However, 'autoreconf' is complaining
    dnl if the 'AM_PROG_CC_C_O' macro is not used.
    dnl
    dnl In case we encounter a compiler that does not support '-c -o' in the
    dnl future, similar modifications to what this macro does are required for
    dnl '$CXX', '$F77', and '$FC'.
    AM_PROG_CC_C_O

    dnl If we get here, we know that the C compiler is working
    AC_CACHE_VAL([afs_cv_prog_cc_works], [afs_cv_prog_cc_works=yes])
    AC_LANG_PUSH([C])

    dnl Feature checks
    m4_ifnblank([$1],
        [AX_C_COMPILE_STDC([$1], [$2], [$3])])

    dnl Vendor & version
    AX_COMPILER_VENDOR
    AX_COMPILER_VERSION

    dnl Compiler-specific quirks
    dnl
    dnl The "classic" Cray C compilers prior to CCE 9.0 enable OpenMP by
    dnl default.  Thus enforce non-OpenMP mode to allow for explicit control,
    dnl which doesn't do any harm for later versions.
    AS_IF([test x${ax_cv_c_compiler_vendor%/*} = xcray], [
        CC="$CC -hnoomp"
    ])

    AC_LANG_POP([C])
])


# AFS_PROG_CC_SUMMARY([PREFIX], [SUFFIX])
# ---------------------------------------
# Prints C compiler information to the configuration summary.  The optional
# PREFIX (e.g., "MPI") can be used to disambiguate compilers in different
# build directories in the summary output, the optional SUFFIX is appended
# to the summary message.
#
# NOTE: Requires a prior invocation of AFS_PROG_CC!
#
AC_DEFUN([AFS_PROG_CC_SUMMARY], [
    AC_LANG_PUSH([C])
    _AFS_COMPILER_SUMMARY([$1], [$2])
    AC_LANG_POP([C])
])


# AFS_PROG_CXX([VERSION], [ext|noext], [mandatory|optional])
# ----------------------------------------------------------
# (Arguments follow the `AX_CXX_COMPILE_STDCXX` macro, i.e., the documentation
# should be kept in sync!)
#
# Determines a C++ compiler to use, its vendor and version, and configures
# common features.
#
# The first argument, if specified, enables a check for baseline language
# coverage in the compiler for the specified version of the C++ standard,
# which adds switches to CXX and CXXCPP to enable support.  VERSION may be
# '11' (for the C++11 standard), '14' (for the C++14 standard), or '17'
# (for the C++17 standard).
#
# The second argument is only significant if a specific language VERSION
# is requested.  If specified, indicates whether you insist on an extended
# mode (e.g., '-std=gnu++11') or a strict conformance mode (e.g.,
# '-std=c++11').  If neither option is specified, you get whatever works,
# with preference for no added switch, then for an extended mode.
#
# The third argument, if set to 'mandatory' or left unspecified, indicates
# that a working C++ compiler with baseline support for the given language
# VERSION is required, and that the macro should error out if either no
# working compiler or no compiler with that support is found.  If set to
# 'optional', then configuration proceeds regardless, after defining
# 'HAVE_CXX${VERSION}' if and only if a working compiler with supporting
# mode is found.
#
# List of configure variables set:
#   `afs_cv_prog_cxx_works`::  'yes' if C++ compiler works, 'no' otherwise
#   `HAVE_CXX<VERSION>`::      'yes' if requested language version is supported,
#                              'no' otherwise (only if VERSION is specified)
#
AC_DEFUN_ONCE([AFS_PROG_CXX], [
    AC_REQUIRE([_AFS_PROG_CXX_COMPILER])
    AC_BEFORE([$0] [AFS_PROG_CXX_SUMMARY])

    m4_case([$3],
        [], [afs_prog_cxx_required=true],
        [mandatory], [afs_prog_cxx_required=true],
        [optional], [afs_prog_cxx_required=false],
        [m4_fatal([invalid third argument `$3' to AFS_PROG_CXX])])

    dnl Store C++ language standard, used by AFS_PROG_CXX_SUMMARY
    _afs_prog_cxx_std="$1"

    AS_IF([test "x$CXX" = x], [
        dnl Error out if C++ compiler is mandatory but could not be found
        AS_IF([test x$afs_prog_cxx_required = xtrue], [
            AC_MSG_FAILURE([no acceptable C++ compiler found in \$PATH])
        ])
        CXX=no
        AC_CACHE_VAL([afs_cv_prog_cxx_works], [afs_cv_prog_cxx_works=no])
    ], [
        AC_LANG_PUSH([C++])

        dnl Check for working compiler
        AC_CACHE_CHECK([whether the C++ compiler works],
            [afs_cv_prog_cxx_works],
            [AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])],
                [afs_cv_prog_cxx_works=yes],
                [afs_cv_prog_cxx_works=no])])
        AS_IF([test x$afs_cv_prog_cxx_works = xno], [
            dnl Error out if C++ compiler is mandatory but does not work
            AS_IF([test x$afs_prog_cxx_required = xtrue], [
                AC_MSG_FAILURE([C++ compiler cannot create executables])
            ])
            CXX=no
        ], [
            dnl Feature checks
            m4_ifnblank([$1],
                [AX_CXX_COMPILE_STDCXX([$1], [$2], [$3])])
            AX_CXX_INTTYPE_MACROS

            dnl Vendor & version
            AX_COMPILER_VENDOR
            AX_COMPILER_VERSION

            dnl Compiler-specific quirks
            dnl
            dnl The "classic" Cray C++ compilers prior to CCE 9.0 enable OpenMP
            dnl by default.  Thus enforce non-OpenMP mode to allow for explicit
            dnl control, which doesn't do any harm for later versions.
            AS_IF([test x${ax_cv_cxx_compiler_vendor%/*} = xcray], [
                CXX="$CXX -hnoomp"
            ])
        ])

        AC_LANG_POP([C++])
    ])
])


# AFS_PROG_CXX_SUMMARY([PREFIX], [SUFFIX])
# ----------------------------------------
# Prints C++ compiler information to the configuration summary.  The optional
# PREFIX (e.g., "MPI") can be used to disambiguate compilers in different
# build directories in the summary output, the optional SUFFIX is appended
# to the summary message.
#
# NOTE: Requires a prior invocation of AFS_PROG_CXX!
#
AC_DEFUN([AFS_PROG_CXX_SUMMARY], [
    AC_LANG_PUSH([C++])
    _AFS_COMPILER_SUMMARY([$1], [$2])
    AC_LANG_POP([C++])
])


# AFS_PROG_F77([mandatory|optional])
# ----------------------------------
# Determines a Fortran 77 compiler to use.
#
# The first argument, if set to 'mandatory' or left unspecified, indicates
# that a working Fortran 77 compiler is required, and that the macro should
# error out if no compiler is found.  If set to 'optional', then configuration
# proceeds regardless.
#
# List of configure variables set:
#   `afs_cv_prog_f77_works`::  'yes' if Fortran 77 compiler works, 'no'
#                              otherwise
#
AC_DEFUN_ONCE([AFS_PROG_F77], [
    AC_REQUIRE([_AFS_PROG_F77_COMPILER])
    AC_BEFORE([$0] [AFS_PROG_F77_SUMMARY])

    m4_case([$1],
        [], [afs_prog_f77_required=true],
        [mandatory], [afs_prog_f77_required=true],
        [optional], [afs_prog_f77_required=false],
        [m4_fatal([invalid first argument `$1' to AFS_PROG_F77])])

    dnl Unset language standard, used by AFS_PROG_F77_SUMMARY
    AS_UNSET([_afs_prog_f77_std])

    AS_IF([test "x$F77" = x], [
        dnl Error out if Fortran 77 compiler is mandatory but could not be found
        AS_IF([test x$afs_prog_f77_required = xtrue], [
            AC_MSG_FAILURE([no acceptable Fortran 77 compiler found in \$PATH])
        ])
        F77=no
        AC_CACHE_VAL([afs_cv_prog_f77_works], [afs_cv_prog_f77_works=no])
    ], [
        AC_LANG_PUSH([Fortran 77])

        dnl Check for working compiler
        AC_CACHE_CHECK([whether the Fortran 77 compiler works],
            [afs_cv_prog_f77_works],
            [AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])],
                [afs_cv_prog_f77_works=yes],
                [afs_cv_prog_f77_works=no])])
        AS_IF([test x$afs_cv_prog_f77_works = xno], [
            dnl Error out if Fortran 77 compiler is mandatory but does not work
            AS_IF([test x$afs_prog_f77_required = xtrue], [
                AC_MSG_FAILURE([Fortran 77 compiler cannot create executables])
            ])
            F77=no
        ], [
            dnl Vendor & version (unset variables for AFS_PROG_F77_SUMMARY)
            AS_UNSET([ax_cv_f77_compiler_vendor])
            AS_UNSET([ax_cv_f77_compiler_version])

            dnl Compiler-specific quirks
            dnl
            dnl The Cray Fortran compilers prior to CCE 9.0 enable OpenMP by
            dnl default.  Thus enforce non-OpenMP mode to allow for explicit
            dnl control, which doesn't do any harm for later versions.
            dnl
            dnl NOTE: This decision is based on the FC compiler vendor, as
            dnl       the 'AX_COMPILER_VENDOR' macro does not work for $F77!
            AS_IF([test "x$ax_cv_fc_compiler_vendor" = x || \
                   test "x$ax_cv_fc_compiler_vendor" = xunknown], [
                AC_MSG_WARN([F77 setting cannot be based on FC vendor. F77 setting might be incorrect.])
            ], [
                AS_IF([test x${ax_cv_fc_compiler_vendor%/*} = xcray], [
                    F77="$F77 -hnoomp"
                ])
            ])
        ])

        AC_LANG_POP([Fortran 77])
    ])
])


# AFS_PROG_F77_SUMMARY([PREFIX], [SUFFIX])
# ----------------------------------------
# Prints F77 compiler information to the configuration summary.  The optional
# PREFIX (e.g., "MPI") can be used to disambiguate compilers in different
# build directories in the summary output, the optional SUFFIX is appended
# to the summary message.
#
# NOTE: Requires a prior invocation of AFS_PROG_F77!
#
AC_DEFUN([AFS_PROG_F77_SUMMARY], [
    AC_LANG_PUSH([Fortran 77])
    _AFS_COMPILER_SUMMARY([$1], [$2])
    AC_LANG_POP([Fortran 77])
])


# AFS_PROG_FC([mandatory|optional])
# ---------------------------------
# Determines a Fortran compiler to use, as well as its vendor.
#
# The first argument, if set to 'mandatory' or left unspecified, indicates
# that a working Fortran compiler is required, and that the macro should
# error out if no compiler is found.  If set to 'optional', then configuration
# proceeds regardless.
#
# List of configure variables set:
#   `afs_cv_prog_fc_works`::  'yes' if Fortran compiler works, 'no' otherwise
#
AC_DEFUN_ONCE([AFS_PROG_FC], [
    AC_REQUIRE([_AFS_PROG_FC_COMPILER])
    AC_BEFORE([$0], [AFS_PROG_FC_SUMMARY])
    AC_BEFORE([$0], [AFS_PROG_F77])dnl Requires 'ax_cv_fc_compiler_vendor'

    m4_case([$1],
        [], [afs_prog_fc_required=true],
        [mandatory], [afs_prog_fc_required=true],
        [optional], [afs_prog_fc_required=false],
        [m4_fatal([invalid first argument `$1' to AFS_PROG_FC])])

    dnl Unset language standard, used by AFS_PROG_F77_SUMMARY
    AS_UNSET([_afs_prog_fc_std])

    AS_IF([test "x$FC" = x], [
        dnl Error out if Fortran compiler is mandatory but could not be found
        AS_IF([test x$afs_prog_fc_required = xtrue], [
            AC_MSG_FAILURE([no acceptable Fortran compiler found in \$PATH])
        ])
        FC=no
        AC_CACHE_VAL([afs_cv_prog_fc_works], [afs_cv_prog_fc_works=no])
    ], [
        AC_LANG_PUSH([Fortran])

        dnl Check for working compiler
        AC_CACHE_CHECK([whether the Fortran compiler works],
            [afs_cv_prog_fc_works],
            [AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])],
                [afs_cv_prog_fc_works=yes],
                [afs_cv_prog_fc_works=no])])
        AS_IF([test x$afs_cv_prog_fc_works = xno], [
            dnl Error out if Fortran compiler is mandatory but does not work
            AS_IF([test x$afs_prog_fc_required = xtrue], [
                AC_MSG_FAILURE([Fortran compiler cannot create executables])
            ])
            FC=no
        ], [
            dnl Feature checks
            AC_FC_PP_SRCEXT([F])

            dnl Vendor & version (unset variables for AFS_PROG_FC_SUMMARY)
            AX_COMPILER_VENDOR
            AS_UNSET([ax_cv_fc_compiler_version])

            dnl Compiler-specific quirks
            dnl
            dnl The Cray Fortran compilers prior to CCE 9.0 enable OpenMP by
            dnl default.  Thus enforce non-OpenMP mode to allow for explicit
            dnl control, which doesn't do any harm for later versions.
            AS_IF([test x${ax_cv_fc_compiler_vendor%/*} = xcray], [
                FC="$FC -hnoomp"
            ])
        ])

        AC_LANG_POP([Fortran])
    ])
])


# AFS_PROG_FC_SUMMARY([PREFIX], [SUFFIX])
# ---------------------------------------
# Prints FC compiler information to the configuration summary.  The optional
# PREFIX (e.g., "MPI") can be used to disambiguate compilers in different
# build directories in the summary output, the optional SUFFIX is appended
# to the summary message.
#
# NOTE: Requires a prior invocation of AFS_PROG_FC!
#
AC_DEFUN([AFS_PROG_FC_SUMMARY], [
    AC_LANG_PUSH([Fortran])
    _AFS_COMPILER_SUMMARY([$1], [$2])
    AC_LANG_POP([Fortran])
])


# _AFS_PROG_CC_COMPILER
# ---------------------
# Wrapper around 'AC_PROG_CC' to avoid "expanded before it was required"
# warnings during bootstrap.
#
AC_DEFUN_ONCE([_AFS_PROG_CC_COMPILER], [
    AC_PROG_CC
])


# _AFS_PROG_CXX_COMPILER
# ----------------------
# Wrapper around 'AC_PROG_CXX' to avoid "expanded before it was required"
# warnings during bootstrap.
#
AC_DEFUN_ONCE([_AFS_PROG_CXX_COMPILER], [
    AC_PROG_CXX
])


# _AFS_PROG_F77_COMPILER
# ----------------------
# Wrapper around 'AC_PROG_F77' to avoid "expanded before it was required"
# warnings during bootstrap.
#
AC_DEFUN_ONCE([_AFS_PROG_F77_COMPILER], [
    AC_PROG_F77
])


# _AFS_PROG_FC_COMPILER
# ---------------------
# Wrapper around 'AC_PROG_FC' to avoid "expanded before it was required"
# warnings during bootstrap.
#
AC_DEFUN_ONCE([_AFS_PROG_FC_COMPILER], [
    AC_PROG_FC
])


# _AFS_COMPILER_SUMMARY([PREFIX], [SUFFIX])
# -----------------------------------------
# Prints compiler information to the configuration summary.  The optional
# PREFIX (e.g., "MPI") can be used to disambiguate compilers in different
# build directories in the summary output, the optional SUFFIX is appended
# to the summary message.  Expects
#   - the language specific compiler variable (e.g., `$CC` or `$FC`) to be
#     set to either the compiler executable name with options, optionally
#     prefixed by the `compile` script, or "no" if no suitable compiler is
#     available.
#   - the shell variable `_afs_prog_<lang>_std` to be set to the requested
#     language standard (C/C++ only).
#
AC_DEFUN([_AFS_COMPILER_SUMMARY], [
    dnl Set disambiguation prefix
    m4_ifblank([$1],
        [AS_UNSET([_afs_compiler_summary_prefix])],
        [_afs_compiler_summary_prefix="m4_normalize($1) "])

    AS_IF([test "x$]_AC_CC[" != "xno"], [
        dnl Strip off compile script prefix potentially added by AM_PROG_CC_C_O
        _afs_compiler_summary_real=${[]_AC_CC[]#$am_aux_dir/compile }

        dnl Split compiler into executable name and options
        _afs_compiler_summary_comp=${_afs_compiler_summary_real%% *}
        AS_IF([test "$_afs_compiler_summary_comp" = "$_afs_compiler_summary_real"], [
            AS_UNSET([_afs_compiler_summary_opts])
        ], [
            _afs_compiler_summary_opts=" ${_afs_compiler_summary_real#* }"
        ])

        dnl Determine full path to executable
        _afs_compiler_summary_path=`which $_afs_compiler_summary_comp`

        AS_UNSET([_afs_compiler_summary_comp])
        AS_UNSET([_afs_compiler_summary_real])
    ], [
        AS_UNSET([_afs_compiler_summary_opts])
        _afs_compiler_summary_path=$[]_AC_CC
    ])

    dnl Compose vendor/version info
    AS_UNSET([_afs_compiler_summary_info])
    AS_IF([test "x$ax_cv_]_AC_LANG_ABBREV[_compiler_vendor" != x], [
        _afs_compiler_summary_info=" ($ax_cv_[]_AC_LANG_ABBREV[]_compiler_vendor"
        AS_IF([test "x$ax_cv_]_AC_LANG_ABBREV[_compiler_version" != x], [
            _afs_compiler_summary_info="$_afs_compiler_summary_info $ax_cv_[]_AC_LANG_ABBREV[]_compiler_version"
        ])
        _afs_compiler_summary_info="$_afs_compiler_summary_info)"
    ])

    AFS_SUMMARY([${_afs_compiler_summary_prefix}]_AC_LANG[$_afs_prog_]_AC_LANG_ABBREV[_std compiler],
                [$_afs_compiler_summary_path$_afs_compiler_summary_opts$_afs_compiler_summary_info$2])

    AS_UNSET([_afs_compiler_summary_info])
    AS_UNSET([_afs_compiler_summary_path])
    AS_UNSET([_afs_compiler_summary_opts])
    AS_UNSET([_afs_compiler_summary_prefix])
])


# AC_SCOREP_PROG_CC_C99
# ---------------------
# Meanwhile identical to AC_PROG_CC_C99, thus mark as obsolete and warn during
# bootstrap.  This macro can be removed once all uses have been converted or
# configure scripts have moved from 'AC_PROG_CC' to 'AFS_PROG_CC'.
#
AU_DEFUN([AC_SCOREP_PROG_CC_C99], [AC_PROG_CC_C99])
