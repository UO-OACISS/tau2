## -*- mode: autoconf -*-

##
## This file is part of the Score-P software ecosystem (http://www.score-p.org)
##
## Copyright (c) 2016,
## Forschungszentrum Juelich GmbH, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##


# AX_CXX_INTTYPE_MACROS
# ---------------------
# Checks whether the system's libc follows the C99 or C11 (or later) standard
# with respect to the fixed-width integer constant/format/limit macros provided
# by <stdint.h> and <inttypes.h> when compiling C++ code.
#
# The ISO/IEC 9899:1999 standard (aka C99) states that for C++, C library
# implementations should define the fixed-width integer
#  - constant macros only if '__STDC_CONSTANT_MACROS' is defined before
#    including <stdint.h> (7.18.4)
#  - format macros only if '__STDC_FORMAT_MACROS' is defined before including
#    <inttypes.h> (7.8.1)
#  - limit macros only if '__STDC_LIMIT_MACROS' is defined before including
#    <stdint.h> (7.18.2)
# However, this was never adopted by any C++ standard, and was subsequently
# removed in the ISO/IEC 9899:2011 standard (aka C11).  Thus, depending on the
# C standard version followed by libc, these macros may be required or not.
#
# List of provided config header defines:
#  `__STDC_CONSTANT_MACROS`::  Set to '1' if needed to make the constant macros
#                              available in C++, unset otherwise
#  `__STDC_FORMAT_MACROS`::    Set to '1' if needed to make the format macros
#                              available in C++, unset otherwise
#  `__STDC_LIMIT_MACROS`::     Set to '1' if needed to make the limit macros
#                              available in C++, unset otherwise
#
AC_DEFUN([AX_CXX_INTTYPE_MACROS], [
AC_LANG_PUSH([C++])
_AX_CXX_INTTYPE_MACRO_TEST([constant])
_AX_CXX_INTTYPE_MACRO_TEST([format])
_AX_CXX_INTTYPE_MACRO_TEST([limit])
AC_LANG_POP([C++])
])


# _AX_CXX_INTTYPE_MACRO_TEST(TYPE)
# --------------------------------
# Helper macro which tests whether '__STDC_<TYPE>_MACROS' is needed to make the
# corresponding fixed-width integer <TYPE> macros available in C++.  It first
# tries to compile a test program without '__STDC_<TYPE>_MACROS'.  If this
# fails, it tries to compile again with '__STDC_<TYPE>_MACROS' set in CPPFLAGS.
# If both attemps fail, it aborts with an error.
#
# Valid values for `TYPE`: constant, format, limit
#
AC_DEFUN([_AX_CXX_INTTYPE_MACRO_TEST], [
m4_pushdef([_MACRO_NAME], [__STDC_[]m4_toupper($1)[]_MACROS])
AC_MSG_CHECKING([whether $CXX needs _MACRO_NAME])
AC_COMPILE_IFELSE([_AX_CXX_INTTYPE_MACRO_testbody_$1],
    [_ax_result=no],
    [_ax_save_CPPFLAGS="$CPPFLAGS"
     CPPFLAGS="-D[]_MACRO_NAME"
     AC_COMPILE_IFELSE([_AX_CXX_INTTYPE_MACRO_testbody_$1],
        [_ax_result=yes
         AC_DEFINE([_MACRO_NAME], [1],
            [Define to 1 if needed to make fixed-width integer $1 macros
             available in C++])],
        [AC_MSG_ERROR([$CXX cannot compile fixed-width integer $1 macros])])
     CPPFLAGS="$_ax_save_CPPFLAGS"])
AC_MSG_RESULT([$_ax_result])
m4_popdef([_MACRO_NAME])
])


# _AX_CXX_INTTYPE_MACRO_testbody_constant
# ---------------------------------------
# Configure test program for fixed-width integer constant macros.
#
AC_DEFUN([_AX_CXX_INTTYPE_MACRO_testbody_constant], [
AC_LANG_PROGRAM([[
    #include <stdint.h>
]], [[
    int8_t   i8  = INT8_C( 42 );
    int16_t  i16 = INT16_C( 42 );
    int32_t  i32 = INT32_C( 42 );
    int64_t  i64 = INT64_C( 42 );
    uint8_t  u8  = UINT8_C( 42 );
    uint16_t u16 = UINT16_C( 42 );
    uint32_t u32 = UINT32_C( 42 );
    uint64_t u64 = UINT64_C( 42 );
]])
])


# _AX_CXX_INTTYPE_MACRO_testbody_format
# -------------------------------------
# Configure test program for fixed-width integer format macros.
#
AC_DEFUN([_AX_CXX_INTTYPE_MACRO_testbody_format], [
AC_LANG_PROGRAM([[
    #include <inttypes.h>
]], [[
    const char* i8  = PRId8;
    const char* i16 = PRId16;
    const char* i32 = PRId32;
    const char* i64 = PRId64;
    const char* u8  = PRIu8;
    const char* u16 = PRIu16;
    const char* u32 = PRIu32;
    const char* u64 = PRIu64;
]])
])


# _AX_CXX_INTTYPE_MACRO_testbody_limit
# ------------------------------------
# Configure test program for fixed-width integer limit macros.
#
AC_DEFUN([_AX_CXX_INTTYPE_MACRO_testbody_limit], [
AC_LANG_PROGRAM([[
    #include <stdint.h>
]], [[
    int8_t   i8  = INT8_MAX;
    int16_t  i16 = INT16_MAX;
    int32_t  i32 = INT32_MAX;
    int64_t  i64 = INT64_MAX;
    uint8_t  u8  = UINT8_MAX;
    uint16_t u16 = UINT16_MAX;
    uint32_t u32 = UINT32_MAX;
    uint64_t u64 = UINT64_MAX;
]])
])
