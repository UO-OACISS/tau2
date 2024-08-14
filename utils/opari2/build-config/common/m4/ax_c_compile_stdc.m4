# SYNOPSIS
#
#   AX_C_COMPILE_STDC(VERSION, [ext|noext], [mandatory|optional])
#
# DESCRIPTION
#
#   Check for baseline language coverage in the compiler for the specified
#   version of the C standard.  If necessary, add switches to CC and CPP to
#   enable support.  VERSION may be '99' (for the C99 standard) or '11'
#   (for the C11 standard).
#
#   The second argument, if specified, indicates whether you insist on an
#   extended mode (e.g. -std=gnu99) or a strict conformance mode (e.g.
#   -std=c99).  If neither is specified, you get whatever works, with
#   preference for no added switch, and then for an extended mode.
#
#   The third argument, if specified 'mandatory' or if left unspecified,
#   indicates that baseline support for the specified C standard is
#   required and that the macro should error out if no mode with that
#   support is found.  If specified 'optional', then configuration proceeds
#   regardless, after defining HAVE_C${VERSION} if and only if a supporting
#   mode is found.
#
# This macro is based on the AX_CXX_COMPILE_STDCXX macro (serial 12) from
# the GNU Autoconf Archive (https://www.gnu.org/software/autoconf-archive/).
#
# LICENSE
#
#   Copyright (c) 2008 Benjamin Kosnik <bkoz@redhat.com>
#   Copyright (c) 2012 Zack Weinberg <zackw@panix.com>
#   Copyright (c) 2013 Roy Stogner <roystgnr@ices.utexas.edu>
#   Copyright (c) 2014, 2015 Google Inc.; contributed by Alexey Sokolov <sokolov@google.com>
#   Copyright (c) 2015 Paul Norman <penorman@mac.com>
#   Copyright (c) 2015 Moritz Klammler <moritz@klammler.eu>
#   Copyright (c) 2016, 2018 Krzesimir Nowak <qdlacz@gmail.com>
#   Copyright (c) 2019 Enji Cooper <yaneurabeya@gmail.com>
#   Copyright (c) 2020 Jason Merrill <jason@redhat.com>
#   Copyright (c) 2021 Markus Geimer <m.geimer@fz-juelich.de>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved.  This file is offered as-is, without any
#   warranty.
#
#
# The language feature test codes are derived from GNU Autoconf
# lib/autoconf/c.m4
# commit 6d38e9fa2b39b3c3a8e4d6d7da38c59909d3f39d
# dated Thu Jan 28 17:54:10 2021 -0500
# which comes with the following license:
#
#   Copyright (C) 2001-2017, 2020-2021 Free Software Foundation, Inc.
#
#   This file is part of Autoconf.  This program is free
#   software; you can redistribute it and/or modify it under the
#   terms of the GNU General Public License as published by the
#   Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   Under Section 7 of GPL version 3, you are granted additional
#   permissions described in the Autoconf Configure Script Exception,
#   version 3.0, as published by the Free Software Foundation.
#
#   You should have received a copy of the GNU General Public License
#   and a copy of the Autoconf Configure Script Exception along with
#   this program; see the files COPYINGv3 and COPYING.EXCEPTION
#   respectively.  If not, see <https://www.gnu.org/licenses/>.
#
#   Written by David MacKenzie, with help from
#   Akim Demaille, Paul Eggert,
#   Franc,ois Pinard, Karl Berry, Richard Pixley, Ian Lance Taylor,
#   Roland McGrath, Noah Friedman, david d zuhn, and many others.


#serial 1


AC_DEFUN([AX_C_COMPILE_STDC], [dnl
  m4_if([$1], [99], [ax_c_compile_alternatives="99 9x"],
        [$1], [11], [ax_c_compile_alternatives="11 1x"],
        [m4_fatal([invalid first argument `$1' to AX_C_COMPILE_STDC])])dnl
  m4_if([$2], [], [],
        [$2], [ext], [],
        [$2], [noext], [],
        [m4_fatal([invalid second argument `$2' to AX_C_COMPILE_STDC])])dnl
  m4_if([$3], [], [ax_c_compile_c$1_required=true],
        [$3], [mandatory], [ax_c_compile_c$1_required=true],
        [$3], [optional], [ax_c_compile_c$1_required=false],
        [m4_fatal([invalid third argument `$3' to AX_C_COMPILE_STDC])])
  AC_LANG_PUSH([C])dnl
  ac_success=no

  m4_if([$2], [], [dnl
    AC_CACHE_CHECK(whether $CC supports C$1 features by default,
                   ax_cv_c_compile_c$1,
      [AC_COMPILE_IFELSE([AC_LANG_SOURCE([_AX_C_COMPILE_STDC_testbody_$1])],
        [ax_cv_c_compile_c$1=yes],
        [ax_cv_c_compile_c$1=no])])
    if test x$ax_cv_c_compile_c$1 = xyes; then
      ac_success=yes
    fi])

  m4_if([$2], [noext], [], [dnl
  if test x$ac_success = xno; then
    for alternative in ${ax_c_compile_alternatives}; do
      for switch in -std=gnu${alternative} -std=std${alternative} -c${alternative} -AC${alternative} -qlanglvl=extc${alternative} -Kc${alternative} -hstd=c${alternative} -D_STDC_C99=; do
          cachevar=AS_TR_SH([ax_cv_c_compile_c$1_$switch])
          AC_CACHE_CHECK(whether $CC supports C$1 features with $switch,
                         $cachevar,
            [ac_save_CC="$CC"
             CC="$CC $switch"
             AC_COMPILE_IFELSE([AC_LANG_SOURCE([_AX_C_COMPILE_STDC_testbody_$1])],
              [eval $cachevar=yes],
              [eval $cachevar=no])
             CC="$ac_save_CC"])
          if eval test x\$$cachevar = xyes; then
            CC="$CC $switch"
            if test "x$CPP" != x; then
              CPP="$CPP $switch"
            fi
            ac_success=yes
            break
          fi
      done
      if test x$ac_success = xyes; then
        break
      fi
    done
  fi])

  m4_if([$2], [ext], [], [dnl
  if test x$ac_success = xno; then
    for alternative in ${ax_c_compile_alternatives}; do
      for switch in -std=c${alternative} -c${alternative} -AC${alternative} -qlanglvl=stdc${alternative} -Kc${alternative} -hstd=c${alternative} -D_STDC_C99=; do
        cachevar=AS_TR_SH([ax_cv_c_compile_c$1_$switch])
        AC_CACHE_CHECK(whether $CC supports C$1 features with $switch,
                       $cachevar,
          [ac_save_CC="$CC"
           CC="$CC $switch"
           AC_COMPILE_IFELSE([AC_LANG_SOURCE([_AX_C_COMPILE_STDC_testbody_$1])],
            [eval $cachevar=yes],
            [eval $cachevar=no])
           CC="$ac_save_CC"])
        if eval test x\$$cachevar = xyes; then
          CC="$CC $switch"
          if test "x$CPP" != x; then
            CPP="$CPP $switch"
          fi
          ac_success=yes
          break
        fi
      done
      if test x$ac_success = xyes; then
        break
      fi
    done
  fi])
  AC_LANG_POP([C])
  if test x$ax_c_compile_c$1_required = xtrue; then
    if test x$ac_success = xno; then
      AC_MSG_ERROR([*** A compiler with support for C$1 language features is required.])
    fi
  fi
  if test x$ac_success = xno; then
    HAVE_C$1=0
    AC_MSG_NOTICE([No compiler with C$1 support was found])
  else
    HAVE_C$1=1
    AC_DEFINE(HAVE_C$1,1,
              [define if the compiler supports basic C$1 syntax])
  fi
  AC_SUBST(HAVE_C$1)
])


dnl  Test code for checking C89 support

m4_define([_AX_C_COMPILE_STDC_testbody_89],
_AX_C_COMPILE_STDC_globals_new_in_89
_AX_C_COMPILE_STDC_testbody_prolog
_AX_C_COMPILE_STDC_main_new_in_89
_AX_C_COMPILE_STDC_testbody_epilog)

dnl  Test code for checking C99 support

m4_define([_AX_C_COMPILE_STDC_testbody_99],
_AX_C_COMPILE_STDC_globals_new_in_89
_AX_C_COMPILE_STDC_globals_new_in_99
_AX_C_COMPILE_STDC_testbody_prolog
_AX_C_COMPILE_STDC_main_new_in_89
_AX_C_COMPILE_STDC_main_new_in_99
_AX_C_COMPILE_STDC_testbody_epilog)

dnl  Test code for checking C11 support

m4_define([_AX_C_COMPILE_STDC_testbody_11],
_AX_C_COMPILE_STDC_globals_new_in_89
_AX_C_COMPILE_STDC_globals_new_in_99
_AX_C_COMPILE_STDC_globals_new_in_11
_AX_C_COMPILE_STDC_testbody_prolog
_AX_C_COMPILE_STDC_main_new_in_89
_AX_C_COMPILE_STDC_main_new_in_99
_AX_C_COMPILE_STDC_main_new_in_11
_AX_C_COMPILE_STDC_testbody_epilog)


dnl  Test code main body prolog/epilog

m4_define([_AX_C_COMPILE_STDC_testbody_prolog],
[[

int
main(int argc, char** argv)
{
  int ok = 0;
]])

m4_define([_AX_C_COMPILE_STDC_testbody_epilog],
[[
  return ok;
}
]])


dnl  Tests for new features in C89

m4_define([_AX_C_COMPILE_STDC_globals_new_in_89], [[
/* Does the compiler advertise C89 conformance?
   Do not test the value of __STDC__, because some compilers set it to 0
   while being otherwise adequately conformant. */
#if !defined __STDC__
# error "Compiler does not advertise C89 conformance"
#endif

#include <stddef.h>
#include <stdarg.h>
struct stat;
/* Most of the following tests are stolen from RCS 5.7 src/conf.sh.  */
struct buf { int x; };
struct buf * (*rcsopen) (struct buf *, struct stat *, int);
static char *e (p, i)
     char **p;
     int i;
{
  return p[i];
}
static char *f (char * (*g) (char **, int), char **p, ...)
{
  char *s;
  va_list v;
  va_start (v,p);
  s = g (p, va_arg (v,int));
  va_end (v);
  return s;
}

/* OSF 4.0 Compaq cc is some sort of almost-ANSI by default.  It has
   function prototypes and stuff, but not \xHH hex character constants.
   These do not provoke an error unfortunately, instead are silently treated
   as an "x".  The following induces an error, until -std is added to get
   proper ANSI mode.  Curiously \x00 != x always comes out true, for an
   array size at least.  It is necessary to write \x00 == 0 to get something
   that is true only with -std.  */
int osf4_cc_array ['\x00' == 0 ? 1 : -1];

/* IBM C 6 for AIX is almost-ANSI by default, but it replaces macro parameters
   inside strings and character constants.  */
#define FOO(x) 'x'
int xlc6_cc_array[FOO(a) == 'x' ? 1 : -1];

int test (int i, double x);
struct s1 {int (*f) (int a);};
struct s2 {int (*f) (double a);};
int pairnames (int, char **, int *(*)(struct buf *, struct stat *, int),
               int, int);
]])

m4_define([_AX_C_COMPILE_STDC_main_new_in_89], [[
ok |= (argc == 0 || f (e, argv, 0) != argv[0] || f (e, argv, 1) != argv[1]);
]])


dnl  Tests for new features in C99

m4_define([_AX_C_COMPILE_STDC_globals_new_in_99], [[
// Does the compiler advertise C99 conformance?
#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 199901L
# error "Compiler does not advertise C99 conformance"
#endif

#include <stdbool.h>
extern int puts (const char *);
extern int printf (const char *, ...);
extern int dprintf (int, const char *, ...);
extern void *malloc (size_t);

// Check varargs macros.  These examples are taken from C99 6.10.3.5.
// dprintf is used instead of fprintf to avoid needing to declare
// FILE and stderr.
#define debug(...) dprintf (2, __VA_ARGS__)
#define showlist(...) puts (#__VA_ARGS__)
#define report(test,...) ((test) ? puts (#test) : printf (__VA_ARGS__))
static void
test_varargs_macros (void)
{
  int x = 1234;
  int y = 5678;
  debug ("Flag");
  debug ("X = %d\n", x);
  showlist (The first, second, and third items.);
  report (x>y, "x is %d but y is %d", x, y);
}

// Check long long types.
#define BIG64 18446744073709551615ull
#define BIG32 4294967295ul
#define BIG_OK (BIG64 / BIG32 == 4294967297ull && BIG64 % BIG32 == 0)
#if !BIG_OK
  #error "your preprocessor is broken"
#endif
#if BIG_OK
#else
  #error "your preprocessor is broken"
#endif
static long long int bignum = -9223372036854775807LL;
static unsigned long long int ubignum = BIG64;

struct incomplete_array
{
  int datasize;
  double data[];
};

struct named_init {
  int number;
  const wchar_t *name;
  double average;
};

typedef const char *ccp;

static inline int
test_restrict (ccp restrict text)
{
  // See if C++-style comments work.
  // Iterate through items via the restricted pointer.
  // Also check for declarations in for loops.
  for (unsigned int i = 0; *(text+i) != '\0'; ++i)
    continue;
  return 0;
}

// Check varargs and va_copy.
static bool
test_varargs (const char *format, ...)
{
  va_list args;
  va_start (args, format);
  va_list args_copy;
  va_copy (args_copy, args);

  const char *str = "";
  int number = 0;
  float fnumber = 0;

  while (*format)
    {
      switch (*format++)
        {
        case 's': // string
          str = va_arg (args_copy, const char *);
          break;
        case 'd': // int
          number = va_arg (args_copy, int);
          break;
        case 'f': // float
          fnumber = va_arg (args_copy, double);
          break;
        default:
          break;
        }
    }
  va_end (args_copy);
  va_end (args);

  return *str && number && fnumber;
}
]])

m4_define([_AX_C_COMPILE_STDC_main_new_in_99], [[
  // Check bool.
  _Bool success = false;
  success |= (argc != 0);

  // Check restrict.
  if (test_restrict ("String literal") == 0)
    success = true;
  char *restrict newvar = "Another string";

  // Check varargs.
  success &= test_varargs ("s, d' f .", "string", 65, 34.234);
  test_varargs_macros ();

  // Check flexible array members.
  struct incomplete_array *ia =
    malloc (sizeof (struct incomplete_array) + (sizeof (double) * 10));
  ia->datasize = 10;
  for (int i = 0; i < ia->datasize; ++i)
    ia->data[i] = i * 1.234;

  // Check named initializers.
  struct named_init ni = {
    .number = 34,
    .name = L"Test wide string",
    .average = 543.34343,
  };

  ni.number = 58;

  int dynamic_array[ni.number];
  dynamic_array[0] = argv[0][0];
  dynamic_array[ni.number - 1] = 543;

  // work around unused variable warnings
  ok |= (!success || bignum == 0LL || ubignum == 0uLL || newvar[0] == 'x'
         || dynamic_array[ni.number - 1] != 543);
]])


dnl  Tests for new features in C11

m4_define([_AX_C_COMPILE_STDC_globals_new_in_11], [[
// Does the compiler advertise C11 conformance?
#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 201112L
# error "Compiler does not advertise C11 conformance"
#endif

// Check _Alignas.
char _Alignas (double) aligned_as_double;
char _Alignas (0) no_special_alignment;
extern char aligned_as_int;
char _Alignas (0) _Alignas (int) aligned_as_int;

// Check _Alignof.
enum
{
  int_alignment = _Alignof (int),
  int_array_alignment = _Alignof (int[100]),
  char_alignment = _Alignof (char)
};
_Static_assert (0 < -_Alignof (int), "_Alignof is signed");

// Check _Noreturn.
int _Noreturn does_not_return (void) { for (;;) continue; }

// Check _Static_assert.
struct test_static_assert
{
  int x;
  _Static_assert (sizeof (int) <= sizeof (long int),
                  "_Static_assert does not work in struct");
  long int y;
};

// Check UTF-8 literals.
#define u8 syntax error!
char const utf8_literal[] = u8"happens to be ASCII" "another string";

// Check duplicate typedefs.
typedef long *long_ptr;
typedef long int *long_ptr;
typedef long_ptr long_ptr;

// Anonymous structures and unions -- taken from C11 6.7.2.1 Example 1.
struct anonymous
{
  union {
    struct { int i; int j; };
    struct { int k; long int l; } w;
  };
  int m;
} v1;
]])

m4_define([_AX_C_COMPILE_STDC_main_new_in_11], [[
  _Static_assert ((offsetof (struct anonymous, i)
                   == offsetof (struct anonymous, w.k)),
                  "Anonymous union alignment botch");
  v1.i = 2;
  v1.w.k = 5;
  ok |= v1.i != 5;
]])
