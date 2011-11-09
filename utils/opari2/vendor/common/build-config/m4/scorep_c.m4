## -*- mode: autoconf -*-

## 
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2011, 
##    RWTH Aachen University, Germany
##    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##    Technische Universitaet Dresden, Germany
##    University of Oregon, Eugene, USA
##    Forschungszentrum Juelich GmbH, Germany
##    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##    Technische Universitaet Muenchen, Germany
##
## See the COPYING file in the package base directory for details.
##


## file       scorep_c.m4 
##            This file contains a modified versions of the following
##            autoconf's 2.65 macros:
##            
##            AC_PROG_CC_C99 (renamed to SCOREP_PROG_CC_C99):
##              In SCOREP we don't want to use the GNU option -std=gnu99
##              (but -std=c99 instead) as this prevents some warnings that
##              may cause portability issues.
##
##            AC_OPENMP (renamed to SCOREP_OPENMP):
##              Add support for NEC SX compiler.
##
##            Please find the autoconf licence below.
##
## maintainer Christian Roessel <c.roessel@fz-juelich.de>
##


# This file is part of Autoconf.			-*- Autoconf -*-
# Programming languages support.
# Copyright (C) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009
# Free Software Foundation, Inc.

# This file is part of Autoconf.  This program is free
# software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# Under Section 7 of GPL version 3, you are granted additional
# permissions described in the Autoconf Configure Script Exception,
# version 3.0, as published by the Free Software Foundation.
#
# You should have received a copy of the GNU General Public License
# and a copy of the Autoconf Configure Script Exception along with
# this program; see the files COPYINGv3 and COPYING.EXCEPTION
# respectively.  If not, see <http://www.gnu.org/licenses/>.

# Written by David MacKenzie, with help from
# Akim Demaille, Paul Eggert,
# Franc,ois Pinard, Karl Berry, Richard Pixley, Ian Lance Taylor,
# Roland McGrath, Noah Friedman, david d zuhn, and many others.


# SCOREP_PROG_CC_C99
# --------------
AC_DEFUN([SCOREP_PROG_CC_C99],
[ AC_REQUIRE([AC_PROG_CC])dnl
  _SCOREP_PROG_CC_C99
])


# _SCOREP_PROG_CC_C99 ([ACTION-IF-AVAILABLE], [ACTION-IF-UNAVAILABLE])
# ----------------------------------------------------------------
# If the C compiler is not in ISO C99 mode by default, try to add an
# option to output variable CC to make it so.  This macro tries
# various options that select ISO C99 on some system or another.  It
# considers the compiler to be in ISO C99 mode if it handles _Bool,
# // comments, flexible array members, inline, long long int, mixed
# code and declarations, named initialization of structs, restrict,
# va_copy, varargs macros, variable declarations in for loops and
# variable length arrays.
AC_DEFUN([_SCOREP_PROG_CC_C99],
[_AC_C_STD_TRY([c99],
[[#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <wchar.h>
#include <stdio.h>

// Check varargs macros.  These examples are taken from C99 6.10.3.5.
#define debug(...) fprintf (stderr, __VA_ARGS__)
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
  your preprocessor is broken;
#endif
#if BIG_OK
#else
  your preprocessor is broken;
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
static void
test_varargs (const char *format, ...)
{
  va_list args;
  va_start (args, format);
  va_list args_copy;
  va_copy (args_copy, args);

  const char *str;
  int number;
  float fnumber;

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
}
]],
[[
  // Check bool.
  _Bool success = false;

  // Check restrict.
  if (test_restrict ("String literal") == 0)
    success = true;
  char *restrict newvar = "Another string";

  // Check varargs.
  test_varargs ("s, d' f .", "string", 65, 34.234);
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
  dynamic_array[ni.number - 1] = 543;

  // work around unused variable warnings
  return (!success || bignum == 0LL || ubignum == 0uLL || newvar[0] == 'x'
	  || dynamic_array[ni.number - 1] != 543);
]],
dnl Try
dnl GCC		-std=c99 (unused restrictive modes: -std=iso9899:1999)
dnl AIX		-qlanglvl=extc99 (unused restrictive mode: -qlanglvl=stdc99)
dnl HP cc	-AC99
dnl Intel ICC	-std=c99, -c99 (deprecated)
dnl IRIX	-c99
dnl Solaris	-xc99=all (Forte Developer 7 C mishandles -xc99 on Solaris 9,
dnl		as it incorrectly assumes C99 semantics for library functions)
dnl Tru64	-c99
dnl NEC SX	-Kc99
dnl with extended modes being tried first.
[[-std=c99 -c99 -AC99 -xc99=all -qlanglvl=extc99 -Kc99]], [$1], [$2])[]dnl
])# _SCOREP_PROG_CC_C99
