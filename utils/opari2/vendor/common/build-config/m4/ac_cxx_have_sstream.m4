# ===========================================================================
#          http://autoconf-archive.cryp.to/ac_cxx_have_sstream.html
# ===========================================================================
#
# SYNOPSIS
#
#   AC_CXX_HAVE_SSTREAM
#
# DESCRIPTION
#
#   If the C++ library has a working stringstream, define HAVE_SSTREAM.
#
# LAST MODIFICATION
#
#   2008-04-12
#
# COPYLEFT
#
#   Copyright (c) 2008 Ben Stanley <Ben.Stanley@exemail.com.au>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved.

AC_DEFUN([AC_CXX_HAVE_SSTREAM],
[AC_CACHE_CHECK(whether the compiler has stringstream,
ac_cv_cxx_have_sstream,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <sstream>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[stringstream message; message << "Hello"; return 0;]])],[ac_cv_cxx_have_sstream=yes],[ac_cv_cxx_have_sstream=no])
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_have_sstream" = yes; then
  AC_DEFINE(HAVE_SSTREAM,,[define if the compiler has stringstream])
fi
])
