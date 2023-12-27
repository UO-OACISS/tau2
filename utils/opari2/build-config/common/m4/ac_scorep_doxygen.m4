## -*- mode: autoconf -*-

##
## This file is part of the Score-P software ecosystem (http://www.score-p.org)
##
## Copyright (c) 2009-2011,
## RWTH Aachen University, Germany
##
## Copyright (c) 2009-2011,
## Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##
## Copyright (c) 2009-2011,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2011,
## Forschungszentrum Juelich GmbH, Germany
##
## Copyright (c) 2009-2011,
## German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##
## Copyright (c) 2009-2011,
## Technische Universitaet Muenchen, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##


# AC_SCOREP_DOXYGEN
#----------------
# Check doxygen related stuff, sets some output variables and the
# AM_CONDITIONALs HAVE_DOXYGEN and HAVE_DOXYGEN_LATEX.
#
AC_DEFUN([AC_SCOREP_DOXYGEN],
[
#if test -z "$have_svnversion"; then
#  AC_CHECK_PROG([have_svnversion], [svnversion], [yes], [no],,)
#fi
AC_PATH_PROG([DOXYGEN], [doxygen])
AM_CONDITIONAL(HAVE_DOXYGEN, test "x${DOXYGEN}" != "x")

have_doxygen_latex="no"
AM_COND_IF([HAVE_DOXYGEN],[
   AC_CHECK_PROG([have_dot], [dot], [yes], [no],,)
   AC_CHECK_PROG([have_pdflatex], [pdflatex], [yes], [no],,)
   AC_CHECK_PROG([have_makeindex], [makeindex], [yes], [no],,)
   if test "x${have_pdflatex}" = xyes && test "x${have_makeindex}" = xyes; then
      have_doxygen_latex="yes"
   fi
])
AC_SUBST([have_doxygen_latex])
AM_CONDITIONAL(HAVE_DOXYGEN_LATEX, test "x${have_doxygen_latex}" = xyes)

# ac_scorep_doxygen_distdir may be passed in from upper level configure
# will end up in 'USER_DOC_DIR = $(top_distdir)$(ac_scorep_doxygen_distdir)/doc'
if test ! -n "$ac_scorep_doxygen_distdir"; then
   ac_scorep_doxygen_distdir=""
fi
AC_SUBST([ac_scorep_doxygen_distdir])

])
