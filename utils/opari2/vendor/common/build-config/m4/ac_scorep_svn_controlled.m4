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


AC_DEFUN([AC_SCOREP_SVN_CONTROLLED],
[
    ac_scorep_svn_controlled="no"
    if test -d $srcdir/.svn; then
        ac_scorep_svn_controlled="yes"
        AC_DEFINE([SCOREP_IN_DEVELOPEMENT], [], [Defined if we are working from svn.])
    else
        AC_DEFINE([SCOREP_IN_PRODUCTION], [], [Defined if we are working from a make dist generated tarball.])
    fi
    AM_CONDITIONAL(SVN_CONTROLLED, test "x${ac_scorep_svn_controlled}" = xyes)  
])
