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


AC_DEFUN([AC_CUTEST_COLOR_TESTS],
[

AC_CHECK_HEADER(
    [unistd.h],
    [AC_CHECK_DECL(
        [STDOUT_FILENO],
        [AC_CHECK_FUNC(
            [isatty],
            [AC_DEFINE(
                [CUTEST_USE_COLOR],
                [1],
                [Try to use colorful output for tests.])])
        ],
        [],
        [#include <unistd.h>])])

])
