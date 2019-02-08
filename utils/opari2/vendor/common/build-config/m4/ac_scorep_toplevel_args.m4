## -*- mode: autoconf -*-

##
## This file is part of the Score-P software (http://www.score-p.org)
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
## Copyright (c) 2009-2011, 2016,
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


# Save user provided arguments for use by sub-configures.
AC_DEFUN([AC_SCOREP_TOPLEVEL_ARGS],
[
# Quote arguments with shell meta characters.
TOPLEVEL_CONFIGURE_ARGUMENTS=
set -- "$progname" "$[@]"
for ac_arg
do
    AS_CASE(["$ac_arg"],
        [*" "*|*"	"*|*[[\[\]\~\#\$\^\&\*\(\)\{\}\\\|\;\<\>\?\']]*],
        [ac_arg=`echo "$ac_arg" | sed "s/'/'\\\\\\\\''/g"`
         # if the argument is of the form -foo=baz, quote the baz part only
         ac_arg=`echo "'$ac_arg'" | sed "s/^'\([[-a-zA-Z0-9]]*=\)/\\1'/"`])
    # Add the quoted argument to the list.
    TOPLEVEL_CONFIGURE_ARGUMENTS="$TOPLEVEL_CONFIGURE_ARGUMENTS
$ac_arg"
done
# Remove the initial space we just introduced.
TOPLEVEL_CONFIGURE_ARGUMENTS=`echo "x$TOPLEVEL_CONFIGURE_ARGUMENTS" | sed -e 's/^x *//'`

echo "$TOPLEVEL_CONFIGURE_ARGUMENTS" > ./user_provided_configure_args
])
