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
## Copyright (c) 2009-2014, 2019,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2013, 2021,
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


AC_DEFUN([AC_SCOREP_REVISION],
[
AC_REQUIRE([_AC_SCOREP_GIT_CONTROLLED])
AC_REQUIRE([AC_SCOREP_PACKAGE_AND_LIBRARY_VERSION])
AC_REQUIRE([AC_PROG_SED])

# When in a working copy, write REVISION* files. The REVISION* files
# are updated during each configure call and also at make
# doxygen-user.

# When working with a make-dist-generated tarball, the REVISION* files
# are provided.

component_revision="invalid"
AS_IF([test "x${ac_scorep_git_controlled}" = xyes &&
       component_revision=$(
            unset $(git rev-parse --local-env-vars 2>/dev/null) &&
            cd ${afs_srcdir} &&
            git describe --long --always --dirty | ${SED} 's/.*-g//' 2>/dev/null)],
      [echo "$component_revision" >${afs_srcdir}/build-config/REVISION],
      [component_revision=external])

# Warn if the REVISION files contain -dirty prefix or is external.
AS_CASE([`cat ${afs_srcdir}/build-config/REVISION`],
        [*-dirty|external|invalid],
        [component_revision=`cat ${afs_srcdir}/build-config/REVISION`
         AC_MSG_WARN([distribution does not match a single, unmodified revision, but $component_revision.])])
])

AC_DEFUN([AC_SCOREP_PACKAGE_AND_LIBRARY_VERSION],
[
    AC_SUBST([PACKAGE_MAJOR],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[build-config/common/generate-package-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$major"]))
    AC_SUBST([PACKAGE_MINOR],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[build-config/common/generate-package-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$minor"]))
    AC_SUBST([PACKAGE_BUGFIX],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[build-config/common/generate-package-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$bugfix"]))
    AC_SUBST([PACKAGE_SUFFIX],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[build-config/common/generate-package-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$suffix"]))

    AC_SUBST([LIBRARY_CURRENT],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[build-config/common/generate-library-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$current"]))
    AC_SUBST([LIBRARY_REVISION],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[build-config/common/generate-library-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$revision"]))
    AC_SUBST([LIBRARY_AGE],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[build-config/common/generate-library-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$age"]))
])




AC_DEFUN([AC_SCOREP_DEFINE_REVISIONS],
[
    AS_IF([test ! -e ${afs_srcdir}/build-config/REVISION],
          [AC_MSG_ERROR([File ${afs_srcdir}/build-config/REVISION must exist.])])

    component_revision=`cat ${afs_srcdir}/build-config/REVISION`
    AC_DEFINE_UNQUOTED([SCOREP_COMPONENT_REVISION], ["${component_revision}"], [Revision of ]AC_PACKAGE_NAME)
])
