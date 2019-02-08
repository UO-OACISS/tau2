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
## Copyright (c) 2009-2014,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2013,
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
AC_REQUIRE([AC_SCOREP_PACKAGE_AND_LIBRARY_VERSION])

# When in a working copy, write REVISION* files. The REVISION* files
# are updated during each configure call and also at make
# doxygen-user.

# When working with a make-dist-generated tarball, the REVISION* files
# are provided.

component_revision="invalid"
common_revision="invalid"
which svnversion 2>&1 >/dev/null
AS_IF([test $? -eq 0],
      [component_revision=`svnversion -c $srcdir | cut -d ':' -f 2`
       common_revision=`svnversion -c $srcdir/vendor/common | cut -d ':' -f 2`
       # If we are in a working copy, update the REVISION* files.
       AS_IF([test "x$component_revision" != "xexported" && \
              test "x$component_revision" != "xUnversioned directory"],
             [echo $component_revision > $srcdir/build-config/REVISION])
       AS_IF([test "x$common_revision" != "xexported" && \
              test "x$component_revision" != "xUnversioned directory"],
             [echo $common_revision > $srcdir/build-config/REVISION_COMMON])])

# Warn if the REVISION* files contain anything but plain numbers.
AS_IF([grep -E [[A-Z]] $srcdir/build-config/REVISION > /dev/null || \
       grep ":" $srcdir/build-config/REVISION > /dev/null || \
       grep -E [[A-Z]] $srcdir/build-config/REVISION_COMMON > /dev/null || \
       grep ":" $srcdir/build-config/REVISION_COMMON > /dev/null],
      [component_revision=`cat $srcdir/build-config/REVISION`
       common_revision=`cat $srcdir/build-config/REVISION_COMMON`
       AC_MSG_WARN([distribution does not match a single, unmodified revision, but $component_revision (${PACKAGE_NAME}) and $common_revision (common).])])
])


AC_DEFUN([AC_SCOREP_PACKAGE_AND_LIBRARY_VERSION],
[
    AC_SUBST([PACKAGE_MAJOR],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[vendor/common/build-config/generate-package-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$major"]))
    AC_SUBST([PACKAGE_MINOR],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[vendor/common/build-config/generate-package-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$minor"]))
    AC_SUBST([PACKAGE_BUGFIX],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[vendor/common/build-config/generate-package-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$bugfix"]))
    AC_SUBST([PACKAGE_SUFFIX],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[vendor/common/build-config/generate-package-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$suffix"]))

    AC_SUBST([LIBRARY_CURRENT],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[vendor/common/build-config/generate-library-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$current"]))
    AC_SUBST([LIBRARY_REVISION],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[vendor/common/build-config/generate-library-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$revision"]))
    AC_SUBST([LIBRARY_AGE],
             m4_esyscmd(AFS_PACKAGE_TO_TOP[vendor/common/build-config/generate-library-version.sh ] AFS_PACKAGE_TO_TOP[build-config/VERSION "echo \$age"]))
])




AC_DEFUN([AC_SCOREP_DEFINE_REVISIONS],
[
    for i in REVISION REVISION_COMMON; do
        if test ! -e ${srcdir}/../build-config/${i}; then
            AC_MSG_ERROR([File ${srcdir}/../build-config/${i} must exist.])
        fi
    done

    component_revision=`cat ${srcdir}/../build-config/REVISION`
    common_revision=`cat ${srcdir}/../build-config/REVISION_COMMON`
    AC_DEFINE_UNQUOTED([SCOREP_COMPONENT_REVISION], ["${component_revision}"], [Revision of ]AC_PACKAGE_NAME)
    AC_DEFINE_UNQUOTED([SCOREP_COMMON_REVISION],    ["${common_revision}"], [Revision of common repository])
])
