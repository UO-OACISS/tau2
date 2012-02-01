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


AC_DEFUN([AC_SCOREP_REVISION],
[
    # When working with a svn checkout, write a REVISION file. The REVISION
    # file is updated during each configure call and also at make doxygen-user
    # and make dist.

    # When working with a make-dist-generated tarball, REVISION is already
    # there.

    component_revision="invalid"
    common_revision="invalid"
    which svnversion > /dev/null; \
    if test $? -eq 0; then
        component_revision=`svnversion $srcdir`
        common_revision=`svnversion $srcdir/vendor/common`
        if test "x$component_revision" != "xexported"; then
            echo $component_revision > $srcdir/build-config/REVISION
        fi
        if test "x$common_revision" != "xexported"; then
            echo $common_revision > $srcdir/build-config/REVISION_COMMON
        fi
    fi

    if grep -E [[A-Z]] $srcdir/build-config/REVISION > /dev/null || \
       grep ":" $srcdir/build-config/REVISION > /dev/null ||
       grep -E [[A-Z]] $srcdir/build-config/REVISION_COMMON > /dev/null || \
       grep ":" $srcdir/build-config/REVISION_COMMON > /dev/null; then
        component_revision=`cat $srcdir/build-config/REVISION`
        common_revision=`cat $srcdir/build-config/REVISION_COMMON`
        AC_MSG_WARN([distribution does not match a single, unmodified revision, but $component_revision (${PACKAGE_NAME}) and $common_revision (common).])
    fi

    AC_SUBST([PACKAGE_MAJOR],
             m4_esyscmd([vendor/common/build-config/generate-package-version.sh build-config/VERSION "echo \$major"]))
    AC_SUBST([PACKAGE_MINOR],
             m4_esyscmd([vendor/common/build-config/generate-package-version.sh build-config/VERSION "echo \$minor"]))
    AC_SUBST([PACKAGE_BUGFIX],
             m4_esyscmd([vendor/common/build-config/generate-package-version.sh build-config/VERSION "echo \$bugfix"]))

    AC_SUBST([LIBRARY_CURRENT],
             m4_esyscmd([vendor/common/build-config/generate-library-version.sh build-config/VERSION "echo \$current"]))
    AC_SUBST([LIBRARY_REVISION],
             m4_esyscmd([vendor/common/build-config/generate-library-version.sh build-config/VERSION "echo \$revision"]))
    AC_SUBST([LIBRARY_AGE],
             m4_esyscmd([vendor/common/build-config/generate-library-version.sh build-config/VERSION "echo \$age"]))
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
    AC_DEFINE_UNQUOTED([SCOREP_COMPONENT_REVISION], ["${component_revision}"], [Revision of ${PACKAGE_NAME}])
    AC_DEFINE_UNQUOTED([SCOREP_COMMON_REVISION],    ["${common_revision}"], [Revision of common repository])
])
