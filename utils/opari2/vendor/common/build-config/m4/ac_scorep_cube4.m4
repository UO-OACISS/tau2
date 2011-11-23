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


AC_DEFUN([AC_SCOREP_CUBE4], [

## Evalute parameters
AC_ARG_WITH(cube4-lib, [AS_HELP_STRING([--with-cube4-lib=path_to_library], [Specifies the path where the Cube 4 library is located])],[
    AC_SUBST(CUBE_LIB_PATH,"$withval")
    LDFLAGS="$LDFLAGS -L$withval"
],[])
AC_ARG_WITH(cube4-header, [AS_HELP_STRING([--with-cube4-header=path_to_header], [Specifies the path where the Cube 4 header files are located])],[CPPFLAGS="$CPPFLAGS -I$withval"],[])

## preliminary error message due to problems with cross-compiling
if test "x${ac_scorep_cross_compiling}" = "xyes"; then
        AC_MSG_NOTICE([Can't reliably determine backend libcubew in cross
compiling mode. Disable Cube 4 support])
        AM_CONDITIONAL(HAVE_CUBE4,[test no = yes])
else
    ## Check presence of cube writer library
    AC_LANG_PUSH([C])
    AC_MSG_CHECKING([for libcubew])    
    scorep_save_libs=$LIBS
    CUBE_LIBS="-lcubew4 -lsc.z -lz -lm"
    LIBS="$LIBS $CUBE_LIBS"
    AC_LINK_IFELSE([AC_LANG_PROGRAM([void* cubew_create(unsigned myrank, unsigned Nthreads, unsigned Nwriters, const char * cubename, int compression);],
                                    [[cubew_create(1,1,1,"test",0);]])],[has_cube4_lib=yes],[CUBE_LIBS=""])
    AC_MSG_RESULT([$CUBE_LIBS])

    ## Check presence of cube writer header
    AC_CHECK_HEADER([cubew.h], [has_cube4_header=yes], [], [])

    ## Set makefile conditional
    AM_CONDITIONAL(HAVE_CUBE4,[test x$has_cube4_lib$has_cube4_header = xyesyes])
    AC_SUBST(CUBE_LIBS, "$CUBE_LIBS")

    ## Clean up
    LIBS=$scorep_savelibs
    AC_LANG_POP([C])
fi
])
