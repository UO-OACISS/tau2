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


## file       ac_scorep_mpi.m4
## maintainer Christian Roessel <c.roessel@fz-juelich.de>

AC_DEFUN([AC_SCOREP_POSIX_FUNCTIONS], [

    AC_LANG_PUSH(C)

    AC_CHECK_DECLS([gethostname, fseeko], [], [], [[
      #include <unistd.h>
      #include <stdio.h>
    ]])

    AC_MSG_CHECKING([for gethostname])
    AC_LINK_IFELSE([
        AC_LANG_SOURCE([
            #include <stddef.h>
            int gethostname(char *name, size_t namelen);
 
            int main()
            {
                char* name;
                return gethostname(name, 256);
            }
            ])],
        [AC_MSG_RESULT(yes);
         AC_DEFINE(HAVE_GETHOSTNAME, 1, [Can link a gethostname function])], 
        [AC_MSG_RESULT(no)]
    ) # AC_COMPILE_IF_ELSE

    AC_MSG_CHECKING([for fseeko])
    AC_LINK_IFELSE([
        AC_LANG_SOURCE([
            #include <stdio.h>
	    #include <fcntl.h>
            int fseeko(FILE *stream, off_t offset, int whence);
 
            int main()
            {
                FILE* stream;
                return fseeko(stream, 256, 0);
            }
            ])],
        [AC_MSG_RESULT(yes);
         AC_DEFINE(HAVE_FSEEKO, 1, [Can link a fseeko function])], 
        [AC_MSG_RESULT(no)]
    ) # AC_COMPILE_IF_ELSE

    AC_LANG_POP(C)

])
