! * This file is part of the Score-P software (http://www.score-p.org)
! *
! * Copyright (c) 2009-2011,
! *    RWTH Aachen University, Germany
! *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
! *    Technische Universitaet Dresden, Germany
! *    University of Oregon, Eugene, USA
! *    Forschungszentrum Juelich GmbH, Germany
! *    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
! *    Technische Universitaet Muenchen, Germany
! *
! * See the COPYING file in the package base directory for details.
! *
! * Testfile for automated testing of OPARI2
! *
! *
! * @brief Test the parsers ability to find directives and filter strings and comments.


program test1_free
  IMPLICIT NONE

  SUBROUTINE foo(A,N)
!The include should be inserted after this line.
    REAL(q),POINTER :: A(:)
    INTEGER N

    IF (ASSOCIATED(A)) THEN
    ENDIF

  END SUBROUTINE SMART_ALLOCATE_REAL
end program test1_free
