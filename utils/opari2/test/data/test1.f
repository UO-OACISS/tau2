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

      program test1
      integer a
!************************************************
!* The following pragmas should be instrumented *
!************************************************
c$OMP PARALLEL
*$oMp BaRrIeR
!$OmP Barrier
!$omp end parallel

!$OmP    parallel
!$OmP&   default(shared)
!$OmP    end
!$OmP+   parallel

!$OmP   parallel
!$OmP&   default(shared)
!$OmP end
!$OmP+
!$OmP+ parallel

!$omp parallel !comment will be deleted
!more comment, which will be deleted
!$omp&private(a)
!and some more comment...

!$omp end
!$omp&parallel


!**************************************
!* The following should be ignored    *
!**************************************
c $omp no
!!$omp mo
c     comment
! $omp parallel
      write(*,*) "!$omp parallel"
      write(*,*) """!$omp parallel"""
      end program test1
