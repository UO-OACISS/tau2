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
! * @brief Test the nowait and untied clauses.

      program test4
      integer i

      real a(5,5), b(5,5), c(5,5)

!$omp parallel
      write(*,*) "parallel"
!$omp do
      do i=1,4
         write(*,*) "do nowait",i
      enddo
!$omp enddo nowait
      
!$omp sections
!$omp section
      write(*,*) "section nowait 1"
!$omp section
      write(*,*) "section nowait 2"
!$omp end sections nowait
      
!$omp single
      write(*,*) "single nowait"
!$omp end single nowait
      
!$omp workshare
      a = b + c
!$omp end workshare nowait

!$omp task untied
      write(*,*) "task"
!$omp end task
            
!$omp end parallel
      end program test4
      
