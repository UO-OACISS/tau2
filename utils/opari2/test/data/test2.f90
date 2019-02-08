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
! * @brief Test the basic instrumentation of all directives.

program test2
  integer i
  integer k
  
  integer, save :: j
  !$omp threadprivate(j)

  !$omp parallel
  write(*,*) "parallel"
  !$omp do
  do i=1,4
     write(*,*) "do",i
  enddo
  !$omp end do
  
  !$omp flush(k)
  
  !$omp do ordered
  do i=1,4
     !$omp ordered
     write(*,*) "do",i
     !$omp end ordered
  enddo
  !$omp end do
  
  !$omp barrier
  
  !$omp sections
  !$omp section
  write(*,*) "section 1"
  !$omp section
  write(*,*) "section 2"
  !$omp end sections
  
  !$omp master
  write(*,*) "master"
  !$omp end master
  
  !$omp critical
  write(*,*) "critical"
  !$omp end critical
  
  !$omp critical(foobar)
  write(*,*) "critical(foobar)"
  !$omp end critical(foobar)
  
  !$omp atomic
  ! do this atomic
  i = i + 1
  
  !$omp single
  write(*,*) "single"
  !$omp end single
  
  !$omp workshare
  a = b + c
  !$omp end workshare
  
  !$omp end parallel

  !$omp parallel
  !$omp task
  write(*,*) "task"
  !$omp end task

  !$omp taskwait
  !$omp end parallel

  ! $ omp this should be ignored by opari and the compiler
  ! $ this too
   end program test2
