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
! * @brief Tests whether specific clauses are found and inserted into the CTC string.

program test8
  integer i
  integer k
  integer num_threads

  integer, save :: j
  !$omp threadprivate(j)

  !$omp parallel if(k.eq.0) num_threads(4) reduction(+:k)
  write(*,*) "parallel"

  !$omp do reduction(+:k) schedule(dynamic, 5) collapse(1)
  do i=1,4
     write(*,*) "do",i
     k = k + 1
  enddo
  !$omp end do

  !$omp sections reduction(+:k)
  !$omp section
  write(*,*) "section 1"
  !$omp section
  write(*,*) "section 2"
  !$omp end sections

  !$omp end parallel

  !$omp parallel
  !$omp task untied
  write(*,*) "task"
  !$omp end task
  !$omp end parallel

  !$omp parallel do num_threads(4) reduction(+:k) schedule(static,chunkif) collapse(1) ordered if(.true.) default(private) shared(i,k)
  do i=1,4
     !$omp ordered
     write(*,*) "do",i
     !$omp end ordered
     k = k + 1
  enddo
  !$omp end parallel do

  !$omp parallel sections if((i+k)>5) num_threads(4) reduction(+:k) default(none)
  !$omp section
  write(*,*) "section 1"
  !$omp section
  write(*,*) "section 2"
  !$omp end parallel sections

  !$omp parallel workshare if(.true.) num_threads(4) reduction(+:k)
  write(*,*) "workshare"
  !$omp end parallel workshare

  !$omp parallel shared(num_threads)
  write(*,*) num_threads
  !$omp end parallel 
end program test8
