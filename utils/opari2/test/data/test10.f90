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
! * @authors Bernd Mohr, Peter Philippen
! *
! * @brief Tests user instrumentation directives.

program test10

integer (kind=omp_lock_kind)      lock

  !$POMP INST INIT

  !$POMP INST OFF

  !$POMP NOINSTRUMENT

  !$OMP PARALLEL
  i = 1
  !$OMP end PARALLEL
  
  !$OMP PARALLEL DO
    do 100, i = 2, 50
     j++
100 continue
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO
    do  i = 3, 50
     j++
    enddo

  !$OMP PARALLEL SECTIONS
  !$OMP SECTION
    i = 4
  !$OMP SECTION
    i = 5
  !$OMP END PARALLEL SECTIONS


  !$OMP PARALLEL WORKSHARE
    i = 6
  !$OMP END PARALLEL WORKSHARE

  !$OMP SINGLE
    i = 7
  !$OMP END SINGLE
 
  !$OMP MASTER
   i = 8
  !$OMP END MASTER

  !$OMP critical
    i = 9
  !$omp end critical

  !$OMP critical test
    i = 10
  !$omp end critical test

  !$OMP workshare 
    i = 11
  !$OMP end workshare

  !$OMP workshare 
    i = 12
  !$OMP end workshare nowait

  !$OMP ordered
    i = 13
  !$OMP end ordered

  !$OMP task
    i = 14
  !$OMP end task

  !$OMP taskwait

  !$OMP atomic update
    i = 15

  !$OMP sections
    i = 16
  !$omp section
    i = 17
  !$omp section 
    i = 18
  !$omp end sections nowait

  !$OMP barrier

  !$OMP flush

  !$OMP threadprivate( i )

  call omp_init_lock( lock )
  call omp_destroy_lock( lock ) 

  !$POMP INSTRUMENT

  !$OMP PARALLEL
  i = 1
  !$OMP end PARALLEL
  
  !$OMP PARALLEL DO
    do 100, i = 2, 50
     j++
100 continue
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO
    do  i = 3, 50
     j++
    enddo

  !$OMP PARALLEL SECTIONS
  !$OMP SECTION
    i = 4
  !$OMP SECTION
    i = 5
  !$OMP END PARALLEL SECTIONS


  !$OMP PARALLEL WORKSHARE
    i = 6
  !$OMP END PARALLEL WORKSHARE

  !$OMP SINGLE
    i = 7
  !$OMP END SINGLE
 
  !$OMP MASTER
   i = 8
  !$OMP END MASTER

  !$OMP critical
    i = 9
  !$omp end critical

  !$OMP critical test
    i = 10
  !$omp end critical test

  !$OMP workshare 
    i = 11
  !$OMP end workshare

  !$OMP workshare 
    i = 12
  !$OMP end workshare nowait

  !$OMP ordered
    i = 13
  !$OMP end ordered

  !$OMP task
    i = 14
  !$OMP end task

  !$OMP taskwait

  !$OMP atomic update
    i = 15

  !$OMP sections
    i = 16
  !$omp section
    i = 17
  !$omp section 
    i = 18
  !$omp end sections nowait

  !$OMP barrier

  !$OMP flush

  !$OMP threadprivate( i )

  call omp_init_lock( lock )
  call omp_destroy_lock( lock ) 


  !$POMP INST FINALIZE
end program test10
