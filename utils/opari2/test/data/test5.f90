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
! * @brief Test the splitting of combined parallel clauses.

program test5
  integer i,j,k,m
  logical l
  integer, dimension(10,10) :: AA, BB, CC
  
  integer, save :: t
  !$omp threadprivate(t)
  
  
  !$omp   parallel            &   !parallel
  !$omp & do                  &   !do  
  !$omp & lastprivate(k)      &   !comment
  !$omp & private(i,j),       &   !schedule
  !$omp & lastprivate         &
  !$omp & (                   &
  !$omp &   l                 &   !comment inside argument
  !$omp & ), schedule(dynamic &
  !$omp & )
  do i=1,4
     write(*,*) "parallel do ", i
     k=k+i
  end do
  !$omp  end parallel do
      
  if(k .gt. 0) l = .true.
  !$omp  parallel sections if(l) num_threads(2) default(shared)   &
  !$omp &firstprivate(j) lastprivate(i) copyin(t) reduction(+:l)
  !$omp  section
  write(*,*) "section1"
  !$omp  section
  write(*,*) "section2"
  !$omp  section
  write(*,*) "section3"
  !$omp  end parallel sections

  !$omp parallel workshare
  AA = BB
  BB = CC
  !$omp end parallel workshare
end program test5
