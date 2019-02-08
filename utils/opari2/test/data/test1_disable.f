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
! * @brief Test the --disable= option


      program test1_disable
      IMPLICIT NONE
      
      SUBROUTINE foo(A,N)
      INTEGER I,N,L,T
      
c$omp parallel

c$omp atomic
      N=0

c$omp critical
      N=1
c$omp end critical

c$omp flush

      CALL OMP_INIT_LOCK(L)
      CALL OMP_SET_LOCK(L)
      T=OMP_TEST_LOCK(L)
      CALL OMP_UNSET_LOCK(L)
      CALL OMP_DESTROY_LOCK(L)

      CALL OMP_INIT_NEST_LOCK(L)
      CALL OMP_SET_NEST_LOCK(L)
      T=OMP_TEST_NEST_LOCK(L)
      CALL OMP_UNSET_NEST_LOCK(L)
      CALL OMP_DESTROY_NEST_LOCK(L)
c$omp master
      N=2
c$omp end master

c$omp do
      DO I=1,5
c$omp ordered
         N=I
c$omp end ordered
      END DO

c$omp single
      N=6
c$omp end single

c$omp task
      N=7
c$omp end task

c$omp end parallel      

c$pomp inst init

c$pomp inst begin(user_region)
      if .false. then
c$pomp inst altend(user_region)
      return
      end
c$pomp inst end(user_region)

      END SUBROUTINE 
      end program test1_free
