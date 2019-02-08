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
! * @brief Special tests for end pragma substitution and nested parallel regions/loops.

      program test3
      integer i, j, k
      real end_level, dolic

c$OMP parallel
c$OMP do
      do 12,i = 1,8
         a=a+1
 12   continue
c$OMP end parallel

c$OMP parallel do
      do 13,i = 1,8
         a=a+1
 13   continue

c$OMP parallel
c$OMP do
      do 14,i = 1,8
         a=a+1
 14   continue
c$OMP atomic
      me = me + omp_get_thread_num()
c$OMP end parallel

c$OMP parallel do
      do 15,i = 1,8
 15      a=a+1
c$OMP parallel private(me,glob)
      me = omp_get_thread_num()
c$OMP end parallel

! **********************
! * nested parallelism *
! **********************

!$omp parallel
!$omp parallel
!$omp parallel
!$omp parallel do
      do i = 1,8
         end_level = end_level + dolic/i
         a=a+1
      end do
!$omp end parallel do
!$omp end parallel
!$omp end parallel
!$omp end parallel

! *******************************************
! * end pragma substitution in nested loops *
! *******************************************
c$OMP parallel do
      do 16, i = 1,8
         do 16, j = 1,8
            a=a+1
 16   continue

      do 17,i = 1,8
c$OMP parallel do
         do 18, j = 1,8
            a=a+1
 18      continue
 17   continue

c$omp parallel do
      do
         if (a .gt. 0) then
            exit
         endif
      enddo
      end program test3

