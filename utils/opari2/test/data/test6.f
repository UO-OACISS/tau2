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
! * @brief Test that the insertion of wrapper functions works correctly, but ONLY on supported functions.

      program test6
      include 'omp_lib.h'

      integer (kind=omp_lock_kind)      lock1
      integer (kind=omp_nest_lock_kind) lock2
      integer (kind=omp_sched_kind)     sched
      integer mod

!     **************************************************
!     * Should be replaced by wrapper functions        *
!     *  regardless of "distractions"                  *
!     **************************************************
      call omp_init_lock(lock1); call omp_init_nest_lock(lock2)
      call omp_set_lock(lock1); write(*,*) "omp_set_lock(lock1)"
      call omp_set_nest_lock(lock2) ! omp_set_nest_lock(lock2);
      call omp_unset_lock(lock1); !omp_unset_lock(lock1);
      call omp_unset_nest_lock(lock2)
!$    mod = omp_test_lock(lock1)
*$    mod = omp_test_nest_lock(lock2)
!P$   mod = omp_test_lock(lock1)
cP$   mod = omp_test_nest_lock(lock2)

      call omp_destroy_lock(lock1)
      call omp_destroy_nest_lock(lock2)

!     **************************************************
!     * Not now, but planned for the future!           *
!     **************************************************

      call omp_set_num_threads(4)
      call omp_set_dynamic(.true.)
      call omp_set_schedule(omp_sched_static, 1)
      call omp_set_nested(.true.)
      call omp_set_max_active_levels(2)

!     **************************************************
!     * No replacement beyond this point!              *
!     **************************************************

!     call omp_init_lock(i)
c     call omp_init_lock(i)
*     call omp_init_lock(i)

      write(*,*) "omp_init_lock(i)",  'omp_init_lock(i)' ! call omp_init_lock(i)
      write(*,*)  "omp_init_lock(i)""test", """omp_init_lock(i)",
     &"omp_init_lock(i)""",  """", """""""","omp_init_lock(i) ",
!        ",&
     & "  + call omp_init_lock(i)"

!     call omp_init_lock(i)       ! call omp_init_lock(i)
!     call omp_init_lock(i) ; call omp_set_lock(i)
!     write(*,*) "call omp_init_lock(i)" ; call omp_init_lock(i)
      end program test6

! Simulation of included header file, nothing should be replaced
#line 1 /some/path/to/include/file/omp_lib.h
      call omp_init_lock(lock1)
      call omp_init_nest_lock(lock2)
      call omp_set_lock(lock1)
      call omp_set_nest_lock(lock2)
      call omp_unset_lock(lock1)
      call omp_unset_nest_lock(lock2)
