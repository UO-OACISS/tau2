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
! * @brief Test the parsers ability to insert a necessary include statement at the right places.


      integer function f0()
      implicit
     &none
      double precision :: d
      write (*,*) "function f0"
      f0 = 5
      return
      end function f0
      
      subroutine s0
      write (*,*) "subroutine s0"
      end subroutine s0
      
      integer function f1(a)    !interface
      implicit
     >none
      integer :: a, result
      write (*,*) "function f1"
      f1 = a
      return
      end function f1

      recursive subroutine s1(a)
      implicit
     $none
      integer :: a
      write (*,*) "subroutine s1"
      write (*,*) "keyword interface inside a string"
      call ss1()

      entry ss1()
      write (*,*) "entry ss1"
      end subroutine s1

      program otest
      integer :: i, f0, f1, function
      
      function = 0
      write (*,*) 'program otest'
      i = f0()
      i = f1(2)
      call s0
      call s1(4)
      end program otest
      
