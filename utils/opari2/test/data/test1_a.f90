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

        module functions_module
          interface
            function if0(a, & ! ) a=b
                         b ) ! end function
              integer(kind=4) :: if0, a
            end function if0
          end interface

          interface
            subroutine is1(a)
              integer :: a
            end subroutine is1
          end interface

          contains
            integer(kind=4) function mf0(a, & ! ) a=b
                                         b )  ! end function
!              integer functionkind     THIS IS TOO EVIL FOR NOW, WILL TAKE CARE OF LATER
!              functionkind = 4
              write (*,*) "function mf0"
              mf0 = 5
              return
            end function mf0

            subroutine ms0
              write (*,*) "subroutine ms0"
            end subroutine ms0
        end module mmm

        module module1
          use functions_module

        contains
          subroutine r1()
            write (*,*) "Subroutine 1"
            call f1()
          end subroutine r1

        end module module1

        integer function f0()
          implicit &
            none
          double precision :: d
          write (*,*) "function f0"
          f0 = 5
          return
        end function f0

        subroutine s0
          write (*,*) "subroutine s0"
        end subroutine s0

        integer function f1(a) !interface
          use mmm
          integer :: a, result
          write (*,*) "function f1"
          f1 = a
          return
        end function f1

        subroutine s1(a)
          integer :: a
          write (*,*) "subroutine s1"
          write (*,*) "keyword interface inside a string"
          call ss1()

          contains
            subroutine ss1()
              write (*,*) "subroutine ss1"
            end subroutine ss1
        end subroutine s1

        subroutine sub_bind(a) bind(c,name='sub_bind_f')
          integer :: a

          write(*,*) "subroutine sub_bind(a) bind(c,name='sub_bind_f')"
        end subroutine sub_bind
        program otest
          use mmm
          use rename_test, bar => foo
          integer :: i, f0, f1, function

          function = 0
          write (*,*) 'program otest'
          i = f0()
          i = f1(2)
          i = mf0()
          call s0
          call s1(4)
          call ms0
        end program otest
