      subroutine F1()
        implicit none

        print *, "Inside f1()"
        call F2()
        print *, "After calling f2()"

      end subroutine F1
