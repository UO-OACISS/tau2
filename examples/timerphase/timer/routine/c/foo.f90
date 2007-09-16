      subroutine bar(x)
      integer :: x

      print *, "inside bar: x = ", x
      if (x .gt. 0) call mysleep(x)

      end subroutine bar
      subroutine foo(x)
      integer :: x

      print *, "inside foo: calling bar, x = ", x

      call bar(x-1)
      end subroutine foo
      
      program main
      integer :: x
      print *, "inside main: calling foo"
      do x=0,4
        call foo(x)
      end do
      end program main
