subroutine bar(arg)
  integer arg

  print *, "inside bar arg = ", arg
end subroutine bar

subroutine foo(iVal)
  integer iVal
  integer j, k
! Do something here...
  print *, "Iteration = ", iVal
        do j = 1, 5
          do k = 1, 2
            print *, "j = ", j
          end do
        end do

        do 10, i = 1, 3
        call bar(i+iVal)
10      continue
        print *, "after calling bar in foo"
      end

program main
  integer i

  print *, "test program"

  do 10, i = 1, 3
    call foo(i)
10  continue
end program main

