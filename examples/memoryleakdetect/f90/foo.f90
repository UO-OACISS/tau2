       subroutine bar(value)
       integer:: value
       integer, dimension(:), allocatable:: X
   
       if (value .gt. 3) then
         print *, "bar called with value=", value
         allocate(X(value))
         X(2) = 2
         ! Do not free X. Create a  memory leak unless value > 15 
         if (value .gt. 15) deallocate(X)
       else
         print *, "bar called from foo!"
         allocate(X(45))
         X(23) = 2
         deallocate(X)
       endif
       end subroutine bar
        
       subroutine g(value)
       integer:: value
     
       print *, "Inside g: value = ", value
       call bar(value)
       end subroutine g
        
       subroutine foo(value)
       integer:: value, ierr
       
       print *, "Inside foo: value = ", value
       
       ! Create two distinct paths to reach bar
       if (value .gt. 5) then
          call g(value)
       else 
          call bar(value)
       endif

       end subroutine foo

       program main
       call TAU_PROFILE_SET_NODE(0)
       call foo(12) ! leak
       call foo(20) ! no leak
       call foo(2)  ! no leak
       call foo(13) ! leak
       call foo(4)  ! leak
       end program main
