module scratch
   integer, parameter :: wp = selected_real_kind(15)
   real(wp), allocatable :: A(:,:), B(:,:), C(:,:)   ! scratch arrays
contains
   subroutine allocate_scratch(n)
      integer, intent(in) :: n
      allocate( A(n,n), B(n,n), C(n,n) )
      return
   end subroutine allocate_scratch
   subroutine use_scratch
      ! use the module arrays A, B, and C.
      call random_number( A ); call random_number( B )
      C = matmul( A, B )
      return
   end subroutine use_scratch
   subroutine deallocate_scratch
      deallocate( A, B )  ! note that C is not included in this list.
      return
   end subroutine deallocate_scratch
end module scratch

program memory_leak
   use scratch
   call allocate_scratch( 100 )
   call use_scratch
   call deallocate_scratch
end program memory_leak
