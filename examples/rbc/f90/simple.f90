module scratch

   integer, parameter :: wp = selected_real_kind(15)
   integer :: N
   real(wp), allocatable :: A(:,:), B(:,:), C(:,:)   ! scratch arrays

contains

   subroutine allocate_scratch(nn)
      integer, intent(in) :: nn
      N = nn
      allocate( A(N,N), B(N,N), C(N,N) )
      return
   end subroutine allocate_scratch

   subroutine use_scratch
      ! use the module arrays A, B, and C.
      call random_number( A ); call random_number( B )
      C = matmul( A, B )
      return
   end subroutine use_scratch

   subroutine overflow_scratch
     ! Write above A's end
     integer :: i
     write (*,*) 'Testing overflow.  Expect a segfault'
     A(N+10, N) = 1
   end subroutine overflow_scratch

   subroutine underflow_scratch
     ! Write below A's beginning
     write (*,*) 'Testing underflow.  Expect a segfault'
     A(0, 0) = 1
   end subroutine underflow_scratch


   subroutine deallocate_scratch
      deallocate( A, B )  ! note that C is not included in this list.
      return
   end subroutine deallocate_scratch

end module scratch


program matrix
   use scratch
   call allocate_scratch( 1000 )
   call use_scratch
   call overflow_scratch
   call underflow_scratch
   call deallocate_scratch
end program matrix
