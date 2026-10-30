module matmult_mod
  use iso_fortran_env, only: dp => real64
  implicit none

  interface
    module subroutine initialize(m)
      real(dp), intent(out) :: m(:,:)
    end subroutine initialize

    module subroutine compute(a, b, c)
      real(dp), intent(in) :: a(:,:), b(:,:)
      real(dp), intent(inout) :: c(:,:)
    end subroutine compute

    module subroutine compute_interchange(a, b, c)
      real(dp), intent(in) :: a(:,:), b(:,:)
      real(dp), intent(inout) :: c(:,:)
    end subroutine compute_interchange
  end interface
end module matmult_mod

program matmult
  use matmult_mod
  implicit none
  integer, parameter :: n = 512
  real(dp), allocatable :: a(:,:), b(:,:), c(:,:)

  allocate(a(n,n), b(n,n), c(n,n))
  call initialize(a)
  call initialize(b)
  c = 0
  call compute(a, b, c)
  call compute_interchange(a, b, c)
  print *, 'c(1,2) = ', c(1,2)
end program matmult
