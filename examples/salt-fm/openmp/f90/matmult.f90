module matmult_mod
  use iso_fortran_env, only: dp => real64
  implicit none

contains

  subroutine initialize(m)
    real(dp), intent(out) :: m(:,:)
    integer :: i, j

    !$omp parallel do
    do j = 1, size(m, 2)
      do i = 1, size(m, 1)
        m(i,j) = i + j
      end do
    end do
    !$omp end parallel do
  end subroutine initialize

  subroutine compute(a, b, c)
    real(dp), intent(in) :: a(:,:), b(:,:)
    real(dp), intent(inout) :: c(:,:)
    integer :: i, j, k

    !$omp parallel do private(j, k)
    do i = 1, size(c, 1)
      do j = 1, size(c, 2)
        do k = 1, size(a, 2)
          c(i,j) = c(i,j) + a(i,k) * b(k,j)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine compute

  subroutine compute_interchange(a, b, c)
    real(dp), intent(in) :: a(:,:), b(:,:)
    real(dp), intent(inout) :: c(:,:)
    integer :: i, j, k

    !$omp parallel do private(i, k)
    do j = 1, size(c, 2)
      do k = 1, size(a, 2)
        do i = 1, size(c, 1)
          c(i,j) = c(i,j) + a(i,k) * b(k,j)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine compute_interchange
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
