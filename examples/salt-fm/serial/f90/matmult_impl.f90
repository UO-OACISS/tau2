submodule (matmult_mod) matmult_impl
  implicit none

contains

  module procedure initialize
    integer :: i, j

    do j = 1, size(m, 2)
      do i = 1, size(m, 1)
        m(i,j) = i + j
      end do
    end do
  end procedure initialize

  module procedure compute
    integer :: i, j, k

    do i = 1, size(c, 1)
      do j = 1, size(c, 2)
        do k = 1, size(a, 2)
          c(i,j) = c(i,j) + a(i,k) * b(k,j)
        end do
      end do
    end do
  end procedure compute

  module procedure compute_interchange
    integer :: i, j, k

    do j = 1, size(c, 2)
      do k = 1, size(a, 2)
        do i = 1, size(c, 1)
          c(i,j) = c(i,j) + a(i,k) * b(k,j)
        end do
      end do
    end do
  end procedure compute_interchange
end submodule matmult_impl
