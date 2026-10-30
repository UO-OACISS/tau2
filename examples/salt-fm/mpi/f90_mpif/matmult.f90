module matmult_mod
  use iso_fortran_env, only: dp => real64
  implicit none

contains

  subroutine initialize(m)
    real(dp), intent(out) :: m(:,:)
    integer :: i, j

    do j = 1, size(m, 2)
      do i = 1, size(m, 1)
        m(i,j) = i + j
      end do
    end do
  end subroutine initialize

  subroutine multiply_row(row, b, answer)
    real(dp), intent(in) :: row(:), b(:,:)
    real(dp), intent(out) :: answer(:)
    integer :: j

    do j = 1, size(b, 2)
      answer(j) = dot_product(row, b(:,j))
    end do
  end subroutine multiply_row
end module matmult_mod

program matmult
  use matmult_mod
  implicit none
  include "mpif.h"
  integer, parameter :: n = 512, master = 0
  real(dp), allocatable :: a(:,:), b(:,:), c(:,:)
  real(dp) :: row(n), answer(n)
  integer :: rank, nprocs, ierr, status(MPI_STATUS_SIZE)
  integer :: i, numsent, sender, tag

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  if (nprocs < 2) then
    print *, 'Run with 2 or more processes'
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  print *, 'Process ', rank, ' of ', nprocs, ' is active'

  allocate(a(n,n), b(n,n), c(n,n))
  if (rank == master) then
    call initialize(a)
    call initialize(b)
  end if
  call MPI_Bcast(b, n*n, MPI_DOUBLE_PRECISION, master, MPI_COMM_WORLD, ierr)

  if (rank == master) then
    numsent = 0
    do i = 1, nprocs - 1
      if (numsent < n) then
        numsent = numsent + 1
        row = a(numsent,:)
        call MPI_Send(row, n, MPI_DOUBLE_PRECISION, i, numsent, MPI_COMM_WORLD, ierr)
      else
        call MPI_Send(row, 0, MPI_DOUBLE_PRECISION, i, 0, MPI_COMM_WORLD, ierr)
      end if
    end do
    do i = 1, n
      call MPI_Recv(answer, n, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, MPI_ANY_TAG, &
                    MPI_COMM_WORLD, status, ierr)
      sender = status(MPI_SOURCE)
      c(status(MPI_TAG),:) = answer
      if (numsent < n) then
        numsent = numsent + 1
        row = a(numsent,:)
        call MPI_Send(row, n, MPI_DOUBLE_PRECISION, sender, numsent, MPI_COMM_WORLD, ierr)
      else
        call MPI_Send(row, 0, MPI_DOUBLE_PRECISION, sender, 0, MPI_COMM_WORLD, ierr)
      end if
    end do
    print *, 'c(1,2) = ', c(1,2)
  else
    do
      call MPI_Recv(row, n, MPI_DOUBLE_PRECISION, master, MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)
      tag = status(MPI_TAG)
      if (tag == 0) exit
      call multiply_row(row, b, answer)
      call MPI_Send(answer, n, MPI_DOUBLE_PRECISION, master, tag, MPI_COMM_WORLD, ierr)
    end do
  end if

  call MPI_Finalize(ierr)
end program matmult
