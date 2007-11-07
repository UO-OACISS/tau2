!**********************************************************************
!     matmult.f90 - simple matrix multiply implementation 
!************************************************************************
      module matrices
      integer, parameter :: wp = selected_real_kind(15)
      real(wp), allocatable :: A(:,:), B(:,:), C(:,:)   ! scratch arrays
      integer MSIZE
      parameter (MSIZE = 1000) 
      contains
      subroutine allocate_matrices
        allocate( A(MSIZE,MSIZE), B(MSIZE,MSIZE), C(MSIZE,MSIZE) )
        return
      end subroutine allocate_matrices
      subroutine deallocate_matrices
        deallocate( A, B )  ! note that C is not included in this list.
        return
      end subroutine deallocate_matrices

      subroutine initialize

! first initialize the A matrix
        do i = 1,MSIZE 
          do j = 1,MSIZE 
            A(j,i) = i 
          end do
        end do

! then initialize the B matrix
        do i = 1,MSIZE 
          do j = 1,MSIZE 
            B(j,i) = i 
          end do
        end do

      end subroutine initialize
      
      subroutine multiply_matrices(answer, buffer)
        double precision buffer(MSIZE), answer(MSIZE)
        integer i, j
! multiply the row with the column 

        do i = 1,MSIZE 
          answer(i) = 0.0 
          do j = 1,MSIZE 
            answer(i) = answer(i) + buffer(j)*B(j,i) 
          end do
        end do
      end subroutine multiply_matrices

      subroutine driver
      include "mpif.h"

! try changing this value to 2000 to get rid of transient effects 
! at startup
      double precision buffer(MSIZE), answer(MSIZE)

      integer myid, master, maxpe, ierr, status(MPI_STATUS_SIZE) 
      integer i, j, numsent, sender 
      integer answertype, row, flag

      call MPI_INIT( ierr ) 
      call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr ) 
      call MPI_COMM_SIZE( MPI_COMM_WORLD, maxpe, ierr ) 
      print *, "Process ", myid, " of ", maxpe, " is active"

      master = 0 
      matsize = MSIZE

      if ( myid .eq. master ) then 
! master initializes and then dispatches 
! initialize a and b 
        call initialize
        numsent = 0

! send b to each other process 
        do i = 1,matsize 
          call MPI_BCAST(B(1,i), matsize, MPI_DOUBLE_PRECISION, master, &
             MPI_COMM_WORLD, ierr) 
        end do

! send a row of a to each other process; tag with row number 
        do i = 1,maxpe-1 
          do j = 1,matsize 
            buffer(j) = A(i,j) 
          end do
          call MPI_SEND(buffer, matsize, MPI_DOUBLE_PRECISION, i,       &
             i, MPI_COMM_WORLD, ierr) 
          numsent = numsent+1 
        end do

        do i = 1,matsize 
          call MPI_RECV(answer, matsize, MPI_DOUBLE_PRECISION,          &
           MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)
          sender = status(MPI_SOURCE) 
          answertype = status(MPI_TAG) 
          do j = 1,matsize 
            c(answertype,j) = answer(j) 
          end do

          if (numsent .lt. matsize) then 
            do j = 1,matsize 
              buffer(j) = A(numsent+1,j) 
            end do
            call MPI_SEND(buffer, matsize, MPI_DOUBLE_PRECISION, sender,&
              numsent+1, MPI_COMM_WORLD, ierr) 
            numsent = numsent+1 
          else 
            call MPI_SEND(1.0, 1, MPI_DOUBLE_PRECISION, sender, 0,      &
                 MPI_COMM_WORLD, ierr) 
          endif 
        end do

! print out one element of the answer
        print *, "c(", matsize, ",", matsize, ") = ", c(matsize,matsize)
      else 
! workers receive B, then compute rows of C until done message 
        do i = 1,matsize 
          call MPI_BCAST(B(1,i), matsize, MPI_DOUBLE_PRECISION, master, &
                 MPI_COMM_WORLD, ierr) 
        end do
        flag = 1
        do while (flag .ne. 0)
          call MPI_RECV(buffer, matsize, MPI_DOUBLE_PRECISION, master,  &
            MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr) 
          row = status(MPI_TAG) 
          flag = row
          if (flag .ne. 0) then
! multiply the matrices here using C(i,j) += sum (A(i,k)* B(k,j))
            call multiply_matrices(answer, buffer)
            call MPI_SEND(answer, matsize, MPI_DOUBLE_PRECISION, master,&
               row, MPI_COMM_WORLD, ierr) 
          endif 
        end do
      endif

      call MPI_FINALIZE(ierr) 
      end subroutine driver
      end module matrices


      program main
      use matrices
      call allocate_matrices
      call driver
      call deallocate_matrices
      end program main
