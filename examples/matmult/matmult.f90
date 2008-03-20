!**********************************************************************
!     matmult.f90 - simple matrix multiply implementation 
!************************************************************************
      subroutine initialize(a, b, n)
        double precision a(n,n)
        double precision b(n,n)
        integer n

! first initialize the A matrix
        do i = 1,n 
          do j = 1,n 
            a(j,i) = i 
          end do
        end do

! then initialize the B matrix
        do i = 1,n 
          do j = 1,n 
            b(j,i) = i 
          end do
        end do

      end subroutine initialize
      
      subroutine multiply_matrices(answer, buffer, b, matsize)
        double precision buffer(matsize), answer(matsize)
        double precision b(matsize, matsize)
        integer i, j
! multiply the row with the column 

        do i = 1,matsize 
          answer(i) = 0.0 
          do j = 1,matsize 
            answer(i) = answer(i) + buffer(j)*b(j,i) 
          end do
        end do
      end subroutine multiply_matrices

      program main
      include "mpif.h"

      integer SIZE_OF_MATRIX
      parameter (SIZE_OF_MATRIX = 2000) 
! try changing this value to 2000 to get rid of transient effects 
! at startup
      double precision a(SIZE_OF_MATRIX,SIZE_OF_MATRIX) 
      double precision b(SIZE_OF_MATRIX,SIZE_OF_MATRIX) 
      double precision c(SIZE_OF_MATRIX,SIZE_OF_MATRIX) 
      double precision buffer(SIZE_OF_MATRIX), answer(SIZE_OF_MATRIX)

      integer myid, master, maxpe, ierr, status(MPI_STATUS_SIZE) 
      integer i, j, numsent, sender 
      integer answertype, row, flag
      integer matsize

      call MPI_INIT( ierr ) 
      call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr ) 
      call MPI_COMM_SIZE( MPI_COMM_WORLD, maxpe, ierr ) 
      print *, "Process ", myid, " of ", maxpe, " is active"

      master = 0 
      matsize = SIZE_OF_MATRIX 

      if ( myid .eq. master ) then 
! master initializes and then dispatches 
! initialize a and b 
        call initialize(a, b, matsize)
        numsent = 0

! send b to each other process 
        do i = 1,matsize 
          call MPI_BCAST(b(1,i), matsize, MPI_DOUBLE_PRECISION, master, &
             MPI_COMM_WORLD, ierr) 
        end do

! send a row of a to each other process; tag with row number 
        do i = 1,maxpe-1 
          do j = 1,matsize 
            buffer(j) = a(i,j) 
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
              buffer(j) = a(numsent+1,j) 
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
          call MPI_BCAST(b(1,i), matsize, MPI_DOUBLE_PRECISION, master, &
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
            call multiply_matrices(answer, buffer, b, matsize)
            call MPI_SEND(answer, matsize, MPI_DOUBLE_PRECISION, master,&
               row, MPI_COMM_WORLD, ierr) 
          endif 
        end do
      endif

      call MPI_FINALIZE(ierr) 
      end program main
