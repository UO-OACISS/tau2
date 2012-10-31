!**********************************************************************
!     matmult.f90 - simple matrix multiply implementation 
!************************************************************************
      subroutine initialize(a, b, n)
        real a(n,n)
        real b(n,n)
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

      subroutine multiply_matrices(a, b, c, matsize)
        IMPLICIT NONE
        real a(matsize, matsize)
        real b(matsize, matsize)
        real c(matsize, matsize)
        real ctemp
        integer i, j, k, l, m, matsize

!$acc data copyin(a,b) copyout(c)
!$acc kernels loop
        do k = 1,matsize
          do i = 1,matsize
            do j = 1,matsize
              c(i,k) = c(i,k) + a(i,j) * b(j,k)
            enddo
          enddo
        enddo
!$acc end kernels loop
!$acc end data 
      end subroutine multiply_matrices
      
      program main

      integer SIZE_OF_MATRIX
      parameter (SIZE_OF_MATRIX = 1000) 
      
      real a(SIZE_OF_MATRIX,SIZE_OF_MATRIX) 
      real b(SIZE_OF_MATRIX,SIZE_OF_MATRIX) 
      real c(SIZE_OF_MATRIX,SIZE_OF_MATRIX) 

      integer matsize

      matsize = SIZE_OF_MATRIX 

      call initialize(a, b, matsize)

! multiply the matrices here using C(i,j) += (A(i,k)* B(k,j)) 
      call multiply_matrices(a, b, c, matsize)
     
      end program main
