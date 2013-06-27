      PROGRAM EXAMPLE

      INTEGER i, lsum, sum

      sum = 0

!$omp parallel private(i, lsum) reduction(+:sum)
      lsum = 0

!$omp do
      do i=1,20
         lsum = lsum + i
      enddo
!$omp end do

      write(*,*) "LOCAL SUM: ", lsum
      sum = sum + lsum
!$omp end parallel

      write(*,*) "TOTAL SUM: ", sum

      END
