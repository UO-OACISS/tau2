      module ppm
      implicit none

      public :: ppmwrite_ascii, ppmwrite

      contains

      subroutine ppmwrite_ascii(fname, iterations, maxiter)

      character(len=*), intent(in)        :: fname
      integer, dimension(:,:), intent(in) :: iterations
      integer, intent(in)                 :: maxiter

      integer :: width, height
      integer :: ix, iy, iz, cnt, idx
      integer, parameter :: maxc = 3
      integer, parameter :: maxcols = (maxc+1)*(maxc+1)*(maxc+1)
      integer, dimension(maxcols,3) :: col
      real :: factor

! --- initialization
      width  = size(iterations,1)
      height = size(iterations,2)
      factor  = real(maxiter) / maxcols

! --- generate color map
      cnt = maxcols
      do ix=0,maxc
        do iy=0,maxc
          do iz=0,maxc
            col(cnt,1) = ix
            col(cnt,2) = iy
            col(cnt,3) = iz
            cnt = cnt-1
          end do
        end do
      end do

! --- generate ppm file
      open(unit=9, file=fname, status="replace", action="write")
      write(unit=9, fmt="('P3 ', 2i5, i2)") width, height, maxc
      do iy=1,height
        do ix=1,width
          if ( iterations(ix,iy) == maxiter ) then
            write(unit=9, fmt="(A)") "0 0 0"
          else
            idx = iterations(ix,iy) / factor + 1
            write(unit=9, fmt="(3i2)") col(idx,1), col(idx,2), col(idx,3)
          end if
        end do
      end do
      close(unit=9)
      end subroutine ppmwrite_ascii

      subroutine ppmwrite(fname, iterations, maxiter)

      character(len=*), intent(in)        :: fname
      integer, dimension(:,:), intent(in) :: iterations
      integer, intent(in)                 :: maxiter

      integer :: width, height
      integer :: ix, iy, iz, cnt, idx, b, r, i, total
      integer, parameter :: maxc = 3, buflen=512
      integer, parameter :: maxcols = (maxc+1)*(maxc+1)*(maxc+1)
      real :: factor
      character(len=1), dimension(maxcols,3) :: col
      character(len=buflen)                  :: buf
      character(len=20)                      :: tmpbuf

! --- initialization
      width  = size(iterations,1)
      height = size(iterations,2)
      factor  = real(maxiter) / maxcols
      b = 1
      total = 0

! --- generate color map
      cnt = maxcols
      do ix=0,maxc
        do iy=0,maxc
          do iz=0,maxc
            col(cnt,1) = char(ix)
            col(cnt,2) = char(iy)
            col(cnt,3) = char(iz)
            cnt = cnt-1
          end do
        end do
      end do

! --- generate ppm file
      open(unit=9, file=fname, access="direct", recl=buflen, &
                   action="write", status="replace")

      write(unit=tmpbuf, fmt="('P6 ', 2i5, i2)") width, height, maxc
      buf(1:15) = tmpbuf(1:15)
      buf(16:16) = char(10)
      b = 17
      r = 1

      do iy=1,height
        do ix=1,width
          total = total + iterations(ix,iy)
          if ( iterations(ix,iy) == maxiter ) then
            do i=1,3
              buf(b:b) = char(0)
              b = b+1
              if ( b > buflen ) then
                write(unit=9, rec=r) buf
                b = 1
                r = r+1
              endif
            enddo
          else
            idx = iterations(ix,iy) / factor + 1
            do i=1,3
              buf(b:b) = col(idx,i)
              b = b+1
              if ( b > buflen ) then
                write(unit=9, rec=r) buf
                b = 1
                r = r+1
              endif
            end do
          end if
        end do
      end do
      write(unit=9, rec=r) buf
      close(unit=9)
      write (unit=*, fmt="(I A)") total, " total iterations"
      end subroutine ppmwrite

      end module ppm
