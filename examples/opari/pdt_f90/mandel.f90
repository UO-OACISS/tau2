      program mandel
      use ppm

      integer, parameter               :: dp=selected_real_kind(13)
      complex(kind=dp) :: c, z
      integer          :: ix, iy, cnt
      integer          :: numpe, maxiter
      integer, parameter               :: width=800, height=800
      integer, dimension(width,height) :: iterations
      real(kind=dp)                    :: xmin, xmax, ymin, ymax, x, y, dx, dy
      external mytimer
!$    integer OMP_GET_NUM_THREADS

! --- init
!     -1.5 0.5 -1.0 1.0
!     -.59 -.54 -.58 -.53
!     -.65 -.4 .475 .725
!     -.65 -.5 .575 .725
!     -.59 -.56 .47 .5
      write(unit=*, fmt="(A)") "xmin xmax ymin ymax maxiter (Sample: -.59 -.56 .47 .5 216) ?"
      read (unit=*, fmt=*) xmin, xmax, ymin, ymax, maxiter

! --- initialization
      dx = (xmax - xmin) / width
      dy = (ymax - ymin) / height
      numpe = 1

! --- calculate mandelbrot set
      call mytimer(0)
!$OMP PARALLEL
!$    numpe = OMP_GET_NUM_THREADS()
!$OMP DO PRIVATE(ix,x,y,c,z,cnt)
      do iy=1,height
        y = ymin + iy*dy
        do ix=1,width
          x = xmin + ix*dx
          c = cmplx(x,y)
          z = (0.0,0.0)
          cnt = 0
          do 
            z = z*z + c
            cnt = cnt+1
            if ( (dble(z)*dble(z) + dimag(z)*dimag(z)) >= 16 .or. &
                 cnt >= maxiter ) then
              exit
            end if
          end do
          iterations(ix,iy) = cnt
        end do
      end do
!$OMP END DO
!$OMP END PARALLEL
      call mytimer(numpe)

! --- generate ppm file
      write (unit=*, fmt="(A)") "generate ppm file..."
      call ppmwrite("mandel.ppm", iterations, maxiter)

      end program mandel
