      program mandel
      use ppm

      implicit none

      integer, parameter               :: dp=selected_real_kind(13)
      complex(kind=dp) :: c, z
      integer          :: ix, iy, cnt
      integer          :: numpe, maxiter
      integer, parameter               :: width=800, height=800
      integer, dimension(width,height) :: iterations
      real(kind=dp)                    :: xmin, xmax, ymin, ymax, x, y, dx, dy
      external mytimer
      integer mandelprofiler(2), parprof(2), loopprof(2)
!$    integer OMP_GET_NUM_THREADS
        save mandelprofiler, parprof, loopprof
        call TAU_PROFILE_INIT()
        call TAU_PROFILE_TIMER(mandelprofiler, 'PROGRAM MANDEL')
        call TAU_PROFILE_TIMER(parprof, 'OpenMP Parallel Region')
        call TAU_PROFILE_TIMER(loopprof, 'OpenMP Loop')
        call TAU_PROFILE_START(mandelprofiler)
        call TAU_PROFILE_SET_NODE(0)


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
      call TAU_PROFILE_START(parprof)
!$    numpe = OMP_GET_NUM_THREADS()
!$OMP DO PRIVATE(ix,x,y,c,z,cnt)
      do iy=1,height
        call TAU_PROFILE_START(loopprof)
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
        call TAU_PROFILE_STOP(loopprof)
      end do
      call TAU_PROFILE_STOP(parprof)
!$OMP END PARALLEL
      call mytimer(numpe)

! --- generate ppm file
      write (unit=*, fmt="(A)") "generate ppm file..."
      call ppmwrite("mandel.ppm", iterations, maxiter)

      call TAU_PROFILE_STOP(mandelprofiler)
      end program mandel
