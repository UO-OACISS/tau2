! Simple test of GPSHMEM Fortran program
      program mainf

! External defs
      integer gpmype, gpnumpes
      external gpshmem_init, gpshmem_finalize, gpshmem_barrier

! Vars
      integer me, numpes, i

      call gpshmem_init()

      me = gpmype()
      numpes = gpnumpes()

      print *, "Hello, world from ", me, " of ", numpes

      do i=1, 10
         call gpshmem_barrier_all()
      enddo

      call gpshmem_finalize()

      end program
