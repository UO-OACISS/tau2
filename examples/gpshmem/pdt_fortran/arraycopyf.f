! version of arraycopy.c in fortran

      program arraycopy
      implicit none

      include 'gps_arrays.Fh'

! External defs
      integer gpmype, gpnumpes
      external gpshmem_init, gpmype, gpnumpes
      external gpshmem_barrier_all
      external gpshmem_fence, gpshmem_get, gpshmem_put
      integer gpshmalloci
      external gpshmalloci, gpshfree_handle

! Variables
      integer N, sizeofint, i, nxt, prv
      integer handlesrc, handletgt, indexsrc, indextgt
      parameter (N=100000, sizeofint=4)
      integer me, npes

! init gpshmem
      call gpshmem_init()
      me = gpmype()
      npes = gpnumpes()
      nxt = mod(me + 1, npes)
      prv = mod(me - 1 + npes, npes)

! allocate memory
      handlesrc = gpshmalloci(0, N*sizeofint, indexsrc)
      handletgt = gpshmalloci(0, N*sizeofint, indextgt)
      call gpshmem_barrier_all()

! test get
      do i=0,N - 1
        gps_int(indexsrc + i) = me
      enddo
      call gpshmem_barrier_all()
      if (me == 0) print *, "Testing get..."
      call gpshmem_barrier_all()
      call gpshmem_get(gps_int(indextgt), gps_int(indexsrc), N, nxt);
      call checkarray(gps_int(indextgt), N, nxt, me)

! test put
      call gpshmem_barrier_all()
      do i=0, N - 1
        gps_int(indexsrc + i) = me
      enddo
      call gpshmem_barrier_all()
      if (me == 0) print *, "Testing put..."
      call gpshmem_put(gps_int(indextgt), gps_int(indexsrc), N, nxt);
      call gpshmem_barrier_all()
      call gpshmem_fence()
      call checkarray(gps_int(indextgt), N, prv, me)
			
! free up the memory
      call gpshmem_barrier_all()
      call gpshfree_handle(handletgt)
      call gpshfree_handle(handlesrc)
       
! we're done
      call gpshmem_finalize()

      end program arraycopy

! Checks the array for expected values
      subroutine checkarray(A, N, val, me)
      integer A(N), bad, val, me

      bad = 0

      do i=1,N
        if (.not. (A(i) == val)) then
          bad = 1
          print *, me, "Index ", i, " failed (got ", A(i), 
     +       ", should be ", val, ")"
        endif
      enddo

      if (bad .eq. 1) then
        print *, me, " Failed!" 
      else 
        print *, me, " Passed!"
      endif

      end subroutine checkarray
      
