! (c) 2011 University of Houston System.  All rights reserved.


!
! rotate PE id to right neighbor (dest), with wrap-around
!
 
program rotput

  integer, save :: dest
  integer src
  integer me, npes
  integer nextpe

  include 'mpp/shmem.fh'

  call start_pes(0)
  me = my_pe()
  npes = num_pes()

  nextpe = MOD(me + 1, npes)

  src = nextpe;

  dest = -1;
  call shmem_barrier_all()

  call shmem_integer_put(dest, src, 1, nextpe)

  call shmem_barrier_all()

  if (dest == me) then
     print *, me, ': got ', dest, 'CORRECT'
  else
     print *, me, ': got ', dest, 'WRONG'
  end if

end program rotput
