
c---------------------------------------------------------------------
c---------------------------------------------------------------------

      integer function nodedim(num)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c
c  compute the exponent where num = 2**nodedim
c  NOTE: assumes a power-of-two number of nodes
c
c---------------------------------------------------------------------

      implicit none

c---------------------------------------------------------------------
c  input parameters
c---------------------------------------------------------------------
      integer num

c---------------------------------------------------------------------
c  local variables
c---------------------------------------------------------------------
      integer profiler(2)
      save profiler
      double precision fnum


      call TAU_PROFILE_TIMER(profiler, 'nodedim');
      call TAU_PROFILE_START(profiler);
      fnum = dble(num)
      nodedim = log(fnum)/log(2.0d+0) + 0.00001

      call TAU_PROFILE_STOP(profiler);
      return
      end


