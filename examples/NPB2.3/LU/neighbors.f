
c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine neighbors ()

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

      include 'applu.incl'

      integer profiler(2)
      save profiler
      call TAU_PROFILE_TIMER(profiler, 'neighbors');
      call TAU_PROFILE_START(profiler);
c---------------------------------------------------------------------
c     figure out the neighbors and their wrap numbers for each processor
c---------------------------------------------------------------------

        south = -1
        east  = -1
        north = -1
        west  = -1

      if (row.gt.1) then
              north = id -1
      else
              north = -1
      end if

      if (row.lt.xdim) then
              south = id + 1
      else
              south = -1
      end if

      if (col.gt.1) then
              west = id- xdim
      else
              west = -1
      end if

      if (col.lt.ydim) then
              east = id + xdim
      else 
              east = -1
      end if

      call TAU_PROFILE_STOP(profiler);
      return
      end
