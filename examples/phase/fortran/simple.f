cc simple.f
cc --------
cc This file contains sample code for phased based profiling with TAU
cc-----------------------------------------------------------------------------


      subroutine F1()
        integer profiler(2) / 0, 0 /
        save    profiler

        call TAU_PROFILE_TIMER(profiler,'f1()')
        call TAU_PROFILE_START(profiler)
        call SLEEP(1)
        call TAU_PROFILE_STOP(profiler)
      end

      subroutine INPUT()
        integer profiler(2) / 0, 0 /
        save    profiler

        call TAU_PROFILE_TIMER(profiler,'input')
        call TAU_PROFILE_START(profiler)
        call SLEEP(1)
        call TAU_PROFILE_STOP(profiler)
      end

      subroutine OUTPUT()
        integer profiler(2) / 0, 0 /
        save    profiler

        call TAU_PROFILE_TIMER(profiler,'output')
        call TAU_PROFILE_START(profiler)
        call SLEEP(1)
        call TAU_PROFILE_STOP(profiler)
      end


      subroutine F2()
        integer profiler(2) / 0, 0 /
        save    profiler

        integer phase(2) / 0, 0 /
        save    phase

        call TAU_PROFILE_TIMER(profiler,'f2()')
        call TAU_PROFILE_START(profiler)

        call TAU_PHASE_CREATE_STATIC(phase,'IO Phase')
        call TAU_PHASE_START(phase)

        call INPUT()
        call OUTPUT()
        
        call TAU_PHASE_STOP(phase)

        call TAU_PROFILE_STOP(profiler)
      end


      subroutine F3(val)
        integer val
        integer profiler(2) / 0, 0 /
        save    profiler

        call TAU_PROFILE_TIMER(profiler,'f3()')
        call TAU_PROFILE_START(profiler)
        call SLEEP(val)
        call TAU_PROFILE_STOP(profiler)
      end


      subroutine F4()
        integer profiler(2) / 0, 0 /
        save    profiler

        call TAU_PROFILE_TIMER(profiler,'f4()')
        call TAU_PROFILE_START(profiler)
        call SLEEP(1)
        call TAU_PROFILE_STOP(profiler)
      end


      subroutine ITERATION(val)
        integer val
        character(13) cvar 
        integer profiler(2) / 0, 0 /
        save profiler

        print *, "Iteration ", val

        write (cvar,'(a9,i2)') 'Iteration', val
        call TAU_PHASE_CREATE_DYNAMIC(profiler, cvar)
        call TAU_PHASE_START(profiler)

        call F1()
        call F2()
        call F3(val)
        call TAU_PHASE_STOP(profiler)
        return
      end 

      program main
        integer i
        integer profiler(2) / 0, 0 /
        save    profiler

        call TAU_PROFILE_INIT()
        call TAU_PROFILE_TIMER(profiler, 'main()')
        call TAU_PROFILE_START(profiler)
        call TAU_PROFILE_SET_NODE(0)


        do 10, i = 0, 4
        call ITERATION(i)
10      continue


        call F1()
        call F4()
        call TAU_PROFILE_STOP(profiler)
      end

