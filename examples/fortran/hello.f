cc TestInit_F
cc --------
cc This file contains code for testing the Fortran interface to just the
cc HPC++ init and exit functions.
cc-----------------------------------------------------------------------------

      subroutine HELLOWORLD(iVal)
      include 'Profile/TauFAPI.h'
        integer iVal
        integer profiler(2)
        save    profiler

        call TAU_PROFILE_TIMER(profiler,'HelloWorld()', 12, '', 0, 
     c TAU_DEFAULT)
        call TAU_PROFILE_START(profiler)
cc Do something here...
 	print *, "Iteration = ", iVal
        call TAU_PROFILE_STOP(profiler)
cc       HelloWorld = iVal
      end

      program main
      include 'Profile/TauFAPI.h'
        integer group(2), error, x, i
        integer profiler(2)
	save profiler
        call TAU_PROFILE_INIT()
        call TAU_PROFILE_TIMER(profiler, 'main()', 6,'', 0, TAU_DEFAULT)
        call TAU_PROFILE_START(profiler)
        call TAU_PROFILE_SET_NODE(0)

      write(6,*) "hello world"

        do 10, i = 1, 10
        call HELLOWORLD(i)
10      continue
        call TAU_PROFILE_STOP(profiler)
      end

