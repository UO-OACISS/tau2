cc TestInit_F
cc --------
cc This file contains code for testing the Fortran interface to just the
cc HPC++ init and exit functions.
cc-----------------------------------------------------------------------------

      program main
      include 'Profile/TauFAPI.h'
        integer group(2), error, x
        integer profiler(2)
	save profiler
cc Remember to 'save' profiler object to ensure static like behavior. This
cc allows name registration overhead to be restricted to the first time 
cc the function is called.

        call TAU_PROFILE_TIMER(profiler, 'main()', '',TAU_DEFAULT)
        call TAU_PROFILE_START(profiler)
        call TAU_PROFILE_SET_NODE(0)

      write(6,*) "hello world"

        do 10, i = 1, 10
        x = HelloWorld(i)
10      continue
        call TAU_PROFILE_STOP(profiler)
      stop
      end

      integer function HelloWorld(iVal)
      include 'Profile/TauFAPI.h'
        integer iVal
        integer profiler(2)
        save    profiler

        call TAU_PROFILE_TIMER(profiler,'HelloWorld()', ' ',TAU_DEFAULT)
        call TAU_PROFILE_START(profiler)
cc Do something here...
 	print *, "Iteration = ", iVal
        call TAU_PROFILE_STOP(profiler)
       HelloWorld = iVal
      end
