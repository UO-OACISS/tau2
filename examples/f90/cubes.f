cc34567 Cubes program
      PROGRAM SUM_OF_CUBES 
      include 'Profile/TauFAPI.h'
       integer profiler(2)
	save profiler
      INTEGER :: H, T, U 
        call TAU_PROFILE_INIT()
        call TAU_PROFILE_TIMER(profiler, 'PROGRAM SUM_OF_CUBES', 20,'', 0, TAU_DEFAULT)
     c  TAU_DEFAULT) 
cc        call TAU_PROFILE_TIMER(profiler, 'main()', 6,'i',1,TAU_DEFAULT)
        call TAU_PROFILE_START(profiler)
        call TAU_PROFILE_SET_NODE(0)
      ! This program prints all 3-digit numbers that 
      ! equal the sum of the cubes of their digits. 
      DO H = 1, 9 
        DO T = 0, 9 
          DO U = 0, 9 
          IF (100*H + 10*T + U == H**3 + T**3 + U**3) THEN
             PRINT "(3I1)", H, T, U 
	  ENDIF
          END DO 
        END DO 
      END DO 
        call TAU_PROFILE_STOP(profiler)
      END PROGRAM SUM_OF_CUBES
