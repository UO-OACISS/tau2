cc34567 Cubes program
      PROGRAM SUM_OF_CUBES 
       integer profiler(2) / 0, 0 /
	save profiler
      INTEGER :: H, T, U 
        call TAU_PROFILE_INIT()
        call TAU_PROFILE_TIMER(profiler, 'PROGRAM SUM_OF_CUBES')
        call TAU_PROFILE_START(profiler)
        call TAU_PROFILE_SET_NODE(0)

        call TAU_PROFILE_PARAM_1L('param', 50)
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
