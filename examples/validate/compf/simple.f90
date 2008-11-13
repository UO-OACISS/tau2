      PROGRAM SUM_OF_CUBES 
      INTEGER :: H, T, U 
      CALL TAU_PROFILE_SET_NODE(0)

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
      END PROGRAM SUM_OF_CUBES
