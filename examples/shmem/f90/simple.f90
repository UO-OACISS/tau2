	PROGRAM REDUCTION
               INCLUDE 'shmem.fh'
               REAL VALUES, SUM
               COMMON /C/ VALUES
               REAL WORK
               CALL START_PES(0)
               VALUES = SHMEM_MY_PE()
               CALL SHMEM_BARRIER_ALL                  ! Synchronize all PEs
               SUM = 0.0
               DO I = 0,SHMEM_N_PES()-1
                  CALL SHMEM_GET(WORK, VALUES, 1, I)   ! Get next value
                  SUM = SUM + WORK                     ! Sum it
               ENDDO
               PRINT*,'PE ',SHMEM_MY_PE(),' COMPUTED       SUM=',SUM
               CALL SHMEM_BARRIER_ALL
               CALL SHMEM_FINALIZE
       END

