      SUBROUTINE FUNC(me, proc)
      INCLUDE 'mpif.h'
      INTEGER i, proc, me, err
      INTEGER f(10)
      INTEGER s(MPI_STATUS_SIZE)

      WRITE(*,*) me, 'started.'
      DO I=1,10
        f(i) = i
      ENDDO

      CALL MPI_Barrier(MPI_COMM_WORLD, err)

      IF (me .EQ. 0) THEN
        CALL MPI_Send(f, 10, MPI_INTEGER, 1, 4711, MPI_COMM_WORLD, err)
        CALL MPI_Recv(f, 10, MPI_INTEGER, proc-1, 4711, MPI_COMM_WORLD, s, err)
      ELSE
        CALL MPI_Recv(f, 10, MPI_INTEGER, me-1, 4711, MPI_COMM_WORLD, s, err)
        IF (me .EQ. proc-1) THEN
          CALL MPI_Send(f, 10, MPI_INTEGER, 0, 4711, MPI_COMM_WORLD, err)
        ELSE
          CALL MPI_Send(f, 10, MPI_INTEGER, me+1, 4711, MPI_COMM_WORLD, err)
        ENDIF
      ENDIF
      CALL MPI_Bcast (f, 10, MPI_INTEGER, 0, MPI_COMM_WORLD, err)
      WRITE(*,*) me, 'done.'
      RETURN
      END SUBROUTINE FUNC

      PROGRAM main
      INCLUDE 'mpif.h'

      INTEGER i, proc, me, err

      CALL MPI_Init (err)
      CALL MPI_Comm_size (MPI_COMM_WORLD, proc, err)
      CALL MPI_Comm_rank (MPI_COMM_WORLD, me, err)

      !CALL FUNC(me, proc)
      CALL MPI_Finalize (err)
      END
