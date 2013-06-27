      SUBROUTINE FOO
      INTEGER i
!$POMP INST BEGIN(FOO)      
      WRITE (*,*) 'Hello from FOO.'
!     work is done here
      if (i.eq.0) THEN
!$POMP INST ALTEND(FOO)
              RETURN
      END IF
!     other work is done here
!$POMP INST END(FOO)

      END


      PROGRAM EXAMPLE_USER_INSTRUMENTATION
!$POMP INST INIT
      WRITE (*,*) 'Hello from PROGRAM.'
      CALL FOO()
      END
