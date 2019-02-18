! * This file is part of the Score-P software (http://www.score-p.org)
! *
! * Copyright (c) 2009-2011,
! *    RWTH Aachen University, Germany
! *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
! *    Technische Universitaet Dresden, Germany
! *    University of Oregon, Eugene, USA
! *    Forschungszentrum Juelich GmbH, Germany
! *    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
! *    Technische Universitaet Muenchen, Germany
! *
! * See the COPYING file in the package base directory for details.
! *
! * Testfile for automated testing of OPARI2
! *
! *
! * @brief Tests user instrumentation directives.

program test7
  !$POMP INST INIT

  !$POMP INST OFF

  !$POMP INST BEGIN(foo)

  !$OMP PARALLEL
  i = 1
  !$OMP end PARALLEL

  !$POMP INST END(foo)

  !$POMP INST ON

  !$POMP NOINSTRUMENT
  !$OMP PARALLEL
  !$OMP DO
  do i = 1,2
     write(*,*) i
  end do
  !$OMP END DO
  !$OMP END PARALLEL

  !$OMP PARALLEL DO
  do i = 1,2
     write(*,*) i
  end do
  !$OMP end PARALLEL DO
  !$POMP INSTRUMENT

  !$OMP PARALLEL
  i = 3
  !$OMP end PARALLEL

  !$POMP INST FINALIZE
end program test7
