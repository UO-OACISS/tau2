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
! * @authors Jie Jiang 
! *
! * @brief Test worskshare directives.

program test11

  !$OMP critical
    i = 2
  !$omp end critical

  !$OMP atomic update
    i = 3

  !$OMP workshare 
    i = 4
  !$OMP critical
    i = 5
  !$omp end critical
  !$OMP atomic update
    i = 6
  !$OMP end workshare

  !$OMP PARALLEL WORKSHARE
    i =  7
  !$OMP critical
    i = 8
  !$omp end critical
  !$OMP atomic update
    i = 9
  !$OMP END PARALLEL WORKSHARE
end program test11
