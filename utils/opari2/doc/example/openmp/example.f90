!>
!> This file is part of the Score-P software (http://www.score-p.org)
!>
!> Copyright (c) 2009-2011,
!> RWTH Aachen University, Germany
!>
!> Copyright (c) 2009-2011,
!> Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
!>
!> Copyright (c) 2009-2011,
!> Technische Universitaet Dresden, Germany
!>
!> Copyright (c) 2009-2011,
!> University of Oregon, Eugene, USA
!>
!> Copyright (c) 2009-2011, 2013
!> Forschungszentrum Juelich GmbH, Germany
!>
!> Copyright (c) 2009-2011,
!> German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
!>
!> Copyright (c) 2009-2011,
!> Technische Universitaet Muenchen, Germany
!>
!> This software may be modified and distributed under the terms of
!> a BSD-style license. See the COPYING file in the package base
!> directory for details.

      PROGRAM EXAMPLE

      INTEGER i, lsum, sum

      sum = 0

!$omp parallel private(i, lsum) reduction(+:sum)
      lsum = 0

!$omp do
      do i=1,20
         lsum = lsum + i
      enddo
!$omp end do

      write(*,*) "LOCAL SUM: ", lsum
      sum = sum + lsum
!$omp end parallel

      write(*,*) "TOTAL SUM: ", sum

      END
