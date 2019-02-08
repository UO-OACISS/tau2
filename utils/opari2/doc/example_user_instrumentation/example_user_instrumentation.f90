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
