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

!>  @internal
!>
!>  @file       getname.f
!>
!>  @brief      This file is needed to check the Fortran name mangling of
!>              the used compiler. foo_foo is called and depending on
!               the mangling, this is linked against a c function
!               foo_foo_, _foo_foo, ...<!
      program getfname
      call foo_foo()
      end
