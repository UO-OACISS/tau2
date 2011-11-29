!>
!> This file is part of the Score-P software (http://www.score-p.org)
!>
!> Copyright (c) 2009-2011,
!>    RWTH Aachen University, Germany
!>    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
!>    Technische Universitaet Dresden, Germany
!>    University of Oregon, Eugene, USA
!>    Forschungszentrum Juelich GmbH, Germany
!>    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
!>    Technische Universitaet Muenchen, Germany
!>
!> See the COPYING file in the package base directory for details. <!

!>  @internal
!>
!>  @file       getname.f
!>  @status     alpha
!>
!>  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
!>
!>  @brief      This file is needed to check the Fortran name mangling of
!>              the used compiler. foo_foo is called and depending on
!               the mangling, this is linked against a c function
!               foo_foo_, _foo_foo, ...<!
      program getfname
      call foo_foo()
      end
