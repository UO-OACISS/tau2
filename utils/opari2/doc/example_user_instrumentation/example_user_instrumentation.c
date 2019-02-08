/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2011,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2011,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2011, 2013
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2011,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

#include <stdio.h>
/*Disable unknown pragma warning for the intel compiler.
 * This avoids warnings for pomp pragmas if the file is
 * not preprocessed with opari.*/
#ifdef __INTEL_COMPILER
#pragma warning disable 161
#endif

int
foo()
{
    int i = 0;
#pragma pomp inst begin(foo)
    //usefull work could be done here which changes i
    printf( "Hello from foo.\n" );
    if ( i == 0 )
    {
#pragma pomp inst altend(foo)
        return 42;
    }
    //other work might be done here
#pragma pomp inst end(foo)
    return i;
}

int
main( int argc, char** argv )
{
#pragma pomp inst init
    printf( "Hello from main.\n" );
    foo();
    return 0;
}
