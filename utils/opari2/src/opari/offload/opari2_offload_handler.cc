/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

/** @internal
 *
 *  @file      opari2_offload_handler.cc
 *
 *  @brief     Handler functions for Intel OFFLOAD directives.
 */

#include <config.h>

#include "opari2.h"
#include "opari2_directive.h"
#include "opari2_directive_manager.h"

void
h_offload_target( OPARI2_Directive* d,
                  ostream&          os )
{
    DirectiveStackPush( d );
    d->PrintPlainDirective( os );
    DisableInstrumentation( D_FULL );
}

void
h_end_offload_target( OPARI2_Directive* d,
                      ostream&          os )
{
    DirectiveStackPop();
    EnableInstrumentation( D_FULL );
}

void
h_offload_declspec( OPARI2_Directive* d,
                    ostream&          os )
{
    DisableInstrumentation( D_FULL );
    // Only go to the directive stack, not the vector
    DirectiveStackPush( d );
}

void
h_end_offload_declspec( OPARI2_Directive* d,
                        ostream&          os )
{
    EnableInstrumentation( D_FULL );
    // maintain stack
    DirectiveStackPop();
}
