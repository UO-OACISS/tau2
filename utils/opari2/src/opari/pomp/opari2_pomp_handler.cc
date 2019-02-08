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
 *  @file      opari2_pomp_handler.cc
 *
 *  @brief     This file contains all handler funtions to instrument and print POMP directives.
 */


#include <config.h>
#include <iostream>
using std::ostream;
using std::cerr;

#include "common.h"
#include "opari2.h"
#include "opari2_directive.h"
#include "opari2_directive_pomp.h"
#include "opari2_directive_manager.h"

extern OPARI2_Option_t opt;

OPARI2_DirectivePomp*
cast2pomp( OPARI2_Directive* d_base )
{
    OPARI2_DirectivePomp* d = dynamic_cast<OPARI2_DirectivePomp*>( d_base );

    if ( d == NULL )
    {
        cerr << "INTERNAL ERROR: OPARI2_DirectivePomp* expected. Please contact user support.\n";
        cleanup_and_exit();
    }

    return d;
}

void
h_pomp_inst( OPARI2_Directive* d_base,
             ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );

    string& name = d->GetName();
    char    c1   = toupper( name.substr( 4 )[ 0 ] );

    if ( opt.lang & L_FORTRAN )
    {
        os << "      call POMP2_" << c1 << name.substr( 5 ) << "()\n";
    }
    else if ( opt.lang & L_C_OR_CXX )
    {
        os << "POMP2_" << c1 << name.substr( 5 ) << "();\n";
    }
    if ( opt.keep_src_info )
    {
        d->ResetSourceInfo( os );
    }
}

/**
 * @brief handling "pomp inst init".
 */
void
h_pomp_instinit( OPARI2_Directive* d_base,
                 ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );

    h_pomp_inst( d, os );
}


/**
 * @brief handling "pomp inst finalize".
 */
void
h_pomp_instfinalize( OPARI2_Directive* d_base,
                     ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );

    h_pomp_inst( d, os );
}


/**
 * @brief handling "pomp inst on".
 */
void
h_pomp_inston( OPARI2_Directive* d_base,
               ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );

    h_pomp_inst( d, os );
}


/**
 * @brief handling "pomp inst off".
 */
void
h_pomp_instoff( OPARI2_Directive* d_base,
                ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );

    h_pomp_inst( d, os );
}



/**
 * @brief handling "pomp instrument" .
 */
void
h_pomp_instrument( OPARI2_Directive* d_base,
                   ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );

    EnableInstrumentation( D_USER );
    if ( opt.keep_src_info )
    {
        d->ResetSourceInfo( os );
    }
}

/**
 * @brief handling "pomp noinstrument" .
 */
void
h_pomp_noinstrument( OPARI2_Directive* d_base,
                     ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );

    DisableInstrumentation( D_USER );
    if ( opt.keep_src_info )
    {
        d->ResetSourceInfo( os );
    }
}


/**
 * @brief handling "pomp inst begin" .
 */
void
h_pomp_instbegin( OPARI2_Directive* d_base,
                  ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );

    d->EnterRegion();
    int id = d->GetID();

    d->SetName( "userRegion" );
    d->FindUserRegionName();

    if ( opt.lang & L_FORTRAN )
    {
        os << "      call POMP2_Begin(" << region_id_prefix << id;
        os << ", " << d->GetCTCStringVariable() << ")\n";
    }
    else if ( opt.lang & L_C_OR_CXX )
    {
        os << "POMP2_Begin(&" << region_id_prefix << id;
        os << ", " << d->GetCTCStringVariable() << ");\n";
    }
    if ( opt.keep_src_info )
    {
        d->ResetSourceInfo( os );
    }
}


/**
 * @brief handling "pomp inst altend" .
 */
void
h_pomp_instaltend( OPARI2_Directive* d_base,
                   ostream&          os )
{
    OPARI2_DirectivePomp* d     = cast2pomp( d_base );
    OPARI2_DirectivePomp* d_top = cast2pomp( DirectiveStackTop( d ) );

    d->FindUserRegionName();
    string& subname     = d->GetUserRegionName();
    string& subname_top = d_top->GetUserRegionName();
    if ( subname != subname_top  )
    {
        string& filename = d->GetFilename();
        cerr << filename << ":" << d_top->GetLineno()
             << ": ERROR: missing inst end(" << subname_top
             << ") pragma/directive\n";
        cerr << filename << ":" << d->GetLineno()
             << ": ERROR: non-matching inst end(" << subname
             << ") pragma/directive\n";
        cleanup_and_exit();
    }

    if ( opt.lang & L_FORTRAN )
    {
        os << "      call POMP2_End(" << region_id_prefix << d_top->GetID() << ")\n";
    }
    else if ( opt.lang & L_C_OR_CXX )
    {
        os << "POMP2_End(&" << region_id_prefix << d_top->GetID() << ");\n";
    }

    if ( opt.keep_src_info )
    {
        d->ResetSourceInfo( os );
    }
}

/**
 * @brief handling "pomp inst end" .
 */
void
h_pomp_instend( OPARI2_Directive* d_base,
                ostream&          os )
{
    OPARI2_DirectivePomp* d = cast2pomp( d_base );
    // change the directive/region name explicity.
    //d->SetRegionName( "endregion" );
    d->SetName( "enduserRegion" );
    d->FindUserRegionName();
    OPARI2_DirectivePomp* d_top       = cast2pomp( DirectiveStackTop( d ) );
    string&               subname     = d->GetUserRegionName();
    string&               subname_top = d_top->GetUserRegionName();
    int                   id          = d->ExitRegion( false );

    if ( subname != subname_top  )
    {
        string& filename = d->GetFilename();
        cerr << filename << ":" << d_top->GetLineno()
             << ": ERROR: missing inst end(" << subname_top
             << ") pragma/directive\n";
        cerr << filename << ":" << d->GetLineno()
             << ": ERROR: non-matching inst end(" << subname
             << ") pragma/directive\n";
        cleanup_and_exit();
    }
    if ( opt.lang & L_FORTRAN )
    {
        os << "      call POMP2_End(" << region_id_prefix << id << ")\n";
    }
    else if ( opt.lang & L_C_OR_CXX )
    {
        os << "POMP2_End(&" << region_id_prefix << id << ");\n";
    }
    if ( opt.keep_src_info )
    {
        d->ResetSourceInfo( os );
    }
}
