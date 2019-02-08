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
 *  @file      opari2_directive_pomp.cc
 *
 *  @brief     Methods of POMP base class.
 */

#include <config.h>

#include <string>
using std::string;
#include <iostream>
using std::ostream;
#include <assert.h>

#include "common.h"
#include "opari2_directive_pomp.h"
#include "opari2_directive_entry_pomp.h"

/** This seems overkill in this case, but it should be consistent with
    the other paradigms*/
typedef struct
{
    uint32_t     mEnum;
    const string mGroupName;
} OPARI2_POMPGroupStringMapEntry;

OPARI2_POMPGroupStringMapEntry pompGroupStringMap[] =
{
    { G_POMP_REGION, "userRegion" },
    { G_POMP_ALL,    "pomp"       },
};

uint32_t
OPARI2_DirectivePomp::String2Group( const string name )
{
    size_t n = sizeof( pompGroupStringMap ) / sizeof( OPARI2_POMPGroupStringMapEntry );

    for ( size_t i = 0; i < n; ++i )
    {
        if ( pompGroupStringMap[ i ].mGroupName.compare( name ) == 0 )
        {
            return pompGroupStringMap[ i ].mEnum;
        }
    }

    return 0;
}

void
OPARI2_DirectivePomp::IncrementRegionCounter( void )
{
    ++s_num_regions;
}

/**
 * @brief Generate CTC string for POMP region.
 */
string
OPARI2_DirectivePomp::generate_ctc_string( OPARI2_Format_t form )
{
    stringstream s;

    if ( m_name == "userRegion" )
    {
        s << "userRegionName=" << m_user_region_name << "*";
    }

    return generate_ctc_string_common( form, s.str() );
}


void
OPARI2_DirectivePomp::GenerateHeader( ostream& os )
{
    if ( s_lang & L_C_OR_CXX )
    {
        if ( !s_preprocessed_file )
        {
            os << "#include <opari2/pomp2_user_lib.h>\n\n";
        }
    }
}


void
OPARI2_DirectivePomp::GenerateDescr( ostream& os )
{
    OPARI2_Directive::generate_descr_common( os );

    if ( s_lang & L_FORTRAN )
    {
        s_init_handle_calls << "         call " << s_paradigm_prefix << "_Assign_handle( "
                            << region_id_prefix << m_id << ", ";
        if ( s_format == F_FIX )
        {
            s_init_handle_calls << "\n     &   ";
        }
        else
        {
            s_init_handle_calls << "&\n         ";
        }

        s_init_handle_calls << m_ctc_string_variable << " )\n";
    }
    else if ( s_lang & L_C_OR_CXX )
    {
        s_init_handle_calls << "    " << s_paradigm_prefix << "_Assign_handle( "
                            << "&" << region_id_prefix << m_id << ", "
                            << m_ctc_string_variable << " );\n";
    }
}

/**
 * @brief Generate a function to allow initialization of all region handles for Fortran.
 *
 * These functions need to be called from POMP2_Init_regions.
 */
void
OPARI2_DirectivePomp::GenerateInitHandleCalls( ostream&     os,
                                               const string incfile )
{
    OPARI2_Directive::GenerateInitHandleCalls( os, incfile, s_paradigm_prefix,
                                               s_init_handle_calls, s_num_regions );
    return;
}


void
OPARI2_DirectivePomp::FindName( void )
{
    find_name_common();

    if ( m_name == "inst" )
    {
        m_name += find_next_word();
    }
}


void
OPARI2_DirectivePomp::SetName( string name_str )
{
    m_name = name_str;
}

void
OPARI2_DirectivePomp::FindUserRegionName( void )
{
    string user_region_name = find_next_word();

    if ( user_region_name == "(" )
    {
        m_user_region_name = find_next_word();
    }
}


string&
OPARI2_DirectivePomp::GetUserRegionName( void )
{
    return m_user_region_name;
}

stringstream OPARI2_DirectivePomp::s_init_handle_calls;
int OPARI2_DirectivePomp::         s_num_regions     = 0;
string OPARI2_DirectivePomp::      s_paradigm_prefix = "POMP2_USER";
