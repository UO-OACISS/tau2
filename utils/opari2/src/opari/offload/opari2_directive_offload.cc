/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013, 2014
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

/** @internal
 *
 *  @file      opari2_directive_offload.cc
 *
 *  @brief     Methods of base class for OFFLOAD directives.
 */

#include <config.h>

#include <string>
using std::string;
#include <iostream>
using std::ostream;
#include <assert.h>

#include "common.h"
#include "opari2_directive_offload.h"
#include "opari2_directive_entry_offload.h"


uint32_t
OPARI2_DirectiveOffload::String2Group( const string name )
{
    return 0;
}

void
OPARI2_DirectiveOffload::IncrementRegionCounter( void )
{
    ++s_num_regions;
}

void
OPARI2_DirectiveOffload::GenerateDescr( ostream& os )
{
    if ( s_lang & L_C_OR_CXX )
    {
        OPARI2_Directive::generate_descr_common( os );

        s_init_handle_calls << "    " << s_paradigm_prefix << "_assign_handle( "
                            << "&" << region_id_prefix << m_id << ", "
                            << m_ctc_string_variable << " );\n";
    }
}


void
OPARI2_DirectiveOffload::SetName( string name_str )
{
    m_name = name_str;
}

int OPARI2_DirectiveOffload::         s_num_regions = 0;
stringstream OPARI2_DirectiveOffload::s_init_handle_calls;
string OPARI2_DirectiveOffload::      s_paradigm_prefix = "POFLD";
