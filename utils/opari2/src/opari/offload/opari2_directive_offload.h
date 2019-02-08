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
 *  @file      opari2_directive_offload.h
 *
 *  @brief Base class for Intel OFFLOAD directives
 */

#ifndef OPARI2_DIRECTIVE_OFFLOAD_H
#define OPARI2_DIRECTIVE_OFFLOAD_H

#include <string>
using std::string;
#include <iostream>
using std::ostream;
#include <vector>
using std::vector;

#include "opari2.h"
#include "opari2_directive.h"

/** @brief Base Class to store and manipulate Intel offload directive and related data. */
class OPARI2_DirectiveOffload : public OPARI2_Directive
{
public:
    OPARI2_DirectiveOffload( const string&   fname,
                             const int       ln,
                             vector<string>& lines,
                             vector<string>& directive_prefix )
        : OPARI2_Directive( fname, ln, lines, directive_prefix )
    {
        m_type =  OPARI2_PT_OFFLOAD;

        if ( directive_prefix.size() )
        {
            string sentinel = directive_prefix[ directive_prefix.size() - 1 ];
            if ( sentinel == "__declspec" )
            {
                SetName( "declspec" );
            }
        }
    }

    virtual
    ~
    OPARI2_DirectiveOffload( void )
    {
    }

    /** Increment region counter */
    void
    IncrementRegionCounter( void );

    void
    SetName( string name_str );

    /** Parse string and return offload group id */
    static uint32_t
    String2Group( const string name );

    virtual void
    GenerateDescr( ostream& os );

private:
    static string       s_paradigm_prefix;
    static stringstream s_init_handle_calls;
    static int          s_num_regions;
};
#endif
