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
 *  @file	opari2_directive_pomp.h
 *
 *  @brief	Class definitions for POMP directives.
 */

#ifndef OPARI2_DIRECTIVE_POMP_H
#define OPARI2_DIRECTIVE_POMP_H

#include <string>
using std::string;
#include <iostream>
using std::ostream;
#include <vector>
using std::vector;

#include "opari2.h"
#include "opari2_directive.h"

/** @brief Base Class to store and manipulate  POMP directive and related data. */
class OPARI2_DirectivePomp : public OPARI2_Directive
{
public:
    OPARI2_DirectivePomp( const string&   fname,
                          const int       ln,
                          vector<string>& lines,
                          vector<string>& directive_prefix )
        : OPARI2_Directive( fname, ln, lines, directive_prefix )
    {
        m_type =  OPARI2_PT_POMP;
    }

    virtual
    ~
    OPARI2_DirectivePomp( void )
    {
    }

    /** Increment region counter */
    void
    IncrementRegionCounter( void );

    void
    GenerateDescr( ostream& os );

    /** @brief Generate first lines of "include" directives. */
    static void
    GenerateHeader( ostream& os );

    /** generate call POMP2_Init_reg_XXX function to initialize
     *  all handles, in the Fortran case
     */
    static void
    GenerateInitHandleCalls( ostream&     os,
                             const string incfile = "" );

    virtual void
    FindName( void );

    /** @brief Change the default directive name.  */
    void
    SetName( string name_str );

    void
    FindUserRegionName( void );

    string&
    GetUserRegionName( void );

    /** Parse string and return POMP group id */
    static uint32_t
    String2Group( const string name );

private:
    string m_user_region_name;

    virtual string
    generate_ctc_string( OPARI2_Format_t form );

    static stringstream s_init_handle_calls;
    static string       s_paradigm_prefix;
    static int          s_num_regions;
};
#endif
