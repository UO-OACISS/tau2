/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013, 2014, 2016,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

/** @internal
 *
 *  @file      opari2_directive_manager.cc
 *
 *  @brief     This file contains APIs to disable, process directive
 *             and runtime entries in a separate namespace.
 */

#include <config.h>
#include <iostream>
using std::cerr;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <map>
using std::map;
#include <stack>
using std::stack;
#include <stdlib.h>
#include <cctype>
using std::toupper;


#include "opari2.h"
#include "opari2_directive_definition.h"
#include "opari2_directive_manager.h"

#include "openmp/opari2_omp_handler.h"
#include "openmp/opari2_directive_openmp.h"
#include "openmp/opari2_directive_entry_openmp.h"

#include "pomp/opari2_pomp_handler.h"
#include "pomp/opari2_directive_pomp.h"
#include "pomp/opari2_directive_entry_pomp.h"

#include "offload/opari2_offload_handler.h"
#include "offload/opari2_directive_offload.h"
#include "offload/opari2_directive_entry_offload.h"

static const OPARI2_MapString2ParadigmNameType paradigm_identifiers[] =
{
    OPARI2_OPENMP_SENTINELS,
    OPARI2_POMP2_USER_SENTINELS,
    OPARI2_OFFLOAD_SENTINELS
};

/**
 * @brief Array holding all definitions for all supported directives
 *        of all supported paradigms.
 */
OPARI2_DirectiveDefinition directive_table[] =
{
    OPARI2_OPENMP_DIRECTIVE_ENTRIES,
    OPARI2_POMP_DIRECTIVE_ENTRIES,
    //OPARI2_OPENACC_DIRECTIVE_ENTRIES,
    //OPARI2_TMSE_DIRECTIVE_ENTRIES,
    OPARI2_OFFLOAD_DIRECTIVE_ENTRIES
};

/**
 * @brief Array holding all definitions for all supported API
 *        functions of all supported paradigms.
 */
OPARI2_RuntimeAPIDefinition api_table[] =
{
    OPARI2_OPENMP_API_ENTRIES
    //OPARI2_OPENACC_API_ENTRIES
};


OPARI2_Directive* saved_single_line_directive = NULL;


/**
 * @brief Convert paradigm string to ParadigmType.
 */
OPARI2_ParadigmType_t
string_to_paradigm_type( const string& str )
{
    int n = sizeof( paradigm_identifiers ) / sizeof( OPARI2_MapString2ParadigmNameType );

    for ( int i = 0; i < n; i++ )
    {
        if ( str.compare( paradigm_identifiers[ i ].mString ) == 0 )
        //      if( str == paradigm_identifiers[ i ].mString )
        {
            return ( OPARI2_ParadigmType_t )paradigm_identifiers[ i ].mEnum;
        }
    }

    return OPARI2_PT_NONE;
}


string
paradigm_type_to_string( OPARI2_ParadigmType_t type )
{
    string str;

    switch ( type )
    {
        case OPARI2_PT_OMP:
            str = "OpenMP";
            break;
        case OPARI2_PT_POMP:
            str = "POMP";
            break;
        case OPARI2_PT_OPENACC:
            str =  "OpenACC";
            break;
        case OPARI2_PT_OFFLOAD:
            str = "Offload";
            break;
        case OPARI2_PT_TMSE:
            str = "TMSE";
            break;
        default:
            str = "Unknown paradigm type";
            break;
    }

    return str;
}

/**
 * @brief Convert group name to group number (uint64_t).
 *
 * In case of the same group names in different paradigm,
 * ParadigmType is used jointly to get a correct group number.
 *
 * @param [in] type		paradigm type
 * @param [in] str		group name in string
 *
 * @return				G_NONE:	conversion failed
 *						others: OK
 */
uint64_t
string_to_group( OPARI2_ParadigmType_t type,
                 const string&         str )
{
    switch ( type )
    {
        case OPARI2_PT_OMP:
            return OPARI2_DirectiveOpenmp::String2Group( str );
        case OPARI2_PT_POMP:
            return OPARI2_DirectivePomp::String2Group( str );
        case OPARI2_PT_OFFLOAD:
            return OPARI2_DirectiveOffload::String2Group( str );
        case OPARI2_PT_TMSE:
        default:
            return G_OMP_NONE;
    }
}

vector<OPARI2_Directive*> tmp_directives;
vector<OPARI2_Directive*> directive_vec;
vector<OPARI2_Directive*> directive_stack;
/**
 * Keep track of the paradigm used in the source file.
 * Can be 'OR'ed with multiple paradigm types
 */
uint32_t instrumented_paradigm_type = 0;
bool     pomp2_header_included      = false;


bool
DisableParadigmDirectiveOrGroup(  const string& paradigm,
                                  const string& directiveOrGroup,
                                  bool          inner )
{
    OPARI2_ParadigmType_t type = string_to_paradigm_type( paradigm );

    if ( type == OPARI2_PT_NONE )
    {
        cerr << "Unknown paradigm " << paradigm << std::endl;
        return false;
    }

    bool                disableWholeParadigm = directiveOrGroup.compare( "" ) == 0;
    OPARI2_ParadigmType par_type             = string_to_paradigm_type( paradigm );
    uint64_t            group                = string_to_group( par_type, directiveOrGroup );

    if ( !disableWholeParadigm && group == 0  && !IsValidDirective( paradigm, ( string& )directiveOrGroup ) )
    {
        cerr << "Unknown group or directive: " << directiveOrGroup << std::endl;
        return false;
    }

    int size = sizeof( directive_table ) / sizeof( OPARI2_DirectiveDefinition );
    for ( int i = 0; i < size; i++ )
    {
        if ( type == directive_table[ i ].type
             && ( ( disableWholeParadigm
                    && directive_table[ i ].disable_with_paradigm )
                  || directiveOrGroup.compare( directive_table[ i ].name ) == 0
                  || directive_table[ i ].group & group ) )
        {
            //            printf( "Disabling directive: %s:%s inner:%d\n", paradigm.c_str(), directive_table[ i ].name.c_str(), inner );
            if ( inner )
            {
                directive_table[ i ].inner_active = false;
            }
            else
            {
                directive_table[ i ].active = false;
            }
        }
    }

    size = sizeof( api_table ) / sizeof( OPARI2_RuntimeAPIDefinition );
    for ( int i = 0; i < size; i++ )
    {
        if ( type == api_table[ i ].type
             && ( disableWholeParadigm
                  || api_table[ i ].group & group ) )
        {
            api_table[ i ].active = false;
        }
    }
    return true;
}

typedef pair<OPARI2_Disable_level_t, OPARI2_Disable_level_t> OPARI2_Disable_level_cur_max_t;
static stack<OPARI2_Disable_level_cur_max_t>
disable_levels( std::deque<OPARI2_Disable_level_cur_max_t>( 1, OPARI2_Disable_level_cur_max_t( D_NONE, D_NONE ) ) );

void
DisableInstrumentation( OPARI2_Disable_level_t l )
{
    OPARI2_Disable_level_cur_max_t level;

    level.first  = l;
    level.second = ( OPARI2_Disable_level_t )( l | disable_levels.top().second );

    disable_levels.push( level );
}

void
EnableInstrumentation( OPARI2_Disable_level_t l )
{
    disable_levels.pop();

    if ( disable_levels.empty() )
    {
        cerr << "Invalid nesting of regions, disabling certain instrumentation. "
             << "Check usage of pomp noinstrument/instrument directives." << std::endl;
        cleanup_and_exit();
    }
}

bool
InstrumentationDisabled( OPARI2_Disable_level_t l )
{
    /* This works for now (only one group and one sublevel */
    return disable_levels.top().second != D_NONE && l <=  disable_levels.top().second;

    /* If more fields are added it might be changed to something like
       the expression below */

    // return ( disable_levels.top().second != D_NONE &&
    //          ( ( l <  disable_levels.top().second && l | disable_levels.top().second ) ||
    //            ( l == disable_levels.top().second ) ) );
}

bool
IsValidSentinel( const string&     lowline,
                 string::size_type p,
                 string&           sentinel )
{
    int n = sizeof( paradigm_identifiers ) / sizeof( OPARI2_MapString2ParadigmNameType );
    for ( int i = 0; i < n; i++ )
    {
        if ( lowline.find( paradigm_identifiers[ i ].mString ) == p )
        {
            sentinel = paradigm_identifiers[ i ].mString;
            return true;
        }
    }
    return false;
}

bool
IsValidDirective( const string& paradigm,
                  string&       directive )
{
    OPARI2_ParadigmType_t type = string_to_paradigm_type( paradigm );
    int                   size = sizeof( directive_table ) / sizeof( OPARI2_DirectiveDefinition );

    for ( int i = 0; i < size; i++ )
    {
        if ( ( type == directive_table[ i ].type )  &&
             ( directive.compare( directive_table[ i ].name ) == 0 ) )
        {
            return true;
        }
    }

    return false;
}

/**
 * @brief	Check if a given paradigm/directive is enabled.
 *
 * @param	type            name of the paradigm type.
 * @param	directive	name of the directive.
 *
 * @return	true	if the given directive is found in the "directive_table" and enabled.
                        false	otherwise.
 */
bool
directive_enabled( OPARI2_ParadigmType_t type,
                   string&               directive )
{
    int size = sizeof( directive_table ) / sizeof( OPARI2_DirectiveDefinition );

    for ( int i = 0; i < size; i++ )
    {
        if ( type == directive_table[ i ].type &&
             ( directive.compare( directive_table[ i ].name ) == 0  ||
               directive.compare( directive_table[ i ].end_name ) == 0 ) )
        {
            // We know the directive.
            // It might be disabled, indicated by the active flag.
            // The active flag is evaluated in ProcessDirective() and
            // HandleSingleLineDirective(); if inactive, the directive is printed
            // preceded by a #line directive. IMO (CF) the #line directive makes
            // no sense but if it is omitted, make check fails. Therefore just
            // return true instead of directive_table[ i ].active.
            return true;
        }
    }
    // We don't know the directive.
    return false;
}


OPARI2_DirectiveDefinition*
get_directive_table_entry( OPARI2_ParadigmType_t type,
                           const std::string&    directive )
{
    int size = sizeof( directive_table ) / sizeof( OPARI2_DirectiveDefinition );

    for ( int i = 0; i < size; i++ )
    {
        if ( type == directive_table[ i ].type &&
             directive.compare( directive_table[ i ].name ) == 0 )
        {
            return ( OPARI2_DirectiveDefinition* )( &( directive_table[ i ] ) );
        }
    }
    return NULL;
}


/**
 * @brief Given a OPARI2_Directive object, get the matching entry in
 *        the directive_table.
 *
 * The entry is matching if both the type and name are the same as the
 * object.
 */
OPARI2_DirectiveDefinition*
get_directive_table_entry( OPARI2_Directive* d )
{
    return get_directive_table_entry( d->GetParadigmType(), d->GetName() );
}


bool
DirectiveActive( OPARI2_ParadigmType_t type,
                 const std::string&    directive )
{
    OPARI2_DirectiveDefinition* d = get_directive_table_entry( type, directive );
    if ( d )
    {
        return d->active;
    }
    return false;
}


OPARI2_Directive*
NewDirective( vector<string>&   lines,
              vector<string>&   directive_prefix,
              OPARI2_Language_t lang,
              const string&     file,
              const int         lineno )
{
    OPARI2_ParadigmType_t type          = OPARI2_PT_NONE;
    OPARI2_Directive*     new_directive = NULL;
    string                sentinel;

    if ( directive_prefix.size() )
    {
        sentinel = directive_prefix[ directive_prefix.size() - 1 ];
        type     = string_to_paradigm_type( sentinel );
    }
    else if ( lang & L_C_OR_CXX )
    {
        /* If the directive_prefix is not provided, this is an
         * automatically generated end region directive */
        type = DirectiveStackTop()->GetParadigmType();
    }
    else
    {
        cerr << "ERROR: Undefined directive type!\n";
        cleanup_and_exit();
    }

    OPARI2_DirectiveDefinition* d_def;
    switch ( type )
    {
        case OPARI2_PT_OMP:
            new_directive = new OPARI2_DirectiveOpenmp( file, lineno,
                                                        lines, directive_prefix );

            /*  Needed for Fortran end do loop detection */
            d_def = get_directive_table_entry( new_directive );
            if ( d_def )
            {
                new_directive->NeedsEndLoopDirective( d_def->loop_block );
            }

            break;
        case OPARI2_PT_POMP:
            new_directive = new OPARI2_DirectivePomp( file, lineno,
                                                      lines, directive_prefix );
            break;
        case OPARI2_PT_OFFLOAD:
            new_directive = new OPARI2_DirectiveOffload( file, lineno,
                                                         lines, directive_prefix );
            break;
        /*no directives supported so far for openacc and tmse*/
        case OPARI2_PT_NONE:
        case OPARI2_PT_OPENACC:
        case OPARI2_PT_TMSE:
            break;
    }
    if ( new_directive )
    {
        if ( directive_enabled( type, new_directive->GetName() ) ||
             new_directive->GetName() == "$END$" )
        {
            tmp_directives.push_back( new_directive );
            return new_directive;
        }
        delete new_directive;
    }
    return NULL;
}


void
ProcessDirective( OPARI2_Directive* d,
                  ostream&          os,
                  bool*             require_end,
                  bool*             is_for )
{
    OPARI2_DirectiveDefinition* d_def = NULL;
    string                      name  = d->GetName();
    //std::cout << "Processing " << name << std::endl;
    if ( name == "$END$" ||                              // end of a directive (block) in C/C++
         ( name.find( "end" ) != string::npos &&         // end of a directive (block) in Fortran
           name != "instaltend" && name != "instend" ) ) // except for POMP directive "inst altend/end"
    {
        // get the directive object at the top of the stack
        OPARI2_Directive* d_top = DirectiveStackTop( d );
        if ( d->GetParadigmType() != d_top->GetParadigmType() )
        {
            cerr << "\nWrong directive type for 'END' processing!\n";
        }

        d_def =  get_directive_table_entry( d_top );
        if ( d_def )
        {
            d->active = d_def->active;
        }
        else
        {
            d->active = true;
        }
        if ( d_def && d_def->active ) // enabled
        {
            if ( d_def->do_exit_transformation )
            {
                d_def->do_exit_transformation( d, os );
            }
        }
        else    //disabled
        {
            if ( d_top->GetName() == "parallelfor" || d_top->GetName() == "paralleldo" || d_top->GetName() == "parallelsections" )
            {
                d_def->do_exit_transformation( d, os );
            }
            else
            {
                //simply output lines without any modification
                d->PrintDirective( os );
            }
            // maintain the directive stack
            if ( d_def->require_end )
            {
                DirectiveStackPop();
            }
        }
    }
    else     // beginning of a directive (block)
    {
        d_def = get_directive_table_entry( d );
        if ( d_def )
        {
            d->active = d_def->active;
        }
        else
        {
            d->active = true;
        }
        if ( d_def && d_def->active )
        {
            instrumented_paradigm_type |= d_def->type;
            //std::cout << "Doing enter transformation" << std::endl;
            d_def->do_enter_transformation( d, os );
            if ( require_end )
            {
                *require_end = d_def->require_end;
            }
            if ( is_for )
            {
                *is_for =  ( name == "for" || name == "parallelfor" );
            }
        }
        else         // the directive is disabled
        {
            if ( d_def == NULL )
            {
                // Unsupported directives should be ignored silently!
                // cerr << "Unsupported / invalid directive keyword \"" << d->GetName() << "\"\n";
            }
            else
            {
                if ( require_end )
                {
                    *require_end = d_def->require_end;
                }

                /*
                 * If the directive is disabled, always set 'is_for' to false
                 */
                if ( is_for )
                {
                    *is_for = false;
                }
                /** Even if the directive is disabled and not
                 * instrumented, The directive, if requires "END",
                 * should still be pushed onto the stack, so the
                 * "END" directive can find the corresponding
                 * end.
                 */
                if ( d_def->require_end )
                {
                    DirectiveStackPush( d );
                }
                if ( d_def->single_statement )
                {
                    SaveSingleLineDirective( d );
                }

                /** Directives that might be implicitly ended by the
                 * end of a do loop*/
                d->NeedsEndLoopDirective( d_def->loop_block );
            }
            if ( name == "parallelfor" || name == "paralleldo" || name == "parallelsections" )
            {
                instrumented_paradigm_type |= d_def->type;
                d_def->do_enter_transformation( d, os );
            }
            else
            {
                // if disabled, simply output lines, without any modification
                d->PrintDirective( os );
            }
        }
    }

    return;
}


/** @brief Check whether the current line is an omp function
    declaration */
bool
is_runtime_decl( const string& file,
                 const string& header_file )
{
    // If it is in the header file, its a declaration
    string::size_type pos = file.length() - header_file.length() - 1;

    return file == header_file ||
           ( file[ pos ] == '/' &&
             file.substr( pos + 1 ) == header_file );
}


/** Replaces a runtime call with the wrapper function. This function
    is currently used only by the Fortran parser */
void
ReplaceRuntimeAPI( string&           lowline,
                   string&           line,
                   const string&     file,
                   OPARI2_Language_t lang )
{
    size_t i   = 0;
    size_t pos = 0;

    if ( InstrumentationDisabled() )
    {
        return;
    }

    //Functions my not be replaced in a line like: use omp_lib, only: omp_init_lock
    if ( lowline.find( ":", lowline.find( "only", lowline.find( "omp_lib", lowline.find( "use" ) ) ) ) != string::npos )
    {
        return;
    }

    for ( i = 0; i < sizeof( api_table ) / sizeof( OPARI2_RuntimeAPIDefinition ); i++ )
    {
        pos = 0;

        if ( api_table[ i ].active &&
             !( lang & L_FORTRAN  && is_runtime_decl( file, api_table[ i ].header_file_f ) ) &&
             !( lang & L_C_OR_CXX && is_runtime_decl( file, api_table[ i ].header_file_c ) ) )
        {
            while ( ( pos = lowline.find( api_table[ i ].name, pos ) ) != string::npos )
            {
                /**
                 * when "omp_test_lock" and "omp_test_nest_lock" functions are defined,
                 * they may not be replaced.
                 */
                if ( ( !api_table[ i ].name.find( "test" ) ) ||
                     !( lowline.find( "logical" ) < pos ) )
                {
                    instrumented_paradigm_type |= api_table[ i ].type;
                    line.replace( pos, 3, "POMP2" );
                    //Keep other letters unchanged, except for line[ pos + 6 ]
                    line[ pos + 6 ] = std::toupper( line[ pos + 6 ] );
                }
                pos += api_table[ i ].name.length();
            }
        }
    }
}


set<string> header_files_f;
set<string> header_files_c;

bool
IsSupportedAPIHeaderFile( const string&     include_file,
                          OPARI2_Language_t lang )
{
    if ( header_files_f.empty() || header_files_c.empty() )
    {
        for ( unsigned long i = 0; i < sizeof( api_table ) / sizeof( OPARI2_RuntimeAPIDefinition ); i++ )
        {
            header_files_f.insert( api_table[ i ].header_file_f );
            header_files_c.insert( api_table[ i ].header_file_c );
        }
    }

    string                file;
    set<string>::iterator it, itb, ite;
    if ( lang & L_FORTRAN )
    {
        itb = header_files_f.begin();
        ite = header_files_f.end();

        if ( ( *( include_file.begin() ) == '\"' && *( include_file.end() - 1 ) == '\"' ) ||
             ( *( include_file.begin() ) == '\''  && *( include_file.end() - 1 ) == '\'' ) )
        {
            file = include_file.substr( 1, include_file.length() - 2 );
        }
    }
    else if ( lang & L_C_OR_CXX )
    {
        itb = header_files_c.begin();
        ite = header_files_c.end();

        if ( ( *( include_file.begin() ) == '\"' && *( include_file.end() - 1 ) == '\"' ) ||
             ( *( include_file.begin() ) == '<'  && *( include_file.end() - 1 ) == '>' ) )
        {
            file = include_file.substr( 1, include_file.length() - 2 );
        }
    }

    if ( file.empty() )
    {
        return false;
    }

    for ( it = itb; it != ite; ++it )
    {
        if ( *it == file )
        {
            return true;
        }
    }
    return false;
}


void
SaveForInit( OPARI2_Directive* d )
{
    if ( !tmp_directives.empty() )
    {
        vector<OPARI2_Directive*>::iterator it = tmp_directives.begin();
        while ( it != tmp_directives.end() )
        {
            if ( *it == d )
            {
                it = tmp_directives.erase( it );
                break;
            }
            else
            {
                ++it;
            }
        }
    }

    directive_vec.push_back( d );
}

void
DirectiveStackPush( OPARI2_Directive* d )
{
    directive_stack.push_back( d );
}

OPARI2_Directive*
DirectiveStackTop( OPARI2_Directive* d )
{
    if ( directive_stack.empty() )
    {
        if ( d )
        {
            cerr << d->GetFilename() << ":" << d->GetLineno() << ":"
                 << "ERROR: unbalanced pragma/directive nesting for "
                 << d->GetName() << " directive \n";
        }
        else
        {
            cerr << "ERROR: unbalanced pragma/directive nesting!\n";
        }
        cleanup_and_exit();
    }
    else
    {
        return directive_stack.back();
    }

    return NULL;
}

void
DirectiveStackPop(  OPARI2_Directive* d )
{
    OPARI2_Directive* d_top = DirectiveStackTop( d );
    if ( !tmp_directives.empty() )
    {
        vector<OPARI2_Directive*>::iterator it = tmp_directives.begin();
        while ( it != tmp_directives.end() )
        {
            if ( *it == d_top )
            {
                /*  This is the only reference left to this directive,
                    so perform cleanup */
                tmp_directives.erase( it );
                directive_stack.pop_back();
                delete d_top;
                return;
            }
            else
            {
                ++it;
            }
        }
    }

    directive_stack.pop_back();
}


void
PrintDirectiveStackTop( void )
{
    if ( directive_stack.empty() )
    {
        cerr << "Error: Directive stack empty \n";
        cleanup_and_exit();
    }
    else
    {
        OPARI2_Directive* d = directive_stack.back();

        if ( d )
        {
            cerr << "       near \"" << paradigm_type_to_string( d->GetParadigmType() )
                 << ": " << d->GetName() << "\" construct at "
                 << d->GetFilename() << ":" << d->GetLineno() << std::endl;
        }
    }
}

void
DirectiveStackInsertDescr( int id )
{
    for ( vector<OPARI2_Directive*>::iterator it = directive_stack.begin(); it != directive_stack.end(); ++it )
    {
        if ( !( ( *it )->DescrsEmpty() ) )
        {
            ( *it )->InsertDescr( id );
        }
    }
}


void
Finalize( OPARI2_Option_t& options )
{
    // check region stack
    if ( !directive_stack.empty() )
    {
        cerr << "ERROR: unbalanced pragma/directive nesting\n";
        while ( !directive_stack.empty() )
        {
            PrintDirectiveStackTop();
            DirectiveStackPop();
        }
        cleanup_and_exit();
    }

    // generate opari include file
    ofstream incs( options.incfile.c_str() );
    if ( !incs )
    {
        cerr << "ERROR: cannot open opari include file " << options.incfile << "\n";
        exit( 1 );
    }

    if (  options.lang & L_C_OR_CXX )
    {
        /** In order to please the PGI compiler, the order of the header
         *  generation between OpenMP and user instrumentation must not
         *  be switched. Take care to test additions in that respect.
         *
         * The error that was encountered was that the
         * threadprivate(pomp_tpd) that is added for the
         * --thread=omp:pomp_tpd option caused:
         * "PGCC-S-0155-pomp_tpd_ is not threadprivate
         * (.../jacobi/cxx/main.cpp: 167) PGCC/x86 Linux 14.1-0:
         * compilation completed with severe errors"
         */

        if ( instrumented_paradigm_type & OPARI2_PT_POMP )
        {
            OPARI2_DirectivePomp::GenerateHeader( incs );
        }

        if ( instrumented_paradigm_type & OPARI2_PT_OMP )
        {
            OPARI2_DirectiveOpenmp::GenerateHeader( incs );
        }
    }

    if ( directive_vec.size() )
    {
        for ( vector<OPARI2_Directive*>::iterator it = directive_vec.begin(); it != directive_vec.end(); ++it )
        {
            if ( ( *it )->GetName() != "offload" &&
                 ( *it )->GetName() != "declspec" )
            {
                ( *it )->GenerateDescr( incs );
            }
        }
    }

    if ( options.lang & L_FORTRAN )
    {
        OPARI2_Directive::FinalizeFortranDescrs( incs );

        OPARI2_DirectiveOpenmp::GenerateInitHandleCalls( options.os, options.incfile_nopath );
        OPARI2_DirectiveOpenmp::FinalizeDescrs( incs );

        OPARI2_DirectivePomp::GenerateInitHandleCalls( options.os, options.incfile_nopath );
    }
    else if (  options.lang & L_C_OR_CXX )
    {
        OPARI2_DirectiveOpenmp::GenerateInitHandleCalls( incs );
        OPARI2_DirectivePomp::GenerateInitHandleCalls( incs );
    }
    /*cleanup tmp_directives and directive_vec vectors*/
    for ( vector<OPARI2_Directive*>::iterator it = tmp_directives.begin(); it != tmp_directives.end(); it++ )
    {
        delete *it;
    }
    for ( vector<OPARI2_Directive*>::iterator it = directive_vec.begin(); it != directive_vec.end(); it++ )
    {
        delete *it;
    }
}

void
SaveSingleLineDirective( OPARI2_Directive* d )
{
    saved_single_line_directive = d;
}

void
HandleSingleLineDirective( const int lineno,
                           ostream&  os )
{
    if ( saved_single_line_directive )
    {
        OPARI2_DirectiveDefinition* d_def = get_directive_table_entry( saved_single_line_directive );
        if ( d_def->active )
        {
            /** @TODO This is currently the only case, nonetheless it
                should be generalized */
            extra_openmp_atomic_handler( saved_single_line_directive,
                                         lineno, os );
        }
        else
        {
            DirectiveStackPop();
            saved_single_line_directive = NULL;
        }
    }
}
