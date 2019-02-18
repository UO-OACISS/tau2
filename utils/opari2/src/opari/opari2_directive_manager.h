/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013, 2016,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

/** @internal
 *
 *  @file      opari2_directive_manager.h
 *
 *  @brief     Interface declaration of 'DirectiveManager'
 */

#ifndef OPARI2_DIRECTIVE_MANAGER_H
#define OPARI2_DIRECTIVE_MANAGER_H

#include <vector>
using std::vector;
#include <map>
using std::map;
#include <stack>
using std::stack;
#include <string>
using std::string;
#include <stdint.h>


#include "opari2_directive.h"

/* Map entry. Matches string to enum */
typedef struct
{
    /** string representation*/
    string mString;
    /** matching region type*/
    int    mEnum;
} OPARI2_MapString2ParadigmNameType;


/** @brief Disables the instrumentation of whole paradigms or specific
 *         directives.
 *
 *  Check if the combination of directive/paradigm is supported by
 *  looking up the directive_table.
 *
 *  @param  paradigm	name of the paradigm.
 *  @param  directive	name of the directive.
 *
 *  @return true   ONLY if the combination of paradigm and directive
 *		   is found in the diretive_table.
 *          false  otherwise.
 */
bool
DisableParadigmDirectiveOrGroup( const string& paradigm,
                                 const string& directiveOrGroup,
                                 bool          inner );

/**
 * @brief Different levels for full or partial disabling if
 *        instrumentation.
 *
 * Bitfield for disabling specific types of instrumentation.  E.g. for
 * OpenMP, when POMP directives are used to disable instrumentation
 * (D_USER) it must still be activated for the parallel regions so a
 * measurement system can manage events for different threads in a
 * threadsafe manner. For the case that code is put on a device where
 * there is no measurement system, even this instrumentation must be
 * disabled, thus the level D_FULL. */
typedef enum
{
    D_NONE = 0x00000000,
    D_USER = 0x00000001,
    D_FULL = 0xFFFFFFFF
} OPARI2_Disable_level_t;

/** @brief Disables instrumentation for a specified level */
void
DisableInstrumentation( OPARI2_Disable_level_t l );

/** @brief Re-enables instrumentation for a specified level */
void
EnableInstrumentation( OPARI2_Disable_level_t l );

/** @brief Checks whether instrumentation for a specified level is
 *        disabled.
 *
 * This function checks whether instrumentation for the requested
 * level or a superior level is disabled.
 *
 * @param l specific level to check for
 *
 * @return true  if current disable level is D_NONE
 *               or l is in the group of current level */
bool
InstrumentationDisabled( OPARI2_Disable_level_t l = D_NONE );


bool
IsValidSentinel( const string&     lowline,
                 string::size_type p,
                 string&           sentinel );

/**
 *  @brief  Check the validality of a given paradigm and directive.
 *
 *  Check if the combination of directive/paradigm is supported by
 *  looking up the directive_table.
 *
 *  @param paradigm  name of the paradigm.
 *  @param directive name of the directive.
 *
 *  @return true  ONLY if the combination of paradigm and directive
 *	          is found in the directive_table.
 *          false otherwise.
 */
bool
IsValidDirective( const string& paradigm,
                  string&       directive );


/**
 *  @brief  Check if a directive's active flag is true.
 *
 *  @param type      type of the paradigm.
 *  @param directive name of the directive.
 *
 *  @return the active member of the to @a directive corresponding
 *  directive_table entry. Return false if no directive_table entry
 *  could be found.
 */
bool
DirectiveActive( OPARI2_ParadigmType_t type,
                 const std::string&    directive );


OPARI2_Directive*
NewDirective( vector<string>&   lines,
              vector<string>&   directive_prefix,
              OPARI2_Language_t lang,
              const string&     file,
              const int         lineno );


void
SetLoopDirective( OPARI2_Directive* d );

OPARI2_Directive*
GetLoopDirective( void );

void
ProcessDirective( OPARI2_Directive* d,
                  ostream&          os,
                  bool*             require_end = NULL,
                  bool*             is_for = NULL );


bool
IsSupportedAPIHeaderFile( const string&     include_file,
                          OPARI2_Language_t lang );

/**
 * @brief Replace a runtime API in the line with its wrapper API if enabled.
 */
void
ReplaceRuntimeAPI( string&           lowline,
                   string&           line,
                   const string&     file,
                   OPARI2_Language_t lang );

void
SaveForInit( OPARI2_Directive* d );

void
DirectiveStackPush( OPARI2_Directive* d );

OPARI2_Directive*
DirectiveStackTop( OPARI2_Directive* d = NULL );

void
DirectiveStackPop( OPARI2_Directive* d = NULL );

void
PrintDirectiveStackTop( void );

void
DirectiveStackInsertDescr( int );

/**
 * @brief Generate the final *.opari.inc file.
 *
 * Need revision if multiple kinds of paradigms are used in the source
 * fle. Current implementation considers OpenMP and POMP ONLY!
 *
 */
void
Finalize( OPARI2_Option_t& options );

void
SaveSingleLineDirective( OPARI2_Directive* d );

void
HandleSingleLineDirective( const int lineno,
                           ostream&  os );


#endif // OPARI2_DIRECTIVE_MANAGER_H
