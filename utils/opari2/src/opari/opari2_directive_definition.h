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
 *  @file		opari2_directive_definition.h
 *
 *  @brief		This file defines data structures to represent directive and runtime API.
 */

#ifndef OPARI2_DIRECTIVE_DEFINITION_H
#define OPARI2_DIRECTIVE_DEFINITION_H

#include <stdint.h>
#include <iostream>
using std::ostream;
#include <string>
using std::string;

#include "opari2.h"
#include "opari2_directive.h"


typedef void ( * DirectiveHandler )( OPARI2_Directive*,
                                     ostream& );

/**
 * @brief Structure definition for a paradigm directive.
 */
typedef struct
{
    OPARI2_ParadigmType_t type;                     /**< paradigm the directive belongs to. */
    const string          name;                     /**< directive name. */
    const string          end_name;                 /**< directive closing name, if any. */
    const string          version;                  /**< version of standard/specification. */
    uint64_t              group;                    /**< group the directive belongs to */
    bool                  active;                   /**< directive enabled? */
    bool                  inner_active;             /**< inner begin/end instrumentation enabled? */
    bool                  require_end;              /**< need end handling? */
    bool                  loop_block;               /**< Directive must be followed by a loop */
    bool                  single_statement;         /**< Directive only applies to the next statement */
    bool                  disable_with_paradigm;    /**< Specfies whether this directive gets instrumented, even when the paradigm is disabled, e.g. omp parallel */
    DirectiveHandler      do_enter_transformation;  /**< handler for region enter instrumentation */
    DirectiveHandler      do_exit_transformation;   /**< handler for region exit instrumentation */
} OPARI2_DirectiveDefinition;



/**
 * @brief Structure definition for a paradigm runtime API.
 */
typedef struct
{
    OPARI2_ParadigmType_t type;          /**< paradigm the API belongs to */
    const string          name;          /**< API name */
    const string          version;       /**< version of standard/specification */
    uint64_t              group;         /**< group the API belongs to */
    bool                  active;        /**< API enabled? */
    const string          wrapper;       /**< name of the wrapper funtion */
    const string          header_file_c; /***< name of the include file of the API */
    const string          header_file_f; /***< name of the include file of the API */
} OPARI2_RuntimeAPIDefinition;

#endif // OPARI2_DIRECTIVE_DEFINITION_H
