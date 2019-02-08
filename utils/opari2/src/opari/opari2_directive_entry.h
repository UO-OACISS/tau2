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
 *  @file  opari2_directive_entry.h
 *
 *  @brief Macro definitions for the creation of the table holding all
 *         supported directives
 */

#ifndef OPARI2_DIRECTIVE_ENTRY_H
#define OPARI2_DIRECTIVE_ENTRY_H


/**
 * @brief MACRO definition for the creation of a directive table entry
 *        with a corresponding end-directive.
 */
#define OPARI2_CREATE_TABLE_ENTRY( type, name, loop, disable_with_paradigm, version, group, paradigm ) \
    { type, #name, "end" #name, #version, group, true, true, true, loop, false, disable_with_paradigm, h_ ## paradigm ## _ ## name, h_end_ ## paradigm ## _ ## name }

/**
 * @brief MACRO definition for the creation of a directive table entry
 *        without a corresponding end-directive.
 */
#define OPARI2_CREATE_TABLE_ENTRY_NOEND( type, name, loop, version, group, paradigm ) \
    { type, #name, "", #version, group, true, true, false, loop, false, true, h_ ## paradigm ## _ ## name, NULL }

/**
 * @brief MACRO definition for the creation of a directive table entry
 *        for a directive that applies only to the next statement.
 */
#define OPARI2_CREATE_TABLE_ENTRY_SINGLE_STATEMENT( type, name, loop, disable_with_paradigm, version, group, paradigm ) \
    { type, #name, "", #version, group, true, true, true, loop, true, disable_with_paradigm, h_ ## paradigm ## _ ## name, h_end_ ## paradigm ## _ ## name }


#endif
