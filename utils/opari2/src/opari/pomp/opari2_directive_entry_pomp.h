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
 *  @file      opari2_directive_entry_pomp.h
 *
 *  @brief Define POMP directive entries.
 */

#ifndef OPARI2_DIRECTIVE_ENTRY_POMP_H
#define OPARI2_DIRECTIVE_ENTRY_POMP_H

#include "opari2.h"
#include "opari2_directive_entry.h"

/**
 * @brief Definition of OpenAcc directive/runtime API group.
 * More directive group definition can be added before G_OMP_ALL.
 */
enum OPARI2_PompGroup
{
    G_POMP_NONE    = 0x00000000,
    G_POMP_REGION  = 0x00000001,
    G_POMP_DEFAULT = 0x00010000,
    G_POMP_ALL     = 0xFFFFFFFF
};

#define OPARI2_POMP2_USER_SENTINELS \
    { "!$pomp", OPARI2_PT_POMP },     \
    { "c$pomp", OPARI2_PT_POMP },     \
    { "*$pomp", OPARI2_PT_POMP },     \
    { "!pomp$", OPARI2_PT_POMP },     \
    { "!pomp$", OPARI2_PT_POMP },     \
    { "cpomp$", OPARI2_PT_POMP },     \
    { "*pomp$", OPARI2_PT_POMP },     \
    { "pomp", OPARI2_PT_POMP }


#define OPARI2_CREATE_POMP_TABLE_ENTRY( name, version, group )    \
    OPARI2_CREATE_TABLE_ENTRY( OPARI2_PT_POMP, name, false, true, version, group, pomp )

#define OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( name, version, group ) \
    OPARI2_CREATE_TABLE_ENTRY_NOEND( OPARI2_PT_POMP, name, false, version, group, pomp )

/**
 * TODO:
 * Set the correct version / group  number for POMP directives
 */
#define OPARI2_POMP_DIRECTIVE_ENTRIES \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( instinit,     1.1, G_POMP_DEFAULT ), \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( instfinalize, 1.1, G_POMP_DEFAULT ), \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( inston,       1.1, G_POMP_DEFAULT ), \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( instoff,      1.1, G_POMP_DEFAULT ), \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( instrument,   1.1, G_POMP_DEFAULT ), \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( noinstrument, 1.1, G_POMP_DEFAULT ), \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( instbegin,    1.1, G_POMP_REGION ), \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( instaltend,   1.1, G_POMP_REGION ), \
    OPARI2_CREATE_POMP_TABLE_ENTRY_NOEND( instend,      1.1, G_POMP_REGION )


#endif // OPARI2_DIRECTIVE_ENTRY_POMP_H
