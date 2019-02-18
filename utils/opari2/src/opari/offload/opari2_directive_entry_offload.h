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
 *  @file		opari2_directive_entry_offload.h
 *
 *  @brief		Define Intel Offload directive entries.
 */

#ifndef OPARI2_DIRECTIVE_ENTRY_OFFLOAD_H
#define OPARI2_DIRECTIVE_ENTRY_OFFLOAD_H

#include "opari2.h"
#include "opari2_directive_entry.h"

/**
 * @brief Definition of Intel Offload directive/runtime API group.
 * More directive group definition can be added before G_OFF_ALL.
 */
enum OPARI2_OffloadGroup
{
    G_OFFLOAD_NONE    = 0x00000000,
    G_OFFLOAD_DEFAULT = 0x00001000,
    G_OFFLOAD_ALL     = 0xFFFFFFFF
};

#define OPARI2_OFFLOAD_SENTINELS \
    { "!dir$",              OPARI2_PT_OFFLOAD }, \
    { "cdir$",              OPARI2_PT_OFFLOAD }, \
    { "*dir$",              OPARI2_PT_OFFLOAD }, \
    { "offload",            OPARI2_PT_OFFLOAD }, \
    { "__declspec",         OPARI2_PT_OFFLOAD }

#define OPARI2_CREATE_OFFLOAD_TABLE_ENTRY( name, version, group ) \
    OPARI2_CREATE_TABLE_ENTRY( OPARI2_PT_OFFLOAD, name, false, true, version, group, offload )


/* *INDENT-OFF* */
#define OPARI2_OFFLOAD_DIRECTIVE_ENTRIES \
	OPARI2_CREATE_OFFLOAD_TABLE_ENTRY( target, 1.0, G_OFFLOAD_DEFAULT ),\
	OPARI2_CREATE_OFFLOAD_TABLE_ENTRY( declspec, 1.0, G_OFFLOAD_DEFAULT )

/* *INDENT-ON* */
#endif
