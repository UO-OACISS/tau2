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
 *  @file		opari2_directive_entry_openmp.h
 *
 *  @brief		Define Openmp directive and runtime API entries.
 */

#ifndef OPARI2_DIRECTIVE_ENTRY_OPENMP_H
#define OPARI2_DIRECTIVE_ENTRY_OPENMP_H

#include "opari2.h"
#include "opari2_directive_entry.h"


/**
 * @brief Definition of Openmp directive/runtime API group.
 * More directive group definition can be added before G_OMP_ALL.
 */
enum OPARI2_OmpGroup
{
    /** invalid group */
    G_OMP_NONE     = 0x00000000,
    G_OMP_ATOMIC   = 0x00000001,
    G_OMP_CRITICAL = 0x00000002,
    G_OMP_MASTER   = 0x00000004,
    G_OMP_SINGLE   = 0x00000008,
    G_OMP_LOCKS    = 0x00000010,
    G_OMP_FLUSH    = 0x00000020,
    G_OMP_TASK     = 0x00000040,
    G_OMP_ORDERED  = 0x00000080,
    G_OMP_SYNC     = 0x000000FF,
    G_OMP_DEFAULT  = 0x00001000,
    G_OMP_OMP      = 0x0000FFFF,
    G_OMP_REGION   = 0x00010000,
    G_OMP_ALL      = 0xFFFFFFFF
};

#define OPARI2_OPENMP_SENTINELS \
    { "!$omp", OPARI2_PT_OMP }, \
    { "c$omp", OPARI2_PT_OMP }, \
    { "*$omp", OPARI2_PT_OMP }, \
    { "omp",   OPARI2_PT_OMP }, \
    { "openmp", OPARI2_PT_OMP }



#define OPARI2_CREATE_OPENMP_TABLE_ENTRY( name, loop, disable_with_paradigm, version, group ) \
    OPARI2_CREATE_TABLE_ENTRY( OPARI2_PT_OMP, name, loop, disable_with_paradigm, version, group, omp )

#define OPARI2_CREATE_OPENMP_TABLE_ENTRY_NOEND( name, loop, version, group ) \
    OPARI2_CREATE_TABLE_ENTRY_NOEND( OPARI2_PT_OMP, name, loop, version, group, omp )

#define OPARI2_CREATE_OPENMP_TABLE_ENTRY_SINGLE_STATEMENT( name, version, group ) \
    OPARI2_CREATE_TABLE_ENTRY_SINGLE_STATEMENT( OPARI2_PT_OMP, name, false, true, version, group, omp )

/* *INDENT-OFF* */
#define OPARI2_OPENMP_DIRECTIVE_ENTRIES \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( parallel,            false, false, 3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( for,                 true,  true,  3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( do,                  true,  true,  3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( sections,            false, true,  3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( section,             false, true,  3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( single,              false, true,  3.0, G_OMP_SINGLE ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( master,              false, true,  3.0, G_OMP_MASTER ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( critical,            false, true,  3.0, G_OMP_CRITICAL ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( parallelfor,         true,  true, 3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( paralleldo,          true,  true, 3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( parallelsections,    false, true, 3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( workshare,           false, true,  3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( parallelworkshare,   false, true,  3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( ordered,             false, true,  3.0, G_OMP_ORDERED ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY( task,                false, true,  3.0, G_OMP_TASK ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY_SINGLE_STATEMENT( atomic, 3.0, G_OMP_ATOMIC ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY_NOEND( barrier,       false, 3.0, G_OMP_DEFAULT ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY_NOEND( flush,         false, 3.0, G_OMP_FLUSH ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY_NOEND( taskwait,      false, 3.0, G_OMP_TASK ), \
        OPARI2_CREATE_OPENMP_TABLE_ENTRY_NOEND( threadprivate, false, 3.0, G_OMP_DEFAULT )
/* *INDENT-ON* */

#define OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( name, version, group, wrapper ) \
    { OPARI2_PT_OMP, #name, #version, group, true, #wrapper, "omp.h", "omp_lib.h" }

#define OPARI2_OPENMP_API_ENTRIES \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_init_lock,         3.0, G_OMP_LOCKS, POMP2_Init_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_destroy_lock,      3.0, G_OMP_LOCKS, POMP2_Destroy_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_set_lock,          3.0, G_OMP_LOCKS, POMP2_Set_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_unset_lock,        3.0, G_OMP_LOCKS, POMP2_Unset_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_test_lock,         3.0, G_OMP_LOCKS, POMP2_Test_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_init_nest_lock,    3.0, G_OMP_LOCKS, POMP2_Init_nest_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_destroy_nest_lock, 3.0, G_OMP_LOCKS, POMP2_Destroy_nest_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_set_nest_lock,     3.0, G_OMP_LOCKS, POMP2_Set_nest_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_unset_nest_lock,   3.0, G_OMP_LOCKS, POMP2_Unset_nest_lock ), \
    OPARI2_CREATE_OPENMP_API_TABLE_ENTRY( omp_test_nest_lock,    3.0, G_OMP_LOCKS, POMP2_Test_nest_lock )
#endif
