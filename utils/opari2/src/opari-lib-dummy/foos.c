/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2011,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2011,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2011, 2013
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2011,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */
/****************************************************************************
**  SCALASCA    http://www.scalasca.org/                                   **
**  KOJAK       http://www.fz-juelich.de/jsc/kojak/                        **
*****************************************************************************
**  Copyright (c) 1998-2009                                                **
**  Forschungszentrum Juelich, Juelich Supercomputing Centre               **
**                                                                         **
**  See the file COPYRIGHT in the package base directory for details       **
****************************************************************************/
/** @internal
 *
 *  @file       foos.c
 *
 *  @brief     This file is used to check the Fotran mangling of the
 *             used compiler.  In getfname.f a Function foo_foo is
 *             called. Here functions with mangled names exist
 *             foo_foo, foo_foo_, _foo_foo_, ... , depending on the
 *             function called a file pomp2_fwrapper_def.h is written
 *             with a macro to mangle function names to fortran
 *             function names.*/

#include <config.h>
#include <stdio.h>
#include <stdlib.h>

static char* header =

    "/*\n"
    " * Fortan subroutine external name setup\n"
    " */\n"
    "\n"
    "#define POMP2_Finalize_U		POMP2_FINALIZE\n"
    "#define POMP2_Init_U		POMP2_INIT\n"
    "#define POMP2_Off_U		POMP2_OFF\n"
    "#define POMP2_On_U		        POMP2_ON\n"
    "#define POMP2_Atomic_enter_U	POMP2_ATOMIC_ENTER\n"
    "#define POMP2_Atomic_exit_U	POMP2_ATOMIC_EXIT\n"
    "#define POMP2_Implicit_barrier_enter_U	POMP2_IMPLICIT_BARRIER_ENTER\n"
    "#define POMP2_Implicit_barrier_exit_U	POMP2_IMPLICIT_BARRIER_EXIT\n"
    "#define POMP2_Barrier_enter_U	POMP2_BARRIER_ENTER\n"
    "#define POMP2_Barrier_exit_U	POMP2_BARRIER_EXIT\n"
    "#define POMP2_Flush_enter_U	POMP2_FLUSH_ENTER\n"
    "#define POMP2_Flush_exit_U	        POMP2_FLUSH_EXIT\n"
    "#define POMP2_Critical_begin_U	POMP2_CRITICAL_BEGIN\n"
    "#define POMP2_Critical_end_U	POMP2_CRITICAL_END\n"
    "#define POMP2_Critical_enter_U	POMP2_CRITICAL_ENTER\n"
    "#define POMP2_Critical_exit_U	POMP2_CRITICAL_EXIT\n"
    "#define POMP2_Do_enter_U		POMP2_DO_ENTER\n"
    "#define POMP2_Do_exit_U		POMP2_DO_EXIT\n"
    "#define POMP2_Master_begin_U	POMP2_MASTER_BEGIN\n"
    "#define POMP2_Master_end_U	        POMP2_MASTER_END\n"
    "#define POMP2_Parallel_begin_U	POMP2_PARALLEL_BEGIN\n"
    "#define POMP2_Parallel_end_U	POMP2_PARALLEL_END\n"
    "#define POMP2_Parallel_enter_U	POMP2_PARALLEL_ENTER\n"
    "#define POMP2_Parallel_exit_U	POMP2_PARALLEL_EXIT\n"
    "#define POMP2_Parallel_fork_U	POMP2_PARALLEL_FORK\n"
    "#define POMP2_Parallel_join_U	POMP2_PARALLEL_JOIN\n"
    "#define POMP2_Section_begin_U	POMP2_SECTION_BEGIN\n"
    "#define POMP2_Section_end_U	POMP2_SECTION_END\n"
    "#define POMP2_Section_enter_U	POMP2_SECTION_ENTER\n"
    "#define POMP2_Section_exit_U	POMP2_SECTION_EXIT\n"
    "#define POMP2_Sections_enter_U	POMP2_SECTIONS_ENTER\n"
    "#define POMP2_Sections_exit_U	POMP2_SECTIONS_EXIT\n"
    "#define POMP2_Single_begin_U	POMP2_SINGLE_BEGIN\n"
    "#define POMP2_Single_end_U	        POMP2_SINGLE_END\n"
    "#define POMP2_Single_enter_U	POMP2_SINGLE_ENTER\n"
    "#define POMP2_Single_exit_U	POMP2_SINGLE_EXIT\n"
    "#define POMP2_Ordered_begin_U	POMP2_ORDERED_BEGIN\n"
    "#define POMP2_Ordered_end_U	POMP2_ORDERED_END\n"
    "#define POMP2_Ordered_enter_U	POMP2_ORDERED_ENTER\n"
    "#define POMP2_Ordered_exit_U	POMP2_ORDERED_EXIT\n"

    "#define POMP2_Task_begin_U         POMP2_TASK_BEGIN\n"
    "#define POMP2_Task_end_U           POMP2_TASK_END\n"
    "#define POMP2_Task_create_begin_U  POMP2_TASK_CREATE_BEGIN\n"
    "#define POMP2_Task_create_end_U    POMP2_TASK_CREATE_END\n"
    "#define POMP2_Untied_task_begin_U         POMP2_UNTIED_TASK_BEGIN\n"
    "#define POMP2_Untied_task_end_U           POMP2_UNTIED_TASK_END\n"
    "#define POMP2_Untied_task_create_begin_U  POMP2_UNTIED_TASK_CREATE_BEGIN\n"
    "#define POMP2_Untied_task_create_end_U    POMP2_UNTIED_TASK_CREATE_END\n"
    "#define POMP2_Taskwait_begin_U     POMP2_TASKWAIT_BEGIN\n"
    "#define POMP2_Taskwait_end_U       POMP2_TASKWAIT_END\n"
    "#define POMP2_Workshare_enter_U	POMP2_WORKSHARE_ENTER\n"
    "#define POMP2_Workshare_exit_U	POMP2_WORKSHARE_EXIT\n"

    "#define POMP2_Begin_U		POMP2_BEGIN\n"
    "#define POMP2_End_U		POMP2_END\n"

    "#define POMP2_Lib_get_max_threads_U        POMP2_LIB_GET_MAX_THREADS\n"
    "#define POMP2_Init_lock_U	        POMP2_INIT_LOCK\n"
    "#define POMP2_Destroy_lock_U	POMP2_DESTROY_LOCK\n"
    "#define POMP2_Set_lock_U		POMP2_SET_LOCK\n"
    "#define POMP2_Unset_lock_U	        POMP2_UNSET_LOCK\n"
    "#define POMP2_Test_lock_U	        POMP2_TEST_LOCK\n"
    "#define POMP2_Init_nest_lock_U	POMP2_INIT_NEST_LOCK\n"
    "#define POMP2_Destroy_nest_lock_U	POMP2_DESTROY_NEST_LOCK\n"
    "#define POMP2_Set_nest_lock_U	POMP2_SET_NEST_LOCK\n"
    "#define POMP2_Unset_nest_lock_U	POMP2_UNSET_NEST_LOCK\n"
    "#define POMP2_Test_nest_lock_U	POMP2_TEST_NEST_LOCK\n"
    "#define POMP2_Assign_handle_U	POMP2_ASSIGN_HANDLE\n"
    "#define POMP2_USER_Assign_handle_U	POMP2_USER_ASSIGN_HANDLE\n"
    "#define omp_init_lock_U		OMP_INIT_LOCK\n"
    "#define omp_destroy_lock_U	        OMP_DESTROY_LOCK\n"
    "#define omp_set_lock_U		OMP_SET_LOCK\n"
    "#define omp_unset_lock_U	        OMP_UNSET_LOCK\n"
    "#define omp_test_lock_U		OMP_TEST_LOCK\n"
    "#define omp_init_nest_lock_U	OMP_INIT_NEST_LOCK\n"
    "#define omp_destroy_nest_lock_U	OMP_DESTROY_NEST_LOCK\n"
    "#define omp_set_nest_lock_U	OMP_SET_NEST_LOCK\n"
    "#define omp_unset_nest_lock_U	OMP_UNSET_NEST_LOCK\n"
    "#define omp_test_nest_lock_U	OMP_TEST_NEST_LOCK\n"
    "\n"
    "#define POMP2_Finalize_L		pomp2_finalize\n"
    "#define POMP2_Init_L		pomp2_init\n"
    "#define POMP2_Off_L		pomp2_off\n"
    "#define POMP2_On_L		        pomp2_on\n"
    "#define POMP2_Atomic_enter_L	pomp2_atomic_enter\n"
    "#define POMP2_Atomic_exit_L	pomp2_atomic_exit\n"
    "#define POMP2_Implicit_barrier_enter_L	pomp2_implicit_barrier_enter\n"
    "#define POMP2_Implicit_barrier_exit_L	pomp2_implicit_barrier_exit\n"
    "#define POMP2_Barrier_enter_L	pomp2_barrier_enter\n"
    "#define POMP2_Barrier_exit_L	pomp2_barrier_exit\n"
    "#define POMP2_Flush_enter_L	pomp2_flush_enter\n"
    "#define POMP2_Flush_exit_L	        pomp2_flush_exit\n"
    "#define POMP2_Critical_begin_L	pomp2_critical_begin\n"
    "#define POMP2_Critical_end_L	pomp2_critical_end\n"
    "#define POMP2_Critical_enter_L	pomp2_critical_enter\n"
    "#define POMP2_Critical_exit_L	pomp2_critical_exit\n"
    "#define POMP2_Do_enter_L		pomp2_do_enter\n"
    "#define POMP2_Do_exit_L		pomp2_do_exit\n"
    "#define POMP2_Master_begin_L	pomp2_master_begin\n"
    "#define POMP2_Master_end_L	        pomp2_master_end\n"
    "#define POMP2_Parallel_begin_L	pomp2_parallel_begin\n"
    "#define POMP2_Parallel_end_L	pomp2_parallel_end\n"
    "#define POMP2_Parallel_enter_L	pomp2_parallel_enter\n"
    "#define POMP2_Parallel_exit_L	pomp2_parallel_exit\n"
    "#define POMP2_Parallel_fork_L	pomp2_parallel_fork\n"
    "#define POMP2_Parallel_join_L	pomp2_parallel_join\n"
    "#define POMP2_Section_begin_L	pomp2_section_begin\n"
    "#define POMP2_Section_end_L	pomp2_section_end\n"
    "#define POMP2_Section_enter_L	pomp2_section_enter\n"
    "#define POMP2_Section_exit_L	pomp2_section_exit\n"
    "#define POMP2_Sections_enter_L	pomp2_sections_enter\n"
    "#define POMP2_Sections_exit_L	pomp2_sections_exit\n"
    "#define POMP2_Single_begin_L	pomp2_single_begin\n"
    "#define POMP2_Single_end_L	        pomp2_single_end\n"
    "#define POMP2_Single_enter_L	pomp2_single_enter\n"
    "#define POMP2_Single_exit_L	pomp2_single_exit\n"
    "#define POMP2_Ordered_begin_L	pomp2_ordered_begin\n"
    "#define POMP2_Ordered_end_L	pomp2_ordered_end\n"
    "#define POMP2_Ordered_enter_L	pomp2_ordered_enter\n"
    "#define POMP2_Ordered_exit_L	pomp2_ordered_exit\n"

    "#define POMP2_Task_begin_L         pomp2_task_begin\n"
    "#define POMP2_Task_end_L           pomp2_task_end\n"
    "#define POMP2_Task_create_begin_L  pomp2_task_create_begin\n"
    "#define POMP2_Task_create_end_L    pomp2_task_create_end\n"
    "#define POMP2_Untied_task_begin_L         pomp2_untied_task_begin\n"
    "#define POMP2_Untied_task_end_L           pomp2_untied_task_end\n"
    "#define POMP2_Untied_task_create_begin_L  pomp2_untied_task_create_begin\n"
    "#define POMP2_Untied_task_create_end_L    pomp2_untied_task_create_end\n"
    "#define POMP2_Taskwait_begin_L     pomp2_taskwait_begin\n"
    "#define POMP2_Taskwait_end_L       pomp2_taskwait_end\n"
    "#define POMP2_Workshare_enter_L	pomp2_workshare_enter\n"
    "#define POMP2_Workshare_exit_L	pomp2_workshare_exit\n"

    "#define POMP2_Begin_L		pomp2_begin\n"
    "#define POMP2_End_L		pomp2_end\n"

    "#define POMP2_Lib_get_max_threads_L        pomp2_lib_get_max_threads\n"
    "#define POMP2_Init_lock_L	        pomp2_init_lock\n"
    "#define POMP2_Destroy_lock_L	pomp2_destroy_lock\n"
    "#define POMP2_Set_lock_L		pomp2_set_lock\n"
    "#define POMP2_Unset_lock_L	        pomp2_unset_lock\n"
    "#define POMP2_Test_lock_L	        pomp2_test_lock\n"
    "#define POMP2_Init_nest_lock_L	pomp2_init_nest_lock\n"
    "#define POMP2_Destroy_nest_lock_L	pomp2_destroy_nest_lock\n"
    "#define POMP2_Set_nest_lock_L	pomp2_set_nest_lock\n"
    "#define POMP2_Unset_nest_lock_L	pomp2_unset_nest_lock\n"
    "#define POMP2_Test_nest_lock_L	pomp2_test_nest_lock\n"
    "#define POMP2_Assign_handle_L	pomp2_assign_handle\n"
    "#define POMP2_USER_Assign_handle_L	pomp2_user_assign_handle\n"
    "#define omp_init_lock_L		omp_init_lock\n"
    "#define omp_destroy_lock_L	        omp_destroy_lock\n"
    "#define omp_set_lock_L		omp_set_lock\n"
    "#define omp_unset_lock_L	        omp_unset_lock\n"
    "#define omp_test_lock_L		omp_test_lock\n"
    "#define omp_init_nest_lock_L	omp_init_nest_lock\n"
    "#define omp_destroy_nest_lock_L	omp_destroy_nest_lock\n"
    "#define omp_set_nest_lock_L	omp_set_nest_lock\n"
    "#define omp_unset_nest_lock_L	omp_unset_nest_lock\n"
    "#define omp_test_nest_lock_L	omp_test_nest_lock\n"
    "\n"
    "#define XSUFFIX(name)  name##_\n"
    "#define XSUFFIX2(name) name##__\n"
    "#define XPREFIX(name)  _##name\n"
    "#define XPREFIX2(name) __##name\n"
    "\n"
    "#define SUFFIX(name)  XSUFFIX(name)\n"
    "#define SUFFIX2(name) XSUFFIX2(name)\n"
    "#define PREFIX(name)  XPREFIX(name)\n"
    "#define PREFIX2(name) XPREFIX2(name)\n"
    "\n"
    "#define UPCASE(name)  name##_U\n"
    "#define LOWCASE(name) name##_L\n"
    "\n";

static void
generate( char* str )
{
    FILE* f = fopen( "pomp2_fwrapper_def.h", "w" );
    if ( f == NULL )
    {
        perror( "pomp2_fwrapper_def.h" );
        exit( 1 );
    }
    fputs( header, f );
    fputs( str, f );
    fputs( "\n", f );
    fclose( f );
}

void
foo_foo()
{
    generate( "#define FSUB(name) LOWCASE(name)" );
}
void
foo_foo_()
{
    generate( "#define FSUB(name) SUFFIX(LOWCASE(name))" );
}
void
foo_foo__()
{
    generate( "#define FSUB(name) SUFFIX2(LOWCASE(name))" );
}
void
_foo_foo()
{
    generate( "#define FSUB(name) PREFIX(LOWCASE(name))" );
}
void
__foo_foo()
{
    generate( "#define FSUB(name) PREFIX2(LOWCASE(name))" );
}
void
_foo_foo_()
{
    generate( "#define FSUB(name) PREFIX(SUFFIX(LOWCASE(name)))" );
}
void
FOO_FOO()
{
    generate( "#define FSUB(name) UPCASE(name)" );
}
void
FOO_FOO_()
{
    generate( "#define FSUB(name) SUFFIX(UPCASE(name))" );
}
void
FOO_FOO__()
{
    generate( "#define FSUB(name) SUFFIX2(UPCASE(name))" );
}
void
_FOO_FOO()
{
    generate( "#define FSUB(name) PREFIX(UPCASE(name))" );
}
void
__FOO_FOO()
{
    generate( "#define FSUB(name) PREFIX2(UPCASE(name))" );
}
void
_FOO_FOO_()
{
    generate( "#define FSUB(name) PREFIX(SUFFIX(UPCASE(name)))" );
}
