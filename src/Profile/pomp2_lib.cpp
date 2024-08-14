/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 *    RWTH Aachen University, Germany
 *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *    Technische Universitaet Dresden, Germany
 *    University of Oregon, Eugene, USA
 *    Forschungszentrum Juelich GmbH, Germany
 *    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *    Technische Universitaet Muenchen, Germany
 *
 * See the COPYING file in the package base directory for details.
 *
 */
/****************************************************************************
**  SCALASCA    http://www.scalasca.org/                                   **
**  KOJAK       http://www.fz-juelich.de/jsc/kojak/            ///            **
*****************************************************************************
**  Copyright (c) 1998-2009                                                **
**  Forschungszentrum Juelich, Juelich Supercomputing Centre               **
**                                                                         **
**  See the file COPYRIGHT in the package base directory for details       **
****************************************************************************/

#include <config.h>
#include "opari2/pomp2_lib.h"
#include "opari2/pomp2_user_region_info.h"
#include "opari2/pomp2_user_lib.h"


#include "pomp2_region_info.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <string>
#include <mutex>
#include <Profile/Profiler.h>
#ifdef TAU_OPENMP
#ifndef _OPENMP
#define _OPENMP
#endif /* _OPENMP */
#endif /* TAU_OPENMP */
using std::string;

/* Define these weakly linked routines so even if the pompregions.o file is
 * missing the TAU libraries will be able to resolve all undefined symbols.
 * */
#ifndef TAU_CLANG
void POMP2_Init_regions() {}
void POMP2_USER_Init_regions() {}
size_t POMP2_Get_num_regions() { return 0; }
size_t POMP2_USER_Get_num_regions() { return 0; }
#endif /* TAU_CLANG */
#pragma weak POMP2_Init_regions
#pragma weak POMP2_Get_num_regions
#pragma weak POMP2_USER_Init_regions
#pragma weak POMP2_USER_Get_num_regions

/* These two defines specify if we want region based views or construct based
views or both */
#ifndef TAU_OPENMP_PARTITION_REGION
#define TAU_OPENMP_PARTITION_REGION
#endif

#ifdef TAU_OPARI_REGION
#define TAU_OPENMP_REGION_VIEW
#elif TAU_OPARI_CONSTRUCT
#define TAU_AGGREGATE_OPENMP_TIMINGS
#else /* in the default mode, define REGION! */
// #define TAU_AGGREGATE_OPENMP_TIMINGS
#define TAU_OPENMP_REGION_VIEW
#endif

omp_lock_t tau_ompregdescr_lock;
#define OpenMP TAU_USER
#define TAU_EMBEDDED_MAPPING 1

#define TAU_OPARI_CONSTRUCT_TIMER(timer, name, type, group) void *TauGlobal##timer(void) \
{ static void *ptr = NULL; \
  Tau_profile_c_timer(&ptr, name, type, group, #group); \
  return ptr; \
}

#define TAU_OPARI_CONSTRUCT_TIMER_START(timer) \
    Tau_start_timer(TauGlobal##timer(), 0, Tau_get_thread());

#define TAU_OPARI_CONSTRUCT_TIMER_STOP(timer) \
    Tau_stop_timer(TauGlobal##timer(), Tau_get_thread());

TAU_OPARI_CONSTRUCT_TIMER(tatomic, "atomic enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tbarrier, "barrier enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tcriticalb, "critical begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tcriticale, "critical enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tfor, "for enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tmaster, "master begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tparallelb, "parallel begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tparallelf, "parallel fork/join", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tsectionb, "section begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tsectione, "sections enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tsingleb, "single begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tsinglee, "single enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tworkshare, "workshare enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER(tregion, "inst region begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER( tflush , "flush enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER( torderedb , "ordered begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER( torderede , "ordered enter/exit", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER( ttaskcreate , "task create begin/create end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER( ttask , "task begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER( tuntiedcreate , "untied task create begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER( tuntied , "untied task begin/end", "[OpenMP]", OpenMP)
TAU_OPARI_CONSTRUCT_TIMER( ttaskwait , "taskwait begin/end", "[OpenMP]", OpenMP)


#define NUM_OMP_TYPES 22

static char const * omp_names[NUM_OMP_TYPES] = {
    "atomic enter/exit",
    "barrier enter/exit",
    "critical begin/end",
    "critical enter/exit",
    "loop body",
    "master begin/end",
    "parallel begin/end",
    "parallel fork/join",
    "section begin/end",
    "sections enter/exit",
    "single begin/end",
    "single enter/exit",
    "workshare enter/exit",
    "inst region begin/end",
    "flush enter/exit",
    "ordered begin/end",
    "ordered enter/exit",
    "task create begin/end",
    "task begin/end",
    "untied task create begin/end",
    "untied task begin/end",
    "taskwait begin/end"
};


#define TAU_OMP_ATOMIC      0
#define TAU_OMP_BARRIER     1
#define TAU_OMP_CRITICAL_BE 2
#define TAU_OMP_CRITICAL_EE 3
#define TAU_OMP_FOR_EE      4
#define TAU_OMP_MASTER_BE   5
#define TAU_OMP_PAR_BE      6
#define TAU_OMP_PAR_FJ      7
#define TAU_OMP_SECTION_BE  8
#define TAU_OMP_SECTION_EE  9
#define TAU_OMP_SINGLE_BE  10
#define TAU_OMP_SINGLE_EE  11
#define TAU_OMP_WORK_EE    12
#define TAU_OMP_INST_BE    13
#define TAU_OMP_FLUSH_EE   14
#define TAU_OMP_ORDERED_BE 15
#define TAU_OMP_ORDERED_EE 16
#define TAU_OMP_TASK_CREATE 17
#define TAU_OMP_TASK       18
#define TAU_OMP_UNTIED_TASK_CREATE_BE 19
#define TAU_OMP_UNTIED_TASK_BE 20
#define TAU_OMP_TASKWAIT_BE  21

/** @name Functions generated by the instrumenter */
/*@{*/
/**
 * Returns the number of instrumented regions.@n
 * The instrumenter scans all opari-created include files with nm and greps
 * the POMP2_INIT_uuid_numRegions() function calls. Here we return the sum of
 * all numRegions.
 */
extern "C" size_t
POMP2_Get_num_regions();

/**
 * Init all opari-created regions.@n
 * The instrumentor scans all opari-created include files with nm and greps
 * the POMP2_INIT_uuid_numRegions() function calls. The instrumentor then
 * defines this functions by calling all grepped functions.
 */
extern "C" void
POMP2_Init_regions();

/**
 * Returns the opari version.
 */
extern "C" const char*
POMP2_Get_opari2_version();

/*@}*/

/** @brief All relevant information for a region is stored here */
typedef struct
{
    /** region type of construct */
    char*  rtype;
    /** critical or user region name */
    char*  name;
    /** sections only: number of sections */
    int    num_sections;

    /** start file name*/
    char*  start_file_name;
    /** line number 1*/
    int    start_line_1;
    /** line number 2*/
    int    start_line_2;

    /** end file name*/
    char*  end_file_name;
    /** line number 1*/
    int    end_line_1;
    /** line number 2*/
    int    end_line_2;
    /** region id*/
    size_t id;

    /** space for performance data*/
    void* data;

} my_pomp2_region;

struct my_pomp2_region_node {
  my_pomp2_region_node* next;
  my_pomp2_region region;
};

my_pomp2_region_node* tau_region_list_top = NULL;



/** Id of the currently executing task*/
POMP2_Task_handle pomp2_current_task = 0;
#pragma omp threadprivate(pomp2_current_task)

/** Counter of tasks used to determine task ids for newly created ta*/
POMP2_Task_handle pomp2_task_counter = 1;
#pragma omp threadprivate(pomp2_task_counter)

extern "C" POMP2_Task_handle
POMP2_Get_new_task_handle()
{
    return ( ( POMP2_Task_handle )omp_get_thread_num() << 32 ) + pomp2_task_counter++;
}

static void
free_my_pomp2_region_member( char** member )
{
    if ( *member )
    {
        free( *member );
        *member = 0;
    }
}

static void
free_my_pomp2_region_members( my_pomp2_region* region )
{
    if ( region )
    {
        free_my_pomp2_region_member( &region->rtype );
        free_my_pomp2_region_member( &region->name );
        free_my_pomp2_region_member( &region->start_file_name );
        free_my_pomp2_region_member( &region->end_file_name );
    }
}

static void
assignString( char**      destination,
              const char* source )
{
    assert( source );
    *destination = strdup(source);
}


static void
initDummyRegionFromPOMP2RegionInfo(
    my_pomp2_region*         pomp2_region,
    const POMP2_Region_info* pomp2RegionInfo )
{
    assignString( &( pomp2_region->rtype ),
                  pomp2RegionType2String( pomp2RegionInfo->mRegionType ) );

    assignString( &pomp2_region->start_file_name,
                  pomp2RegionInfo->mStartFileName );
    pomp2_region->start_line_1 = pomp2RegionInfo->mStartLine1;
    pomp2_region->start_line_2 = pomp2RegionInfo->mStartLine2;

    assignString( &pomp2_region->end_file_name,
                  pomp2RegionInfo->mEndFileName );
    pomp2_region->end_line_1 = pomp2RegionInfo->mEndLine1;
    pomp2_region->end_line_2 = pomp2RegionInfo->mEndLine2;

/* OLD CODE :
    if ( pomp2RegionInfo->mRegionType == POMP2_User_region )
    {
        assignString( &pomp2_region->name,
                      pomp2RegionInfo->mUserRegionName );
    } else
*/
    if ( pomp2RegionInfo->mRegionType == POMP2_Critical && pomp2RegionInfo->mCriticalName )
    {
        assignString( &pomp2_region->name,
                      pomp2RegionInfo->mCriticalName );
    }

    pomp2_region->num_sections = pomp2RegionInfo->mNumSections;
}


static void
initDummyRegionFromPOMP2UserRegionInfo(
    my_pomp2_region*              pomp2_region,
    const POMP2_USER_Region_info* pomp2RegionInfo )
{
    assignString( &( pomp2_region->rtype ),
                  pomp2UserRegionType2String( (POMP2_USER_Region_type)pomp2RegionInfo->mRegionType ) );

    assignString( &pomp2_region->start_file_name,
                  pomp2RegionInfo->mStartFileName );
    pomp2_region->start_line_1 = pomp2RegionInfo->mStartLine1;
    pomp2_region->start_line_2 = pomp2RegionInfo->mStartLine2;

    assignString( &pomp2_region->end_file_name,
                  pomp2RegionInfo->mEndFileName );
    pomp2_region->end_line_1 = pomp2RegionInfo->mEndLine1;
    pomp2_region->end_line_2 = pomp2RegionInfo->mEndLine2;

    if ( pomp2RegionInfo->mRegionType == POMP2_USER_Region )
    {
        assignString( &pomp2_region->name,
                      pomp2RegionInfo->mUserRegionName );
    }
}


/* TAU specific calls */
int tau_openmp_init(void)
{
  omp_init_lock(&tau_ompregdescr_lock);
  return 0;
}


int tau_openmp_initialized = tau_openmp_init();

extern FunctionInfo * Tau_make_openmp_timer(const char * n, const char * t);

void TauStartOpenMPRegionTimer(my_pomp2_region *r, int index)
{
/* For any region, create a mapping between a region r and timer t and
   start the timer. */

  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  if(r == NULL) {
    printf("TAU WARNING: a POMP2 Region was not initialized.  Something went wrong during the creation of pompregions.c\n");
  }

  FunctionInfo **flist = (FunctionInfo **)(r->data);
  static std::mutex mtx;

#ifdef TAU_OPENMP_PARTITION_REGION
  if (!r->data) {
    // only one thread should create the array.
    std::lock_guard<std::mutex> guard(mtx);
    // make sure some other thread hasn't created it.
    if (!r->data) {
      flist = new FunctionInfo*[NUM_OMP_TYPES];
	  // initialize the array to be null pointers
      for (int i=0; i < NUM_OMP_TYPES; i++) {
	    flist[i] = NULL;
      }
      // save the list of timers to the region
      r->data = (void*)flist;
	}
    flist = (FunctionInfo **)(r->data);
  }

  // does the timer we want exist?
  if (flist[index] == NULL) {
    // only one thread should create the timers.
    std::lock_guard<std::mutex> guard(mtx);
     // make sure some other thread hasn't created it.
    if (flist[index] == NULL) {
      char rname[1024], rtype[1024];
      snprintf(rname, sizeof(rname),  "%s (%s)",  r->rtype, omp_names[index]);
      snprintf(rtype, sizeof(rtype),  "[OpenMP location: file:%s <%d, %d>]",
      r->start_file_name, r->start_line_1, r->end_line_1);
      flist[index] = Tau_make_openmp_timer(rname, rtype);
    }
  }

  FunctionInfo *f = flist[index];

#else // not TAU_OPENMP_PARTITION_REGION

  if (!r->data) {
    std::lock_guard<std::mutex> guard(mtx);
    // make sure some other thread hasn't created the timer.
    if (!r->data) {
      // create the timer for this region
      char rname[1024], rtype[1024];
      snprintf(rname, sizeof(rname),  "%s", r->rtype);
      snprintf(rtype, sizeof(rtype),  "[OpenMP location: file:%s <%d, %d>]",
	      r->start_file_name, r->start_line_1, r->end_line_1);

      FunctionInfo *f = Tau_make_openmp_timer(rname, rtype);
      r->data = (void*)f;
    }
  }
  FunctionInfo *f = (FunctionInfo *)r->data;
#endif
  Tau_start_timer(f, 0, Tau_get_thread());
}

void TauStopOpenMPRegionTimer(my_pomp2_region  *r, int index)
{

#ifdef TAU_OPENMP_PARTITION_REGION
    FunctionInfo *f = ((FunctionInfo **)r->data)[index];
#else
    FunctionInfo *f = (FunctionInfo *)r->data;
#endif
      Tau_stop_timer(f, Tau_get_thread());
//This silently ignored bugs,
//Let the measurement layer deal with problems with the profiler
//And report any errors
/*    TauGroup_t gr = f->GetProfileGroup();

    int tid = RtsLayer::myThread();
    Profiler *p =TauInternal_CurrentProfiler(tid);
    if(p == NULL){
      // nothing, it must have been disabled/throttled
    } else if (p->ThisFunction == f) {

      Tau_stop_timer(f, Tau_create_tid());
    } else {
      // nothing, it must have been disabled/throttled
    }*/
}
/*
 * Global variables
 */

int              pomp2_tracing = 0;
my_pomp2_region* my_pomp2_regions;

/*
 * C pomp2 function library
 */



void POMP2_Finalize()
{
  static int pomp2_finalize_called = 0;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  size_t i;
  const size_t nRegions = POMP2_Get_num_regions();

  if (my_pomp2_regions) {
    for (i = 0; i < nRegions; ++i) {
      free_my_pomp2_region_members(&my_pomp2_regions[i]);
    }
    free(my_pomp2_regions);
    my_pomp2_regions = 0;
  }

  while ( tau_region_list_top != NULL) {
    my_pomp2_region_node * next = tau_region_list_top->next;
    free (tau_region_list_top);
    tau_region_list_top = next;
  }

  if (!pomp2_finalize_called) {
    pomp2_finalize_called = 1;
#ifdef DEBUG_PROF
    TAU_VERBOSE( "  0: finalize\n" );
#endif /* DEBUG_PROF */
  }
}

void POMP2_Init()
{
  static int pomp2_init_called = 0;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  if (!pomp2_init_called) {
    pomp2_init_called = 1;

    atexit(POMP2_Finalize);
#ifdef DEBUG_PROF
    TAU_VERBOSE( "  0: init  code\n" );
#endif /* DEBUG_PROF */

    /* Allocate memory for your POMP2_Get_num_regions() regions */
#ifdef OLD_OPARI2
    my_pomp2_regions = (my_pomp2_region *)(calloc(POMP2_Get_num_regions(), sizeof(my_pomp2_region)));
    //pomp2_tpd_ = ( void* )malloc( sizeof( int ) );
    //pomp2_tpd_ = ( long )0;

    POMP2_Init_regions();
#endif /* OLD_OPARI2 */

    int n_pomp2_regions = POMP2_Get_num_regions() + POMP2_USER_Get_num_regions();
    int n_pomp2_user_regions = POMP2_USER_Get_num_regions();

    my_pomp2_regions = (my_pomp2_region *) calloc( n_pomp2_regions + n_pomp2_user_regions,
                                   sizeof( my_pomp2_region ) );

    if ( n_pomp2_regions > 0 ) {
      POMP2_Init_regions();
    }

    if ( n_pomp2_regions > 0 ) {
      POMP2_USER_Init_regions();
    }

    pomp2_tracing = 1;
  }
}

extern "C" void
POMP2_Off()
{
    pomp2_tracing = 0;
}

extern "C" void
POMP2_On()
{
    pomp2_tracing = 1;
}

extern "C" void
POMP2_Begin( POMP2_USER_Region_handle* pomp2_handle,
             const char                ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
  }
  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;
//my_pomp2_region* region = *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tregion);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_INST_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: begin region %s\n",
        omp_get_thread_num(), region->name );
  }
#endif /* DEBUG_PROF */
}

void POMP2_End(POMP2_Region_handle* pomp2_handle)
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
  }
  //my_pomp2_region* region = *pomp2_handle;
  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_INST_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tregion);
  /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: end   region %s\n",
        omp_get_thread_num(), region->name );
  }
#endif /* DEBUG_PROF */
}


my_pomp2_region* TauAllocateRegionOnTheFly(void) {
   my_pomp2_region_node * node = (my_pomp2_region_node *)malloc( sizeof(my_pomp2_region_node));
 //  my_pomp2_region * new_region = (my_pomp2_region *)malloc( sizeof(my_pomp2_region));
   node->next = tau_region_list_top;
   tau_region_list_top  = node;
   node->region.data = (void *) NULL;
   return &node->region;
}

void POMP2_Assign_handle(POMP2_Region_handle* pomp2_handle, const char ctc_string[])
{
  static size_t count = 0;
  //printf( "%d POMP2_Assign_handle: \"%s\"\n", (int)count, ctc_string );

  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  my_pomp2_region* new_handle = (count < POMP2_Get_num_regions() ?
	& my_pomp2_regions[count] : TauAllocateRegionOnTheFly() );

  POMP2_Region_info pomp2RegionInfo;
  ctcString2RegionInfo(ctc_string, &pomp2RegionInfo);

  initDummyRegionFromPOMP2RegionInfo(new_handle, &pomp2RegionInfo);
  new_handle->id = count;
#ifdef DEBUG_PROF
  TAU_VERBOSE( "assign_handle %d %s\n", ( int )count, new_handle->rtype );
#endif /* DEBUG_PROF */

  *pomp2_handle = new_handle;

  freePOMP2RegionInfoMembers(&pomp2RegionInfo);
  ++count;

}

extern "C" void
POMP2_USER_Assign_handle( POMP2_USER_Region_handle* pomp2_handle,
                          const char                ctc_string[] )
{
    static size_t count = 0;
    assert( count < POMP2_Get_num_regions() );

    POMP2_USER_Region_info pomp2RegionInfo;
    ctcString2UserRegionInfo( ctc_string, &pomp2RegionInfo );

    initDummyRegionFromPOMP2UserRegionInfo( &my_pomp2_regions[ count ], &pomp2RegionInfo );
    my_pomp2_regions[ count ].id = count;

    *pomp2_handle = &my_pomp2_regions[ count ];

    freePOMP2UserRegionInfoMembers( &pomp2RegionInfo );
    ++count;
}

extern "C" void POMP2_Atomic_enter(POMP2_Region_handle* pomp2_handle, const char ctc_string[])
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
    POMP2_Assign_handle(pomp2_handle, ctc_string);
  }
#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tatomic);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer((my_pomp2_region*)*pomp2_handle, TAU_OMP_ATOMIC);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: enter atomic\n", omp_get_thread_num() );
  }
#endif /* DEBUG_PROF */
}

extern "C" void POMP2_Atomic_exit(POMP2_Region_handle* pomp2_handle)
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
  }

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer((my_pomp2_region*)*pomp2_handle, TAU_OMP_ATOMIC);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tatomic);
  /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: exit  atomic\n", omp_get_thread_num() );
  }
#endif /* DEBUG_PROF */
}



extern "C" void POMP2_Barrier_enter(POMP2_Region_handle* pomp2_handle, POMP2_Task_handle* pomp2_old_task, const char ctc_string[])
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  *pomp2_old_task = pomp2_current_task;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
    POMP2_Assign_handle(pomp2_handle, ctc_string);
  }
  //my_pomp2_region* region = *pomp2_handle;
  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tbarrier);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_BARRIER);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    if ( region->rtype[ 0 ] == 'b' )
    {
      TAU_VERBOSE( "%3d: enter barrier\n", omp_get_thread_num() );
    }
    else
    {
      TAU_VERBOSE( "%3d: enter implicit barrier of %s\n",
          omp_get_thread_num(), region->rtype );
    }
  }
#endif /* DEBUG_PROF */
}

extern "C" void POMP2_Barrier_exit(POMP2_Region_handle* pomp2_handle, POMP2_Task_handle pomp2_old_task)
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  pomp2_old_task = pomp2_current_task;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
  }
  //my_pomp2_region* region = *pomp2_handle;
  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_BARRIER);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tbarrier);
  /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    if ( region->rtype[ 0 ] == 'b' )
    {
      TAU_VERBOSE( "%3d: exit  barrier\n", omp_get_thread_num() );
    }
    else
    {
      TAU_VERBOSE( "%3d: exit  implicit barrier of %s\n",
          omp_get_thread_num(), region->rtype );
    }
  }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Implicit_barrier_enter( POMP2_Region_handle* pomp2_handle,POMP2_Task_handle*   pomp2_old_task )
{
  TauInternalFunctionGuard protects_this_function;
  POMP2_Barrier_enter( pomp2_handle, pomp2_old_task,  "" );
}

extern "C" void
POMP2_Implicit_barrier_exit( POMP2_Region_handle* pomp2_handle, POMP2_Task_handle   pomp2_old_task )
{
  TauInternalFunctionGuard protects_this_function;
  POMP2_Barrier_exit( pomp2_handle, pomp2_old_task );
}

extern "C" void POMP2_Flush_enter(POMP2_Region_handle* pomp2_handle, const char ctc_string[])
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
    POMP2_Assign_handle(pomp2_handle, ctc_string);
  }

  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;
#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tflush);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_FLUSH_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: enter flush\n", omp_get_thread_num() );
  }
#endif /* DEBUG_PROF */
}

extern "C" void POMP2_Flush_exit(POMP2_Region_handle* pomp2_handle)
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
  }

  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;
#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_FLUSH_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  //TAU_UGLOBAL_TIMER_STOP(tflush);
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tflush);
  /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: exit  flush\n", omp_get_thread_num() );
  }
#endif /* DEBUG_PROF */
}

extern "C" void POMP2_Critical_begin(POMP2_Region_handle* pomp2_handle)
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  if (*pomp2_handle == NULL) {
    POMP2_Init();
  }
  //my_pomp2_region* region = *pomp2_handle;
  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;
#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tcriticalb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_CRITICAL_BE);
#endif /* TAU_OPENMP_REGION_VIEW */
#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: begin critical %s\n",
        omp_get_thread_num(), region->rtype );
  }
#endif /* DEBUG_PROF */
}

extern "C" void POMP2_Critical_end(POMP2_Region_handle* pomp2_handle)
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  if (*pomp2_handle == NULL) {
    POMP2_Init();
  }
  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;
  //my_pomp2_region* region = *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_CRITICAL_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tcriticalb);
  /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: end   critical %s\n",
        omp_get_thread_num(), region->name );
  }
#endif /* DEBUG_PROF */
}

extern "C" void POMP2_Critical_enter(POMP2_Region_handle* pomp2_handle, const char ctc_string[])
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
  if (*pomp2_handle == NULL) {
    POMP2_Init();
    POMP2_Assign_handle(pomp2_handle, ctc_string);
  }
  //my_pomp2_region* region = *pomp2_handle;
  my_pomp2_region* region = (my_pomp2_region*)*pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tcriticale);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_CRITICAL_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( pomp2_tracing )
  {
    TAU_VERBOSE( "%3d: enter critical %s\n",
        omp_get_thread_num(), region->name );
  }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Critical_exit( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    //my_pomp2_region* region = *pomp2_handle;
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_CRITICAL_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tcriticale); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */



#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: exit  critical %s\n",
                 omp_get_thread_num(), region->name );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_For_enter( POMP2_Region_handle* pomp2_handle, const char ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
        POMP2_Assign_handle(pomp2_handle, ctc_string);
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;
#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tfor);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_FOR_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: enter for\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_For_exit( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;
#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_FOR_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tfor); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  // as in a stack. lifo

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: exit  for\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Master_begin( POMP2_Region_handle* pomp2_handle, const char ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
        POMP2_Assign_handle(pomp2_handle, ctc_string);
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tmaster);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_MASTER_BE);
#endif /* TAU_OPENMP_REGION_VIEW */


#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: begin master\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Master_end( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_MASTER_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tmaster); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: end   master\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Parallel_begin( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tparallelb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_PAR_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: begin parallel\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Parallel_end( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_PAR_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tparallelb); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */


#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: end   parallel\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Parallel_fork( POMP2_Region_handle* pomp2_handle,
                     int                  if_clause,
                     int                  num_threads,
                     POMP2_Task_handle*   pomp2_old_task,
                     const char           ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    *pomp2_old_task = pomp2_current_task;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
        POMP2_Assign_handle(pomp2_handle, ctc_string);
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

  Tau_create_top_level_timer_if_necessary_task(Tau_get_thread());

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tparallelf);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_PAR_FJ);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: fork  parallel\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Parallel_join( POMP2_Region_handle* pomp2_handle, POMP2_Task_handle   pomp2_old_task )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    pomp2_old_task = pomp2_current_task;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_PAR_FJ);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tparallelf); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: join  parallel\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Section_begin( POMP2_Region_handle* pomp2_handle, const char ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
        POMP2_Assign_handle(pomp2_handle, ctc_string);
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

  Tau_create_top_level_timer_if_necessary_task(Tau_get_thread());

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tsectionb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_SECTION_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: begin section\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Section_end( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_SECTION_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tsectionb); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: end   section\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Sections_enter( POMP2_Region_handle* pomp2_handle, const char ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
        POMP2_Assign_handle(pomp2_handle, ctc_string);
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tsectione);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_SECTION_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: enter sections (%d)\n",
                 omp_get_thread_num(), region->num_sections );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Sections_exit( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_SECTION_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tsectione); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: exit  sections\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Single_begin( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tsingleb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_SINGLE_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: begin single\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Single_end( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_SINGLE_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tsingleb); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: end   single\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Single_enter( POMP2_Region_handle* pomp2_handle, const char ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
        POMP2_Assign_handle(pomp2_handle, ctc_string);
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tsinglee);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_SINGLE_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: enter single\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Single_exit( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_SINGLE_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tsinglee); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: exit  single\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Workshare_enter( POMP2_Region_handle* pomp2_handle, const char ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
        POMP2_Assign_handle(pomp2_handle, ctc_string);
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tworkshare);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_WORK_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: enter workshare\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Workshare_exit( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_WORK_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tworkshare); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: exit  workshare\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
}

extern "C" void
POMP2_Ordered_begin( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(torderedb); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_ORDERED_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF

    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: begin ordered\n", omp_get_thread_num() );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Ordered_end( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_ORDERED_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(torderedb); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF


    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: end ordered\n", omp_get_thread_num() );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Ordered_enter( POMP2_Region_handle* pomp2_handle,
                    const char           ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
        POMP2_Assign_handle(pomp2_handle, ctc_string);
    }

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(torderede); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_ORDERED_EE);
#endif /* TAU_OPENMP_REGION_VIEW */


#ifdef DEBUG_PROF


    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: enter ordered\n", omp_get_thread_num() );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Ordered_exit( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

#pragma omp critical
    if ( *pomp2_handle == NULL )
    {
        POMP2_Init();
    }

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_ORDERED_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(torderede); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF

    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: exit ordered\n", omp_get_thread_num() );
    }
#endif /*DEBUG_PROF*/
}


extern "C" void
POMP2_Task_create_begin( POMP2_Region_handle* pomp2_handle,
                         POMP2_Task_handle*   pomp2_old_task,
                         POMP2_Task_handle*   pomp2_new_task,
                         int                  pomp2_if,
                         const char           ctc_string[])
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    *pomp2_old_task = pomp2_current_task;
    *pomp2_new_task = POMP2_Get_new_task_handle();

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(ttaskcreate); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_TASK_CREATE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: task create begin\n", omp_get_thread_num() );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Task_create_end( POMP2_Region_handle* pomp2_handle,
                       POMP2_Task_handle    pomp2_old_task )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    pomp2_current_task = pomp2_old_task;
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_TASK_CREATE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(ttaskcreate); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF


   if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: task create end\n", omp_get_thread_num() );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Task_begin( POMP2_Region_handle* pomp2_handle,
                  POMP2_Task_handle    pomp2_task )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    pomp2_current_task = pomp2_task;

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(ttaskcreate); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_TASK);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF


    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: task begin\n", omp_get_thread_num() );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Task_end( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_TASK);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(ttaskcreate); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF

    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: task end\n", omp_get_thread_num());
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Untied_task_create_begin( POMP2_Region_handle* pomp2_handle,
                                POMP2_Task_handle*   pomp2_new_task,
                                POMP2_Task_handle*   pomp2_old_task,
                                int                  pomp2_if,
                                const char           ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    *pomp2_new_task = POMP2_Get_new_task_handle();
    *pomp2_old_task = pomp2_current_task;

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tuntiedcreate); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_UNTIED_TASK_CREATE_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF


    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: create  untied task\n", omp_get_thread_num() );
        TAU_VERBOSE( "%3d:         suspend task %lld\n", omp_get_thread_num(), pomp2_current_task );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Untied_task_create_end( POMP2_Region_handle* pomp2_handle,
                              POMP2_Task_handle    pomp2_old_task )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    pomp2_current_task = pomp2_old_task;

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_UNTIED_TASK_CREATE_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tuntiedcreate); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF


    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: created  untied task\n", omp_get_thread_num() );
        TAU_VERBOSE( "%3d:          resume task %lld\n", omp_get_thread_num(), pomp2_current_task );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Untied_task_begin( POMP2_Region_handle* pomp2_handle,
                         POMP2_Task_handle    pomp2_parent_task )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    pomp2_current_task = POMP2_Get_new_task_handle();


    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(tuntied); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_UNTIED_TASK_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF

    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: start  untied task %lld\n", omp_get_thread_num(), pomp2_current_task );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Untied_task_end( POMP2_Region_handle* pomp2_handle )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_UNTIED_TASK_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(tuntied); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF

    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: end  untied task %lld\n", omp_get_thread_num(), pomp2_current_task );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Taskwait_begin( POMP2_Region_handle* pomp2_handle,
                      POMP2_Task_handle*   pomp2_old_task,
                      const char           ctc_string[] )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    *pomp2_old_task = pomp2_current_task;

    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_START(ttaskwait); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(region, TAU_OMP_TASKWAIT_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF


    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: begin  taskwait\n", omp_get_thread_num() );
        TAU_VERBOSE( "%3d:  suspend task: %lld\n", omp_get_thread_num(), pomp2_current_task );
    }
#endif /*DEBUG_PROF*/
}

extern "C" void
POMP2_Taskwait_end( POMP2_Region_handle* pomp2_handle,
                    POMP2_Task_handle    pomp2_old_task )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

    pomp2_current_task = pomp2_old_task;
    my_pomp2_region* region = ( my_pomp2_region*) *pomp2_handle;

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(region, TAU_OMP_TASKWAIT_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_OPARI_CONSTRUCT_TIMER_STOP(ttaskwait); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF



    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: end  taskwait\n", omp_get_thread_num() );
        TAU_VERBOSE( "%3d: resume task: %lld\n", omp_get_thread_num(), pomp2_current_task );
    }
#endif /*DEBUG_PROF*/
}


/*
   *----------------------------------------------------------------
 * C Wrapper for OpenMP API
 ******----------------------------------------------------------------
 */

extern "C" void
POMP2_Init_lock( omp_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_init_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: init lock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    omp_init_lock( s );
}

extern "C" void
POMP2_Destroy_lock( omp_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_destroy_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: destroy lock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    omp_destroy_lock( s );
}

extern "C" void
POMP2_Set_lock( omp_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_set_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: set lock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    omp_set_lock( s );
}

extern "C" void
POMP2_Unset_lock( omp_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_unset_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: unset lock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    omp_unset_lock( s );
}

extern "C" int
POMP2_Test_lock( omp_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_test_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: test lock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    return omp_test_lock( s );
}

extern "C" void
POMP2_Init_nest_lock( omp_nest_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_init_nest_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: init nestlock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    omp_init_nest_lock( s );
}

extern "C" void
POMP2_Destroy_nest_lock( omp_nest_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_destroy_nest_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: destroy nestlock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    omp_destroy_nest_lock( s );
}

extern "C" void
POMP2_Set_nest_lock( omp_nest_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_set_nest_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: set nestlock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    omp_set_nest_lock( s );
}

extern "C" void
POMP2_Unset_nest_lock( omp_nest_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_unset_nest_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: unset nestlock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    omp_unset_nest_lock( s );
}

extern "C" int
POMP2_Test_nest_lock( omp_nest_lock_t* s )
{
  // Automatically increment and decrement insideTAU
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("omp_test_nest_lock", "[OpenMP]", OpenMP);

#ifdef DEBUG_PROF
    if ( pomp2_tracing )
    {
        TAU_VERBOSE( "%3d: test nestlock\n", omp_get_thread_num() );
    }
#endif /* DEBUG_PROF */
    return omp_test_nest_lock( s );
}

/* *INDENT-OFF*  */
/** @todo include again if c part is ready */
#if 0
/*
   *----------------------------------------------------------------
 * Fortran  Wrapper for OpenMP API
 ******----------------------------------------------------------------
 */
/* *INDENT-OFF*  */
#if defined(__ICC) || defined(__ECC) || defined(_SX)
#define CALLFSUB(a) a
#else
#define CALLFSUB(a) FSUB(a)
#endif

void FSUB(POMP2_Init_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: init lock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_init_lock)(s);
}

void FSUB(POMP2_Destroy_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: destroy lock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_destroy_lock)(s);
}

void FSUB(POMP2_Set_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: set lock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_set_lock)(s);
}

void FSUB(POMP2_Unset_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: unset lock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_unset_lock)(s);
}

int  FSUB(POMP2_Test_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: test lock\n", omp_get_thread_num());
  }
  return CALLFSUB(omp_test_lock)(s);
}

#ifndef __osf__
void FSUB(POMP2_Init_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: init nestlock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_init_nest_lock)(s);
}

void FSUB(POMP2_Destroy_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: destroy nestlock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_destroy_nest_lock)(s);
}

void FSUB(POMP2_Set_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: set nestlock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_set_nest_lock)(s);
}

void FSUB(POMP2_Unset_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: unset nestlock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_unset_nest_lock)(s);
}

int  FSUB(POMP2_Test_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    TAU_VERBOSE("%3d: test nestlock\n", omp_get_thread_num());
  }
  return CALLFSUB(omp_test_nest_lock)(s);
}
#endif
#endif /*0*/
