/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2013,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2013,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2014,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2013,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2013,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2013,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */


/**
 *  @file
 *
 *  @brief      This tool prints out all event files of an archive to console.
 *
 *  @return                  Returns EXIT_SUCCESS if successful, EXIT_FAILURE
 *                           if an error occures.
 */

#include <config.h>

#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>

#include <otf2/otf2.h>
extern "C" {
#include "otf2_hash_table.h"
}
#include "otf2_vector.h"

#include <UTILS_IO.h>

#include <trace2profile.h>
#include <handlers.h>


/* ___ Shorthand macros. ____________________________________________________ */


#define PRIUint8      PRIu8
#define PRIInt8       PRId8
#define PRIUint16     PRIu16
#define PRIInt16      PRId16
#define PRIUint32     PRIu32
#define PRIInt32      PRId32
#define PRIUint64     PRIu64
#define PRIUint64Full PRIu64
#define PRIInt64      PRId64
#define PRIFloat      "f"
#define PRIDouble     "f"

#define BUFFER_SIZE 128


/** @internal
 *  @brief Type used to indicate a reference to a @eref{String} definition */
typedef uint64_t OTF2_StringRef64;

/* ___ Global variables. ____________________________________________________ */


/** @internal
 *  @brief Name of the program. */
static const char* otf2_NAME;

/** @internal
 *  @brief Defines if debug is turned on (1) or off (0). */
static bool otf2_DEBUG;

/** @internal
 *  @brief Defines if all data is printed (1) or not (0). */
static bool otf2_ALL;

/** @internal
 *  @brief Defines if global definitions are printed (1) or not (0). */
static bool otf2_GLOBDEFS;

/** @internal
 *  @brief Defines if information from anchor file are printed (1) or not (0). */
static bool otf2_ANCHORFILE_INFO;

/** @internal
 *  @brief Defines if thumbnail headers should be printed. */
static bool otf2_THUMBNAIL_INFO;

/** @internal
 *  @brief Defines if thumbnail headers should be printed. */
static bool otf2_THUMBNAIL_SAMPLES;

/** @internal
 *  @brief Defines if a single location is selected. */
static uint64_t otf2_LOCAL = OTF2_UNDEFINED_LOCATION;

/** @internal
 *  @brief Tell if a local location was found (1) or not (0). */
static bool otf2_LOCAL_FOUND;

/** @internal
 *  @brief Defines lower bound of selected time interval. */
static uint64_t otf2_MINTIME;

/** @internal
 *  @brief Defines upper bound of selected time interval. */
static uint64_t otf2_MAXTIME = OTF2_UNDEFINED_UINT64;

/** @internal
 *  @brief Defines number of printed events in each step (UINT64_MAX means unlimited). */
static uint64_t otf2_STEP = OTF2_UNDEFINED_UINT64;

/** @internal
 *  @brief Defines if events are printed or not. */
static bool otf2_SILENT = false;
static bool print_SILENT = true;

/** @internal
 *  @brief Defines if dot output is selected. */
static bool otf2_DOT;

/** @internal
 *  @brief Defines if we want to see the mapping tables. */
static bool otf2_MAPPINGS;

/** @internal
 *  @brief Defines if we want to see timer synchronizations. */
static bool otf2_CLOCK_OFFSETS;

/** @internal
 *  @brief Don't read local defs, to prevent the reader to apply mappings
 *         and clock corrections. */
static bool otf2_NOLOCALDEFS;

/** @internal
 *  @brief Print also any snapshots. */
static bool otf2_NOSNAPSHOTS;

/** @internal
 *  @brief width of the column with the anchor file information. */
static int otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH = 30;


/** @internal
 *  @brief width of the column with the anchor file information. */
static int otf2_THUMBNAIL_HEADER_COLUMN_WIDTH = 16;


/** @internal
 *  @brief print the backtrace for a calling context sample. */
static bool otf2_UNWIND_CALLING_CONTEXT = false;


static enum otf2_print_timestamp_format
{
    TIMESTAMP_PLAIN,
    TIMESTAMP_OFFSET
} otf2_TIMESTAMP_FORMAT;


/* ___ Structures. __________________________________________________________ */


/** @internal
 *  @brief Keeps all data for the callbacks. */
struct otf2_print_data
{
    /** @brief Reader handle. */
    OTF2_Reader* reader;
    /** @brief List of locations to process. */
    otf2_vector* locations_to_read;

    /** @brief Collected definitions. */
    struct otf2_print_defs* defs;

    /** @brief Clock properties. */
    uint64_t timer_resolution;
    uint64_t global_offset;
    uint64_t trace_length;

    /** @brief Number of the artifical string refs in the hash table. */
    OTF2_StringRef64 artificial_string_refs;

    /** @brief Defined COMM_LOCATIONS groups. Indexed via OTF2_Paradigm. */
    otf2_vector* comm_paradigms;

    /** @brief File handle for dot output. */
    FILE* dot_file;
};

/** @internal
 *  @brief Paradigm definition element. */
struct otf2_print_paradigm_def
{
    OTF2_Paradigm      paradigm;
    char*              name;
    OTF2_ParadigmClass paradigmClass;
};


/** @internal
 *  @brief Region definition element. */
struct otf2_print_def_name
{
    /** @brief The ID of the definition. */
    uint64_t def;
    /** @brief The name if the definition. */
    char*    name;
};


/** @internal
 *  @brief Metric definition element. */
struct otf2_print_metric_def
{
    struct otf2_print_def_name def;
    uint8_t                    number_of_members;
    OTF2_MetricMemberRef       members[];
};


/** @internal
 *  @brief Group definition element. */
struct otf2_print_group_def
{
    struct otf2_print_def_name def;
    OTF2_GroupType             type;
    OTF2_GroupFlag             flags;
    OTF2_Paradigm              paradigm;
    /** COMM_LOCATIONS => pointer to the COMM_SELF gorup
        COMM_GROUP,COMM_SELF => pointer to the COMM_LOCATIONS group */
    struct otf2_print_group_def* comm_data;
    uint32_t                     number_of_members;
    uint64_t                     members[];
};


/** @internal
 *  @brief Comm definition element. */
struct otf2_print_comm_def
{
    struct otf2_print_def_name         def;
    const struct otf2_print_group_def* comm_group;
};


/** @internal
 *  @brief RmaWin definition element. */
struct otf2_print_rma_win_def
{
    struct otf2_print_def_name        def;
    const struct otf2_print_comm_def* comm;
};


/** @internal
 *  @brief CartTopology definition element. */
struct otf2_print_cart_topology_def
{
    struct otf2_print_def_name        def;
    const struct otf2_print_comm_def* comm;
};


/** @internal
 *  @brief Calling context definition property. */
struct otf2_print_calling_context_property
{
    struct otf2_print_calling_context_property* next;
    OTF2_StringRef                              name;
    OTF2_Type                                   type;
    OTF2_AttributeValue                         value;
};


/** @internal
 *  @brief Calling context definition element. */
struct otf2_print_calling_context_def
{
    struct otf2_print_def_name def;
    uint64_t                   ip;
    OTF2_RegionRef             region;
    OTF2_SourceCodeLocationRef scl;
    OTF2_CallingContextRef     parent;

    /* property list */
    struct otf2_print_calling_context_property*  properties_head;
    struct otf2_print_calling_context_property** properties_tail;
};


/* ___ Prototypes for static functions. _____________________________________ */

static void
otf2_print_die( const char* fmt,
                ... )
{
    if ( fmt )
    {
        va_list va;
        fprintf( stderr, "%s: ", otf2_NAME );
        va_start( va, fmt );
        vfprintf( stderr, fmt, va );
        va_end( va );
    }
    fprintf( stderr, "Try '%s --help' for more information.\n", otf2_NAME );
    exit( EXIT_FAILURE );
}

static void
otf2_print_warn( const char* fmt,
                 ... )
{
    va_list va;
    fprintf( stderr, "%s: warning: ", otf2_NAME );
    va_start( va, fmt );
    vfprintf( stderr, fmt, va );
    fflush( stderr );
    va_end( va );
}

static void
otf2_print_anchor_file_information( OTF2_Reader* reader );

static void
otf2_print_thumbnails( OTF2_Reader* reader );

static void
otf2_get_parameters( int    argc,
                     char** argv,
                     char** anchorFile );

static void
check_pointer( void* pointer,
               const char* description,
               ... );

static void
check_status( OTF2_ErrorCode status,
              const char*          description,
              ... );

static void
check_condition( bool  condition,
                 const char* description,
                 ... );

static void
otf2_print_add_clock_properties( struct otf2_print_data* data,
                                 uint64_t                timerResolution,
                                 uint64_t                globalOffset,
                                 uint64_t                traceLength );

static const char*
otf2_print_get_timestamp( struct otf2_print_data* data,
                          OTF2_TimeStamp          time );

static void
otf2_print_add_location_to_read( struct otf2_print_data* data,
                                 OTF2_LocationRef        location );

static void
otf2_print_add_string( otf2_hash_table* strings,
                       OTF2_StringRef64 string,
                       size_t           content_len,
                       const char*      content_fmt,
                       ... );

static void
otf2_print_add_def64_name( const char*      def_name,
                           otf2_hash_table* defs,
                           otf2_hash_table* strings,
                           uint64_t         def,
                           OTF2_StringRef64 string );

static void
otf2_print_add_metric( otf2_hash_table*            metrics,
                       OTF2_MetricRef              metric,
                       OTF2_MetricRef              metricClass,
                       uint8_t                     numberOfMembers,
                       const OTF2_MetricMemberRef* metricMembers );

static const struct otf2_print_metric_def*
otf2_print_get_metric( otf2_hash_table* metrics,
                       OTF2_MetricRef   metric );

static void
otf2_print_add_group( struct otf2_print_data* data,
                      OTF2_GroupRef           group,
                      OTF2_StringRef          name,
                      OTF2_GroupType          type,
                      OTF2_Paradigm           paradigm,
                      OTF2_GroupFlag          flags,
                      uint32_t                numberOfMembers,
                      const uint64_t*         members );

static const struct otf2_print_group_def*
otf2_print_get_group( otf2_hash_table* groups,
                      OTF2_GroupRef    group );

static void
otf2_print_add_comm( struct otf2_print_data* data,
                     OTF2_CommRef            comm,
                     OTF2_StringRef          name,
                     OTF2_GroupRef           group,
                     OTF2_CommRef            parent );

static const struct otf2_print_comm_def*
otf2_print_get_comm( otf2_hash_table* comms,
                     OTF2_CommRef     comm );

static const char*
otf2_print_comm_get_rank_name( struct otf2_print_defs* defs,
                               OTF2_LocationRef        location,
                               OTF2_CommRef            comm,
                               uint32_t                rank );

static void
otf2_print_add_rma_win( struct otf2_print_data* data,
                        OTF2_RmaWinRef          rmaWin,
                        OTF2_StringRef          name,
                        OTF2_CommRef            comm );

static const struct otf2_print_rma_win_def*
otf2_print_get_rma_win( otf2_hash_table* rmaWins,
                        OTF2_RmaWinRef   rmaWin );

static const char*
otf2_print_rma_win_get_rank_name( struct otf2_print_defs* defs,
                                  OTF2_LocationRef        location,
                                  OTF2_RmaWinRef          rmaWin,
                                  uint32_t                rank );

static void
otf2_print_add_cart_topology( struct otf2_print_data* data,
                              OTF2_CartTopologyRef    cartTopology,
                              OTF2_StringRef          name,
                              OTF2_CommRef            comm );

static const struct otf2_print_cart_topology_def*
otf2_print_get_cart_topology( otf2_hash_table*     cartTopologies,
                              OTF2_CartTopologyRef cartTopology );

static const char*
otf2_print_cart_topology_get_rank_name( struct otf2_print_defs* defs,
                                        OTF2_LocationRef        location,
                                        OTF2_CartTopologyRef    cartTopology,
                                        uint32_t                rank );

static void
otf2_print_add_calling_context( struct otf2_print_data*    data,
                                OTF2_CallingContextRef     self,
                                OTF2_RegionRef             region,
                                OTF2_SourceCodeLocationRef sourceCodeLocation,
                                OTF2_CallingContextRef     parent );

static struct otf2_print_calling_context_def*
otf2_print_get_calling_context( otf2_hash_table*       callingContexts,
                                OTF2_CallingContextRef callingContext );

static void
otf2_print_add_calling_context_property( struct otf2_print_defs* defs,
                                         OTF2_CallingContextRef  callingContext,
                                         OTF2_StringRef          name,
                                         OTF2_Type               type,
                                         OTF2_AttributeValue     value );

static const char*
otf2_print_get_rank_name( uint64_t    rank,
                          const char* rankName );

static char*
otf2_print_get_buffer( size_t len );

static const char*
otf2_print_get_id64( uint64_t ID );

static const char*
otf2_print_get_name( const char* name,
                     uint64_t    ID );

static const char*
otf2_print_get_def64_name( const otf2_hash_table* defs,
                           uint64_t               def );

static const char*
otf2_print_get_def_name( const otf2_hash_table* defs,
                         uint32_t               def );

static const char*
otf2_print_get_def64_raw_name( const otf2_hash_table* defs,
                               uint64_t               def );

static const char*
otf2_print_get_def_raw_name( const otf2_hash_table* defs,
                             uint32_t               def );

static const char*
otf2_print_get_paradigm_name( const otf2_hash_table* paradigms,
                              OTF2_Paradigm          paradigm );

static char*
otf2_print_get_string( const otf2_hash_table* strings,
                       OTF2_StringRef64       string );

static const char*
otf2_print_get_attribute_value( struct otf2_print_defs* defs,
                                OTF2_Type               type,
                                OTF2_AttributeValue     value
                                #if OTF2_VERSION_MAJOR > 2
				,
				bool                    raw 
				#endif
				);

static const char*
otf2_print_get_paradigm_property_value( struct otf2_print_defs* defs,
                                        OTF2_ParadigmProperty   property,
                                        OTF2_Type               type,
                                        OTF2_AttributeValue     attributeValue,
                                        const char**            typeString );

static const char*
otf2_print_get_invalid( uint64_t ID );

#if OTF2_VERSION_MAJOR > 2
extern "C" {
#include "otf2_print_types.h"
}
#else
#include "otf2-print-2.3/otf2_print_types.h"
#endif

/* ___ Prototypes for event callbacks. ______________________________________ */


static OTF2_CallbackCode
print_unknown( OTF2_LocationRef    location,
               OTF2_TimeStamp      time,
               void*               userData,
               OTF2_AttributeList* attributes );


static OTF2_CallbackCode
print_global_def_unknown( void* userData );

#if OTF2_VERSION_MAJOR > 2
#include "otf2_print_inc.c"
#else
#include  "otf2-print-2.3/otf2_print_inc.c"
#endif

static OTF2_CallbackCode
print_def_mapping_table( void*             userData,
                         OTF2_MappingType  mapType,
                         const OTF2_IdMap* iDMap );

static OTF2_CallbackCode
print_def_clock_offset( void*    userData,
                        uint64_t time,
                        int64_t  offset,
                        double   stddev );


/* ___ main _________________________________________________________________ */


int
//main( int    argc, char** argv )
ReadTraceFile( int    argc, char** argv )
{

	StateGroupDef(0, "Empty" );
    otf2_NAME = strrchr( argv[ 0 ], '/' );
    if ( otf2_NAME )
    {
        otf2_NAME++;
    }
    else
    {
        otf2_NAME = argv[ 0 ];
    }

    char* anchor_file = NULL;
    otf2_get_parameters( argc, argv, &anchor_file );

    if ( otf2_NOLOCALDEFS && ( otf2_MAPPINGS || otf2_CLOCK_OFFSETS ) )
    {
        otf2_print_die( "--no-local-defs is mutual exclusive to --show-mappings and --show-clock-offsets\n" );
    }

    //printf( "\n=== OTF2-PRINT ===\n" );

    /* Get a reader handle. */
    OTF2_Reader* reader = OTF2_Reader_Open( anchor_file );
    check_pointer( reader, "Create new reader handle." );

    OTF2_ErrorCode status = OTF2_Reader_SetSerialCollectiveCallbacks( reader );
    check_status( status, "Set serial mode." );

    OTF2_Boolean global_reader_hint = OTF2_TRUE;
    status = OTF2_Reader_SetHint( reader,
                                  OTF2_HINT_GLOBAL_READER,
                                  &global_reader_hint );
    check_status( status, "Setting global-reader hint." );

    if ( otf2_ANCHORFILE_INFO )
    {
        otf2_print_anchor_file_information( reader );
    }


    if ( otf2_THUMBNAIL_INFO )
    {
        otf2_print_thumbnails( reader );
    }

    /* Only exit if --show-info was given. */
    if ( ( otf2_ANCHORFILE_INFO || otf2_THUMBNAIL_INFO ) && !otf2_ALL )
    {
        OTF2_Reader_Close( reader );

        /* This is just to add a message to the debug output. */
        check_status( OTF2_SUCCESS, "Delete reader handle." );
        check_status( OTF2_SUCCESS, "Program finished." );

        return EXIT_SUCCESS;
    }
/* ___ Read Global Definitions _______________________________________________*/

    uint32_t number_of_snapshots;
    status = OTF2_Reader_GetNumberOfSnapshots( reader, &number_of_snapshots );
    check_status( status, "Read number of snapshots." );

    /* Add a nice table header. */
    if ( otf2_GLOBDEFS )
    {
        printf( "\n" );
        printf( "=== Global Definitions =========================================================" );
        printf( "\n\n" );
        printf( "%-*s %12s  Attributes\n", otf2_DEF_COLUMN_WIDTH, "Definition", "ID" );
        printf( "--------------------------------------------------------------------------------\n" );
    }
    /* Define definition callbacks. */
    OTF2_GlobalDefReaderCallbacks* def_callbacks = otf2_print_create_global_def_callbacks();

    /* Get number of locations from the anchor file. */
    uint64_t num_locations = 0;
    status = OTF2_SUCCESS;
    status = OTF2_Reader_GetNumberOfLocations( reader, &num_locations );
    check_status( status, "Get number of locations. Number of locations: %" PRIu64,
                  num_locations );


    /* User data for callbacks. */
    struct otf2_print_data user_data;
    struct otf2_print_defs user_defs;
    memset( &user_data, 0, sizeof( user_data ) );
    memset( &user_defs, 0, sizeof( user_defs ) );
    user_data.reader            = reader;
    user_data.locations_to_read = otf2_vector_create();
    user_data.defs              = &user_defs;
    otf2_print_def_create_hash_tables( user_data.defs );
    user_data.comm_paradigms = otf2_vector_create();
    otf2_vector_resize( user_data.comm_paradigms, otf2_max_known_paradigm );
    user_data.dot_file               = NULL;
    user_data.artificial_string_refs = ( OTF2_StringRef64 )OTF2_UNDEFINED_STRING;


    /* If in dot output mode open dot file. */
    char dot_path[ 1024 ] = "";
    if ( otf2_DOT )
    {
        snprintf( dot_path, sizeof( dot_path),  "%.*s.SystemTree.dot", ( int )strlen( anchor_file ) - 5, anchor_file );

        user_data.dot_file = fopen( dot_path, "w" );
        if ( user_data.dot_file == NULL )
        {
            fprintf( stderr,
                     "%s: cannot open dot file for system tree\n",
                     otf2_NAME );
            return EXIT_FAILURE;
        }

        fprintf( user_data.dot_file, "/* This is the graph representation of the system tree. */\n" );
        fprintf( user_data.dot_file, "digraph SystemTree\n" );
        fprintf( user_data.dot_file, "{\n" );
    }


    /* Read global definitions. */
    uint64_t              definitions_read  = 0;
    OTF2_GlobalDefReader* global_def_reader = OTF2_Reader_GetGlobalDefReader( reader );
    check_pointer( global_def_reader, "Create global definition reader handle." );

    status = OTF2_Reader_RegisterGlobalDefCallbacks( reader, global_def_reader,
                                                     def_callbacks,
                                                     &user_data );
    check_status( status, "Register global definition callbacks." );
    OTF2_GlobalDefReaderCallbacks_Delete( def_callbacks );

    status = OTF2_Reader_ReadGlobalDefinitions( reader, global_def_reader,
                                                OTF2_UNDEFINED_UINT64,
                                                &definitions_read );
    check_status( status, "Read global definitions. Number of definitions: %" PRIu64,
                  definitions_read );

    OTF2_Reader_CloseGlobalDefReader( reader,
                                      global_def_reader );

    /* Check if number of global definitions read matches the number of global
     * definitions defined in the anchor file. */
    uint64_t defs_anchor = 0;
    status = OTF2_Reader_GetNumberOfGlobalDefinitions( reader, &defs_anchor );
    check_status( status, "Get number of global definitions: %" PRIu64,
                  defs_anchor );

    if ( defs_anchor != definitions_read )
    {
        check_status( OTF2_ERROR_INTEGRITY_FAULT,
                      "Number of global definitions read and defined in anchor file do not match!" );
    }

    /* Check if a requested local trace file was found. */
    if ( otf2_LOCAL == OTF2_UNDEFINED_LOCATION )
    {
        check_condition( otf2_LOCAL_FOUND, "Find location definitions in global definition file." );
    }
    else
    {
        check_condition( otf2_LOCAL_FOUND, "Find event file for selected location." );
    }

    OTF2_DefReaderCallbacks* local_def_callbacks = OTF2_DefReaderCallbacks_New();
    check_pointer( def_callbacks, "Create global definition callback handle." );
    if ( otf2_MAPPINGS )
    {
        OTF2_DefReaderCallbacks_SetMappingTableCallback( local_def_callbacks, print_def_mapping_table );
    }
    if ( otf2_CLOCK_OFFSETS )
    {
        OTF2_DefReaderCallbacks_SetClockOffsetCallback( local_def_callbacks, print_def_clock_offset );
    }

    /* Open a new local event reader for each found location ID. */
    if ( otf2_MAPPINGS || otf2_CLOCK_OFFSETS )
    {
        printf( "\n" );
        printf( "=== Per Location Definitions ===================================================" );
        printf( "\n\n" );
        printf( "%-*s %12s  Attributes\n", otf2_DEF_COLUMN_WIDTH, "Definition", "Location" );
        printf( "--------------------------------------------------------------------------------\n" );
    }

    for ( size_t i = 0; i < otf2_vector_size( user_data.locations_to_read ); i++ )
    {
        uint64_t* location_item = (uint64_t*)otf2_vector_at( user_data.locations_to_read, i );
        status = OTF2_Reader_SelectLocation( reader, *location_item );
        check_status( status, "Select location to read." );
    }

    bool local_def_files_exists = false;
    if ( !otf2_NOLOCALDEFS && ( !otf2_GLOBDEFS || otf2_ALL ) )
    {
        status = OTF2_Reader_OpenDefFiles( reader );
        /* Will fail if no local def files were written. Remember it and skip
           reading the local def files and closing them. */
        local_def_files_exists = OTF2_SUCCESS == status;
        if ( OTF2_ERROR_ENOENT == status )
        {
            /* Ignore ENOENT in the next check */
            status = OTF2_SUCCESS;
        }
        check_status( status, "Open local definition files for reading." );
    }
    if ( !otf2_GLOBDEFS || otf2_ALL )
    {
        status = OTF2_Reader_OpenEvtFiles( reader );
        check_status( status, "Open event files for reading." );
    }
    for ( size_t i = 0; i < otf2_vector_size( user_data.locations_to_read ); i++ )
    {
        uint64_t* location_item      = (uint64_t*)otf2_vector_at( user_data.locations_to_read, i );
        uint64_t  locationIdentifier = *location_item;

        /* Do not open the event reader, when only showing the global defs */
        if ( !otf2_GLOBDEFS || otf2_ALL )
        {
            OTF2_EvtReader* evt_reader = OTF2_Reader_GetEvtReader( reader,
                                                                   locationIdentifier );
            check_pointer( evt_reader, "Create local event reader for location %" PRIu64 ".",
                           locationIdentifier );
        }

        if ( !otf2_NOSNAPSHOTS && number_of_snapshots > 0 )
        {
            OTF2_SnapReader* snap_reader = OTF2_Reader_GetSnapReader( reader,
                                                                      locationIdentifier );
            check_pointer( snap_reader, "Create local snapshot reader for location %" PRIu64 ".",
                           locationIdentifier );
        }

        if ( otf2_NOLOCALDEFS || ( otf2_GLOBDEFS && !otf2_ALL ) )
        {
            continue;
        }

        if ( local_def_files_exists )
        {
            /* Also open a definition reader and read all local definitions. */
            OTF2_DefReader* def_reader = OTF2_Reader_GetDefReader( reader,
                                                                   locationIdentifier );
            /* a local def file is not mandatory */
            if ( def_reader )
            {
                if ( otf2_MAPPINGS || otf2_CLOCK_OFFSETS )
                {
                    status = OTF2_Reader_RegisterDefCallbacks( reader,
                                                               def_reader,
                                                               local_def_callbacks,
                                                               &locationIdentifier );
                    check_status( status, "Register local definition callbacks." );
                }

                uint64_t definitions_read = 0;
                status = OTF2_SUCCESS;
                do
                {
                    uint64_t def_reads = 0;
                    status = OTF2_Reader_ReadAllLocalDefinitions( reader,
                                                                  def_reader,
                                                                  &def_reads );
                    definitions_read += def_reads;

                    /* continue reading, if we have a duplicate mapping table */
                    if ( OTF2_ERROR_DUPLICATE_MAPPING_TABLE != status )
                    {
                        break;
                    }
                }
                while ( true );
                check_status( status,
                              "Read %" PRIu64 " definitions for location %" PRIu64,
                              definitions_read,
                              locationIdentifier );

                /* Close def reader, it is no longer useful and occupies memory */
                status = OTF2_Reader_CloseDefReader( reader, def_reader );
                check_status( status, "Close local definition reader." );
            }
        }
    }
    OTF2_DefReaderCallbacks_Delete( local_def_callbacks );
    if ( !otf2_NOLOCALDEFS && !( otf2_GLOBDEFS && !otf2_ALL ) && local_def_files_exists )
    {
        status = OTF2_Reader_CloseDefFiles( reader );
        check_status( status, "Close local definition files for reading." );
    }


    /* If in dot output mode close dot file and terminate. */
    if ( otf2_DOT )
    {
        fprintf( user_data.dot_file, "}\n" );
        fclose( user_data.dot_file );

        printf( "\nGenerate system tree dot graph for \"%s\".\n\n", anchor_file );
        printf( "Dot file written to \"%s\".\n\n", dot_path );
        printf( "To generate an image from the dot file run:\n" );
        printf( "\"dot -Tpng %s -o SystemTree.png\"\n\n", dot_path );

        OTF2_Reader_Close( reader );

        /* This is just to add a message to the debug output. */
        check_status( OTF2_SUCCESS, "Delete reader handle." );
        check_status( OTF2_SUCCESS, "Program finished." );

        return EXIT_SUCCESS;
    }

    if ( ( otf2_GLOBDEFS || otf2_MAPPINGS || otf2_CLOCK_OFFSETS ) && !otf2_ALL )
    {
        OTF2_Reader_Close( reader );

        /* This is just to add a message to the debug output. */
        check_status( OTF2_SUCCESS, "Delete reader handle." );
        check_status( OTF2_SUCCESS, "Program finished." );

        return EXIT_SUCCESS;
    }

    ProcessDefs();

/* ___ Read Event Records ____________________________________________________*/



    /* Add a nice table header. */
    if ( !otf2_SILENT && !print_SILENT)
    {
        printf( "=== Events =====================================================================\n" );
    }

    /* Define event callbacks. */
    OTF2_GlobalEvtReaderCallbacks* evt_callbacks = otf2_print_create_global_evt_callbacks();

    /* Get global event reader. */
    OTF2_GlobalEvtReader* global_evt_reader = OTF2_Reader_GetGlobalEvtReader( reader );
    check_pointer( global_evt_reader, "Create global event reader." );


    /* Register the above defined callbacks to the global event reader. */
    if ( !otf2_SILENT )
    {
        status = OTF2_Reader_RegisterGlobalEvtCallbacks( reader,
                                                         global_evt_reader,
                                                         evt_callbacks,
                                                         &user_data );
        check_status( status, "Register global event callbacks." );
    }
    OTF2_GlobalEvtReaderCallbacks_Delete( evt_callbacks );


    /* Read until events are all read. */
    uint64_t events_read = otf2_STEP;

    while ( events_read == otf2_STEP )
    {
        if ( !otf2_SILENT && !print_SILENT)
        {
            printf( "%-*s %15s %20s  Attributes\n",
                    otf2_EVENT_COLUMN_WIDTH, "Event", "Location", "Timestamp" );
            printf( "--------------------------------------------------------------------------------\n" );
        }

        status = OTF2_Reader_ReadGlobalEvents( reader,
                                               global_evt_reader,
                                               otf2_STEP,
                                               &events_read );
        check_status( status, "Read %" PRIu64 " events.", events_read );

        /* Step through output if otf2_STEP is defined. */
        if ( otf2_STEP != OTF2_UNDEFINED_UINT64 )
        {
            printf( "Press ENTER to print next %" PRIu64 " events.", otf2_STEP );
            getchar();
        }
    }
    status = OTF2_Reader_CloseGlobalEvtReader( reader,
                                               global_evt_reader );
    check_status( status, "Close global definition reader." );
    status = OTF2_Reader_CloseEvtFiles( reader );
    check_status( status, "Close event files for reading." );

/* ___ Read Snapshot Records ____________________________________________________*/



    if ( !otf2_NOSNAPSHOTS && number_of_snapshots > 0 )
    {
        /* Add a nice table header. */
        if ( !otf2_SILENT && !print_SILENT)
        {
            printf( "=== Snapshots ==================================================================\n" );
        }

        /* Define snapshot callbacks. */
        OTF2_GlobalSnapReaderCallbacks* snap_callbacks = otf2_print_create_global_snap_callbacks();

        /* Get global snapshots reader. */
        OTF2_GlobalSnapReader* global_snap_reader = OTF2_Reader_GetGlobalSnapReader( reader );
        check_pointer( global_snap_reader, "Create global snapshots reader." );


        /* Register the above defined callbacks to the global snapshots reader. */
        if ( !otf2_SILENT )
        {
            status = OTF2_Reader_RegisterGlobalSnapCallbacks( reader,
                                                              global_snap_reader,
                                                              snap_callbacks,
                                                              &user_data );
            check_status( status, "Register global snapshots callbacks." );
        }
        OTF2_GlobalSnapReaderCallbacks_Delete( snap_callbacks );


        /* Read until snapshots are all read. */
        uint64_t records_read = otf2_STEP;
        while ( records_read == otf2_STEP )
        {
            if ( !otf2_SILENT && !print_SILENT)
            {
                printf( "%-*s %15s %20s  Attributes\n",
                        otf2_EVENT_COLUMN_WIDTH, "Snapshot", "Location", "Timestamp" );
                printf( "--------------------------------------------------------------------------------\n" );
            }

            status = OTF2_Reader_ReadGlobalSnapshots( reader,
                                                      global_snap_reader,
                                                      otf2_STEP,
                                                      &records_read );
            check_status( status, "Read %" PRIu64 " snapshot records.", records_read );

            /* Step through output if otf2_STEP is defined. */
            if ( otf2_STEP != OTF2_UNDEFINED_UINT64 )
            {
                printf( "Press ENTER to print next %" PRIu64 " snapshot records.", otf2_STEP );
                getchar();
            }
        }
    }

    OTF2_Reader_Close( reader );

    /* This is just to add a message to the debug output. */
    check_status( OTF2_SUCCESS, "Delete reader handle." );
    check_status( OTF2_SUCCESS, "Program finished." );

    otf2_vector_foreach( user_data.locations_to_read, free );
    otf2_vector_free( user_data.locations_to_read );
    otf2_print_def_destroy_hash_tables( user_data.defs );
    otf2_vector_free( user_data.comm_paradigms );

    return EXIT_SUCCESS;
}



/* ___ Implementation of static functions ___________________________________ */

static void
otf2_parse_number_argument( const char* option,
                            const char* argument,
                            uint64_t*   number )
{
    const char* p = argument;
    *number = 0;
    while ( *p )
    {
        if ( ( *p < '0' ) || ( *p > '9' ) )
        {
            otf2_print_die( "invalid number argument for %s: %s\n",
                            option, argument );
        }
        uint64_t new_number = *number * 10 + *p - '0';
        if ( new_number < *number )
        {
            otf2_print_die( "number argument to large for '%s': %s\n",
                            option, argument );
        }
        *number = new_number;
        p++;
    }
    if ( p == argument )
    {
        otf2_print_die( "empty number argument for '%s'\n", option );
    }
}

/** @internal
 *  @brief Get command line parameters.
 *
 *  Parses command line parameters and checks for their existence.
 *  Prints help for parameters '-h' or '--help'.
 *
 *  @param argc             Programs argument counter.
 *  @param argv             Programs argument values.
 *  @param anchorFile       Return pointer for the anchor file path.
 */
void
otf2_get_parameters( int    argc,
                     char** argv,
                     char** anchorFile )
{
    bool process_options = true;
    int  i;
    for ( i = 1; process_options && i < argc; i++ )
    {
        if ( !strcmp( argv[ i ], "--help" ) || !strcmp( argv[ i ], "-h" ) )
        {
            printf(
                #include "otf2_print_usage.h"
                "\n"
                "Report bugs to <%s>\n",
                PACKAGE_BUGREPORT );
            exit( EXIT_SUCCESS );
        }

        else if ( !strcmp( argv[ i ], "--version" ) || !strcmp( argv[ i ], "-V" ) )
        {
            printf( "%s: version %s\n", otf2_NAME, OTF2_VERSION );
            exit( EXIT_SUCCESS );
        }

        else if ( !strcmp( argv[ i ], "--debug" ) || !strcmp( argv[ i ], "-d" ) )
        {
            otf2_DEBUG = true;
        }

        /* Check for requested system tree dot output. */
        else if ( !strcmp( argv[ i ], "--system-tree" ) )
        {
            otf2_DOT = true;
        }

        else if ( !strcmp( argv[ i ], "--show-all" ) || !strcmp( argv[ i ], "-A" ) )
        {
            otf2_ANCHORFILE_INFO = true;
            otf2_THUMBNAIL_INFO  = true;
            otf2_ALL             = true;
            otf2_GLOBDEFS        = true;
        }

        else if ( !strcmp( argv[ i ], "--show-global-defs" ) || !strcmp( argv[ i ], "-G" ) )
        {
            otf2_GLOBDEFS = true;
        }

        else if ( !strcmp( argv[ i ], "--show-info" ) || !strcmp( argv[ i ], "-I" ) )
        {
            otf2_ANCHORFILE_INFO = true;
        }

        else if ( !strcmp( argv[ i ], "--show-thumbnails" ) || !strcmp( argv[ i ], "-T" ) )
        {
            otf2_THUMBNAIL_INFO = true;
        }

        else if ( !strcmp( argv[ i ], "--show-thumbnail-samples" ) )
        {
            otf2_THUMBNAIL_INFO    = true;
            otf2_THUMBNAIL_SAMPLES = true;
        }

        else if ( !strcmp( argv[ i ], "--show-mappings" ) || !strcmp( argv[ i ], "-M" ) )
        {
            otf2_MAPPINGS = true;
        }

        else if ( !strcmp( argv[ i ], "--show-clock-offsets" ) || !strcmp( argv[ i ], "-C" ) )
        {
            otf2_CLOCK_OFFSETS = true;
        }

        else if ( !strcmp( argv[ i ], "--no-local-defs" ) )
        {
            otf2_NOLOCALDEFS = true;
        }

        else if ( !strcmp( argv[ i ], "--no-snapshots" ) )
        {
            otf2_NOSNAPSHOTS = true;
        }

        else if ( !strcmp( argv[ i ], "--silent" ) )
        {
            otf2_SILENT = true;
        }

        else if ( !strcmp( argv[ i ], "--timestamps" )
                  || !strncmp( argv[ i ], "--timestamps=", 13 ) )
        {
            char* opt = argv[ i ];
            char* arg = &opt[ 12 ]; /* points to '=' or '\0' */
            if ( !*arg++ )
            {
                if ( i + 1 >= argc )
                {
                    otf2_print_die( "missing argument for '%s'\n", opt );
                }
                arg = argv[ i + 1 ];
                i++;
            }
            if ( !strcmp( "plain", arg ) )
            {
                otf2_TIMESTAMP_FORMAT = TIMESTAMP_PLAIN;
            }
            else if ( !strcmp( "offset", arg ) )
            {
                otf2_TIMESTAMP_FORMAT = TIMESTAMP_OFFSET;
            }
            else
            {
                otf2_print_die( "invalid argument for option '%.12s': %s\n", opt, arg );
            }
        }

        else if ( !strcmp( argv[ i ], "--location" )
                  || !strncmp( argv[ i ], "--location=", 11 )
                  || !strcmp( argv[ i ], "-L" ) )
        {
            char* opt = argv[ i ];
            char* arg;
            if ( opt[ 1 ] == '-' && opt[ 10 ] == '=' )
            {
                opt[ 10 ] = '\0';
                arg       = &opt[ 11 ];
            }
            else
            {
                if ( i + 1 >= argc )
                {
                    otf2_print_die( "missing argument for '%s'\n", opt );
                }
                arg = argv[ i + 1 ];
                i++;
            }

            otf2_parse_number_argument( opt, arg, &otf2_LOCAL );
        }

        else if ( !strcmp( argv[ i ], "--time" )
                  || !strncmp( argv[ i ], "--time=", 7 ) )
        {
            char* opt = argv[ i ];
            char* arg1;
            char* arg2;
            if ( opt[ 6 ] == '=' )
            {
                opt[ 6 ] = '\0';
                arg1     = &opt[ 7 ];
                arg2     = strchr( arg1, ',' );
                if ( !arg2 )
                {
                    otf2_print_die( "missing argument for '%s'\n", opt );
                }
                *arg2++ = '\0';
            }
            else
            {
                if ( i + 2 >= argc )
                {
                    otf2_print_die( "missing argument for '%s'\n", opt );
                }
                arg1 = argv[ i + 1 ];
                arg2 = argv[ i + 2 ];
                i   += 2;
            }

            otf2_parse_number_argument( opt, arg1, &otf2_MINTIME );
            otf2_parse_number_argument( opt, arg2, &otf2_MAXTIME );
        }

        else if ( !strcmp( argv[ i ], "--step" )
                  || !strncmp( argv[ i ], "--step=", 7 )
                  || !strcmp( argv[ i ], "-s" ) )
        {
            char* opt = argv[ i ];
            char* arg;
            if ( opt[ 1 ] == '-' && opt[ 6 ] == '=' )
            {
                opt[ 6 ] = '\0';
                arg      = &opt[ 7 ];
            }
            else
            {
                if ( i + 1 >= argc )
                {
                    otf2_print_die( "missing argument for '%s'\n", opt );
                }
                arg = argv[ i + 1 ];
                i++;
            }

            otf2_parse_number_argument( opt, arg, &otf2_STEP );
        }

        else if ( !strcmp( argv[ i ], "--unwind-calling-context" ) )
        {
            otf2_UNWIND_CALLING_CONTEXT = true;
        }

        else if ( !strcmp( argv[ i ], "--" ) )
        {
            process_options = false;
        }

        else if ( argv[ i ][ 0 ] == '-' )
        {
            otf2_print_die( "unrecognized option '%s'\n", argv[ i ] );
        }

        else
        {
            break;
        }
    }

    if ( 1 != argc - i )
    {
        otf2_print_die( "missing or too many anchorfile argument(s)\n" );
    }

    *anchorFile = argv[ i ];
}

static void
otf2_print_anchor_file_information( OTF2_Reader* reader )
{
    OTF2_ErrorCode status;

    uint8_t major;
    uint8_t minor;
    uint8_t bugfix;
    status = OTF2_Reader_GetVersion( reader, &major, &minor, &bugfix );
    check_status( status, "Read version." );

    printf( "\nContent of OTF2 anchor file:\n" );
    printf( "%-*s %u.%u.%u\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Version",
            major, minor, bugfix );

    uint64_t chunk_size_events;
    uint64_t chunk_size_definitions;
    status = OTF2_Reader_GetChunkSize( reader,
                                       &chunk_size_events,
                                       &chunk_size_definitions );
    check_status( status, "Read chunk size." );

    printf( "%-*s %" PRIu64 "\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Chunk size events",
            chunk_size_events );
    printf( "%-*s %" PRIu64 "\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Chunk size definitions",
            chunk_size_definitions );

    OTF2_FileSubstrate substrate;
    status = OTF2_Reader_GetFileSubstrate( reader, &substrate );
    check_status( status, "Read file substrate." );

    printf( "%-*s %s\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH,
            "File substrate",
            otf2_print_get_file_substrate( substrate ) );

    OTF2_Compression compression;
    status = OTF2_Reader_GetCompression( reader, &compression );
    check_status( status, "Read compression mode." );

    printf( "%-*s ", otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Compression" );
    switch ( compression )
    {
        case OTF2_COMPRESSION_NONE:
            printf( "NONE\n" );
            break;
        case OTF2_COMPRESSION_UNDEFINED:
            printf( "UNDEFINED\n" );
            break;
        default:
            printf( "%s\n", otf2_print_get_invalid( compression ) );
    }

    uint64_t number_of_locations;
    status = OTF2_Reader_GetNumberOfLocations( reader, &number_of_locations );
    check_status( status, "Read number of locations." );

    printf( "%-*s %" PRIu64 "\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Number of locations",
            number_of_locations );

    uint64_t number_of_global_definitions;
    status = OTF2_Reader_GetNumberOfGlobalDefinitions(
        reader,
        &number_of_global_definitions );
    check_status( status, "Read number of global definitions." );

    printf( "%-*s %" PRIu64 "\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Number of global definitions",
            number_of_global_definitions );

    char* string_buffer;
    status = OTF2_Reader_GetMachineName( reader, &string_buffer );
    check_status( status, "Read machine name." );

    printf( "%-*s %s\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Machine name",
            string_buffer );
    free( string_buffer );

    status = OTF2_Reader_GetCreator( reader, &string_buffer );
    check_status( status, "Read creator." );

    printf( "%-*s %s\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Creator",
            string_buffer );
    free( string_buffer );

    status = OTF2_Reader_GetDescription( reader, &string_buffer );
    check_status( status, "Read description." );

    printf( "%-*s %s\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Description",
            string_buffer );
    free( string_buffer );

    uint32_t number_of_properties;
    char**   property_names;
    status = OTF2_Reader_GetPropertyNames( reader, &number_of_properties, &property_names );
    check_status( status, "Read names of properties." );

    printf( "%-*s %" PRIu32 "\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Number of properties",
            number_of_properties );

    for ( uint32_t property_index = 0; property_index < number_of_properties; property_index++ )
    {
        printf( "%-*s %s\n",
                otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Property name",
                property_names[ property_index ] );

        status = OTF2_Reader_GetProperty( reader, property_names[ property_index ], &string_buffer );
        check_status( status, "Read value of property." );

        printf( "%-*s %s\n",
                otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Property value",
                string_buffer );
        free( string_buffer );
    }
    free( property_names );

    uint64_t trace_id;
    status = OTF2_Reader_GetTraceId( reader, &trace_id );
    check_status( status, "Read trace identifier." );

    printf( "%-*s %" PRIx64 "\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Trace identifier",
            trace_id );

    uint32_t number;
    status = OTF2_Reader_GetNumberOfSnapshots( reader,  &number );
    check_status( status, "Read number of snapshots." );

    printf( "%-*s %" PRIu32 "\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Number of snapshots:",
            number );

    status = OTF2_Reader_GetNumberOfThumbnails( reader,  &number );
    check_status( status, "Read Number of thumbnails." );

    printf( "%-*s %" PRIu32 "\n",
            otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Number of thumbnails:",
            number );
}

void
otf2_print_thumbnails( OTF2_Reader* reader )
{
    /* Do we have thumbnails at all? */
    uint32_t number_of_thumbnails;
    if ( OTF2_Reader_GetNumberOfThumbnails( reader,
                                            &number_of_thumbnails ) != OTF2_SUCCESS
         || number_of_thumbnails == 0 )
    {
        return;
    }

    printf( "\nThumbnail headers:\n" );
    OTF2_ErrorCode status;
    for ( uint32_t i = 0; i < number_of_thumbnails; i++ )
    {
        OTF2_ThumbReader* thumb_reader = OTF2_Reader_GetThumbReader( reader, i );
        if ( !thumb_reader )
        {
            continue;
        }

        char*              name        = NULL;
        char*              description = NULL;
        OTF2_ThumbnailType type;
        uint32_t           number_of_samples;
        uint32_t           number_of_metrics;
        uint64_t*          refs_to_defs = NULL;

        status = OTF2_ThumbReader_GetHeader( thumb_reader,
                                             &name,
                                             &description,
                                             &type,
                                             &number_of_samples,
                                             &number_of_metrics,
                                             &refs_to_defs );
        check_status( status, "Reading thumbnail header." );

        printf( "%-*s "
                "%12" PRIUint32
                "  "
                "Name: \"%s\", "
                "Description: \"%s\", "
                "Type: %s, "
                "Samples: %" PRIUint32 ", "
                "%" PRIUint32 " Metrics: ",
                otf2_THUMBNAIL_HEADER_COLUMN_WIDTH, "THUMBNAIL_HEADER",
                i,
                name,
                description,
                otf2_print_get_thumbnail_type( type ),
                number_of_samples,
                number_of_metrics );

        const char* sep = "";
        for ( uint32_t ref = 0; ref < number_of_metrics; ref++ )
        {
            printf( "%s%" PRIUint64, sep, refs_to_defs[ ref ] );
            sep = ", ";
        }
        printf( "\n" );

        free( name );
        free( description );
        free( refs_to_defs );

        if ( !otf2_THUMBNAIL_SAMPLES )
        {
            OTF2_Reader_CloseThumbReader( reader, thumb_reader );
            continue;
        }

        uint64_t* one_sample;
        one_sample = (uint64_t*)malloc( number_of_metrics * sizeof( *one_sample ) );
        check_pointer( one_sample, "Allocating sample array." );

        uint64_t baseline;
        for ( uint32_t sample = 0; sample < number_of_samples; sample++ )
        {
            status = OTF2_ThumbReader_ReadSample( thumb_reader,
                                                  &baseline,
                                                  number_of_metrics,
                                                  one_sample );
            check_status( status, "Reading thumbnail sample." );

            printf( "%-*s "
                    "%12" PRIUint32
                    "  "
                    "Baseline: %" PRIUint64 ", "
                    "Metrics: ",
                    otf2_THUMBNAIL_HEADER_COLUMN_WIDTH, "THUMBNAIL_SAMPLE",
                    sample,
                    baseline );

            const char* sep = "";
            for ( uint32_t metric = 0; metric < number_of_metrics; metric++ )
            {
                printf( "%s%" PRIUint64, sep, one_sample[ metric ] );
                sep = ", ";
            }
            printf( "\n" );
        }
        free( one_sample );

        OTF2_Reader_CloseThumbReader( reader, thumb_reader );
    }
}


/** @internal
 *  @brief Check if pointer is NULL.
 *
 *  Checks if a pointer is NULL. If so it prints an error with the passed
 *  description and exits the program.
 *  If in debug mode, it prints a debug message with the passed description.
 *  It is possible to passed a variable argument list like e.g. in printf.
 *
 *  @param pointer          Pointer to be checked.
 *  @param description      Description for error/debug message.
 *  @param ...              Variable argument list like e.g. in printf.
 */
void
check_pointer( void* pointer,
               const char* description,
               ... )
{
    va_list va;
    va_start( va, description );

    if ( pointer == NULL )
    {
        printf( "==ERROR== " );
        vfprintf( stdout, description, va );
        printf( "\n" );
        exit( EXIT_FAILURE );
    }

    if ( otf2_DEBUG )
    {
        printf( "==DEBUG== " );
        vfprintf( stdout, description, va );
        printf( "\n" );
    }

    va_end( va );
}


/** @internal
 *  @brief Check if status is not OTF2_SUCCESS.
 *
 *  Checks if status is not OTF2_SUCCESS. If so it prints an error with the
 *  passed description and exits the program.
 *  If in debug mode, it prints a debug message with the passed description.
 *  It is possible to passed a variable argument list like e.g. in printf.
 *
 *  @param status           Status to be checked.
 *  @param description      Description for error/debug message.
 *  @param ...              Variable argument list like e.g. in printf.
 */
void
check_status( OTF2_ErrorCode status,
              const char*          description,
              ... )
{
    va_list va;
    va_start( va, description );

    if ( status != OTF2_SUCCESS )
    {
        printf( "==ERROR== %s ", OTF2_Error_GetName( status ) );
        vfprintf( stdout, description, va );
        printf( "\n" );
        exit( EXIT_FAILURE );
    }

    if ( otf2_DEBUG )
    {
        printf( "==DEBUG== " );
        vfprintf( stdout, description, va );
        printf( "\n" );
    }

    va_end( va );
}


/** @internal
 *  @brief Check if condition is true. Otherwise fail.
 *
 *  Checks if condition is true. If not it prints an error with the
 *  passed description and exits the program.
 *  If in debug mode, it prints a debug message with the passed description.
 *  It is possible to passed a variable argument list like e.g. in printf.
 *
 *  @param condition        Condition which must hold true.
 *  @param description      Description for error/debug message.
 *  @param ...              Variable argument list like e.g. in printf.
 */
void
check_condition( bool  condition,
                 const char* description,
                 ... )
{
    va_list va;
    va_start( va, description );

    if ( !condition )
    {
        fprintf( stderr, "==ERROR== " );
        vfprintf( stderr, description, va );
        fprintf( stderr, "\n" );
        exit( EXIT_FAILURE );
    }

    if ( otf2_DEBUG )
    {
        printf( "==DEBUG== " );
        vfprintf( stdout, description, va );
        printf( "\n" );
    }

    va_end( va );
}


void
otf2_print_add_clock_properties( struct otf2_print_data* data,
                                 uint64_t                timerResolution,
                                 uint64_t                globalOffset,
                                 uint64_t                traceLength )
{
    if ( data->timer_resolution != 0 )
    {
        otf2_print_warn( "duplicate ClockProperties\n" );
        /* overwriting the current definition */
    }

    if ( timerResolution == 0 )
    {
        otf2_print_warn( "invalid timer resolution in ClockProperties: %" PRIu64 "\n",
                         timerResolution );
        return;
    }

    data->timer_resolution = timerResolution;
    data->global_offset    = globalOffset;
    data->trace_length     = traceLength;
}


const char*
otf2_print_get_timestamp( struct otf2_print_data* data,
                          OTF2_TimeStamp          time )
{
    if ( time == OTF2_UNDEFINED_TIMESTAMP )
    {
        return "UNDEFINED";
    }

    char* buffer = otf2_print_get_buffer( 0 );

    if ( otf2_TIMESTAMP_FORMAT == TIMESTAMP_OFFSET &&
         data->timer_resolution != 0 )
    {
        if ( time < data->global_offset )
        {
            snprintf( buffer, BUFFER_SIZE, "-%" PRIu64, data->global_offset - time );
        }
        else
        {
            snprintf( buffer, BUFFER_SIZE, "%" PRIu64, time - data->global_offset );
        }
    }
    else
    {
        snprintf( buffer, BUFFER_SIZE, "%" PRIu64, time );
    }

    return buffer;
}


/** @internal
 *  @brief Add a locations to the list of locations to read events from.
 *
 *  @param locations        List of regions.
 *  @param location         Location ID of the location.
 */
void
otf2_print_add_location_to_read( struct otf2_print_data* data,
                                 OTF2_LocationRef        location )
{
    uint64_t* location_item = (uint64_t*)malloc( sizeof( *location_item ) );
    assert( location_item );

    *location_item = location;

    otf2_vector_push_back( data->locations_to_read, location_item );
}


/** @internal
 *  @brief Add a string to the set of strings.
 *
 *  @param strings          Set of strings.
 *  @param string           String ID of new element.
 *  @param content          Content of the new element.
 */
void
otf2_print_add_string( otf2_hash_table* strings,
                       OTF2_StringRef64 string,
                       size_t           content_len,
                       const char*      content_fmt,
                       ... )
{
    if ( string == OTF2_UNDEFINED_STRING )
    {
        return;
    }

    bool use_vl = true;
    if ( content_len == 0 )
    {
        content_len = strlen( content_fmt ) + 1;
        use_vl      = false;
    }

    size_t                 hint;
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( strings, &string, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate String definition: %s\n",
                         otf2_print_get_def_name( strings, string ) );
        /* overwriting the current definition */
    }

    struct otf2_print_def_name* new_string = (otf2_print_def_name*)malloc( sizeof( *new_string )
                                                     + content_len );
    check_pointer( new_string, "Allocating memory for String definition." );

    new_string->def  = string;
    new_string->name = ( char* )new_string + sizeof( *new_string );

    if ( use_vl )
    {
        va_list vl;
        va_start( vl, content_fmt );

        vsnprintf( new_string->name, content_len, content_fmt, vl );

        va_end( vl );
    }
    else
    {
        memcpy( new_string->name, content_fmt, strlen( content_fmt ) + 1 );
    }

    otf2_hash_table_insert( strings,
                            &new_string->def,
                            new_string,
                            &hint );

    entry = otf2_hash_table_find( strings, &string, &hint );
    assert( entry );
}


/** @internal
 *  @brief Add a def with id tye uint64_t to the set of defs.
 *
 *  @param regions          Set of regions.
 *  @param region           Region ID of new region.
 *  @param string           String ID of new region.
 */
void
otf2_print_add_def64_name( const char*      defClass,
                           otf2_hash_table* defs,
                           otf2_hash_table* strings,
                           uint64_t         def,
                           OTF2_StringRef64 string )
{
    if ( def == OTF2_UNDEFINED_UINT64 )
    {
        return;
    }

    size_t                 hint;
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( defs, &def, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate %s definition: %s\n",
                         defClass,
                         otf2_print_get_def64_name( defs, def ) );
        /* overwriting the current definition */
    }

    struct otf2_print_def_name* new_def = (otf2_print_def_name*)malloc( sizeof( *new_def ) );
    check_pointer( new_def, "Allocating memory for %s definition.", defClass );

    new_def->def  = def;
    new_def->name = otf2_print_get_string( strings, string );

    otf2_hash_table_insert( defs, &new_def->def, new_def, &hint );
}

/** @internal
 *  @brief Add a def with id type uint32_t to the set of defs.
 *
 *  @param regions          Set of regions.
 *  @param region           Region ID of new region.
 *  @param string           String ID of new region.
 */
static void
otf2_print_add_def_name( const char*      def_name,
                         otf2_hash_table* defs,
                         otf2_hash_table* strings,
                         uint32_t         def,
                         OTF2_StringRef64 string )
{
    if ( def == OTF2_UNDEFINED_UINT32 )
    {
        return;
    }
    otf2_print_add_def64_name( def_name, defs, strings, def, string );
}


/** @internal
 *  @brief Add a metric class or metric instances to the defs.
 */
void
otf2_print_add_metric( otf2_hash_table*            metrics,
                       OTF2_MetricRef              metric,
                       OTF2_MetricRef              metricClass,
                       uint8_t                     numberOfMembers,
                       const OTF2_MetricMemberRef* metricMembers )
{
    if ( metric == OTF2_UNDEFINED_METRIC )
    {
        return;
    }

    uint64_t               metric64 = metric;
    size_t                 hint;
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( metrics, &metric64, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate Metric definition: %s\n",
                         otf2_print_get_def_name( metrics, metric ) );
        /* overwriting the current definition */
    }

    /* Resolve metric class in case of metric instance */
    if ( metricClass != OTF2_UNDEFINED_METRIC )
    {
        uint64_t               metric_class64 = metricClass;
        size_t                 class_hint;
        otf2_hash_table_entry* class_entry =
            otf2_hash_table_find( metrics, &metric_class64, &class_hint );
        if ( class_entry )
        {
            struct otf2_print_metric_def* class_def = (otf2_print_metric_def*)class_entry->value;
            numberOfMembers = class_def->number_of_members;
            metricMembers   = class_def->members;
        }
    }

    struct otf2_print_metric_def* new_def =
        (otf2_print_metric_def*)malloc( sizeof( *new_def ) +
                ( numberOfMembers * sizeof( OTF2_MetricMemberRef ) ) );
    check_pointer( new_def, "Allocating memory for Metric definition." );

    new_def->def.def  = metric64;
    new_def->def.name = NULL;

    new_def->number_of_members = numberOfMembers;
    memcpy( new_def->members,
            metricMembers,
            numberOfMembers * sizeof( OTF2_MetricMemberRef ) );

    otf2_hash_table_insert( metrics, &new_def->def.def, new_def, &hint );
}

/** @internal
 *  @brief Get a metric class or metric instances defs.
 */
const struct otf2_print_metric_def*
otf2_print_get_metric( otf2_hash_table* metrics,
                       OTF2_MetricRef   metric )
{
    if ( metric == OTF2_UNDEFINED_METRIC )
    {
        return NULL;
    }

    uint64_t               metric64 = metric;
    otf2_hash_table_entry* entry    =
        otf2_hash_table_find( metrics, &metric64, NULL );
    if ( entry )
    {
        return (otf2_print_metric_def*)entry->value;
    }

    return NULL;
}


void
otf2_print_add_group( struct otf2_print_data* data,
                      OTF2_GroupRef           group,
                      OTF2_StringRef          name,
                      OTF2_GroupType          type,
                      OTF2_Paradigm           paradigm,
                      OTF2_GroupFlag          flags,
                      uint32_t                numberOfMembers,
                      const uint64_t*         members )
{
    struct otf2_print_defs* defs = data->defs;

    if ( group == OTF2_UNDEFINED_GROUP )
    {
        return;
    }

    uint64_t               group64 = group;
    size_t                 hint;
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( defs->groups, &group64, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate Group definition: %s\n",
                         otf2_print_get_def_name( defs->groups, group ) );
        /* overwriting the current definition */
    }

    struct otf2_print_group_def* new_def =
        (otf2_print_group_def*)malloc( sizeof( *new_def )
                + numberOfMembers * sizeof( *new_def->members ) );
    check_pointer( new_def, "Allocating memory for Group definition." );

    /* initialize base definition */
    new_def->def.def           = group64;
    new_def->def.name          = otf2_print_get_string( defs->strings, name );
    new_def->type              = type;
    new_def->flags             = flags;
    new_def->paradigm          = paradigm;
    new_def->comm_data         = NULL;
    new_def->number_of_members = numberOfMembers;
    memcpy( new_def->members,
            members,
            numberOfMembers * sizeof( *members ) );

    otf2_hash_table_insert( defs->groups, &new_def->def.def, new_def, &hint );

    /* handle the special communication group definions */
    if ( type == OTF2_GROUP_TYPE_COMM_LOCATIONS )
    {
        if ( otf2_vector_size( data->comm_paradigms ) <= ( size_t )paradigm )
        {
            otf2_vector_resize( data->comm_paradigms, ( size_t )paradigm + 1 );
        }

        const struct otf2_print_group_def* group_def =
        		(otf2_print_group_def*)otf2_vector_at( data->comm_paradigms, ( size_t )paradigm );
        if ( group_def )
        {
            otf2_print_warn( "duplicate Group(COMM_LOCATIONS) for paradigm %s: \n",
                             otf2_print_get_paradigm_name( defs->paradigms, paradigm ),
                             otf2_print_get_def_name( defs->groups, group ),
                             otf2_print_get_def64_name( defs->groups, group_def->def.def ) );
        }
        otf2_vector_set( data->comm_paradigms, ( size_t )paradigm, new_def );
    }

    if ( type == OTF2_GROUP_TYPE_COMM_GROUP )
    {
        if ( otf2_vector_size( data->comm_paradigms ) <= ( size_t )paradigm )
        {
            otf2_vector_resize( data->comm_paradigms, ( size_t )paradigm + 1 );
        }

        struct otf2_print_group_def* comm_paradigm =
        		(otf2_print_group_def*)otf2_vector_at( data->comm_paradigms, ( size_t )paradigm );
        if ( !comm_paradigm )
        {
            otf2_print_warn( "undefined Group(COMM_LOCATIONS) for paradigm %s "
                             "for Group(COMM_GROUP): %s\n",
                             otf2_print_get_paradigm_name( defs->paradigms, paradigm ),
                             otf2_print_get_def_name( defs->groups, group ) );
            return;
        }
        new_def->comm_data = comm_paradigm;

        for ( uint32_t i = 0; i < numberOfMembers; ++i )
        {
            if ( members[ i ] >= comm_paradigm->number_of_members )
            {
                otf2_print_warn( "invalid rank in Group(COMM_GROUP) %s: %s\n",
                                 otf2_print_get_def_name( defs->groups, group ),
                                 otf2_print_get_id64( members[ i ] ) );
            }
        }
    }

    if ( type == OTF2_GROUP_TYPE_COMM_SELF )
    {
        if ( otf2_vector_size( data->comm_paradigms ) <= ( size_t )paradigm )
        {
            otf2_vector_resize( data->comm_paradigms, ( size_t )paradigm + 1 );
        }

        struct otf2_print_group_def* comm_paradigm =
        		(otf2_print_group_def*)otf2_vector_at( data->comm_paradigms, ( size_t )paradigm );
        if ( !comm_paradigm )
        {
            otf2_print_warn( "undefined Group(COMM_LOCATIONS) for paradigm %s "
                             "for Group(COMM_SELF): %s\n",
                             otf2_print_get_paradigm_name( defs->paradigms, paradigm ),
                             otf2_print_get_def_name( defs->groups, group ) );
        }
        else
        {
            if ( comm_paradigm->comm_data )
            {
                otf2_print_warn( "duplicate Group(COMM_SELF) for paradigm %s: %s\n",
                                 otf2_print_get_paradigm_name( defs->paradigms, paradigm ),
                                 otf2_print_get_def_name( defs->groups, group ) );
                /* overwriting the COMM_SELF group */
            }
            new_def->comm_data       = comm_paradigm;
            comm_paradigm->comm_data = new_def;
        }
    }
}

const struct otf2_print_group_def*
otf2_print_get_group( otf2_hash_table* groups,
                      OTF2_GroupRef    group )
{
    if ( group == OTF2_UNDEFINED_GROUP )
    {
        return NULL;
    }

    uint64_t               group64 = group;
    otf2_hash_table_entry* entry   =
        otf2_hash_table_find( groups, &group64, NULL );
    if ( entry )
    {
        return (otf2_print_group_def*)entry->value;
    }

    return NULL;
}


void
otf2_print_add_comm( struct otf2_print_data* data,
                     OTF2_CommRef            comm,
                     OTF2_StringRef          name,
                     OTF2_GroupRef           group,
                     OTF2_CommRef            parent )
{
    struct otf2_print_defs* defs = data->defs;

    if ( comm == OTF2_UNDEFINED_COMM )
    {
        return;
    }

    uint64_t               comm64 = comm;
    size_t                 hint;
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( defs->comms, &comm64, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate Comm definition: %s\n",
                         otf2_print_get_def_name( defs->comms, comm ) );
        /* overwriting the current definition */
    }

    struct otf2_print_comm_def* new_def = (otf2_print_comm_def*)malloc( sizeof( *new_def ) );
    check_pointer( new_def, "Allocating memory for Comm definition." );

    new_def->def.def  = comm64;
    new_def->def.name = otf2_print_get_string( defs->strings, name );
    otf2_hash_table_insert( defs->comms, &new_def->def.def, new_def, &hint );

    new_def->comm_group = otf2_print_get_group( defs->groups, group );
    if ( ( new_def->comm_group &&
           ( OTF2_GROUP_TYPE_COMM_GROUP != new_def->comm_group->type &&
             OTF2_GROUP_TYPE_COMM_SELF != new_def->comm_group->type ) ) ||
         !new_def->comm_group )
    {
        otf2_print_warn( "invalid Group reference in Comm %s definition: %s\n",
                         otf2_print_get_def_name( defs->comms, comm ),
                         otf2_print_get_def_name( defs->groups, group ) );
        new_def->comm_group = NULL;
    }
}

const struct otf2_print_comm_def*
otf2_print_get_comm( otf2_hash_table* comms,
                     OTF2_CommRef     comm )
{
    if ( comm == OTF2_UNDEFINED_COMM )
    {
        return NULL;
    }

    uint64_t               comm64 = comm;
    otf2_hash_table_entry* entry  =
        otf2_hash_table_find( comms, &comm64, NULL );
    if ( entry )
    {
        return (otf2_print_comm_def*)entry->value;
    }

    return NULL;
}

const char*
otf2_print_comm_get_rank_name( struct otf2_print_defs* defs,
                               OTF2_LocationRef        location,
                               OTF2_CommRef            communicator,
                               uint32_t                rank )
{
    if ( rank == OTF2_UNDEFINED_UINT32 )
    {
        /* used in collectives without a root rank */
        return "UNDEFINED";
    }

    const struct otf2_print_comm_def* comm_def =
        otf2_print_get_comm( defs->comms, communicator );

    const char* rank_name = "INVALID";
    if ( comm_def && comm_def->comm_group )
    {
        const struct otf2_print_group_def* group_def = comm_def->comm_group;
        if ( group_def->type == OTF2_GROUP_TYPE_COMM_SELF )
        {
            if ( 0 == rank )
            {
                rank_name = otf2_print_get_def64_name( defs->locations, location );
            }
        }

        if ( group_def->type == OTF2_GROUP_TYPE_COMM_GROUP
             && group_def->comm_data )
        {
            const struct otf2_print_group_def* comm_locations_def =
                group_def->comm_data;
            if ( group_def->flags & OTF2_GROUP_FLAG_GLOBAL_MEMBERS )
            {
                if ( rank < comm_locations_def->number_of_members )
                {
                    rank_name = otf2_print_get_def64_name(
                        defs->locations,
                        comm_locations_def->members[ rank ] );
                }
            }
            else if ( rank < group_def->number_of_members
                      && group_def->members[ rank ] < comm_locations_def->number_of_members )
            {
                rank_name = otf2_print_get_def64_name(
                    defs->locations,
                    comm_locations_def->members[ group_def->members[ rank ] ] );
            }
        }
    }

    return otf2_print_get_rank_name( rank, rank_name );
}


void
otf2_print_add_rma_win( struct otf2_print_data* data,
                        OTF2_RmaWinRef          rmaWin,
                        OTF2_StringRef          name,
                        OTF2_CommRef            comm )
{
    struct otf2_print_defs* defs = data->defs;

    if ( rmaWin == OTF2_UNDEFINED_RMA_WIN )
    {
        return;
    }

    uint64_t               rma_win64 = rmaWin;
    size_t                 hint;
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( defs->rma_wins, &rma_win64, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate RmaWin definition: %s\n",
                         otf2_print_get_def_name( defs->rma_wins, rmaWin ) );
        /* overwriting the current definition */
    }

    struct otf2_print_rma_win_def* new_def = (otf2_print_rma_win_def*)malloc( sizeof( *new_def ) );
    check_pointer( new_def, "Allocating memory for RmaWin definition." );

    new_def->def.def  = rma_win64;
    new_def->def.name = otf2_print_get_string( defs->strings, name );
    otf2_hash_table_insert( defs->rma_wins, &new_def->def.def, new_def, &hint );

    new_def->comm = otf2_print_get_comm( defs->comms, comm );
    if ( !new_def->comm )
    {
        otf2_print_warn( "undefined Comm definition for RmaWin %s: %s\n",
                         otf2_print_get_def_name( defs->rma_wins, rmaWin ),
                         otf2_print_get_def_name( defs->comms, comm ) );
    }
}

const struct otf2_print_rma_win_def*
otf2_print_get_rma_win( otf2_hash_table* rmaWins,
                        OTF2_RmaWinRef   rmaWin )
{
    if ( rmaWin == OTF2_UNDEFINED_RMA_WIN )
    {
        return NULL;
    }

    uint64_t               rma_win64 = rmaWin;
    otf2_hash_table_entry* entry     =
        otf2_hash_table_find( rmaWins, &rma_win64, NULL );
    if ( entry )
    {
        return (otf2_print_rma_win_def*)entry->value;
    }

    return NULL;
}

const char*
otf2_print_rma_win_get_rank_name( struct otf2_print_defs* defs,
                                  OTF2_LocationRef        location,
                                  OTF2_RmaWinRef          rmaWin,
                                  uint32_t                rank )
{
    if ( rank == OTF2_UNDEFINED_UINT32 )
    {
        /* used in collectives without a root rank */
        return "UNDEFINED";
    }

    const struct otf2_print_rma_win_def* rma_win_def =
        otf2_print_get_rma_win( defs->rma_wins, rmaWin );
    if ( rma_win_def && rma_win_def->comm )
    {
        return otf2_print_comm_get_rank_name( defs,
                                              location,
                                              rma_win_def->comm->def.def,
                                              rank );
    }

    return otf2_print_get_rank_name( rank, "INVALID" );
}


void
otf2_print_add_cart_topology( struct otf2_print_data* data,
                              OTF2_CartTopologyRef    cartTopology,
                              OTF2_StringRef          name,
                              OTF2_CommRef            comm )
{
    struct otf2_print_defs* defs = data->defs;

    if ( cartTopology == OTF2_UNDEFINED_CART_TOPOLOGY )
    {
        return;
    }

    uint64_t               cart_topology64 = cartTopology;
    size_t                 hint;
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( defs->cart_topologys, &cart_topology64, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate CartTopology definition: %s\n",
                         otf2_print_get_def_name( defs->cart_topologys, cartTopology ) );
        /* overwriting the current definition */
    }

    struct otf2_print_cart_topology_def* new_def = (otf2_print_cart_topology_def*)malloc( sizeof( *new_def ) );
    check_pointer( new_def, "Allocating memory for CartTopology definition." );

    new_def->def.def  = cart_topology64;
    new_def->def.name = otf2_print_get_string( defs->strings, name );
    otf2_hash_table_insert( defs->cart_topologys, &new_def->def.def, new_def, &hint );

    new_def->comm = otf2_print_get_comm( defs->comms, comm );
    if ( !new_def->comm )
    {
        otf2_print_warn( "undefined Comm definition for CartTopology %s: %s\n",
                         otf2_print_get_def_name( defs->cart_topologys, cartTopology ),
                         otf2_print_get_def_name( defs->comms, comm ) );
    }
}

const struct otf2_print_cart_topology_def*
otf2_print_get_cart_topology( otf2_hash_table*     cartTopologies,
                              OTF2_CartTopologyRef cartTopology )
{
    if ( cartTopology == OTF2_UNDEFINED_CART_TOPOLOGY )
    {
        return NULL;
    }

    uint64_t               cart_topology64 = cartTopology;
    otf2_hash_table_entry* entry           =
        otf2_hash_table_find( cartTopologies, &cart_topology64, NULL );
    if ( entry )
    {
        return (otf2_print_cart_topology_def*)entry->value;
    }

    return NULL;
}

const char*
otf2_print_cart_topology_get_rank_name( struct otf2_print_defs* defs,
                                        OTF2_LocationRef        location,
                                        OTF2_CartTopologyRef    cartTopology,
                                        uint32_t                rank )
{
    if ( rank == OTF2_UNDEFINED_UINT32 )
    {
        /* used in collectives without a root rank */
        return "UNDEFINED";
    }

    const struct otf2_print_cart_topology_def* cart_topology_def =
        otf2_print_get_cart_topology( defs->cart_topologys, cartTopology );
    if ( cart_topology_def && cart_topology_def->comm )
    {
        return otf2_print_comm_get_rank_name( defs,
                                              location,
                                              cart_topology_def->comm->def.def,
                                              rank );
    }

    return otf2_print_get_rank_name( rank, "INVALID" );
}


/** @internal
 *  @brief Add a calling context def.
 */
void
otf2_print_add_calling_context( struct otf2_print_data*    data,
                                OTF2_CallingContextRef     callingContext,
                                OTF2_RegionRef             region,
                                OTF2_SourceCodeLocationRef sourceCodeLocation,
                                OTF2_CallingContextRef     parent )
{
    if ( callingContext == OTF2_UNDEFINED_CALLING_CONTEXT )
    {
        return;
    }

    struct otf2_print_defs* defs = data->defs;

    uint64_t               region64    = region;
    const char*            region_name = "UNDEFINED";
    otf2_hash_table_entry* entry       =
        otf2_hash_table_find( defs->regions, &region64, NULL );
    if ( entry )
    {
        struct otf2_print_def_name* def = (otf2_print_def_name*)entry->value;
        region_name = def->name;
    }
    size_t length = strlen( region_name ) + 1;

    const char* scl_prefix = "";
    const char* scl        = "";
    if ( sourceCodeLocation != OTF2_UNDEFINED_SOURCE_CODE_LOCATION )
    {
        const char* full_scl = otf2_print_get_def_raw_name( defs->source_code_locations,
                                                            sourceCodeLocation );
        if ( full_scl )
        {
            scl_prefix = "@";
            scl        = UTILS_IO_GetWithoutPath( full_scl );
        }
    }
    length += strlen( scl_prefix ) + strlen( scl );

    OTF2_StringRef64 cct_name_id = ++data->artificial_string_refs;
    otf2_print_add_string( defs->strings,
                           cct_name_id,
                           length, "%s%s%s", region_name, scl_prefix, scl );

    uint64_t calling_context64 = callingContext;
    size_t   hint;
    entry = otf2_hash_table_find( defs->calling_contexts, &calling_context64, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate CallingContext definition: %s\n",
                         otf2_print_get_def_name( defs->calling_contexts,
                                                  callingContext ) );
        /* overwriting the current definition */
    }

    struct otf2_print_calling_context_def* new_def = (otf2_print_calling_context_def*)calloc( 1, sizeof( *new_def ) );
    check_pointer( new_def, "Out of memory for a CallingContext definition." );

    new_def->def.def  = calling_context64;
    new_def->def.name = otf2_print_get_string( defs->strings, cct_name_id );
    new_def->region   = region;
    new_def->scl      = sourceCodeLocation;
    new_def->parent   = parent;

    new_def->properties_tail = &new_def->properties_head;

    otf2_hash_table_insert( defs->calling_contexts, &new_def->def.def, new_def, &hint );
}

/** @internal
 *  @brief Get a calling context def by ref.
 */
struct otf2_print_calling_context_def*
otf2_print_get_calling_context( otf2_hash_table*       callingContexts,
                                OTF2_CallingContextRef callingContext )
{
    if ( callingContext == OTF2_UNDEFINED_CALLING_CONTEXT )
    {
        return NULL;
    }

    uint64_t               calling_context64 = callingContext;
    otf2_hash_table_entry* entry             =
        otf2_hash_table_find( callingContexts, &calling_context64, NULL );
    if ( entry )
    {
        return (otf2_print_calling_context_def*)entry->value;
    }

    return NULL;
}


/** @internal
 *  @brief Add a calling context property.
 */
void
otf2_print_add_calling_context_property( struct otf2_print_defs* defs,
                                         OTF2_CallingContextRef  callingContext,
                                         OTF2_StringRef          name,
                                         OTF2_Type               type,
                                         OTF2_AttributeValue     value )
{
    struct otf2_print_calling_context_def* def =
        otf2_print_get_calling_context( defs->calling_contexts,
                                        callingContext );
    if ( !def )
    {
        otf2_print_warn( "invalid CallingContext reference: %s\n",
                         otf2_print_get_def_name( defs->calling_contexts,
                                                  callingContext ) );
        return;
    }

    struct otf2_print_calling_context_property* property = (otf2_print_calling_context_property*)calloc( 1, sizeof( property ) );
    check_pointer( property, "Out of memory for an CallingContextProperty definition." );

    property->next  = NULL;
    property->name  = name;
    property->type  = type;
    property->value = value;

    *def->properties_tail = property;
    def->properties_tail  = &property->next;
}

const char*
otf2_print_get_rank_name( uint64_t    rank,
                          const char* rankName )
{
    if ( rank == OTF2_UNDEFINED_UINT64 )
    {
        return "UNDEFINED";
    }

    const char* rank_str = otf2_print_get_id64( rank );
    size_t      len      = strlen( rank_str ) + 4 + strlen( rankName );
    char*       buffer   = otf2_print_get_buffer( len );
    snprintf( buffer, len, "%s (%s)", rank_str, rankName );

    return buffer;
}


/** The returned buffer has a size of at least BUFFER_SIZE */
char*
otf2_print_get_buffer( size_t len )
{
    #define NR_ENTRIES 8
    static uint32_t next_idx;
    static struct otf2_print_buffer
    {
        char*  buffer;
        size_t size;
    } buffers[ NR_ENTRIES ];
    struct otf2_print_buffer* next = &buffers[ next_idx ];

    next_idx++;
    if ( next_idx == NR_ENTRIES )
    {
        next_idx = 0;
    }

    if ( len < BUFFER_SIZE )
    {
        len = BUFFER_SIZE;
    }
    if ( next->size <= len )
    {
        next->buffer = (char*)realloc( next->buffer, len );
        assert( next->buffer );
        next->size = len;
    }

    *next->buffer = '\0';

    return next->buffer;
}


static const char*
otf2_print_get_id( uint32_t ID )
{
    uint64_t id64 = ID;
    if ( ID == OTF2_UNDEFINED_UINT32 )
    {
        id64 = OTF2_UNDEFINED_UINT64;
    }
    return otf2_print_get_id64( id64 );
}


const char*
otf2_print_get_id64( uint64_t ID )
{
    ( void )otf2_print_get_id;

    if ( ID == OTF2_UNDEFINED_UINT64 )
    {
        return "UNDEFINED";
    }

    char* buffer = otf2_print_get_buffer( 0 );

    snprintf( buffer, BUFFER_SIZE, "%" PRIu64, ID );

    return buffer;
}


const char*
otf2_print_get_invalid_( const char* invalid, uint64_t ID )
{
    size_t len = strlen( invalid ) + 1;
    /* add size for id */
    len += 32;

    char* buffer = otf2_print_get_buffer( len );

    snprintf( buffer, len, "%s <%" PRIu64 ">", invalid, ID );
    return buffer;
}


const char*
otf2_print_get_invalid( uint64_t ID )
{
    return otf2_print_get_invalid_( "INVALID", ID );
}


const char*
otf2_print_get_name( const char* name,
                     uint64_t    ID )
{
    if ( !name )
    {
        return otf2_print_get_id64( ID );
    }

    size_t len = strlen( name ) + 1;
    /* add size for id */
    len += 32;

    char* buffer = otf2_print_get_buffer( len );

    snprintf( buffer, len, "\"%s\" <%" PRIu64 ">", name, ID );

    return buffer;
}


/** @internal
 *  @brief Get the name of a definition.
 *
 *  @param regions          Set of regions.
 *  @param strings          Set of strings.
 *  @param region           Region ID.
 *
 *  @return                 Returns the name of a region if successful, NULL
 *                          otherwise.
 */
const char*
otf2_print_get_def64_name( const otf2_hash_table* defs,
                           uint64_t               def )
{
    if ( def == OTF2_UNDEFINED_UINT64 )
    {
        return "UNDEFINED";
    }

    otf2_hash_table_entry* entry =
        otf2_hash_table_find( defs, &def, NULL );
    if ( !entry )
    {
        return otf2_print_get_invalid( def );
    }

    struct otf2_print_def_name* any_def = (otf2_print_def_name*)entry->value;

    return otf2_print_get_name( any_def->name, def );
}


/** @internal
 *  @brief Get the name of a definition.
 *
 *  @param regions          Set of regions.
 *  @param strings          Set of strings.
 *  @param region           Region ID.
 *
 *  @return                 Returns the name of a region if successful, NULL
 *                          otherwise.
 */
static const char*
otf2_print_get_def_name( const otf2_hash_table* defs,
                         uint32_t               def )
{
    uint64_t def64 = def;
    if ( def == OTF2_UNDEFINED_UINT32 )
    {
        def64 = OTF2_UNDEFINED_UINT64;
    }
    return otf2_print_get_def64_name( defs, def64 );
}


const char*
otf2_print_get_def64_raw_name( const otf2_hash_table* defs,
                               uint64_t               def )
{
    if ( def == OTF2_UNDEFINED_UINT64 )
    {
        return "UNDEFINED";
    }

    otf2_hash_table_entry* entry =
        otf2_hash_table_find( defs, &def, NULL );
    if ( !entry )
    {
        return NULL;
    }

    struct otf2_print_def_name* any_def = (otf2_print_def_name*)entry->value;

    return any_def->name;
}


static const char*
otf2_print_get_def_raw_name( const otf2_hash_table* defs,
                             uint32_t               def )
{
    uint64_t def64 = def;
    if ( def == OTF2_UNDEFINED_UINT32 )
    {
        def64 = OTF2_UNDEFINED_UINT64;
    }
    return otf2_print_get_def64_raw_name( defs, def64 );
}


/** @internal
 *  @brief Add a paradigm def.
 */
static void
otf2_print_add_paradigm_name( struct otf2_print_data* data,
                              OTF2_Paradigm           paradigm,
                              OTF2_StringRef          string,
                              OTF2_ParadigmClass      paradigmClass )
{
    struct otf2_print_defs* defs = data->defs;

    size_t                 hint;
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( defs->paradigms, &paradigm, &hint );
    if ( entry )
    {
        otf2_print_warn( "duplicate Paradigm definition: %s\n",
                         otf2_print_get_paradigm_name( defs->paradigms, paradigm ) );
        /* overwriting the current definition */
    }

    struct otf2_print_paradigm_def* new_paradigm = (otf2_print_paradigm_def*)malloc( sizeof( *new_paradigm ) );
    check_pointer( new_paradigm, "Allocating memory for Paradigm definition." );

    new_paradigm->paradigm      = paradigm;
    new_paradigm->name          = otf2_print_get_string( defs->strings, string );
    new_paradigm->paradigmClass = paradigmClass;

    otf2_hash_table_insert( defs->paradigms, &new_paradigm->paradigm, new_paradigm, &hint );

    if ( otf2_vector_size( data->comm_paradigms ) <= ( size_t )paradigm )
    {
        otf2_vector_resize( data->comm_paradigms, ( size_t )paradigm + 1 );
    }
}


/** @internal
 *  @brief Get the name of a paradigm.
 *
 *  @return                 Returns the name of a paradigm if successful, NULL
 *                          otherwise.
 */
const char*
otf2_print_get_paradigm_name( const otf2_hash_table* paradigms,
                              OTF2_Paradigm          paradigm )
{
    otf2_hash_table_entry* entry =
        otf2_hash_table_find( paradigms, &paradigm, NULL );
    if ( !entry )
    {
        return otf2_print_get_paradigm( paradigm );
    }

    struct otf2_print_paradigm_def* paradigm_def = (otf2_print_paradigm_def*)entry->value;

    return otf2_print_get_name( paradigm_def->name, paradigm );
}


const char*
otf2_print_get_paradigm_property_value( struct otf2_print_defs* defs,
                                        OTF2_ParadigmProperty   property,
                                        OTF2_Type               type,
                                        OTF2_AttributeValue     attributeValue,
                                        const char**            typeString )
{
    *typeString = otf2_print_get_type( type );

    switch ( property )
    {
        case OTF2_PARADIGM_PROPERTY_COMM_NAME_TEMPLATE:
        case OTF2_PARADIGM_PROPERTY_RMA_WIN_NAME_TEMPLATE:
            if ( type != OTF2_TYPE_STRING )
            {
                return otf2_print_get_invalid_( "TYPE MISSMATCH", attributeValue.uint64 );
            }
            return otf2_print_get_def_name( defs->strings, attributeValue.stringRef );

        case OTF2_PARADIGM_PROPERTY_RMA_ONLY:
        {
            OTF2_Boolean   property_value;
            OTF2_ErrorCode ret = OTF2_AttributeValue_GetBoolean(
                type,
                attributeValue,
                &property_value );
            if ( OTF2_ERROR_INVALID_ATTRIBUTE_TYPE == ret )
            {
                return otf2_print_get_invalid_( "TYPE MISSMATCH", attributeValue.uint64 );
            }
            *typeString = "BOOLEAN";
            return otf2_print_get_boolean( property_value );
        }

        default:
            return otf2_print_get_invalid_( "UNKNOWN PROPERTY",  attributeValue.uint64 );
    }
}

/** @internal
 *  @brief Get the content of a string.
 *
 *  @param strings          Set of strings.
 *  @param string           String ID.
 *
 *  @return                 Returns the content of a string if successful, NULL
 *                          otherwise.
 */
char*
otf2_print_get_string( const otf2_hash_table* strings,
                       OTF2_StringRef64       string )
{
    if ( string == OTF2_UNDEFINED_STRING )
    {
        return NULL;
    }

    otf2_hash_table_entry* entry =
        otf2_hash_table_find( strings, &string, NULL );
    if ( !entry )
    {
        return NULL;
    }

    struct otf2_print_def_name* def = (otf2_print_def_name*)entry->value;

    return def->name;
}


static inline const char*
otf2_print_get_scope_name( struct otf2_print_defs* defs,
                           OTF2_MetricScope        scopeType,
                           uint64_t                scope )
{
    switch ( scopeType )
    {
        #define scope_case( SCOPE_TYPE, scope_type ) \
    case OTF2_SCOPE_ ## SCOPE_TYPE: \
        return otf2_print_get_def64_name( defs->scope_type, scope )

        scope_case( LOCATION, locations );
        scope_case( LOCATION_GROUP, location_groups );
        scope_case( SYSTEM_TREE_NODE, system_tree_nodes );
        scope_case( GROUP, groups );

        #undef scope_case

        default:
            return otf2_print_get_id64( scope );
    }
}


static void
otf2_print_attribute_list( struct otf2_print_data* data,
                           OTF2_AttributeList*     attributes )
{
    struct otf2_print_defs* defs = data->defs;

    /* Print additional attributes. */
    uint32_t count = OTF2_AttributeList_GetNumberOfElements( attributes );
    if ( count == 0 )
    {
        return;
    }

    const char* sep = "";
    printf( "%-*s ADDITIONAL ATTRIBUTES: ", otf2_EVENT_COLUMN_WIDTH + 38, "" );
    for ( uint32_t i = 0; i < count; i++ )
    {
        uint32_t            id;
        OTF2_Type           type;
        OTF2_AttributeValue value;

        OTF2_AttributeList_PopAttribute( attributes, &id, &type, &value );

        printf( "%s(%s; %s; %s)",
                sep,
                otf2_print_get_def_name( defs->attributes, id ),
                otf2_print_get_type( type ),
                otf2_print_get_attribute_value( defs, type, value
			#if OTF2_VERSION_MAJOR > 2
			, false
			#endif
			) );
        sep = ", ";
    }
    printf( "\n" );
}


/* ___ Implementation of callbacks __________________________________________ */



OTF2_CallbackCode
print_unknown( OTF2_LocationRef    location,
               OTF2_TimeStamp      time,
               void*               userData,
               OTF2_AttributeList* attributes )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "UNKNOWN",
            location, time );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

/** @internal
 *  @name Callbacks for events.
 *
 *  @param location         Location ID.
 *  @param time             Timestamp of the event.
 *  @param userData         Optional user data.
 *
 *  @return                 Returns OTF2_SUCCESS if successful, an error code
 *                          if an error occures.
 *  @{
 */
OTF2_CallbackCode
print_buffer_flush( OTF2_LocationRef    location,
                    OTF2_TimeStamp      time,
                    void*               userData,
                    OTF2_AttributeList* attributes,
                    OTF2_TimeStamp      stopTime )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Stop Time: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "BUFFER_FLUSH",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_timestamp( data, stopTime ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_measurement_on_off( OTF2_LocationRef     location,
                          OTF2_TimeStamp       time,
                          void*                userData,
                          OTF2_AttributeList*  attributes,
                          OTF2_MeasurementMode measurementMode )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Mode: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MEASUREMENT_ON_OFF",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_measurement_mode( measurementMode ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_enter( OTF2_LocationRef    location,
             OTF2_TimeStamp      time,
             void*               userData,
             OTF2_AttributeList* attributes,
             OTF2_RegionRef      region )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
    {printf( "%-*s %15" PRIu64 " %20s  "
            "Region: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "ENTER",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->regions, region ),
            "\n" );
    }
    otf2_print_attribute_list( data, attributes );

    //cout << "Entering at time: " << time << endl;

    //printf(":%d --- %d\n", location, region);

    EnterStateDef(time,location,region);

    //double ctime = (double)time;
    //cout << "OTIME: " << time << ", CTIME: " << ctime << ", DTIME: " << time-ctime << endl;

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_leave( OTF2_LocationRef    location,
             OTF2_TimeStamp      time,
             void*               userData,
             OTF2_AttributeList* attributes,
             OTF2_RegionRef      region )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    if(!print_SILENT)
       {printf( "%-*s %15" PRIu64 " %20s  "
            "Region: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "LEAVE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->regions, region ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );
    //cout << "Leaving at time: " << time << endl;
    LeaveStateDef(time,location);

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_send( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            receiver,
                OTF2_CommRef        communicator,
                uint32_t            msgTag,
                uint64_t            msgLength )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Receiver: %s, "
            "Communicator: %s, "
            "Tag: %" PRIUint32 ", "
            "Length: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_SEND",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_comm_get_rank_name( defs, location, communicator, receiver ),
            otf2_print_get_def_name( defs->comms, communicator ),
            msgTag,
            msgLength,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_isend( OTF2_LocationRef    location,
                 OTF2_TimeStamp      time,
                 void*               userData,
                 OTF2_AttributeList* attributes,
                 uint32_t            receiver,
                 OTF2_CommRef        communicator,
                 uint32_t            msgTag,
                 uint64_t            msgLength,
                 uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Receiver: %s, "
            "Communicator: %s, "
            "Tag: %" PRIUint32 ", "
            "Length: %" PRIUint64 ", "
            "Request: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_ISEND",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_comm_get_rank_name( defs, location, communicator, receiver ),
            otf2_print_get_def_name( defs->comms, communicator ),
            msgTag,
            msgLength,
            requestID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_isend_complete( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Request: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_ISEND_COMPLETE",
            location,
            otf2_print_get_timestamp( data, time ),
            requestID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_irecv_request( OTF2_LocationRef    location,
                         OTF2_TimeStamp      time,
                         void*               userData,
                         OTF2_AttributeList* attributes,
                         uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Request: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_IRECV_REQUEST",
            location,
            otf2_print_get_timestamp( data, time ),
            requestID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_recv( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            sender,
                OTF2_CommRef        communicator,
                uint32_t            msgTag,
                uint64_t            msgLength )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Sender: %s, "
            "Communicator: %s, "
            "Tag: %" PRIUint32 ", "
            "Length: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_RECV",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_comm_get_rank_name( defs, location, communicator, sender ),
            otf2_print_get_def_name( defs->comms, communicator ),
            msgTag,
            msgLength,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_irecv( OTF2_LocationRef    location,
                 OTF2_TimeStamp      time,
                 void*               userData,
                 OTF2_AttributeList* attributes,
                 uint32_t            sender,
                 OTF2_CommRef        communicator,
                 uint32_t            msgTag,
                 uint64_t            msgLength,
                 uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Sender: %s, "
            "Communicator: %s, "
            "Tag: %" PRIUint32 ", "
            "Length: %" PRIUint64 ", "
            "Request: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_IRECV",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_comm_get_rank_name( defs, location, communicator, sender ),
            otf2_print_get_def_name( defs->comms, communicator ),
            msgTag,
            msgLength,
            requestID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_request_test( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Request: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_REQUEST_TEST",
            location,
            otf2_print_get_timestamp( data, time ),
            requestID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_request_cancelled( OTF2_LocationRef    location,
                             OTF2_TimeStamp      time,
                             void*               userData,
                             OTF2_AttributeList* attributes,
                             uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Request: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_REQUEST_CANCELLED",
            location,
            otf2_print_get_timestamp( data, time ),
            requestID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_collective_begin( OTF2_LocationRef    location,
                            OTF2_TimeStamp      time,
                            void*               userData,
                            OTF2_AttributeList* attributes )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_COLLECTIVE_BEGIN",
            location,
            otf2_print_get_timestamp( data, time ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_mpi_collective_end( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_CollectiveOp   collectiveOp,
                          OTF2_CommRef        communicator,
                          uint32_t            root,
                          uint64_t            sizeSent,
                          uint64_t            sizeReceived )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Operation: %s, "
            "Communicator: %s, "
            "Root: %s, "
            "Sent: %" PRIUint64 ", "
            "Received: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_COLLECTIVE_END",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_collective_op( collectiveOp ),
            otf2_print_get_def_name( defs->comms, communicator ),
            otf2_print_comm_get_rank_name( defs, location, communicator, root ),
            sizeSent,
            sizeReceived,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_omp_fork( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            numberOfRequestedThreads )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "# Requested Threads: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_FORK",
            location,
            otf2_print_get_timestamp( data, time ),
            numberOfRequestedThreads,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_omp_join( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_JOIN",
            location,
            otf2_print_get_timestamp( data, time ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_omp_acquire_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        uint32_t            lockID,
                        uint32_t            acquisitionOrder )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Lock: %" PRIUint32 ", "
            "Acquisition Order: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_ACQUIRE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            lockID,
            acquisitionOrder,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_omp_release_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        uint32_t            lockID,
                        uint32_t            acquisitionOrder )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Lock: %" PRIUint32 ", "
            "Acquisition Order: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_RELEASE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            lockID,
            acquisitionOrder,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_omp_task_create( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint64_t            taskID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Task: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_TASK_CREATE",
            location,
            otf2_print_get_timestamp( data, time ),
            taskID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_omp_task_switch( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint64_t            taskID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Task: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_TASK_SWITCH",
            location,
            otf2_print_get_timestamp( data, time ),
            taskID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_omp_task_complete( OTF2_LocationRef    location,
                         OTF2_TimeStamp      time,
                         void*               userData,
                         OTF2_AttributeList* attributes,
                         uint64_t            taskID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Task: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_TASK_COMPLETE",
            location,
            otf2_print_get_timestamp( data, time ),
            taskID,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_metric( OTF2_LocationRef        location,
              OTF2_TimeStamp          time,
              void*                   userData,
              OTF2_AttributeList*     attributes,
              OTF2_MetricRef          metric,
              uint8_t                 numberOfMetrics,
              const OTF2_Type*        typeIDs,
              const OTF2_MetricValue* metricValues )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Metric: %s, "
            "%" PRIUint8 " Values: ",
            otf2_EVENT_COLUMN_WIDTH, "METRIC",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->metrics, metric ),
            numberOfMetrics );
       }
    const struct otf2_print_metric_def* metric_def =
        otf2_print_get_metric( defs->metrics, metric );

    const char* sep = "";
    for ( uint8_t i = 0; i < numberOfMetrics; i++ )
    {
        const char* metric_member_name = "INVALID";
        if ( metric_def && i < metric_def->number_of_members )
        {
            metric_member_name = otf2_print_get_def_name(
                defs->metric_members,
                metric_def->members[ i ] );
        }
        switch ( typeIDs[ i ] )
        {
            case OTF2_TYPE_INT64:
            	 if(!print_SILENT)
            	    {
                printf( "%s(%s; INT64; %" PRId64 ")", sep,
                        metric_member_name,
                        metricValues[ i ].signed_int );
            	    }
                break;
            case OTF2_TYPE_UINT64:
            	 if(!print_SILENT)
            	    {
                printf( "%s(%s; UINT64; %" PRIu64 ")", sep,
                        metric_member_name,
                        metricValues[ i ].unsigned_int );
            	    }
                break;
            case OTF2_TYPE_DOUBLE:
            	 if(!print_SILENT)
            	    {
                printf( "%s(%s; DOUBLE; %f)", sep,
                        metric_member_name,
                        metricValues[ i ].floating_point );
            	    }
                break;
            default:
            {
            	 if(!print_SILENT)
            	    {
                printf( "%s(%s; %s; %08" PRIx64 ")", sep,
                        metric_member_name,
                        otf2_print_get_invalid( typeIDs[ i ] ),
                        metricValues[ i ].unsigned_int );
            	    }
            }
        }
        sep = ", ";
    }
    if(!print_SILENT)
       {
    printf( "\n" );
       }

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_parameter_string( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        OTF2_ParameterRef   parameter,
                        OTF2_StringRef      string )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Parameter: %s, "
            "Value: \"%s\""
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_STRING",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->parameters, parameter ),
            otf2_print_get_def_name( defs->strings, string ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_parameter_int( OTF2_LocationRef    location,
                     OTF2_TimeStamp      time,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     OTF2_ParameterRef   parameter,
                     int64_t             value )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Parameter: %s, "
            "Value: %" PRIInt64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_INT64",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->parameters, parameter ),
            value,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_parameter_unsigned_int( OTF2_LocationRef    location,
                              OTF2_TimeStamp      time,
                              void*               userData,
                              OTF2_AttributeList* attributes,
                              OTF2_ParameterRef   parameter,
                              uint64_t            value )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Parameter: %s, "
            "Value: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_UINT64",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->parameters, parameter ),
            value,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
print_rma_win_create( OTF2_LocationRef    location,
                      OTF2_TimeStamp      time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      OTF2_RmaWinRef      win )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_WIN_CREATE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_win_destroy( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       OTF2_RmaWinRef      win )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_WIN_DESTROY",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_collective_begin( OTF2_LocationRef    location,
                            OTF2_TimeStamp      time,
                            void*               userData,
                            OTF2_AttributeList* attributes )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_COLLECTIVE_BEGIN",
            location,
            otf2_print_get_timestamp( data, time ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_collective_end( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_CollectiveOp   collectiveOp,
                          OTF2_RmaSyncLevel   syncLevel,
                          OTF2_RmaWinRef      win,
                          uint32_t            root,
                          uint64_t            bytesSent,
                          uint64_t            bytesReceived )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Operation: %s, "
            "Window: %s, "
            "Level of Synchronicity: %s, "
            "Root: %s, "
            "Sent: %" PRIUint64 ", "
            "Received: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_COLLECTIVE_END",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_collective_op( collectiveOp ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_get_rma_sync_level( syncLevel ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              root ),
            bytesSent,
            bytesReceived,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_group_sync( OTF2_LocationRef    location,
                      OTF2_TimeStamp      time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      OTF2_RmaSyncLevel   syncLevel,
                      OTF2_RmaWinRef      win,
                      OTF2_GroupRef       group )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Level of Synchronicity: %s, "
            "Window: %s, "
            "Group: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_GROUP_SYNC",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_rma_sync_level( syncLevel ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_get_def_name( defs->groups, group ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_request_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        OTF2_RmaWinRef      win,
                        uint32_t            remote,
                        uint64_t            lockId,
                        OTF2_LockType       lockType )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Remote: %s, "
            "Lock: %" PRIUint64 ", "
            "Type: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_REQUEST_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              remote ),
            lockId,
            otf2_print_get_lock_type( lockType ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_acquire_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        OTF2_RmaWinRef      win,
                        uint32_t            remote,
                        uint64_t            lockId,
                        OTF2_LockType       lockType )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Remote: %s, "
            "Lock: %" PRIUint64 ", "
            "Type: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_ACQUIRE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              remote ),
            lockId,
            otf2_print_get_lock_type( lockType ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_try_lock( OTF2_LocationRef    location,
                    OTF2_TimeStamp      time,
                    void*               userData,
                    OTF2_AttributeList* attributes,
                    OTF2_RmaWinRef      win,
                    uint32_t            remote,
                    uint64_t            lockId,
                    OTF2_LockType       lockType )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Remote: %s, "
            "Lock: %" PRIUint64 ", "
            "Type: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_TRY_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              remote ),
            lockId,
            otf2_print_get_lock_type( lockType ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_release_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        OTF2_RmaWinRef      win,
                        uint32_t            remote,
                        uint64_t            lockId )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Remote: %s, "
            "Lock: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_RELEASE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              remote ),
            lockId,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_sync( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes,
                OTF2_RmaWinRef      win,
                uint32_t            remote,
                OTF2_RmaSyncType    syncType )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Remote: %s, "
            "Sync Type: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_SYNC",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              remote ),
            otf2_print_get_rma_sync_type( syncType ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_wait_change( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       OTF2_RmaWinRef      win )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_WAIT_CHANGE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_put( OTF2_LocationRef    location,
               OTF2_TimeStamp      time,
               void*               userData,
               OTF2_AttributeList* attributes,
               OTF2_RmaWinRef      win,
               uint32_t            remote,
               uint64_t            bytes,
               uint64_t            matchingId )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Remote: %s, "
            "Bytes: %" PRIUint64 ", "
            "Matching: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_PUT",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              remote ),
            bytes,
            matchingId,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_get( OTF2_LocationRef    location,
               OTF2_TimeStamp      time,
               void*               userData,
               OTF2_AttributeList* attributes,
               OTF2_RmaWinRef      win,
               uint32_t            remote,
               uint64_t            bytes,
               uint64_t            matchingId )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Remote: %s, "
            "Bytes: %" PRIUint64 ", "
            "Matching: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_GET",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              remote ),
            bytes,
            matchingId,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_atomic( OTF2_LocationRef    location,
                  OTF2_TimeStamp      time,
                  void*               userData,
                  OTF2_AttributeList* attributes,
                  OTF2_RmaWinRef      win,
                  uint32_t            remote,
                  OTF2_RmaAtomicType  type,
                  uint64_t            bytesSent,
                  uint64_t            bytesReceived,
                  uint64_t            matchingId )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Remote: %s, "
            "Type: %s, "
            "Sent: %" PRIUint64 ", "
            "Received: %" PRIUint64 ", "
            "Matching: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_ATOMIC",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_rma_win_get_rank_name( defs,
                                              location,
                                              win,
                                              remote ),
            otf2_print_get_rma_atomic_type( type ),
            bytesSent,
            bytesReceived,
            matchingId,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_op_complete_blocking( OTF2_LocationRef    location,
                                OTF2_TimeStamp      time,
                                void*               userData,
                                OTF2_AttributeList* attributes,
                                OTF2_RmaWinRef      win,
                                uint64_t            matchingId )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Matching: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_OP_COMPLETE_BLOCKING",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            matchingId,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_op_complete_non_blocking( OTF2_LocationRef    location,
                                    OTF2_TimeStamp      time,
                                    void*               userData,
                                    OTF2_AttributeList* attributes,
                                    OTF2_RmaWinRef      win,
                                    uint64_t            matchingId )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Matching: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_OP_COMPLETE_NON_BLOCKING",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            matchingId,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_op_test( OTF2_LocationRef    location,
                   OTF2_TimeStamp      time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_RmaWinRef      win,
                   uint64_t            matchingId )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Matching: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_OP_TEST",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            matchingId,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_rma_op_complete_remote( OTF2_LocationRef    location,
                              OTF2_TimeStamp      time,
                              void*               userData,
                              OTF2_AttributeList* attributes,
                              OTF2_RmaWinRef      win,
                              uint64_t            matchingId )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Window: %s, "
            "Matching: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_OP_COMPLETE_REMOTE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            matchingId,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
print_thread_fork( OTF2_LocationRef    location,
                   OTF2_TimeStamp      time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_Paradigm       model,
                   uint32_t            numberOfRequestedThreads )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Model: %s, "
            "# Requested Threads: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_FORK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_paradigm_name( defs->paradigms, model ),
            numberOfRequestedThreads,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_join( OTF2_LocationRef    location,
                   OTF2_TimeStamp      time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_Paradigm       model )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Model: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_JOIN",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_paradigm_name( defs->paradigms, model ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_team_begin( OTF2_LocationRef    location,
                         OTF2_TimeStamp      time,
                         void*               userData,
                         OTF2_AttributeList* attributes,
                         OTF2_CommRef        threadTeam )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Team: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TEAM_BEGIN",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_team_end( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       OTF2_CommRef        threadTeam )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Team: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TEAM_END",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_acquire_lock( OTF2_LocationRef    location,
                           OTF2_TimeStamp      time,
                           void*               userData,
                           OTF2_AttributeList* attributes,
                           OTF2_Paradigm       model,
                           uint32_t            lockID,
                           uint32_t            acquisitionOrder )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Model: %s, "
            "Lock: %" PRIUint32 ", "
            "Acquisition Order: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_ACQUIRE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_paradigm_name( defs->paradigms, model ),
            lockID,
            acquisitionOrder,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_release_lock( OTF2_LocationRef    location,
                           OTF2_TimeStamp      time,
                           void*               userData,
                           OTF2_AttributeList* attributes,
                           OTF2_Paradigm       model,
                           uint32_t            lockID,
                           uint32_t            acquisitionOrder )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Model: %s, "
            "Lock: %" PRIUint32 ", "
            "Acquisition Order: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_RELEASE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_paradigm_name( defs->paradigms, model ),
            lockID,
            acquisitionOrder,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_task_create( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_CommRef        threadTeam,
                          uint32_t            creatingThread,
                          uint32_t            generationNumber )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Team: %s, "
            "Creating Thread: %s, "
            "Generation Number: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TASK_CREATE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            otf2_print_comm_get_rank_name( defs,
                                           location,
                                           threadTeam,
                                           creatingThread ),
            generationNumber,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_task_switch( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_CommRef        threadTeam,
                          uint32_t            creatingThread,
                          uint32_t            generationNumber )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Team: %s, "
            "Creating Thread: %s, "
            "Generation Number: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TASK_SWITCH",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            otf2_print_comm_get_rank_name( defs,
                                           location,
                                           threadTeam,
                                           creatingThread ),
            generationNumber,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_task_complete( OTF2_LocationRef    location,
                            OTF2_TimeStamp      time,
                            void*               userData,
                            OTF2_AttributeList* attributes,
                            OTF2_CommRef        threadTeam,
                            uint32_t            creatingThread,
                            uint32_t            generationNumber )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Team: %s, "
            "Creating Thread: %s, "
            "Generation Number: %" PRIUint32
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TASK_COMPLETE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            otf2_print_comm_get_rank_name( defs,
                                           location,
                                           threadTeam,
                                           creatingThread ),
            generationNumber,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_create( OTF2_LocationRef    location,
                     OTF2_TimeStamp      time,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     OTF2_CommRef        threadContingent,
                     uint64_t            sequenceCount )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Contingent: %s, "
            "Sequence Count: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_CREATE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadContingent ),
            sequenceCount,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_begin( OTF2_LocationRef    location,
                    OTF2_TimeStamp      time,
                    void*               userData,
                    OTF2_AttributeList* attributes,
                    OTF2_CommRef        threadContingent,
                    uint64_t            sequenceCount )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Contingent: %s, "
            "Sequence Count: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_BEGIN",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadContingent ),
            sequenceCount,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_wait( OTF2_LocationRef    location,
                   OTF2_TimeStamp      time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_CommRef        threadContingent,
                   uint64_t            sequenceCount )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Contingent: %s, "
            "Sequence Count: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_WAIT",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadContingent ),
            sequenceCount,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_thread_end( OTF2_LocationRef    location,
                  OTF2_TimeStamp      time,
                  void*               userData,
                  OTF2_AttributeList* attributes,
                  OTF2_CommRef        threadContingent,
                  uint64_t            sequenceCount )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Thread Contingent: %s, "
            "Sequence Count: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_END",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadContingent ),
            sequenceCount,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


static void
unwind_calling_context( struct otf2_print_defs* defs,
                        OTF2_CallingContextRef  callingContext,
                        uint32_t                unwindDistance )
{
    while ( callingContext != OTF2_UNDEFINED_CALLING_CONTEXT )
    {
        const struct otf2_print_calling_context_def* def =
            otf2_print_get_calling_context( defs->calling_contexts,
                                            callingContext );
        if ( !def )
        {
            printf( "%-*s %15s %20s  "
                    "%s"
                    "%s",
                    otf2_EVENT_COLUMN_WIDTH, "", "", "",
                    otf2_print_get_invalid( callingContext ),
                    "\n" );
            break;
        }

        printf( "%-*s %15s %20s  "
                "%s%s",
                otf2_EVENT_COLUMN_WIDTH, "", "", "",
                unwindDistance == OTF2_UNDEFINED_UINT32
                ? "?"
                : unwindDistance > 1
                ? "+"
                : unwindDistance == 1
                ? "*"
                : " ",
                otf2_print_get_def_name( defs->calling_contexts, callingContext ) );

        printf( "\n" );

        callingContext = def->parent;

        if ( unwindDistance != OTF2_UNDEFINED_UINT32 && unwindDistance > 0 )
        {
            unwindDistance--;
        }
    }
}


OTF2_CallbackCode
print_calling_context_sample( OTF2_LocationRef           location,
                              OTF2_TimeStamp             time,
                              void*                      userData,
                              OTF2_AttributeList*        attributes,
                              OTF2_CallingContextRef     callingContext,
                              uint32_t                   unwindDistance,
                              OTF2_InterruptGeneratorRef interruptGenerator )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Calling Context: %s, "
            "Unwind Distance: %s, "
            "Interrupt Generator: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "CALLING_CONTEXT_SAMPLE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->calling_contexts, callingContext ),
            otf2_print_get_id( unwindDistance ),
            otf2_print_get_def_name( defs->interrupt_generators, interruptGenerator ),
            "\n" );
       }

    if ( otf2_UNWIND_CALLING_CONTEXT )
    {
        unwind_calling_context( defs,
                                callingContext,
                                unwindDistance );
    }

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_calling_context_enter( OTF2_LocationRef       location,
                             OTF2_TimeStamp         time,
                             void*                  userData,
                             OTF2_AttributeList*    attributes,
                             OTF2_CallingContextRef callingContext,
                             uint32_t               unwindDistance )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Calling Context: %s, "
            "Unwind Distance: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "CALLING_CONTEXT_ENTER",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->calling_contexts, callingContext ),
            otf2_print_get_id( unwindDistance ),
            "\n" );
       }
    if ( otf2_UNWIND_CALLING_CONTEXT )
    {
        unwind_calling_context( defs,
                                callingContext,
                                unwindDistance );
    }

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_calling_context_leave( OTF2_LocationRef       location,
                             OTF2_TimeStamp         time,
                             void*                  userData,
                             OTF2_AttributeList*    attributes,
                             OTF2_CallingContextRef callingContext )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Calling Context: %s"
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "CALLING_CONTEXT_LEAVE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->calling_contexts, callingContext ),
            "\n" );
       }
    if ( otf2_UNWIND_CALLING_CONTEXT )
    {
        unwind_calling_context( defs,
                                callingContext,
                                1 /* there is progress in the function we leave */ );
    }

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


/** @} */


static OTF2_CallbackCode
print_snap_snapshot_start( OTF2_LocationRef    location,
                           OTF2_TimeStamp      snapTime,
                           void*               userData,
                           OTF2_AttributeList* attributes,
                           uint64_t            numberOfRecords )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "# Events: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "SNAPSHOT_START",
            location,
            otf2_print_get_timestamp( data, snapTime ),
            numberOfRecords,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_snap_snapshot_end( OTF2_LocationRef    location,
                         OTF2_TimeStamp      snapTime,
                         void*               userData,
                         OTF2_AttributeList* attributes,
                         uint64_t            contReadPos )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = (otf2_print_data*)userData;
    if(!print_SILENT)
       {
    printf( "%-*s %15" PRIu64 " %20s  "
            "Cont. Read Position: %" PRIUint64
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "SNAPSHOT_END",
            location,
            otf2_print_get_timestamp( data, snapTime ),
            contReadPos,
            "\n" );
       }
    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


/* ___ Global definitions ____________________________________________________*/


OTF2_CallbackCode
print_global_def_io_paradigm( void*                          userData,
                              OTF2_IoParadigmRef             self,
                              OTF2_StringRef                 identification,
                              OTF2_StringRef                 name,
                              OTF2_IoParadigmClass           ioParadigmClass,
                              OTF2_IoParadigmFlag            ioParadigmFlags,
                              uint8_t                        numberOfProperties,
                              const OTF2_IoParadigmProperty* properties,
                              const OTF2_Type*               types,
                              const OTF2_AttributeValue*     values )
{
    /*TODO*/
 return OTF2_CALLBACK_SUCCESS;
}




OTF2_CallbackCode
print_global_def_unknown( void* userData )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s\n",
            otf2_DEF_COLUMN_WIDTH, "UNKNOWN" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_clock_properties( void*    userData,
                                   uint64_t timerResolution,
                                   uint64_t globalOffset,
                                   uint64_t traceLength
				   #if OTF2_VERSION_MAJOR > 2
				   ,
                                   uint64_t realtimeTimestamp 
				   #endif
				   )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;

    otf2_print_add_clock_properties( data,
                                     timerResolution,
                                     globalOffset,
                                     traceLength );
    ClockPeriodDef(timerResolution);

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s %12s  "
            "Ticks per Seconds: %" PRIUint64 ", "
            "Global Offset: %" PRIUint64 ", "
            "Length: %" PRIUint64
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CLOCK_PROPERTIES",
            "",
            timerResolution,
            globalOffset,
            traceLength,
            "\n" );


    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_paradigm( void*              userData,
                           OTF2_Paradigm      paradigm,
                           OTF2_StringRef     name,
                           OTF2_ParadigmClass paradigmClass )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_paradigm_name( data,
                                  paradigm,
                                  name,
                                  paradigmClass );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint8
            "  "
            "%s, "
            "Name: %s, "
            "Class: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "PARADIGM",
            paradigm,
            otf2_print_get_paradigm( paradigm ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_paradigm_class( paradigmClass ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_paradigm_property( void*                 userData,
                                    OTF2_Paradigm         paradigm,
                                    OTF2_ParadigmProperty property,
                                    OTF2_Type             type,
                                    OTF2_AttributeValue   attributeValue )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    const char* type_string;
    const char* value_string =
        otf2_print_get_paradigm_property_value( defs,
                                                property,
                                                type,
                                                attributeValue,
                                                &type_string );

    printf( "%-*s "
            "%12s"
            "  "
            "Paradigm: %s, "
            "Property: %s, "
            "Type: %s, "
            "Value: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "PARADIGM_PROPERTY",
            "",
            otf2_print_get_paradigm_name( defs->paradigms, paradigm ),
            otf2_print_get_paradigm_property( property ),
            type_string,
            value_string,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_string( void*          userData,
                         OTF2_StringRef self,
                         const char*    string )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_string( defs->strings, self, 0, string );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s %12u  \"%s\""
            "%s",
            otf2_DEF_COLUMN_WIDTH, "STRING",
            self,
            string,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_attribute( void*             userData,
                            OTF2_AttributeRef self,
                            OTF2_StringRef    name,
                            OTF2_StringRef    description,
                            OTF2_Type         type )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Attribute",
                             defs->attributes,
                             defs->strings,
                             self,
                             name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Description: %s, "
            "Type: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "ATTRIBUTE",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->strings, description ),
            otf2_print_get_type( type ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_system_tree_node( void*                  userData,
                                   OTF2_SystemTreeNodeRef self,
                                   OTF2_StringRef         name,
                                   OTF2_StringRef         className,
                                   OTF2_SystemTreeNodeRef parent )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "SystemTreeNode",
                             defs->system_tree_nodes,
                             defs->strings,
                             self,
                             name );

    if ( otf2_DOT )
    {
        /* Generate dot node entry. */
        fprintf( data->dot_file, "    n%u [label=\"%s %s (Node ID: %u)\"];\n",
                 self,
                 otf2_print_get_string( defs->strings, className ),
                 otf2_print_get_string( defs->strings, name ),
                 self );

        /* Generate dot edge entry. */
        if ( parent != OTF2_UNDEFINED_SYSTEM_TREE_NODE )
        {
            fprintf( data->dot_file, "    n%u -> n%u;\n", parent, self );
        }
    }

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Class: %s, "
            "Parent: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "SYSTEM_TREE_NODE",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->strings, className ),
            otf2_print_get_def_name( defs->system_tree_nodes, parent ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_location_group( void*                  userData,
                                 OTF2_LocationGroupRef  self,
                                 OTF2_StringRef         name,
                                 OTF2_LocationGroupType locationGroupType,
                                 OTF2_SystemTreeNodeRef systemTreeParent
				 #if OTF2_VERSION_MAJOR > 2
				 , 
                                 OTF2_LocationGroupRef  creatingLocationGroup 
				 #endif
				 )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "LocationGroup",
                             defs->location_groups,
                             defs->strings,
                             self,
                             name );

    if ( otf2_DOT )
    {
        /* Generate dot node entry. */
        fprintf( data->dot_file,
                 "    g%u [label=\"%s (LG ID: %u)\", shape=box];\n",
                 self,
                 otf2_print_get_string( defs->strings, name ),
                 self );

        /* Generate dot edge entry. */
        if ( systemTreeParent != OTF2_UNDEFINED_SYSTEM_TREE_NODE )
        {
            fprintf( data->dot_file, "      n%u -> g%u;\n",
                     systemTreeParent,
                     self );
        }
    }

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Type: %s, "
            "Parent: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "LOCATION_GROUP",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_location_group_type( locationGroupType ),
            otf2_print_get_def_name( defs->system_tree_nodes, systemTreeParent ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_location( void*                 userData,
                           OTF2_LocationRef      self,
                           OTF2_StringRef        name,
                           OTF2_LocationType     locationType,
                           uint64_t              numberOfEvents,
                           OTF2_LocationGroupRef locationGroup )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def64_name( "Location",
                               defs->locations,
                               defs->strings,
                               self,
                               name );

    /* Print definition if selected. */
    if ( otf2_DOT )
    {
        /* Generate dot node entry. */
        fprintf( data->dot_file,
                 "    l%" PRIu64 " [label=\"%s %s (L ID: %" PRIu64 ")\", shape=diamond];\n",
                 self,
                 otf2_print_get_location_type( locationType ),
                 otf2_print_get_string( defs->strings, name ),
                 self );

        /* Generate dot edge entry. */
        if ( locationGroup != OTF2_UNDEFINED_LOCATION_GROUP )
        {
            fprintf( data->dot_file, "    g%u -> l%" PRIu64 ";\n",
                     locationGroup,
                     self );
        }
    }

    if ( otf2_GLOBDEFS )
    {
        printf( "%-*s "
                "%12" PRIUint64
                "  "
                "Name: %s, "
                "Type: %s, "
                "# Events: %" PRIUint64 ", "
                "Group: %s"
                "%s",
                otf2_DEF_COLUMN_WIDTH, "LOCATION",
                self,
                otf2_print_get_def_name( defs->strings, name ),
                otf2_print_get_location_type( locationType ),
                numberOfEvents,
                otf2_print_get_def_name( defs->location_groups, locationGroup ),
                "\n" );
    }

    /* Only proceed if either no local location is selected (i.e. read all) or
     * location ID matches provided location ID. */
    if ( otf2_LOCAL != OTF2_UNDEFINED_LOCATION && otf2_LOCAL != self )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    char* myname = (char*)otf2_print_get_def_name( defs->strings, name );
    //printf("Init name: %s\n",myname);
    char* tok = strtok(myname," .\"");
    char* nodeC=strtok(NULL," .\"");
    char* threadC=strtok(NULL," .\"");
    //printf("%s --- %s --- %s\n",myname, nodeC,threadC);

    int node = 0;
    if(nodeC==NULL)
	printf("Warning: invalid node from location string %s\n",myname);
    else
	node=atoi(nodeC);

    int thread = 0;
    if(threadC==NULL)
	    printf("Warning: invalid node from location string %s\n",myname);
    else
	    thread=atoi(threadC);
    //unsigned int convSelf = (unsigned int)self;
    //cout << "Thread Def: "<< (char*)otf2_print_get_def_name( defs->strings, name ) <<", Node: " << node << ", Thread: " << thread <<", Self: " <<self << endl;//", ConvSelf: " <<convSelf <<", trunkname: " << myname<<endl;
    ThreadDef(node,thread,self,myname);//TODO: Self is the only node?

    //printf("%s --- %d\n",myname,self);

    /* add location to the list of locations to read events from */
    otf2_print_add_location_to_read( data, self );

    otf2_LOCAL_FOUND = true;

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_region( void*           userData,
                         OTF2_RegionRef  self,
                         OTF2_StringRef  name,
                         OTF2_StringRef  canonicalName,
                         OTF2_StringRef  description,
                         OTF2_RegionRole regionRole,
                         OTF2_Paradigm   paradigm,
                         OTF2_RegionFlag regionFlags,
                         OTF2_StringRef  sourceFile,
                         uint32_t        beginLineNumber,
                         uint32_t        endLineNumber )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Region",
                             defs->regions,
                             defs->strings,
                             self,
                             name );


    const char * myname;
        int i;

        char *getname = (char *) otf2_print_get_def_name(defs->strings, name);

        //cout << "Read name: " << getname << endl;

        int quotebreak=-1;
        bool findingFirstQuotes=true;
        for (i = 0; i < strlen(getname); i++) {
        	if(findingFirstQuotes){
        		if(getname[i]!='"'){
        			quotebreak=i;
        			findingFirstQuotes=false;
        		}
        	}
        	else
           if (getname[i] == '"') {
              getname[i] = '\0';
              break;
           }
        }
        myname = &getname[quotebreak];

        //cout << "Saving name: " << myname << endl;
    StateDef(self,myname, 0);

    /* Print definition if selected.
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }
    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s "
            "(Aka. %s), "
            "Descr.: %s, "
            "Role: %s, "
            "Paradigm: %s, "
            "Flags: %s, "
            "File: %s, "
            "Begin: %" PRIUint32 ", "
            "End: %" PRIUint32
            "%s",
            otf2_DEF_COLUMN_WIDTH, "REGION",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->strings, canonicalName ),
            otf2_print_get_def_name( defs->strings, description ),
            otf2_print_get_region_role( regionRole ),
            otf2_print_get_paradigm_name( defs->paradigms, paradigm ),
            otf2_print_get_region_flag( regionFlags ),
            otf2_print_get_def_name( defs->strings, sourceFile ),
            beginLineNumber,
            endLineNumber,
            "\n" );
*/

    //printf(":%s --- %d\n", myname, self);

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_callsite( void*            userData,
                           OTF2_CallsiteRef self,
                           OTF2_StringRef   sourceFile,
                           uint32_t         lineNumber,
                           OTF2_RegionRef   enteredRegion,
                           OTF2_RegionRef   leftRegion )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Callsite",
                             defs->callsites,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "File: %s, "
            "Line Number: %" PRIUint32 ", "
            "Entered Region: %s, "
            "Left Region: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CALLSITE",
            self,
            otf2_print_get_def_name( defs->strings, sourceFile ),
            lineNumber,
            otf2_print_get_def_name( defs->regions, enteredRegion ),
            otf2_print_get_def_name( defs->regions, leftRegion ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_callpath( void*            userData,
                           OTF2_CallpathRef self,
                           OTF2_CallpathRef parent,
                           OTF2_RegionRef   region )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Callpath",
                             defs->callpaths,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Parent: %s, "
            "Region: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CALLPATH",
            self,
            otf2_print_get_def_name( defs->callpaths, parent ),
            otf2_print_get_def_name( defs->regions, region ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_group( void*           userData,
                        OTF2_GroupRef   self,
                        OTF2_StringRef  name,
                        OTF2_GroupType  groupType,
                        OTF2_Paradigm   paradigm,
                        OTF2_GroupFlag  groupFlags,
                        uint32_t        numberOfMembers,
                        const uint64_t* members )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_group( data,
                          self,
                          name,
                          groupType,
                          paradigm,
                          groupFlags,
                          numberOfMembers,
                          members );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Type: %s, "
            "Paradigm: %s, "
            "Flags: %s, "
            "%" PRIUint32 " Members:",
            otf2_DEF_COLUMN_WIDTH, "GROUP",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_group_type( groupType ),
            otf2_print_get_paradigm_name( defs->paradigms, paradigm ),
            otf2_print_get_group_flag( groupFlags ),
            numberOfMembers );

    struct otf2_print_group_def* comm_paradigm = NULL;
    if ( groupType == OTF2_GROUP_TYPE_COMM_GROUP )
    {
        comm_paradigm = (otf2_print_group_def*)otf2_vector_at( data->comm_paradigms, ( size_t )paradigm );
    }

    const char* sep = " ";
    for ( uint32_t i = 0; i < numberOfMembers; i++ )
    {
        switch ( groupType )
        {
            case OTF2_GROUP_TYPE_UNKNOWN:
                printf( "%s%s", sep, otf2_print_get_id64( members[ i ] ) );
                break;

            case OTF2_GROUP_TYPE_LOCATIONS:
                printf( "%s%s", sep, otf2_print_get_def64_name( defs->locations,
                                                                members[ i ] ) );
                break;

            case OTF2_GROUP_TYPE_REGIONS:
                printf( "%s%s", sep, otf2_print_get_def64_name( defs->regions,
                                                                members[ i ] ) );
                break;

            case OTF2_GROUP_TYPE_METRIC:
                printf( "%s%s", sep, otf2_print_get_def64_name( defs->metric_members,
                                                                members[ i ] ) );
                break;

            case OTF2_GROUP_TYPE_COMM_LOCATIONS:
                printf( "%s%s", sep, otf2_print_get_def64_name( defs->locations,
                                                                members[ i ] ) );
                break;

            case OTF2_GROUP_TYPE_COMM_GROUP:
            {
                const char* location_name = "INVALID";
                if ( comm_paradigm &&
                     members[ i ] < comm_paradigm->number_of_members )
                {
                    location_name = otf2_print_get_def64_name(
                        defs->locations,
                        comm_paradigm->members[ members[ i ] ] );
                }
                printf( "%s%s", sep,
                        otf2_print_get_rank_name( members[ i ],
                                                  location_name ) );
                break;
            }

            case OTF2_GROUP_TYPE_COMM_SELF:
                printf( "%s%s", sep, otf2_print_get_id64( members[ i ] ) );
                break;

            default:
                printf( "%s%s", sep, otf2_print_get_id64( members[ i ] ) );
        }
        sep = ", ";
    }
    printf( "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static inline const char*
otf2_print_get_metric_value_type( OTF2_Type type )
{
    switch ( type )
    {
        case OTF2_TYPE_UINT64:
            return "UINT64";
        case OTF2_TYPE_INT64:
            return "INT64";
        case OTF2_TYPE_DOUBLE:
            return "DOUBLE";
        default:
            return otf2_print_get_invalid( type );
    }
}


OTF2_CallbackCode
print_global_def_metric_member( void*                userData,
                                OTF2_MetricMemberRef self,
                                OTF2_StringRef       name,
                                OTF2_StringRef       description,
                                OTF2_MetricType      metricType,
                                OTF2_MetricMode      metricMode,
                                OTF2_Type            valueType,
                                OTF2_Base            base,
                                int64_t              exponent,
                                OTF2_StringRef       unit )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "MetricMember",
                             defs->metric_members,
                             defs->strings,
                             self,
                             name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }


    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Descr.: %s, "
            "Type: %s, "
            "Mode: %s, "
            "Value Type: %s, "
            "Base: %s, "
            "Exponent: %" PRIInt64 ", "
            "Unit: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "METRIC_MEMBER",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->strings, description ),
            otf2_print_get_metric_type( metricType ),
            otf2_print_get_metric_mode( metricMode ),
            otf2_print_get_metric_value_type( valueType ),
            otf2_print_get_base( base ),
            exponent,
            otf2_print_get_def_name( defs->strings, unit ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_metric_class( void*                       userData,
                               OTF2_MetricRef              self,
                               uint8_t                     numberOfMetrics,
                               const OTF2_MetricMemberRef* metricMembers,
                               OTF2_MetricOccurrence       metricOccurrence,
                               OTF2_RecorderKind           recorderKind )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_metric( defs->metrics,
                           self,
                           OTF2_UNDEFINED_METRIC,
                           numberOfMetrics,
                           metricMembers );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Occurrence: %s, "
            "Kind: %s, "
            "%" PRIUint8 " Members:",
            otf2_DEF_COLUMN_WIDTH, "METRIC_CLASS",
            self,
            otf2_print_get_metric_occurrence( metricOccurrence ),
            otf2_print_get_recorder_kind( recorderKind ),
            numberOfMetrics );

    const char* sep = " ";
    for ( uint8_t i = 0; i < numberOfMetrics; i++ )
    {
        printf( "%s%s",
                sep,
                otf2_print_get_def_name( defs->metric_members, metricMembers[ i ] ) );
        sep = ", ";
    }
    printf( "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_metric_instance( void*            userData,
                                  OTF2_MetricRef   self,
                                  OTF2_MetricRef   metricClass,
                                  OTF2_LocationRef recorder,
                                  OTF2_MetricScope metricScope,
                                  uint64_t         scope )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_metric( defs->metrics,
                           self,
                           metricClass,
                           0,
                           NULL );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Class: %s, "
            "Recorder: %s, "
            "Scope: %s %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "METRIC_INSTANCE",
            self,
            otf2_print_get_def_name( defs->metrics, metricClass ),
            otf2_print_get_def64_name( defs->locations, recorder ),
            otf2_print_get_metric_scope( metricScope ),
            otf2_print_get_scope_name( defs, metricScope, scope ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_comm( void*          userData,
                       OTF2_CommRef   self,
                       OTF2_StringRef name,
                       OTF2_GroupRef  group,
                       OTF2_CommRef   parent
		       #if OTF2_VERSION_MAJOR > 2
		       ,
                       OTF2_CommFlag  flags 
		       #endif
		       )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_comm( data,
                         self,
                         name,
                         group,
                         parent );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Group: %s, "
            "Parent: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "COMM",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->groups, group ),
            otf2_print_get_def_name( defs->comms, parent ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_parameter( void*              userData,
                            OTF2_ParameterRef  self,
                            OTF2_StringRef     name,
                            OTF2_ParameterType parameterType )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Parameter",
                             defs->parameters,
                             defs->strings,
                             self,
                             name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Type: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "PARAMETER",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_parameter_type( parameterType ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_rma_win( void*          userData,
                          OTF2_RmaWinRef self,
                          OTF2_StringRef name,
                          OTF2_CommRef   comm
			  #if OTF2_VERSION_MAJOR > 2
			  ,
                          OTF2_RmaWinFlag flags 
			  #endif
			  )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_rma_win( data,
                            self,
                            name,
                            comm );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Communicator: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "RMA_WIN",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->comms, comm ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_global_def_metric_class_recorder( void*            userData,
                                        OTF2_MetricRef   metricClass,
                                        OTF2_LocationRef recorder )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "Class: %s, "
            "Recorder: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "METRIC_CLASS_RECORDER",
            "",
            otf2_print_get_def_name( defs->metrics, metricClass ),
            otf2_print_get_def64_name( defs->locations, recorder ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_system_tree_node_property( void*                  userData,
                                            OTF2_SystemTreeNodeRef systemTreeNode,
                                            OTF2_StringRef         name,
                                            OTF2_Type              type,
                                            OTF2_AttributeValue    value )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    if ( otf2_DOT )
    {
        static uint32_t prop_id;
        /* Generate dot node entry. */
        fprintf( data->dot_file,
                 "    subgraph {\n"
                 "        rank = same;\n"
                 "        n%u;\n"
                 "        np%u [label=\"%s=%s\" shape=box];\n"
                 "        np%u -> n%u;\n"
                 "    }\n",
                 systemTreeNode,
                 prop_id,
                 otf2_print_get_string( defs->strings, name ),
                 otf2_print_get_attribute_value( defs, type, value
			 #if OTF2_VERSION_MAJOR > 2
			 , false
			#endif
			 ),
                 prop_id, systemTreeNode );
        prop_id++;
    }

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "System Tree Node: %s, "
            "Name: %s, "
            "Type: %s, "
            "Value: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "SYSTEM_TREE_NODE_PROPERTY",
            "",
            otf2_print_get_def_name( defs->system_tree_nodes, systemTreeNode ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_type( type ),
            otf2_print_get_attribute_value( defs, type, value
		    #if OTF2_VERSION_MAJOR > 2
		    , false
		    #endif
		    ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_system_tree_node_domain( void*                  userData,
                                          OTF2_SystemTreeNodeRef systemTreeNode,
                                          OTF2_SystemTreeDomain  systemTreeDomain )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    if ( otf2_DOT )
    {
        static uint32_t domain_id;
        /* Generate dot node entry. */
        fprintf( data->dot_file,
                 "    subgraph {\n"
                 "        rank = same;\n"
                 "        n%u;\n"
                 "        nd%u [label=\"%s\" shape=box];\n"
                 "        nd%u -> n%u;\n"
                 "    }\n",
                 systemTreeNode,
                 domain_id,
                 otf2_print_get_system_tree_domain( systemTreeDomain ),
                 domain_id, systemTreeNode );
        domain_id++;
    }

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "System Tree Node: %s, "
            "Domain: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "SYSTEM_TREE_NODE_DOMAIN",
            "",
            otf2_print_get_def_name( defs->system_tree_nodes, systemTreeNode ),
            otf2_print_get_system_tree_domain( systemTreeDomain ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_location_group_property( void*                 userData,
                                          OTF2_LocationGroupRef locationGroup,
                                          OTF2_StringRef        name,
                                          OTF2_Type             type,
                                          OTF2_AttributeValue   value )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "Location Group: %s, "
            "Name: %s, "
            "Type: %s, "
            "Value: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "LOCATION_GROUP_PROPERTY",
            "",
            otf2_print_get_def_name( defs->location_groups, locationGroup ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_type( type ),
            otf2_print_get_attribute_value( defs, type, value
		    #if OTF2_VERSION_MAJOR > 2
		    , false
		    #endif
		    ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_location_property( void*               userData,
                                    OTF2_LocationRef    location,
                                    OTF2_StringRef      name,
                                    OTF2_Type           type,
                                    OTF2_AttributeValue value )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "Location: %s, "
            "Name: %s, "
            "Type: %s, "
            "Value: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "LOCATION_PROPERTY",
            "",
            otf2_print_get_def64_name( defs->locations, location ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_type( type ),
            otf2_print_get_attribute_value( defs, type, value
		    #if OTF2_VERSION_MAJOR > 2
		    , false
		    #endif
		    ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_cart_dimension( void*                 userData,
                                 OTF2_CartDimensionRef self,
                                 OTF2_StringRef        name,
                                 uint32_t              size,
                                 OTF2_CartPeriodicity  cartPeriodicity )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "CartDimension",
                             defs->cart_dimensions,
                             defs->strings,
                             self,
                             name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Size: %" PRIUint32 ", "
            "Periodicity: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CART_DIMENSION",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            size,
            otf2_print_get_cart_periodicity( cartPeriodicity ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_cart_topology( void*                        userData,
                                OTF2_CartTopologyRef         self,
                                OTF2_StringRef               name,
                                OTF2_CommRef                 communicator,
                                uint8_t                      numberOfDimensions,
                                const OTF2_CartDimensionRef* cartDimensions )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_cart_topology( data,
                                  self,
                                  name,
                                  communicator );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Communicator: %s, "
            "%" PRIUint8 " Dimensions: (",
            otf2_DEF_COLUMN_WIDTH, "CART_TOPOLOGY",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->comms, communicator ),
            numberOfDimensions );

    const char* sep = "";
    for ( uint8_t i = 0; i < numberOfDimensions; i++ )
    {
        printf( "%s%s",
                sep,
                otf2_print_get_def_name( defs->cart_dimensions,
                                         cartDimensions[ i ] ) );
        sep = ", ";
    }
    printf( ")\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_cart_coordinate( void*                userData,
                                  OTF2_CartTopologyRef cartTopology,
                                  uint32_t             rank,
                                  uint8_t              numberOfDimensions,
                                  const uint32_t*      coordinates )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "Cartesian Topology: %s, "
            "Rank: %s, "
            "Coordinates: (",
            otf2_DEF_COLUMN_WIDTH, "CART_COORDINATE",
            "",
            otf2_print_get_def_name( defs->cart_topologys, cartTopology ),
            otf2_print_cart_topology_get_rank_name( defs,
                                                    OTF2_UNDEFINED_LOCATION,
                                                    cartTopology,
                                                    rank ) );

    const char* sep = "";
    for ( uint8_t i = 0; i < numberOfDimensions; i++ )
    {
        printf( "%s%" PRIUint32, sep, coordinates[ i ] );
        sep = ", ";
    }
    printf( ")\n" );

    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
print_global_def_source_code_location( void*                      userData,
                                       OTF2_SourceCodeLocationRef self,
                                       OTF2_StringRef             file,
                                       uint32_t                   lineNumber )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    data->artificial_string_refs++;
    uint64_t scl_name_id = data->artificial_string_refs;
    char*    file_name   = otf2_print_get_string( defs->strings, file );
    size_t   length      = strlen( file_name ) + strlen( ":" ) + 10 + 1;

    otf2_print_add_string( defs->strings,
                           scl_name_id,
                           length, "%s:%d", file_name, lineNumber );

    otf2_print_add_def_name( "SourceCodeLocation",
                             defs->source_code_locations,
                             defs->strings,
                             self,
                             scl_name_id );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "File: %s, "
            "Line Number: %" PRIUint32
            "%s",
            otf2_DEF_COLUMN_WIDTH, "SOURCE_CODE_LOCATION",
            self,
            otf2_print_get_def_name( defs->strings, file ),
            lineNumber,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
print_global_def_calling_context_property( void*                  userData,
                                           OTF2_CallingContextRef callingContext,
                                           OTF2_StringRef         name,
                                           OTF2_Type              type,
                                           OTF2_AttributeValue    value )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_calling_context_property( defs,
                                             callingContext,
                                             name,
                                             type,
                                             value );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "Calling Context: %s, "
            "Name: %s, "
            "Type: %s, "
            "Value: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CALLING_CONTEXT_PROPERTY",
            "",
            otf2_print_get_def_name( defs->calling_contexts, callingContext ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_type( type ),
            otf2_print_get_attribute_value( defs, type, value
		    #if OTF2_VERSION_MAJOR > 2
		    , false
                    #endif
		    ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_calling_context( void*                      userData,
                                  OTF2_CallingContextRef     self,
                                  OTF2_RegionRef             region,
                                  OTF2_SourceCodeLocationRef sourceCodeLocation,
                                  OTF2_CallingContextRef     parent )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_calling_context( data,
                                    self,
                                    region,
                                    sourceCodeLocation,
                                    parent );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Region: %s, "
            "Source code location: %s, "
            "Parent: %s"
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CALLING_CONTEXT",
            self,
            otf2_print_get_def_name( defs->regions, region ),
            otf2_print_get_def_name( defs->source_code_locations, sourceCodeLocation ),
            otf2_print_get_def_name( defs->calling_contexts, parent ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
print_global_def_interrupt_generator( void*                       userData,
                                      OTF2_InterruptGeneratorRef  self,
                                      OTF2_StringRef              name,
                                      OTF2_InterruptGeneratorMode interruptGeneratorMode,
                                      OTF2_Base                   base,
                                      int64_t                     exponent,
                                      uint64_t                    period )
{
    struct otf2_print_data* data = (otf2_print_data*)userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "InterruptGenerator",
                             defs->interrupt_generators,
                             defs->strings,
                             self,
                             name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "Name: %s, "
            "Mode: %s, "
            "Base: %s, "
            "Exponent: %" PRIInt64 ", "
            "Period: %" PRIUint64
            "%s",
            otf2_DEF_COLUMN_WIDTH, "INTERRUPT_GENERATOR",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_interrupt_generator_mode( interruptGeneratorMode ),
            otf2_print_get_base( base ),
            exponent,
            period,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


static void
map_traverse_dense( uint64_t localId,
                    uint64_t globalId,
                    void*    userData )
{
    const char* sep = *( char** )userData;
    printf( "%s%" PRIu64, sep, globalId );
    *(const char** )userData = ",";
}

static void
map_traverse_sparse( uint64_t localId,
                     uint64_t globalId,
                     void*    userData )
{
    const char* sep = *( char** )userData;
    printf( "%s%" PRIu64 "=>%" PRIu64, sep, localId, globalId );
    *( const char** )userData = ",";
}

OTF2_CallbackCode
print_def_mapping_table( void*             userData,
                         OTF2_MappingType  mapType,
                         const OTF2_IdMap* iDMap )
{
    uint64_t* location_ptr = (uint64_t*)userData;

    printf( "%-*s %12" PRIu64 "  Type: %s, ",
            otf2_DEF_COLUMN_WIDTH, "MAPPING_TABLE",
            *location_ptr,
            otf2_print_get_mapping_type( mapType ) );

    OTF2_IdMapMode map_mode;
    OTF2_IdMap_GetMode( iDMap, &map_mode );

    const char*                 sep;
    OTF2_IdMap_TraverseCallback traverse_cb;
    const char*                 end;
    if ( map_mode == OTF2_ID_MAP_DENSE )
    {
        sep         = "[";
        traverse_cb = map_traverse_dense;
        end         = "]";
    }
    else
    {
        sep         = "{";
        traverse_cb = map_traverse_sparse;
        end         = "}";
    }

    OTF2_IdMap_Traverse( iDMap, traverse_cb, &sep );

    /* includes "\n" */
    puts( end );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
print_def_clock_offset( void*    userData,
                        uint64_t time,
                        int64_t  offset,
                        double   stddev )
{
    uint64_t* location_ptr = (uint64_t*)userData;

    printf( "%-*s %12" PRIu64 "  Time: %" PRIu64 ", Offset: %+" PRIi64 ", "
            "StdDev: %f\n",
            otf2_DEF_COLUMN_WIDTH, "CLOCK_OFFSET",
            *location_ptr, time, offset, stddev );

    return OTF2_CALLBACK_SUCCESS;
}
