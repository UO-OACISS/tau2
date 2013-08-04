/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2012,
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

/**
 *  @file       otf2_print.c
 *  @status     alpha
 *
 *  @maintainer Michael Wagner <michael.wagner@zih.tu-dresden.de>
 *  @authors    Dominic Eschweiler <d.eschweiler@fz-juelich.de>,
 *              Michael Wagner <michael.wagner@zih.tu-dresden.de>
 *
 *  @brief      This tool prints out all event files of an archive to console.
 *
 *  Usage: otf2profile [OPTION]... ANCHORFILE \n
 *  Print the content of all files of an OTF2 archive with the ANCHORFILE.
 *
 *    -A, --show-all           Print all output including definitions and anchor file.
 *    -G, --show-global-defs   Print global definitions.
 *    -I  --show-info          Print information from the anchor file.
 *    -M, --show-mappings      Print mappings to global definitions.
 *    -C, --show-clock-offsets Print clock offsets to global timer.
 *    -L, --location LID       Limit output to location LID.
 *    -s, --step N             Step through output by steps of N events.
 *        --silent             Only validate trace and do not print any events.
 *        --time MIN MAX       Limit output to events within time interval.
 *    -d, --debug              Turn on debug mode.
 *    -V, --version            Print version information.
 *    -h, --help               Print this help information.
 *
 *  @param argc              Programs argument counter.
 *  @param argv              Programs argument values.
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
#include <inttypes.h>
#include <assert.h>

#include "otf2/otf2.h"

#include "otf2_hash_table.h"
#include "otf2_vector.h"
#include "otf2/OTF2_GeneralDefinitions.h"

#include <trace2profile.h>
#include <handlers.h>


#define dprintf if(0) printf
/* ___ Global variables. ____________________________________________________ */



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
static bool otf2_SILENT;

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
 *  @brief width of the column with the anchor file information. */
static int otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH = 30;

/* ___ Structures. __________________________________________________________ */


/** @internal
 *  @brief Keeps all data for the callbacks.
 *  Please see otf2_print_data_struct for a detailed description. */
typedef struct otf2_print_data_struct otf2_print_data;

/** @internal
 *  @brief Region definition element.
 *  Please see otf2_print_region_struct for a detailed description. */
typedef struct otf2_print_def_name_struct otf2_print_def_name;

/** @internal
 *  @brief Keeps all data for the callbacks. */
struct otf2_print_data_struct
{
    /** @brief Reader handle. */
    OTF2_Reader* reader;
    /** @brief List of locations to process. */
    otf2_vector* locations_to_read;

    /** @brief Set of string definitions. */
    otf2_hash_table* strings;
    /** @brief Set of system tree node definitions. */
    otf2_hash_table* system_tree_nodes;
    /** @brief Set of location group definitions. */
    otf2_hash_table* location_groups;
    /** @brief Set of region definitions. */
    otf2_hash_table* locations;
    /** @brief Set of region definitions. */
    otf2_hash_table* regions;
    /** @brief Set of group definitions. */
    otf2_hash_table* groups;
    /** @brief Set of metric definitions. */
    otf2_hash_table* metrics;
    /** @brief Set of MPI comms definitions. */
    otf2_hash_table* mpi_comms;
    /** @brief Set of attribute definitions. */
    otf2_hash_table* attributes;
    /** @brief Set of parameter definitions. */
    otf2_hash_table* parameters;

    /** @brief File handle for dot output. */
    FILE* dot_file;
};

/** @internal
 *  @brief Region definition element. */
struct otf2_print_def_name_struct
{
    /** @brief The ID of the definition. */
    uint64_t def_id;
    /** @brief The name if the definition. */
    char*    name;
};


/* ___ Prototypes for static functions. _____________________________________ */

static void
otf2_print_anchor_file_information( OTF2_Reader* reader );

static void
otf2_get_parameters( int    argc,
                     char** argv,
                     char** anchorFile );

static void
check_pointer( void* pointer,
               char* description,
               ... );

static void
check_status( OTF2_ErrorCode status,
              char*          description,
              ... );

static void
check_condition( bool  condition,
                 char* description,
                 ... );

static void
otf2_print_add_location_to_read( otf2_print_data* data,
                                 uint64_t         locationID );

static void
otf2_print_add_string( otf2_hash_table* strings,
                       uint32_t         stringID,
                       const char*      content );

static void
otf2_print_add_def64_name( otf2_hash_table* defs,
                           otf2_hash_table* strings,
                           uint64_t         defID,
                           uint32_t         stringID );

static char*
otf2_print_get_buffer( size_t len );

static const char*
otf2_print_get_id64( uint64_t ID );

static const char*
otf2_print_get_name( const char* name,
                     uint64_t    ID );

static const char*
otf2_print_get_def64_name( const otf2_hash_table* defs,
                           uint64_t               defID );

static char*
otf2_print_get_string( const otf2_hash_table* strings,
                       uint32_t               stringID );

static void
otf2_print_attributes( otf2_print_data*    data,
                       OTF2_AttributeList* attributes );

static const char*
otf2_print_get_region_flags( OTF2_RegionFlag regionFlags );


static const char*
otf2_print_get_invalid( uint64_t ID );


#include "otf2_print_types.h"


/* ___ Prototypes for event callbacks. ______________________________________ */



static OTF2_CallbackCode
BufferFlush_print( uint64_t            locationID,
                   uint64_t            time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_TimeStamp      stopTime );

static OTF2_CallbackCode
MeasurementOnOff_print( uint64_t             locationID,
                        uint64_t             time,
                        void*                userData,
                        OTF2_AttributeList*  attributes,
                        OTF2_MeasurementMode mode );

static OTF2_CallbackCode
Enter_print( uint64_t            locationID,
             uint64_t            time,
             void*               userData,
             OTF2_AttributeList* attributes,
             uint32_t            regionID );

static OTF2_CallbackCode
Leave_print( uint64_t            locationID,
             uint64_t            time,
             void*               userData,
             OTF2_AttributeList* attributes,
             uint32_t            regionID );

static OTF2_CallbackCode
MpiSend_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes,
               uint32_t            receiver,
               uint32_t            communicator,
               uint32_t            msgTag,
               uint64_t            msgLength );

static OTF2_CallbackCode
MpiIsend_print( uint64_t            locationID,
                uint64_t            time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            receiver,
                uint32_t            communicator,
                uint32_t            msgTag,
                uint64_t            msgLength,
                uint64_t            requestID );

static OTF2_CallbackCode
MpiIsendComplete_print( uint64_t            locationID,
                        uint64_t            time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        uint64_t            requestID );

static OTF2_CallbackCode
MpiIrecvRequest_print( uint64_t            locationID,
                       uint64_t            time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint64_t            requestID );

static OTF2_CallbackCode
MpiRecv_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes,
               uint32_t            sender,
               uint32_t            communicator,
               uint32_t            msgTag,
               uint64_t            msgLength );

static OTF2_CallbackCode
MpiIrecv_print( uint64_t            locationID,
                uint64_t            time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            sender,
                uint32_t            communicator,
                uint32_t            msgTag,
                uint64_t            msgLength,
                uint64_t            requestID );

static OTF2_CallbackCode
MpiRequestTest_print( uint64_t            locationID,
                      uint64_t            time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      uint64_t            requestID );

static OTF2_CallbackCode
MpiRequestCancelled_print( uint64_t            locationID,
                           uint64_t            time,
                           void*               userData,
                           OTF2_AttributeList* attributes,
                           uint64_t            requestID );

static OTF2_CallbackCode
MpiCollectiveBegin_print( uint64_t            locationID,
                          uint64_t            time,
                          void*               userData,
                          OTF2_AttributeList* attributes );

static OTF2_CallbackCode
MpiCollectiveEnd_print( uint64_t               locationID,
                        uint64_t               time,
                        void*                  userData,
                        OTF2_AttributeList*    attributes,
                        OTF2_MpiCollectiveType type,
                        uint32_t               commId,
                        uint32_t               root,
                        uint64_t               sizeSent,
                        uint64_t               sizeReceived );

static OTF2_CallbackCode
OmpFork_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes,
               uint32_t            numberOfRequestedThreads );

static OTF2_CallbackCode
OmpJoin_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes );

static OTF2_CallbackCode
OmpAcquireLock_print( uint64_t            locationID,
                      uint64_t            time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      uint32_t            lockID,
                      uint32_t            acquisitionOrder );

static OTF2_CallbackCode
OmpReleaseLock_print( uint64_t            locationID,
                      uint64_t            time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      uint32_t            lockID,
                      uint32_t            acquisitionOrder );

static OTF2_CallbackCode
OmpTaskCreate_print( uint64_t            locationID,
                     uint64_t            time,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     uint64_t            taskID );

static OTF2_CallbackCode
OmpTaskSwitch_print( uint64_t            locationID,
                     uint64_t            time,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     uint64_t            taskID );

static OTF2_CallbackCode
OmpTaskComplete_print( uint64_t            locationID,
                       uint64_t            time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint64_t            taskID );

static OTF2_CallbackCode
Metric_print( uint64_t                locationID,
              uint64_t                time,
              void*                   userData,
              OTF2_AttributeList*     attributes,
              uint32_t                metric_id,
              uint8_t                 number_of_metrics,
              const OTF2_Type*        types,
              const OTF2_MetricValue* values );

static OTF2_CallbackCode
ParameterString_print( uint64_t            locationID,
                       uint64_t            time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint32_t            parameter,
                       uint32_t            value );

static OTF2_CallbackCode
ParameterInt_print( uint64_t            locationID,
                    uint64_t            time,
                    void*               userData,
                    OTF2_AttributeList* attributes,
                    uint32_t            parameter,
                    int64_t             value );

static OTF2_CallbackCode
ParameterUnsignedInt_print( uint64_t            locationID,
                            uint64_t            time,
                            void*               userData,
                            OTF2_AttributeList* attributes,
                            uint32_t            parameter,
                            uint64_t            value );

static OTF2_CallbackCode
Unknown_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes );



/* ___ Prototypes for global definition callbacks. __________________________ */



static OTF2_CallbackCode
GlobDefString_print( void*       userData,
                     uint32_t    stringID,
                     const char* string );

static OTF2_CallbackCode
GlobDefLocation_print( void*             userData,
                       uint64_t          locationID,
                       uint32_t          stringID,
                       OTF2_LocationType locationType,
                       uint64_t          numberOfEvents,
                       uint32_t          locationGroup );

static OTF2_CallbackCode
GlobDefLocationGroup_print( void*                  userdata,
                            uint32_t               group_id,
                            uint32_t               name,
                            OTF2_LocationGroupType type,
                            uint32_t               system_tree_parent );

static OTF2_CallbackCode
GlobDefRegion_print( void*           userData,
                     uint32_t        regionID,
                     uint32_t        name,
                     uint32_t        canonicalName,
                     uint32_t        description,
                     OTF2_RegionRole regionRole,
                     OTF2_Paradigm   paradigm,
                     OTF2_RegionFlag regionFlags,
                     uint32_t        sourceFile,
                     uint32_t        beginLineNumber,
                     uint32_t        endLineNumber );

static OTF2_CallbackCode
GlobDefGroup_print( void*           userData,
                    uint32_t        groupID,
                    uint32_t        name,
                    OTF2_GroupType  type,
                    uint32_t        numberOfMembers,
                    const uint64_t* members );

static OTF2_CallbackCode
GlobDefCallsite_print( void*    userdata,
                       uint32_t callsite_identifier,
                       uint32_t source_file,
                       uint32_t line_number,
                       uint32_t region_entered,
                       uint32_t region_left );

static OTF2_CallbackCode
GlobDefCallpath_print( void*    userdata,
                       uint32_t callpath_identifier,
                       uint32_t parent_callpath,
                       uint32_t region_identifier );


static OTF2_CallbackCode
GlobDefMpiComm_print( void*    userdata,
                      uint32_t comm_id,
                      uint32_t comm_name,
                      uint32_t group_id,
                      uint32_t comm_parent );

static OTF2_CallbackCode
GlobDefMetricMember_print( void*           userData,
                           uint32_t        metric_member_id,
                           uint32_t        name,
                           uint32_t        description,
                           OTF2_MetricType type,
                           OTF2_MetricMode mode,
                           OTF2_Type       value_type,
                           OTF2_MetricBase base,
                           int64_t         exponent,
                           uint32_t        unit );

static OTF2_CallbackCode
GlobDefMetricClass_print( void*                 userdata,
                          uint32_t              metric_class_id,
                          uint8_t               number_of_metrics,
                          const uint32_t*       metric_members,
                          OTF2_MetricOccurrence occurrence );

static OTF2_CallbackCode
GlobDefMetricInstance_print( void*            userdata,
                             uint32_t         metricInstanceID,
                             uint32_t         metricClass,
                             uint64_t         recorder,
                             OTF2_MetricScope scope_type,
                             uint64_t         scope );

static OTF2_CallbackCode
GlobDefSystemTreeNode_print( void*    userData,
                             uint32_t nodeID,
                             uint32_t name,
                             uint32_t className,
                             uint32_t nodeParent );

static OTF2_CallbackCode
GlobDefAttribute_print( void*     userData,
                        uint32_t  attributeID,
                        uint32_t  name,
                        OTF2_Type type );

static OTF2_CallbackCode
GlobDefClockProperties_print( void*    userData,
                              uint64_t timer_resolution,
                              uint64_t global_offset,
                              uint64_t trace_length );

OTF2_CallbackCode
GlobDefParameter_print( void*              userData,
                        uint32_t           parameterID,
                        uint32_t           name,
                        OTF2_ParameterType type );

static OTF2_CallbackCode
GlobDefUnknown_print( void* userData );

static OTF2_CallbackCode
DefMappingTable_print( void*             userData,
                       OTF2_MappingType  mapType,
                       const OTF2_IdMap* iDMap );

static OTF2_CallbackCode
DefClockOffset_print( void*    userData,
                      uint64_t time,
                      int64_t  offset,
                      double   stddev );


/* ___ main _________________________________________________________________ */



/** @internal
 *  @brief This tool prints out all event files of an archive to console.
 *
 *  Usage: otf2profile [OPTION]... ANCHORFILE \n
 *  Print the content of all files of an OTF2 archive with the ANCHORFILE.
 *
 *    -A, --show-all           Print all output including definitions and anchor file.
 *    -G, --show-global-defs   Print global definitions.
 *    -I  --show-info          Print information from the anchor file.
 *    -M, --show-mappings      Print mappings to global definitions.
 *    -C, --show-clock-offsets Print clock offsets to global timer.
 *    -L, --location LID       Limit output to location LID.
 *    -s, --step N             Step through output by steps of N events.
 *        --silent             Only validate trace and do not print any events.
 *        --time MIN MAX       Limit output to events within time interval.
 *    -d, --debug              Turn on debug mode.
 *    -V, --version            Print version information.
 *    -h, --help               Print this help information.
 *
 *  @param argc              Programs argument counter.
 *  @param argv              Programs argument values.
 *
 *  @return                  Returns EXIT_SUCCESS if successful, EXIT_FAILURE
 *                           if an error occures.
 */
void
ReadTraceFile()
{
    char* anchor_file = NULL;
    //otf2_get_parameters( argc, argv, &anchor_file );
    //otf2_ANCHORFILE_INFO = true;
    otf2_ALL             = true;
    otf2_GLOBDEFS        = true;
    StateGroupDef(0, "Empty" );
    anchor_file=Converter::trc;

    if ( otf2_NOLOCALDEFS && ( otf2_MAPPINGS || otf2_CLOCK_OFFSETS ) )
    {
        printf( "ERROR: --no-local-defs is mutual exclusive to --show-mappings and --show-clock-offsets.\n" );
        printf( "Try 'otf2profile --help' for information on usage.\n\n" );
        exit( EXIT_FAILURE );
    }

    dprintf( "\n=== OTF2-PRINT ===\n" );

    /* Get a reader handle. */
    OTF2_Reader* reader = OTF2_Reader_Open( anchor_file );
    check_pointer( reader, "Create new reader handle." );

    if ( otf2_ANCHORFILE_INFO )
    {
        otf2_print_anchor_file_information( reader );

        /* Only exit if --show-info was given. */
        if ( !otf2_ALL )
        {
            OTF2_Reader_Close( reader );

            /* This is just to add a message to the debug output. */
            check_status( OTF2_SUCCESS, "Delete reader handle." );
            check_status( OTF2_SUCCESS, "Programm finished successfully." );

            return EXIT_SUCCESS;
        }
    }


/* ___ Read Global Definitions _______________________________________________*/



    /* Add a nice table header. */
    if ( otf2_GLOBDEFS )
    {
        dprintf( "\n" );
        dprintf( "=== Global Definitions =========================================================" );
        dprintf( "\n\n" );
        dprintf( "%-*s %12s  Attributes\n", otf2_DEF_COLUMN_WIDTH, "Definition", "ID" );
        dprintf( "--------------------------------------------------------------------------------\n" );
    }
    /* Define definition callbacks. */
    OTF2_GlobalDefReaderCallbacks* def_callbacks = OTF2_GlobalDefReaderCallbacks_New();
    check_pointer( def_callbacks, "Create global definition callback handle." );
    OTF2_GlobalDefReaderCallbacks_SetUnknownCallback( def_callbacks, GlobDefUnknown_print );
    OTF2_GlobalDefReaderCallbacks_SetStringCallback( def_callbacks, GlobDefString_print );
    OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeCallback( def_callbacks, GlobDefSystemTreeNode_print );
    OTF2_GlobalDefReaderCallbacks_SetLocationGroupCallback( def_callbacks, GlobDefLocationGroup_print );
    OTF2_GlobalDefReaderCallbacks_SetLocationCallback( def_callbacks, GlobDefLocation_print );
    OTF2_GlobalDefReaderCallbacks_SetRegionCallback( def_callbacks, GlobDefRegion_print );
    OTF2_GlobalDefReaderCallbacks_SetGroupCallback( def_callbacks, GlobDefGroup_print );
    OTF2_GlobalDefReaderCallbacks_SetMpiCommCallback( def_callbacks, GlobDefMpiComm_print );
    OTF2_GlobalDefReaderCallbacks_SetMetricMemberCallback( def_callbacks, GlobDefMetricMember_print );
    OTF2_GlobalDefReaderCallbacks_SetAttributeCallback( def_callbacks, GlobDefAttribute_print );
    OTF2_GlobalDefReaderCallbacks_SetParameterCallback( def_callbacks, GlobDefParameter_print );


    /* Only register these callbacks if selected. */
    if ( otf2_GLOBDEFS )
    {
        OTF2_GlobalDefReaderCallbacks_SetClockPropertiesCallback( def_callbacks, GlobDefClockProperties_print );
        OTF2_GlobalDefReaderCallbacks_SetCallsiteCallback( def_callbacks, GlobDefCallsite_print );
        OTF2_GlobalDefReaderCallbacks_SetCallpathCallback( def_callbacks, GlobDefCallpath_print );
        OTF2_GlobalDefReaderCallbacks_SetMetricClassCallback( def_callbacks, GlobDefMetricClass_print );
        OTF2_GlobalDefReaderCallbacks_SetMetricInstanceCallback( def_callbacks, GlobDefMetricInstance_print );
    }


    /* Get number of locations from the anchor file. */
    uint64_t       num_locations = 0;
    OTF2_ErrorCode status        = OTF2_SUCCESS;
    status = OTF2_Reader_GetNumberOfLocations( reader, &num_locations );
    check_status( status, "Get number of locations. Number of locations: %" PRIu64,
                  num_locations );


    /* User data for callbacks. */
    otf2_print_data user_data;
    user_data.reader            = reader;
    user_data.locations_to_read = otf2_vector_create();

    user_data.strings = otf2_hash_table_create_size( 512,
                                                     otf2_hash_table_hash_int64,
                                                     otf2_hash_table_compare_uint64 );
    user_data.system_tree_nodes = otf2_hash_table_create_size( 128,
                                                               otf2_hash_table_hash_int64,
                                                               otf2_hash_table_compare_uint64 );
    user_data.location_groups = otf2_hash_table_create_size( 128,
                                                             otf2_hash_table_hash_int64,
                                                             otf2_hash_table_compare_uint64 );
    user_data.locations = otf2_hash_table_create_size( 128,
                                                       otf2_hash_table_hash_int64,
                                                       otf2_hash_table_compare_uint64 );
    user_data.regions = otf2_hash_table_create_size( 128,
                                                     otf2_hash_table_hash_int64,
                                                     otf2_hash_table_compare_uint64 );
    user_data.groups = otf2_hash_table_create_size( 128,
                                                    otf2_hash_table_hash_int64,
                                                    otf2_hash_table_compare_uint64 );
    user_data.mpi_comms = otf2_hash_table_create_size( 128,
                                                       otf2_hash_table_hash_int64,
                                                       otf2_hash_table_compare_uint64 );
    user_data.metrics = otf2_hash_table_create_size( 128,
                                                     otf2_hash_table_hash_int64,
                                                     otf2_hash_table_compare_uint64 );
    user_data.attributes = otf2_hash_table_create_size( 128,
                                                        otf2_hash_table_hash_int64,
                                                        otf2_hash_table_compare_uint64 );
    user_data.parameters = otf2_hash_table_create_size( 128,
                                                        otf2_hash_table_hash_int64,
                                                        otf2_hash_table_compare_uint64 );

    user_data.dot_file = NULL;


    /* If in dot output mode open dot file. */
    char dot_path[ 1024 ] = "";
    if ( otf2_DOT )
    {
        sprintf( dot_path, "%.*s.SystemTree.dot", ( int )strlen( anchor_file ) - 5, anchor_file );

        user_data.dot_file = fopen( dot_path, "w" );
        if ( user_data.dot_file == NULL )
        {
            printf( "ERROR: Could not open dot file.\n" );
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
        OTF2_DefReaderCallbacks_SetMappingTableCallback( local_def_callbacks, DefMappingTable_print );
    }
    if ( otf2_CLOCK_OFFSETS )
    {
        OTF2_DefReaderCallbacks_SetClockOffsetCallback( local_def_callbacks, DefClockOffset_print );
    }

    /* Open a new local event reader for each found location ID. */
    if ( otf2_MAPPINGS || otf2_CLOCK_OFFSETS )
    {
        dprintf( "\n" );
        dprintf( "=== Per Location Definitions ===================================================" );
        dprintf( "\n\n" );
        dprintf( "%-*s %12s  Attributes\n", otf2_DEF_COLUMN_WIDTH, "Definition", "Location" );
        dprintf( "--------------------------------------------------------------------------------\n" );
    }

    for ( size_t i = 0; i < otf2_vector_size( user_data.locations_to_read ); i++ )
    {
        uint64_t* location_item      = otf2_vector_at( user_data.locations_to_read, i );
        uint64_t  locationIdentifier = *location_item;

        /* Do not open the event reader, when only showing the global defs */
        if ( !otf2_GLOBDEFS || otf2_ALL )
        {
            OTF2_EvtReader* evt_reader = OTF2_Reader_GetEvtReader( reader,
                                                                   locationIdentifier );
            check_pointer( evt_reader, "Create local event reader for location %" PRIu64 ".",
                           locationIdentifier );
        }

        if ( otf2_NOLOCALDEFS || ( otf2_GLOBDEFS && !otf2_ALL ) )
        {
            continue;
        }

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

                /* continue reading, if we have an duplicate mapping table */
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
    OTF2_DefReaderCallbacks_Delete( local_def_callbacks );


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
        check_status( OTF2_SUCCESS, "Programm finished successfully." );

        return EXIT_SUCCESS;
    }

    if ( ( otf2_GLOBDEFS || otf2_MAPPINGS || otf2_CLOCK_OFFSETS ) && !otf2_ALL )
    {
        OTF2_Reader_Close( reader );

        /* This is just to add a message to the debug output. */
        check_status( OTF2_SUCCESS, "Delete reader handle." );
        check_status( OTF2_SUCCESS, "Programm finished successfully." );

        return EXIT_SUCCESS;
    }

    ProcessDefs();

/* ___ Read Event Records ____________________________________________________*/



    /* Add a nice table header. */
    if ( otf2_GLOBDEFS && !otf2_SILENT )
    {
        dprintf( "\n" );
        dprintf( "=== Events =====================================================================\n" );
    }

    /* Define event callbacks. */
    OTF2_GlobalEvtReaderCallbacks* evt_callbacks = OTF2_GlobalEvtReaderCallbacks_New();
    check_pointer( evt_callbacks, "Create event reader callbacks." );

    OTF2_GlobalEvtReaderCallbacks_SetUnknownCallback( evt_callbacks, Unknown_print );
    OTF2_GlobalEvtReaderCallbacks_SetBufferFlushCallback( evt_callbacks, BufferFlush_print );
    OTF2_GlobalEvtReaderCallbacks_SetMeasurementOnOffCallback( evt_callbacks, MeasurementOnOff_print );
    OTF2_GlobalEvtReaderCallbacks_SetEnterCallback( evt_callbacks, Enter_print );
    OTF2_GlobalEvtReaderCallbacks_SetLeaveCallback( evt_callbacks, Leave_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiSendCallback( evt_callbacks, MpiSend_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCallback( evt_callbacks, MpiIsend_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCompleteCallback( evt_callbacks, MpiIsendComplete_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvRequestCallback( evt_callbacks, MpiIrecvRequest_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiRecvCallback( evt_callbacks, MpiRecv_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvCallback( evt_callbacks, MpiIrecv_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiRequestTestCallback( evt_callbacks, MpiRequestTest_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiRequestCancelledCallback( evt_callbacks, MpiRequestCancelled_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveBeginCallback( evt_callbacks, MpiCollectiveBegin_print );
    OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveEndCallback( evt_callbacks, MpiCollectiveEnd_print );
    OTF2_GlobalEvtReaderCallbacks_SetOmpForkCallback( evt_callbacks, OmpFork_print );
    OTF2_GlobalEvtReaderCallbacks_SetOmpJoinCallback( evt_callbacks, OmpJoin_print );
    OTF2_GlobalEvtReaderCallbacks_SetOmpAcquireLockCallback( evt_callbacks, OmpAcquireLock_print );
    OTF2_GlobalEvtReaderCallbacks_SetOmpReleaseLockCallback( evt_callbacks, OmpReleaseLock_print );
    OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCreateCallback( evt_callbacks, OmpTaskCreate_print );
    OTF2_GlobalEvtReaderCallbacks_SetOmpTaskSwitchCallback( evt_callbacks, OmpTaskSwitch_print );
    OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCompleteCallback( evt_callbacks, OmpTaskComplete_print );
    OTF2_GlobalEvtReaderCallbacks_SetMetricCallback( evt_callbacks, Metric_print );
    OTF2_GlobalEvtReaderCallbacks_SetParameterStringCallback( evt_callbacks, ParameterString_print );
    OTF2_GlobalEvtReaderCallbacks_SetParameterIntCallback( evt_callbacks, ParameterInt_print );
    OTF2_GlobalEvtReaderCallbacks_SetParameterUnsignedIntCallback( evt_callbacks, ParameterUnsignedInt_print );


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
        if ( !otf2_SILENT )
        {
            dprintf( "\n" );
            dprintf( "%-*s %15s %20s  Attributes\n",
                    otf2_EVENT_COLUMN_WIDTH, "Event", "Location", "Timestamp" );
            dprintf( "--------------------------------------------------------------------------------\n" );
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

    OTF2_Reader_Close( reader );

    /* This is just to add a message to the debug output. */
    check_status( OTF2_SUCCESS, "Delete reader handle." );
    check_status( OTF2_SUCCESS, "Programm finished successfully." );

    if ( !otf2_SILENT )
    {
        dprintf( "\n" );
    }

    otf2_vector_foreach( user_data.locations_to_read, free );
    otf2_vector_free( user_data.locations_to_read );
    otf2_hash_table_free_all( user_data.strings, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.regions, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.system_tree_nodes, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.location_groups, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.locations, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.groups, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.mpi_comms, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.metrics, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.attributes, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( user_data.parameters, otf2_hash_table_delete_none, free );

    return EXIT_SUCCESS;
}



/* ___ Implementation of static functions ___________________________________ */

static uint64_t
otf2_parse_number_argument( const char* option,
                            const char* argument )
{
    uint64_t number = 0;
    for ( uint8_t j = 0; argument[ j ]; j++ )
    {
        if ( ( argument[ j ] < '0' ) || ( argument[ j ] > '9' ) )
        {
            printf( "ERROR: Invalid number argument for %s: %s\n",
                    option, argument );
            printf( "Try 'otf2profile --help' for information on usage.\n\n" );
            exit( EXIT_FAILURE );
        }
        uint64_t new_number = number * 10 + argument[ j ] - '0';
        if ( new_number < number )
        {
            printf( "ERROR: Number argument to large for %s: %s\n",
                    option, argument );
            printf( "Try 'otf2profile --help' for information on usage.\n\n" );
            exit( EXIT_FAILURE );
        }
        number = new_number;
    }

    return number;
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
    /* Check if there is at least one command line parameter. */
    if ( argc < 2 )
    {
        printf( "ERROR: No input parameters specified.\n" );
        printf( "Try 'otf2profile --help' for information on usage.\n\n" );
        exit( EXIT_FAILURE );
    }

    /* Check for passed '--help' or '-h' and print help. */
    if ( !strcmp( argv[ 1 ], "--help" ) || !strcmp( argv[ 1 ], "-h" ) )
    {
        printf(
        #include "otf2_print_usage.h"
            );
        printf( "\n" );
        exit( EXIT_SUCCESS );
    }

    /* Check for passed '--version' or '-v' and print version information. */
    if ( !strcmp( argv[ 1 ], "--version" ) || !strcmp( argv[ 1 ], "-V" ) )
    {
        printf( "otf2profile - Version %s\n", PACKAGE_VERSION );
        exit( EXIT_SUCCESS );
    }

    /* Check for requested system tree dot output. */
    if ( !strcmp( argv[ 1 ], "--system-tree" ) )
    {
        if ( argc < 3 )
        {
            printf( "ERROR: No path to an anchor file specified.\n" );
            printf( "Try 'otf2profile --help' for information on usage.\n\n" );
            exit( EXIT_FAILURE );
        }

        *anchorFile = argv[ argc - 1 ];
        /* Check if anchor file path was passed. */
        if ( *anchorFile == NULL )
        {
            printf( "ERROR: No path to an anchor file specified.\n" );
            printf( "Try 'otf2profile --help' for information on usage.\n\n" );
            exit( EXIT_FAILURE );
        }

        otf2_DOT = true;
        return;
    }

    if ( argc == 2 && !strncmp( argv[ 1 ], "-", 1 ) )
    {
        printf( "ERROR: Unknown parameter.\n" );
        printf( "Try 'otf2profile --help' for information on usage.\n\n" );
        exit( EXIT_FAILURE );
    }

    if ( !strncmp( argv[ argc - 1 ], "-", 1 ) )
    {
        printf( "ERROR: No OTF2 anchor file specified.\n" );
        printf( "Try 'otf2profile --help' for information on usage.\n\n" );
        exit( EXIT_FAILURE );
    }

    /* Last parameter is anchor file path. */
    *anchorFile = argv[ argc - 1 ];

    /* Check if anchor file path was passed. */
    if ( *anchorFile == NULL )
    {
        printf( "ERROR: No path to an anchor file specified.\n" );
        printf( "Try 'otf2profile --help' for information on usage.\n\n" );
        exit( EXIT_FAILURE );
    }

    for ( uint8_t i = 1; i < argc - 1; i++ )
    {
        if ( !strcmp( argv[ i ], "--debug" ) || !strcmp( argv[ i ], "-d" ) )
        {
            otf2_DEBUG = true;
        }

        else if ( !strcmp( argv[ i ], "--show-all" ) || !strcmp( argv[ i ], "-A" ) )
        {
            otf2_ANCHORFILE_INFO = true;
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

        else if ( !strcmp( argv[ i ], "--silent" ) )
        {
            otf2_SILENT = true;
        }

        else if ( !strcmp( argv[ i ], "--location" ) || !strcmp( argv[ i ], "-L" ) )
        {
            if ( i + 1 == argc - 1 )
            {
                printf( "ERROR: Missing argument for %s.\n", argv[ i ] );
                printf( "Try 'otf2profile --help' for information on usage.\n\n" );
                exit( EXIT_FAILURE );
            }

            otf2_LOCAL = otf2_parse_number_argument( argv[ i ], argv[ i + 1 ] );
            i++;
        }

        else if ( !strcmp( argv[ i ], "--time" ) )
        {
            if ( i + 2 >= argc - 1 )
            {
                printf( "ERROR: Missing argument for %s.\n", argv[ i ] );
                printf( "Try 'otf2profile --help' for information on usage.\n\n" );
                exit( EXIT_FAILURE );
            }

            otf2_MINTIME = otf2_parse_number_argument( argv[ i ], argv[ i + 1 ] );
            otf2_MAXTIME = otf2_parse_number_argument( argv[ i ], argv[ i + 2 ] );
            i           += 2;
        }

        else if ( !strcmp( argv[ i ], "--step" ) || !strcmp( argv[ i ], "-s" ) )
        {
            if ( i + 1 == argc - 1 )
            {
                printf( "ERROR: Missing argument for %s.\n", argv[ i ] );
                printf( "Try 'otf2profile --help' for information on usage.\n\n" );
                exit( EXIT_FAILURE );
            }

            otf2_STEP = otf2_parse_number_argument( argv[ i ], argv[ i + 1 ] );
            i++;
        }
        else
        {
            printf( "Skipped unknown control option %s.\n", argv[ i ] );
        }
    }

    return;
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

    printf( "%-*s ", otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "File substrate" );
    switch ( substrate )
    {
        case OTF2_SUBSTRATE_NONE:
            printf( "NONE\n" );
            break;
        case OTF2_SUBSTRATE_SION:
            printf( "SION\n" );
            break;
        case OTF2_SUBSTRATE_POSIX:
            printf( "POSIX\n" );
            break;
        default:
            printf( "%s\n", otf2_print_get_invalid( substrate ) );
    }

    OTF2_Compression compression;
    status = OTF2_Reader_GetCompression( reader, &compression );
    check_status( status, "Read compression mode." );

    printf( "%-*s ", otf2_ANCHOR_FILE_INFO_COLUMN_WIDTH, "Compression" );
    switch ( compression )
    {
        case OTF2_COMPRESSION_NONE:
            printf( "NONE\n" );
            break;
        case OTF2_COMPRESSION_ZLIB:
            printf( "ZLIB\n" );
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
               char* description,
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
              char*          description,
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
                 char* description,
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


/** @internal
 *  @brief Add a locations to the list of locations to read events from.
 *
 *  @param locations        List of regions.
 *  @param locationID       Location ID of the location.
 */
void
otf2_print_add_location_to_read( otf2_print_data* data,
                                 uint64_t         locationID )
{
    uint64_t* location_item = malloc( sizeof( *location_item ) );
    assert( location_item );

    *location_item = locationID;

    otf2_vector_push_back( data->locations_to_read, location_item );
}


/** @internal
 *  @brief Add a string to the set of strings.
 *
 *  @param strings          Set of strings.
 *  @param stringID         String ID of new element.
 *  @param content          Content of the new element.
 */
void
otf2_print_add_string( otf2_hash_table* strings,
                       uint32_t         stringID,
                       const char*      content )
{
    size_t                 hint;
    otf2_hash_table_entry* entry;

    if ( stringID == OTF2_UNDEFINED_UINT32 )
    {
        return;
    }

    uint64_t string_id_64 = stringID;
    entry = otf2_hash_table_find( strings, &string_id_64, &hint );

    if ( entry )
    {
        /* duplicate */
        return;
    }

    otf2_print_def_name* new_string = malloc( sizeof( *new_string )
                                              + strlen( content ) + 1 );
    assert( new_string );

    new_string->def_id = string_id_64;
    new_string->name   = ( char* )new_string + sizeof( *new_string );
    memcpy( new_string->name, content, strlen( content ) + 1 );

    otf2_hash_table_insert( strings,
                            &new_string->def_id,
                            new_string,
                            &hint );

    entry = otf2_hash_table_find( strings, &string_id_64, &hint );

    assert( entry );
}


/** @internal
 *  @brief Add a def with id tye uint64_t to the set of defs.
 *
 *  @param regions          Set of regions.
 *  @param regionID         Region ID of new region.
 *  @param stringID         String ID of new region.
 */
void
otf2_print_add_def64_name( otf2_hash_table* defs,
                           otf2_hash_table* strings,
                           uint64_t         defID,
                           uint32_t         stringID )
{
    size_t                 hint;
    otf2_hash_table_entry* entry;

    if ( defID == OTF2_UNDEFINED_UINT64 )
    {
        return;
    }

    entry = otf2_hash_table_find( defs, &defID, &hint );

    if ( entry )
    {
        return;
    }

    otf2_print_def_name* new_def = malloc( sizeof( *new_def ) );
    assert( new_def );

    new_def->def_id = defID;

    new_def->name = otf2_print_get_string( strings, stringID );

    otf2_hash_table_insert( defs, &new_def->def_id, new_def, &hint );
}

/** @internal
 *  @brief Add a def with id tye uint32_t to the set of defs.
 *
 *  @param regions          Set of regions.
 *  @param regionID         Region ID of new region.
 *  @param stringID         String ID of new region.
 */
static void
otf2_print_add_def_name( otf2_hash_table* defs,
                         otf2_hash_table* strings,
                         uint32_t         defID,
                         uint32_t         stringID )
{
    uint64_t def64_id = defID;
    if ( defID == OTF2_UNDEFINED_UINT32 )
    {
        def64_id = OTF2_UNDEFINED_UINT64;
    }
    otf2_print_add_def64_name( defs, strings, def64_id, stringID );
}


static void
otf2_print_add_system_tree_node( otf2_print_data* data,
                                 uint32_t         defID,
                                 uint32_t         stringID )
{
    otf2_print_add_def_name( data->system_tree_nodes,
                             data->strings,
                             defID, stringID );
}


static void
otf2_print_add_location_group( otf2_print_data* data,
                               uint32_t         defID,
                               uint32_t         stringID )
{
    otf2_print_add_def_name( data->location_groups,
                             data->strings,
                             defID, stringID );
}


static void
otf2_print_add_location( otf2_print_data* data,
                         uint64_t         defID,
                         uint32_t         stringID )
{
    otf2_print_add_def64_name( data->locations,
                               data->strings,
                               defID, stringID );
}


static void
otf2_print_add_region( otf2_print_data* data,
                       uint32_t         defID,
                       uint32_t         stringID )
{
    otf2_print_add_def_name( data->regions,
                             data->strings,
                             defID, stringID );
}


static void
otf2_print_add_group( otf2_print_data* data,
                      uint32_t         defID,
                      uint32_t         stringID )
{
    otf2_print_add_def_name( data->groups,
                             data->strings,
                             defID, stringID );
}


static void
otf2_print_add_mpi_comm( otf2_print_data* data,
                         uint32_t         defID,
                         uint32_t         stringID )
{
    otf2_print_add_def_name( data->mpi_comms,
                             data->strings,
                             defID, stringID );
}


static void
otf2_print_add_metric( otf2_print_data* data,
                       uint32_t         defID,
                       uint32_t         stringID )
{
    otf2_print_add_def_name( data->metrics,
                             data->strings,
                             defID, stringID );
}


static void
otf2_print_add_attribute( otf2_print_data* data,
                          uint32_t         defID,
                          uint32_t         stringID )
{
    otf2_print_add_def_name( data->attributes,
                             data->strings,
                             defID, stringID );
}


static void
otf2_print_add_parameter( otf2_print_data* data,
                          uint32_t         defID,
                          uint32_t         stringID )
{
    otf2_print_add_def_name( data->parameters,
                             data->strings,
                             defID, stringID );
}


#define BUFFER_SIZE 128

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

    if ( next->size <= len )
    {
        next->buffer = realloc( next->buffer, len );
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
    if ( id64 == OTF2_UNDEFINED_UINT32 )
    {
        id64 = OTF2_UNDEFINED_UINT64;
    }
    return otf2_print_get_id64( id64 );
}


const char*
otf2_print_get_id64( uint64_t ID )
{
    if ( ID == OTF2_UNDEFINED_UINT64 )
    {
        return "UNDEFINED";
    }

    char* buffer = otf2_print_get_buffer( BUFFER_SIZE );

    snprintf( buffer, BUFFER_SIZE, "%" PRIu64, ID );

    return buffer;
}


const char*
otf2_print_get_invalid( uint64_t ID )
{
    char* buffer = otf2_print_get_buffer( BUFFER_SIZE );

    snprintf( buffer, BUFFER_SIZE, "INVALID <%" PRIu64 ">", ID );
    return buffer;
}


const char*
otf2_print_get_name( const char* name,
                     uint64_t    ID )
{
    if ( !name )
    {
        return otf2_print_get_invalid( ID );
    }

    size_t len = name ? strlen( name ) + 1 : 0;
    /* add size for id */
    len += BUFFER_SIZE;

    char* buffer = otf2_print_get_buffer( len );

    snprintf( buffer, len, "\"%s\" <%" PRIu64 ">", name, ID );

    return buffer;
}


/** @internal
 *  @brief Get the name of a definition.
 *
 *  @param regions          Set of regions.
 *  @param strings          Set of strings.
 *  @param regionID         Region ID.
 *
 *  @return                 Returns the name of a region if successful, NULL
 *                          otherwise.
 */
static const char*
otf2_print_get_def_name( const otf2_hash_table* defs,
                         uint32_t               defID )
{
    uint64_t def64_id = defID;
    if ( defID == OTF2_UNDEFINED_UINT32 )
    {
        def64_id = OTF2_UNDEFINED_UINT64;
    }
    return otf2_print_get_def64_name( defs, def64_id );
}


/** @internal
 *  @brief Get the name of a definition.
 *
 *  @param regions          Set of regions.
 *  @param strings          Set of strings.
 *  @param regionID         Region ID.
 *
 *  @return                 Returns the name of a region if successful, NULL
 *                          otherwise.
 */
const char*
otf2_print_get_def64_name( const otf2_hash_table* defs,
                           uint64_t               defID )
{
    otf2_hash_table_entry* entry;

    if ( defID == OTF2_UNDEFINED_UINT64 )
    {
        return "UNDEFINED";
    }

    entry = otf2_hash_table_find( defs, &defID, NULL );

    if ( !entry )
    {
        return otf2_print_get_invalid( defID );
    }

    otf2_print_def_name* def = entry->value;

    return otf2_print_get_name( def->name, defID );
}


/** @internal
 *  @brief Get the content of a string.
 *
 *  @param strings          Set of strings.
 *  @param stringID         String ID.
 *
 *  @return                 Returns the content of a string if successful, NULL
 *                          otherwise.
 */
char*
otf2_print_get_string( const otf2_hash_table* strings,
                       uint32_t               stringID )
{
    otf2_hash_table_entry* entry;

    uint64_t string_id_64 = stringID;
    entry = otf2_hash_table_find( strings, &string_id_64, NULL );

    if ( !entry )
    {
        return NULL;
    }

    otf2_print_def_name* def = entry->value;

    return def->name;
}


/** @internal
 *  @brief Get the content of a region flag.
 *
 *  @param regionFlags      Bitset of region flags.
 *
 *  @return                 Returns the content of a region flag bit set.
 *
 */
static const char*
otf2_print_get_region_flags( OTF2_RegionFlag regionFlags )
{
    static char string[ 20 ];
    char*       sep = "";

    string[ 0 ] = '\0';
    if ( regionFlags == OTF2_REGION_FLAG_NONE )
    {
        strcat( string, "NONE" );
        return string;
    }
    if ( regionFlags & OTF2_REGION_FLAG_DYNAMIC )
    {
        strcat( string, "DYNAMIC" );
        sep = ",";
    }
    if ( regionFlags & OTF2_REGION_FLAG_PHASE )
    {
        strcat( string, sep );
        strcat( string, "PHASE" );
    }

    return string;
}


/** @internal
 *  @brief Print the attribute list of an event, if any.
 *
 *  @param attributes   The attributes.
 *
 */
void
otf2_print_attributes( otf2_print_data*    data,
                       OTF2_AttributeList* attributes )
{
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
        char*               value_buffer = otf2_print_get_buffer( BUFFER_SIZE );
        const char*         value_type_string;
        uint32_t            id;
        OTF2_Type           type;
        OTF2_AttributeValue value;

        OTF2_AttributeList_PopAttribute( attributes, &id, &type, &value );

        switch ( type )
        {
            case OTF2_TYPE_UINT8:
                value_type_string = "UINT8";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%" PRIu8, value.uint8 );
                break;

            case OTF2_TYPE_UINT16:
                value_type_string = "UINT16";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%" PRIu16, value.uint16 );
                break;

            case OTF2_TYPE_UINT32:
                value_type_string = "UINT32";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%" PRIu32, value.uint32 );
                break;

            case OTF2_TYPE_UINT64:
                value_type_string = "UINT64";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%" PRIu64, value.uint64 );
                break;

            case OTF2_TYPE_INT8:
                value_type_string = "INT8";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%" PRId8, value.int8 );
                break;

            case OTF2_TYPE_INT16:
                value_type_string = "INT16";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%" PRId16, value.int16 );
                break;

            case OTF2_TYPE_INT32:
                value_type_string = "INT32";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%" PRId32, value.int32 );
                break;

            case OTF2_TYPE_INT64:
                value_type_string = "INT64";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%" PRId64, value.int64 );
                break;

            case OTF2_TYPE_FLOAT:
                value_type_string = "FLOAT";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%f", value.float32 );
                break;

            case OTF2_TYPE_DOUBLE:
                value_type_string = "DOUBLE";
                snprintf( value_buffer, BUFFER_SIZE,
                          "%f", value.float64 );
                break;

            case OTF2_TYPE_STRING:
                value_type_string = "STRING";
                value_buffer      = ( char* )otf2_print_get_def_name( data->strings, value.uint32 );
                break;

            case OTF2_TYPE_ATTRIBUTE:
                value_type_string = "ATTRIBUTE";
                value_buffer      = ( char* )otf2_print_get_def_name( data->attributes, value.uint32 );
                break;

            case OTF2_TYPE_LOCATION:
                value_type_string = "LOCATION";
                value_buffer      = ( char* )otf2_print_get_def64_name( data->locations, value.uint64 );
                break;

            case OTF2_TYPE_REGION:
                value_type_string = "REGION";
                value_buffer      = ( char* )otf2_print_get_def_name( data->regions, value.uint32 );
                break;

            case OTF2_TYPE_GROUP:
                value_type_string = "GROUP";
                value_buffer      = ( char* )otf2_print_get_def_name( data->groups, value.uint32 );
                break;

            case OTF2_TYPE_METRIC:
                value_type_string = "METRIC";
                value_buffer      = ( char* )otf2_print_get_id( value.uint32 );
                break;

            case OTF2_TYPE_MPI_COMM:
                value_type_string = "MPI_COMM";
                value_buffer      = ( char* )otf2_print_get_def_name( data->mpi_comms, value.uint32 );
                break;

            case OTF2_TYPE_PARAMETER:
                value_type_string = "PARAMETER";
                value_buffer      = ( char* )otf2_print_get_def_name( data->parameters, value.uint32 );
                break;

            default:
                value_type_string = otf2_print_get_invalid( type );
                snprintf( value_buffer, BUFFER_SIZE,
                          "%08" PRIx64, value.uint64 );
                break;
        }
        printf( "%s(%s; %s; %s)",
                sep,
                otf2_print_get_def_name( data->attributes, id ),
                value_type_string,
                value_buffer );
        sep = ", ";
    }
    printf( "\n" );
}

/* ___ Implementation of callbacks __________________________________________ */



/** @internal
 *  @name Callbacks for events.
 *
 *  @param locationID       Location ID.
 *  @param time             Timestamp of the event.
 *  @param userData         Optional user data.
 *
 *  @return                 Returns OTF2_SUCCESS if successful, an error code
 *                          if an error occures.
 *  @{
 */
OTF2_CallbackCode
BufferFlush_print( uint64_t            locationID,
                   uint64_t            time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_TimeStamp      stopTime )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    printf( "%-*s %15" PRIu64 " %20" PRIu64 "  Stop Time: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "BUFFER_FLUSH",
            locationID, time, stopTime );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MeasurementOnOff_print( uint64_t             locationID,
                        uint64_t             time,
                        void*                userData,
                        OTF2_AttributeList*  attributes,
                        OTF2_MeasurementMode mode )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    printf( "%-*s %15" PRIu64 " %20" PRIu64 "  Mode: %s\n",
            otf2_EVENT_COLUMN_WIDTH, "MEASUREMENT_ON_OFF",
            locationID, time,
            otf2_print_get_measurement_mode( mode ) );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
Enter_print( uint64_t            locationID,
             uint64_t            time,
             void*               userData,
             OTF2_AttributeList* attributes,
             uint32_t            regionID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Region: %s\n",
            otf2_EVENT_COLUMN_WIDTH, "ENTER",
            locationID, time,
            otf2_print_get_def_name( data->regions, regionID ) );

    otf2_print_attributes( data, attributes );

    EnterStateDef(time,locationID,regionID);

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
Leave_print( uint64_t            locationID,
             uint64_t            time,
             void*               userData,
             OTF2_AttributeList* attributes,
             uint32_t            regionID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Region: %s\n",
            otf2_EVENT_COLUMN_WIDTH, "LEAVE",
            locationID, time,
            otf2_print_get_def_name( data->regions, regionID ) );

    otf2_print_attributes( data, attributes );

    LeaveStateDef(time,locationID);

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiSend_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes,
               uint32_t            receiver,
               uint32_t            communicator,
               uint32_t            msgTag,
               uint64_t            msgLength )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Receiver: %u, "
            "Communicator: %s, "
            "Tag: %u, Length: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_SEND",
            locationID, time,
            receiver,
            otf2_print_get_def_name( data->mpi_comms, communicator ),
            msgTag,
            msgLength );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiIsend_print( uint64_t            locationID,
                uint64_t            time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            receiver,
                uint32_t            communicator,
                uint32_t            msgTag,
                uint64_t            msgLength,
                uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;


    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Receiver: %u, Communicator: %s, "
            "Tag: %u, Length: %" PRIu64 ", Request: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_ISEND",
            locationID, time,
            receiver,
            otf2_print_get_def_name( data->mpi_comms, communicator ),
            msgTag,
            msgLength,
            requestID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiIsendComplete_print( uint64_t            locationID,
                        uint64_t            time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Request: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_ISEND_COMPLETE",
            locationID, time,
            requestID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiIrecvRequest_print( uint64_t            locationID,
                       uint64_t            time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Request: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_IRECV_REQUEST",
            locationID, time,
            requestID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiRecv_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes,
               uint32_t            sender,
               uint32_t            communicator,
               uint32_t            msgTag,
               uint64_t            msgLength )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Sender: %u, communicator: %s, "
            "Tag: %u, Length: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_RECV",
            locationID, time,
            sender,
            otf2_print_get_def_name( data->mpi_comms, communicator ),
            msgTag,
            msgLength );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiIrecv_print( uint64_t            locationID,
                uint64_t            time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            sender,
                uint32_t            communicator,
                uint32_t            msgTag,
                uint64_t            msgLength,
                uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Sender: %u, Communicator: %s, "
            "Tag: %u, Length: %" PRIu64 ", Request: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_IRECV",
            locationID, time,
            sender,
            otf2_print_get_def_name( data->mpi_comms, communicator ),
            msgTag,
            msgLength,
            requestID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiRequestTest_print( uint64_t            locationID,
                      uint64_t            time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Request: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_REQUEST_TEST",
            locationID, time,
            requestID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiRequestCancelled_print( uint64_t            locationID,
                           uint64_t            time,
                           void*               userData,
                           OTF2_AttributeList* attributes,
                           uint64_t            requestID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Request: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_REQUEST_CANCELLED",
            locationID, time,
            requestID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiCollectiveBegin_print( uint64_t            locationID,
                          uint64_t            time,
                          void*               userData,
                          OTF2_AttributeList* attributes )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_COLLECTIVE_BEGIN",
            locationID, time );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
MpiCollectiveEnd_print( uint64_t               locationID,
                        uint64_t               time,
                        void*                  userData,
                        OTF2_AttributeList*    attributes,
                        OTF2_MpiCollectiveType type,
                        uint32_t               commId,
                        uint32_t               root,
                        uint64_t               sizeSent,
                        uint64_t               sizeReceived )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Type: %s, Communicator: %s, "
            "Root: %s, Sent: %" PRIu64 ", Received: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "MPI_COLLECTIVE_END",
            locationID, time,
            otf2_print_get_mpi_collective_type( type ),
            otf2_print_get_def_name( data->mpi_comms, commId ),
            otf2_print_get_id( root ),
            sizeSent,
            sizeReceived );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
OmpFork_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes,
               uint32_t            numberOfRequestedThreads )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  # Requested Threads: %u\n",
            otf2_EVENT_COLUMN_WIDTH, "OPENMP_FORK",
            locationID, time,
            numberOfRequestedThreads );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
OmpJoin_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "OPENMP_JOIN",
            locationID, time );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
OmpAcquireLock_print( uint64_t            locationID,
                      uint64_t            time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      uint32_t            lockID,
                      uint32_t            acquisitionOrder )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Lock: %u, "
            "Acquisition Order: %u\n",
            otf2_EVENT_COLUMN_WIDTH, "OPENMP_ACQUIRE_LOCK",
            locationID, time,
            lockID,
            acquisitionOrder );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
OmpReleaseLock_print( uint64_t            locationID,
                      uint64_t            time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      uint32_t            lockID,
                      uint32_t            acquisitionOrder )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Lock: %u, "
            "Acquisition Order: %u\n",
            otf2_EVENT_COLUMN_WIDTH, "OPENMP_RELEASE_LOCK",
            locationID, time,
            lockID,
            acquisitionOrder );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
OmpTaskCreate_print( uint64_t            locationID,
                     uint64_t            time,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     uint64_t            taskID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Task: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "OPENMP_TASK_CREATE",
            locationID, time,
            taskID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
OmpTaskSwitch_print( uint64_t            locationID,
                     uint64_t            time,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     uint64_t            taskID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Task: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "OPENMP_TASK_SWITCH",
            locationID, time,
            taskID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
OmpTaskComplete_print( uint64_t            locationID,
                       uint64_t            time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint64_t            taskID )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Task: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "OPENMP_TASK_COMPLETE",
            locationID, time,
            taskID );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
Metric_print( uint64_t                locationID,
              uint64_t                time,
              void*                   userData,
              OTF2_AttributeList*     attributes,
              uint32_t                metricID,
              uint8_t                 numberOfMetrics,
              const OTF2_Type*        typeIDs,
              const OTF2_MetricValue* values )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Metric: %s, "
            "%u Values: ",
            otf2_EVENT_COLUMN_WIDTH, "METRIC",
            locationID, time,
            otf2_print_get_id( metricID ),
            numberOfMetrics );

    const char* sep = "";
    for ( uint8_t i = 0; i < numberOfMetrics; i++ )
    {
        switch ( typeIDs[ i ] )
        {
            case OTF2_TYPE_INT64:
                dprintf( "%s(INT64; %" PRId64 ")", sep, values[ i ].signed_int );
                break;
            case OTF2_TYPE_UINT64:
                dprintf( "%s(UINT64; %" PRIu64 ")", sep, values[ i ].unsigned_int );
                break;
            case OTF2_TYPE_DOUBLE:
                dprintf( "%s(DOUBLE; %f)", sep, values[ i ].floating_point );
                break;
            default:
            {
                dprintf( "%s(%s; %08" PRIx64 ")",
                        sep,
                        otf2_print_get_invalid( typeIDs[ i ] ),
                        values[ i ].unsigned_int );
            }
        }
        sep = ", ";
    }
    dprintf( "\n" );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
ParameterString_print( uint64_t            locationID,
                       uint64_t            time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint32_t            parameter,
                       uint32_t            value )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    /*printf( "%-*s %15" PRIu64 " %20" PRIu64 "  Parameter: %s, "
            "Value: %s\n",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_STRING",
            locationID, time,
            otf2_print_get_def_name( data->parameters, parameter ),
            otf2_print_get_def_name( data->strings, value ) );*/

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
ParameterInt_print( uint64_t            locationID,
                    uint64_t            time,
                    void*               userData,
                    OTF2_AttributeList* attributes,
                    uint32_t            parameter,
                    int64_t             value )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Parameter: %s, "
            "Value: %" PRIi64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_INT64",
            locationID, time,
            otf2_print_get_def_name( data->parameters, parameter ),
            value );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
ParameterUnsignedInt_print( uint64_t            locationID,
                            uint64_t            time,
                            void*               userData,
                            OTF2_AttributeList* attributes,
                            uint32_t            parameter,
                            uint64_t            value )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "  Parameter: %s, "
            "Value: %" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_UINT64",
            locationID, time,
            otf2_print_get_def_name( data->parameters, parameter ),
            value );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
Unknown_print( uint64_t            locationID,
               uint64_t            time,
               void*               userData,
               OTF2_AttributeList* attributes )
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %15" PRIu64 " %20" PRIu64 "\n",
            otf2_EVENT_COLUMN_WIDTH, "UNKNOWN",
            locationID, time );

    otf2_print_attributes( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
/** @} */



/* ___ Global definitions ____________________________________________________*/



OTF2_CallbackCode
GlobDefString_print( void*       userData,
                     uint32_t    stringID,
                     const char* string )
{
    /* Print definition if selected. */
    if ( otf2_GLOBDEFS )
    {
        /*printf( "%-*s %12u  \"%s\"\n",
                otf2_DEF_COLUMN_WIDTH, "STRING",
                stringID, string );*/
    }

    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_string( data->strings, stringID, string );

    return OTF2_CALLBACK_SUCCESS;
}


/** @internal
 *  @brief Callbacks for location definition.
 *
 *  @param userData             Optional user data.
 *  @param locationIdentifier   Location ID.
 *  @param stringID             String ID for the Location description.
 *  @param locationType         Type of the location.

 *
 *  @return                     Returns OTF2_SUCCESS if successful, an error
 *                              code if an error occures.
 */
OTF2_CallbackCode
GlobDefLocation_print( void*             userData,
                       uint64_t          locationID,
                       uint32_t          name,
                       OTF2_LocationType locationType,
                       uint64_t          numberOfEvents,
                       uint32_t          locationGroup )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_location( data, locationID, name );

    /* Print definition if selected. */
    if ( otf2_GLOBDEFS )
    {
        dprintf( "%-*s %12" PRIu64 "  Name: %s, Type: %s, "
                "# Events: %" PRIu64 ", Group: %s\n",
                otf2_DEF_COLUMN_WIDTH, "LOCATION",
                locationID,
                otf2_print_get_def_name( data->strings, name ),
                otf2_print_get_location_type( locationType ),
                numberOfEvents,
                otf2_print_get_def_name( data->location_groups, locationGroup ) );
    }

    if ( otf2_DOT )
    {
        /* Generate dot node entry. */
        fprintf( data->dot_file,
                 "  l%" PRIu64 " [label=\"%s %s (L ID: %" PRIu64 ")\", shape=diamond];\n",
                 locationID,
                 otf2_print_get_location_type( locationType ),
                 otf2_print_get_string( data->strings, name ),
                 locationID );

        /* Generate dot edge entry. */
        if ( locationGroup != OTF2_UNDEFINED_LOCATION_GROUP )
        {
            fprintf( data->dot_file, "  g%u -> l%" PRIu64 ";\n",
                     locationGroup,
                     locationID );
        }
    }

    /* Only proceed if either no local location is selected (i.e. read all) or
     * location ID matches provided location ID. */
    if ( otf2_LOCAL != OTF2_UNDEFINED_LOCATION && otf2_LOCAL != locationID )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    /* add location to the list of locations to read events from */
    otf2_print_add_location_to_read( data, locationID );

    otf2_LOCAL_FOUND = true;

    ThreadDef(locationID,0,locationID,otf2_print_get_def_name( data->strings, name ));

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
GlobDefLocationGroup_print( void*                  userData,
                            uint32_t               groupID,
                            uint32_t               name,
                            OTF2_LocationGroupType type,
                            uint32_t               systemTreeParent )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_location_group( data,
                                   groupID,
                                   name );

    /* Print definition if selected. */
    if ( otf2_GLOBDEFS )
    {
        dprintf( "%-*s %12u  Name: %s, Type: %s, Parent: %s\n",
                otf2_DEF_COLUMN_WIDTH, "LOCATION_GROUP",
                groupID,
                otf2_print_get_def_name( data->strings, name ),
                otf2_print_get_location_group_type( type ),
                otf2_print_get_def_name( data->system_tree_nodes,
                                         systemTreeParent ) );
    }

    if ( otf2_DOT )
    {
        /* Generate dot node entry. */
        fprintf( data->dot_file,
                 "  g%u [label=\"%s (LG ID: %u)\", shape=box];\n",
                 groupID,
                 otf2_print_get_string( data->strings, name ),
                 groupID );


        /* Generate dot edge entry. */
        if ( systemTreeParent != OTF2_UNDEFINED_SYSTEM_TREE_NODE )
        {
            fprintf( data->dot_file, "  n%u -> g%u;\n",
                     systemTreeParent,
                     groupID );
        }
    }

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
GlobDefRegion_print( void*           userData,
                     uint32_t        regionID,
                     uint32_t        name,
                     uint32_t        canonicalName,
                     uint32_t        description,
                     OTF2_RegionRole regionRole,
                     OTF2_Paradigm   paradigm,
                     OTF2_RegionFlag regionFlags,
                     uint32_t        sourceFile,
                     uint32_t        beginLineNumber,
                     uint32_t        endLineNumber )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_region( data,
                           regionID,
                           name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    /*printf( "%-*s %12u  Name: %s (Aka. %s), Descr: %s, "
            "Role: %s, Paradigm: %s, Flags: %s, "
            "File: %s, Begin: %u, End: %u\n",
            otf2_DEF_COLUMN_WIDTH, "REGION",
            regionID,
            otf2_print_get_def_name( data->strings, name ),
            otf2_print_get_def_name( data->strings, canonicalName ),
            otf2_print_get_def_name( data->strings, description ),
            otf2_print_get_region_role( regionRole ),
            otf2_print_get_paradigm( paradigm ),
            otf2_print_get_region_flags( regionFlags ),
            otf2_print_get_def_name( data->strings, sourceFile ),
            beginLineNumber, endLineNumber );*/

    const char * myname;
    int i;
    char *getname = otf2_print_get_def_name(data->strings, name);
    for (i = 1; i < strlen(getname); i++) {
       if (getname[i] == '"') {
          getname[i] = '\0';
          break;
       }
    }
    myname = &getname[1];

    StateDef(regionID,myname, 0);

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
GlobDefGroup_print( void*           userData,
                    uint32_t        groupID,
                    uint32_t        name,
                    OTF2_GroupType  type,
                    uint32_t        numberOfMembers,
                    const uint64_t* members )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_group( data,
                          groupID,
                          name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    dprintf( "%-*s %12u  Name: %s, Type: %s, %u Members:",
            otf2_DEF_COLUMN_WIDTH, "GROUP",
            groupID,
            otf2_print_get_def_name( data->strings, name ),
            otf2_print_get_group_type( type ),
            numberOfMembers );

    for ( uint32_t i = 0; i < numberOfMembers; i++ )
    {
        dprintf( " %" PRIu64, members[ i ] );
    }
    dprintf( "\n" );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
GlobDefCallsite_print( void*    userdata,
                       uint32_t callsite_identifier,
                       uint32_t source_file,
                       uint32_t line_number,
                       uint32_t region_entered,
                       uint32_t region_left )
{
    otf2_print_data* data = ( otf2_print_data* )userdata;

    dprintf( "%-*s %12u  File: %s, Line: %u, "
            "Region entered: %s, Region left: %s\n",
            otf2_DEF_COLUMN_WIDTH, "CALLSITE",
            callsite_identifier,
            otf2_print_get_def_name( data->strings, source_file ),
            line_number,
            otf2_print_get_def_name( data->regions, region_entered ),
            otf2_print_get_def_name( data->regions, region_left ) );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
GlobDefCallpath_print( void*    userdata,
                       uint32_t callpath_identifier,
                       uint32_t parent_callpath,
                       uint32_t region_identifier )
{
    otf2_print_data* data = ( otf2_print_data* )userdata;

    dprintf( "%-*s %12u  Region: %s, Parent: %s\n",
            otf2_DEF_COLUMN_WIDTH, "CALLPATH",
            callpath_identifier,
            otf2_print_get_def_name( data->regions, region_identifier ),
            otf2_print_get_id( parent_callpath ) );

    return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode
GlobDefMpiComm_print( void*    userdata,
                      uint32_t comm_id,
                      uint32_t comm_name,
                      uint32_t group_id,
                      uint32_t comm_parent )
{
    otf2_print_data* data = ( otf2_print_data* )userdata;

    otf2_print_add_mpi_comm( data,
                             comm_id,
                             comm_name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    dprintf( "%-*s %12u  Name: %s, Group: %s, Parent Communicator: %s\n",
            otf2_DEF_COLUMN_WIDTH, "MPI_COMM",
            comm_id,
            otf2_print_get_def_name( data->strings, comm_name ),
            otf2_print_get_def_name( data->groups, group_id ),
            otf2_print_get_def_name( data->mpi_comms, comm_parent ) );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
GlobDefMetricMember_print( void*           userData,
                           uint32_t        metric_member_id,
                           uint32_t        name,
                           uint32_t        description,
                           OTF2_MetricType type,
                           OTF2_MetricMode mode,
                           OTF2_Type       value_type,
                           OTF2_MetricBase base,
                           int64_t         exponent,
                           uint32_t        unit )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_metric( data,
                           metric_member_id,
                           name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    uint8_t base_number = 0;
    switch ( base )
    {
        case OTF2_BASE_BINARY:
            base_number = 2;
            break;
        case OTF2_BASE_DECIMAL:
            base_number = 10;
            break;
    }

    dprintf( "%-*s %12u  Name: %s, Descr.: %s, Type: %s, "
            "Mode: %s, Value Type: %s, Base: %u, Exponent: %" PRId64 ", "
            "Unit: %s\n",
            otf2_DEF_COLUMN_WIDTH, "METRIC_MEMBER",
            metric_member_id,
            otf2_print_get_def_name( data->strings, name ),
            otf2_print_get_def_name( data->strings, description ),
            otf2_print_get_metric_type( type ),
            otf2_print_get_metric_mode( mode ),
            otf2_print_get_type( value_type ),
            base_number, exponent,
            otf2_print_get_def_name( data->strings, unit ) );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
GlobDefMetricClass_print( void*                 userData,
                          uint32_t              metric_class_id,
                          uint8_t               number_of_metrics,
                          const uint32_t*       metric_members,
                          OTF2_MetricOccurrence occurrence )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %12u  Occurrence: %s, %u Members: ",
            otf2_DEF_COLUMN_WIDTH, "METRIC_CLASS",
            metric_class_id,
            otf2_print_get_metric_occurrence( occurrence ),
            number_of_metrics );

    const char* sep = "";
    for ( uint8_t i = 0; i < number_of_metrics; i++ )
    {
        dprintf( "%s%s",
                sep,
                otf2_print_get_def_name( data->metrics, metric_members[ i ] ) );
        sep = ", ";
    }
    dprintf( "\n" );

    return OTF2_CALLBACK_SUCCESS;
}

static const char*
otf2_print_get_scope_name( otf2_print_data* data,
                           OTF2_MetricScope scopeType,
                           uint64_t         scope )
{
    switch ( scopeType )
    {
        #define scope_case( SCOPE_TYPE, scope_type ) \
    case OTF2_SCOPE_ ## SCOPE_TYPE: \
        return otf2_print_get_def64_name( data->scope_type, scope )

        scope_case( LOCATION, locations );
        scope_case( LOCATION_GROUP, location_groups );
        scope_case( SYSTEM_TREE_NODE, system_tree_nodes );
        scope_case( GROUP, groups );

        #undef scope_case

        default:
            return otf2_print_get_id64( scope );
    }
}

OTF2_CallbackCode
GlobDefMetricInstance_print( void*            userData,
                             uint32_t         metricInstanceID,
                             uint32_t         metricClass,
                             uint64_t         recorder,
                             OTF2_MetricScope scopeType,
                             uint64_t         scope )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    dprintf( "%-*s %12u  Class: %u, Recorder: %s, "
            "Scope: %s %s\n",
            otf2_DEF_COLUMN_WIDTH, "METRIC_INSTANCE",
            metricInstanceID,
            metricClass,
            otf2_print_get_def64_name( data->locations, recorder ),
            otf2_print_get_metric_scope( scopeType ),
            otf2_print_get_scope_name( data, scopeType, scope ) );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
GlobDefSystemTreeNode_print( void*    userData,
                             uint32_t nodeID,
                             uint32_t name,
                             uint32_t className,
                             uint32_t nodeParent )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_system_tree_node( data,
                                     nodeID,
                                     name );

    /* Print definition if selected. */
    if ( otf2_GLOBDEFS )
    {
        dprintf( "%-*s %12u  Name: %s, Class: %s, Parent: %s\n",
                otf2_DEF_COLUMN_WIDTH, "SYSTEM_TREE_NODE",
                nodeID,
                otf2_print_get_def_name( data->strings, name ),
                otf2_print_get_def_name( data->strings, className ),
                otf2_print_get_def_name( data->system_tree_nodes, nodeParent ) );
    }

    if ( otf2_DOT )
    {
        /* Generate dot node entry. */
        fprintf( data->dot_file, "  n%u [label=\"%s (Node ID: %u)\"];\n",
                 nodeID,
                 otf2_print_get_string( data->strings, name ),
                 nodeID );

        /* Generate dot edge entry. */
        if ( nodeParent != OTF2_UNDEFINED_SYSTEM_TREE_NODE )
        {
            fprintf( data->dot_file, "  n%u -> n%u;\n", nodeParent, nodeID );
        }
    }

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
GlobDefAttribute_print( void*     userData,
                        uint32_t  attributeID,
                        uint32_t  name,
                        OTF2_Type type )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_attribute( data,
                              attributeID,
                              name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    dprintf( "%-*s %12u  Name: %s, Type: %s\n",
            otf2_DEF_COLUMN_WIDTH, "ATTRIBUTE",
            attributeID,
            otf2_print_get_def_name( data->strings, name ),
            otf2_print_get_type( type ) );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
GlobDefClockProperties_print( void*    userData,
                              uint64_t timer_resolution,
                              uint64_t global_offset,
                              uint64_t trace_length )
{
    /* Dummies to suppress compiler warnings for unused parameters. */
    ( void )userData;

    dprintf( "%-*s %12s  Ticks per Seconds: %" PRIu64 ", "
            "Global Offset: %" PRIu64 ", Length: %" PRIu64 "\n",
            otf2_DEF_COLUMN_WIDTH, "CLOCK_PROPERTIES", "",
            timer_resolution,
            global_offset, trace_length );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
GlobDefParameter_print( void*              userData,
                        uint32_t           parameterID,
                        uint32_t           name,
                        OTF2_ParameterType type )
{
    otf2_print_data* data = ( otf2_print_data* )userData;

    otf2_print_add_parameter( data,
                              parameterID,
                              name );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    dprintf( "%-*s %12u  Name: %s, Type: %s\n",
            otf2_DEF_COLUMN_WIDTH, "PARAMETER",
            parameterID,
            otf2_print_get_def_name( data->strings, name ),
            otf2_print_get_parameter_type( type ) );

    return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode
GlobDefUnknown_print( void* userData )
{
    /* Dummies to suppress compiler warnings for unused parameters. */
    ( void )userData;

    dprintf( "%-*s\n",
            otf2_DEF_COLUMN_WIDTH, "UNKNOWN" );

    return OTF2_CALLBACK_SUCCESS;
}

static void
map_traverse_dense( uint64_t localId,
                    uint64_t globalId,
                    void*    userData )
{
    const char* sep = *( char** )userData;
    dprintf( "%s%" PRIu64, sep, globalId );
    *( char** )userData = ",";
}

static void
map_traverse_sparse( uint64_t localId,
                     uint64_t globalId,
                     void*    userData )
{
    const char* sep = *( char** )userData;
    dprintf( "%s%" PRIu64 "=>%" PRIu64, sep, localId, globalId );
    *( char** )userData = ",";
}

OTF2_CallbackCode
DefMappingTable_print( void*             userData,
                       OTF2_MappingType  mapType,
                       const OTF2_IdMap* iDMap )
{
    uint64_t* location_id_ptr = userData;

    dprintf( "%-*s %12" PRIu64 "  Type: %s, ",
            otf2_DEF_COLUMN_WIDTH, "MAPPING_TABLE",
            *location_id_ptr,
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
DefClockOffset_print( void*    userData,
                      uint64_t time,
                      int64_t  offset,
                      double   stddev )
{
    uint64_t* location_id_ptr = userData;

    dprintf( "%-*s %12" PRIu64 "  Time: %" PRIu64 ", Offset: %+" PRIi64 ", "
            "StdDev: %f\n",
            otf2_DEF_COLUMN_WIDTH, "CLOCK_OFFSET",
            *location_id_ptr, time, offset, stddev );

    return OTF2_CALLBACK_SUCCESS;
}
