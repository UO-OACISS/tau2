/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2013,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2013,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2014, 2016,
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

/** @internal
 *  @brief width of the column with the definition names. */
static int otf2_DEF_COLUMN_WIDTH = 27;

/** @internal
 *  @brief width of the column with the event names. */
static int otf2_EVENT_COLUMN_WIDTH = 32;

/** @internal
 *  @brief max value of an OTF2_Paradigm entry + 1. */
static int otf2_max_known_paradigm = 24 + 1;

static inline const char*
otf2_print_get_raw_boolean( OTF2_Boolean boolean )
{
    switch ( boolean )
    {
        case OTF2_FALSE:
            return "FALSE";
        case OTF2_TRUE:
            return "TRUE";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_boolean( OTF2_Boolean boolean )
{
    const char* result = otf2_print_get_raw_boolean( boolean );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( boolean );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_file_type( OTF2_FileType fileType )
{
    switch ( fileType )
    {
        case OTF2_FILETYPE_ANCHOR:
            return "ANCHOR";
        case OTF2_FILETYPE_GLOBAL_DEFS:
            return "GLOBAL_DEFS";
        case OTF2_FILETYPE_LOCAL_DEFS:
            return "LOCAL_DEFS";
        case OTF2_FILETYPE_EVENTS:
            return "EVENTS";
        case OTF2_FILETYPE_SNAPSHOTS:
            return "SNAPSHOTS";
        case OTF2_FILETYPE_THUMBNAIL:
            return "THUMBNAIL";
        case OTF2_FILETYPE_MARKER:
            return "MARKER";
        case OTF2_FILETYPE_SIONRANKMAP:
            return "SIONRANKMAP";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_file_type( OTF2_FileType fileType )
{
    const char* result = otf2_print_get_raw_file_type( fileType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( fileType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_file_substrate( OTF2_FileSubstrate fileSubstrate )
{
    switch ( fileSubstrate )
    {
        case OTF2_SUBSTRATE_UNDEFINED:
            return "UNDEFINED";
        case OTF2_SUBSTRATE_POSIX:
            return "POSIX";
        case OTF2_SUBSTRATE_SION:
            return "SION";
        case OTF2_SUBSTRATE_NONE:
            return "NONE";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_file_substrate( OTF2_FileSubstrate fileSubstrate )
{
    const char* result = otf2_print_get_raw_file_substrate( fileSubstrate );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( fileSubstrate );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_mapping_type( OTF2_MappingType mappingType )
{
    switch ( mappingType )
    {
        case OTF2_MAPPING_STRING:
            return "STRING";
        case OTF2_MAPPING_ATTRIBUTE:
            return "ATTRIBUTE";
        case OTF2_MAPPING_LOCATION:
            return "LOCATION";
        case OTF2_MAPPING_REGION:
            return "REGION";
        case OTF2_MAPPING_GROUP:
            return "GROUP";
        case OTF2_MAPPING_METRIC:
            return "METRIC";
        case OTF2_MAPPING_COMM:
            return "COMM";
        case OTF2_MAPPING_PARAMETER:
            return "PARAMETER";
        case OTF2_MAPPING_RMA_WIN:
            return "RMA_WIN";
        case OTF2_MAPPING_SOURCE_CODE_LOCATION:
            return "SOURCE_CODE_LOCATION";
        case OTF2_MAPPING_CALLING_CONTEXT:
            return "CALLING_CONTEXT";
        case OTF2_MAPPING_INTERRUPT_GENERATOR:
            return "INTERRUPT_GENERATOR";
        case OTF2_MAPPING_IO_FILE:
            return "IO_FILE";
        case OTF2_MAPPING_IO_HANDLE:
            return "IO_HANDLE";
        case OTF2_MAPPING_LOCATION_GROUP:
            return "LOCATION_GROUP";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_mapping_type( OTF2_MappingType mappingType )
{
    const char* result = otf2_print_get_raw_mapping_type( mappingType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( mappingType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_type( OTF2_Type type )
{
    switch ( type )
    {
        case OTF2_TYPE_NONE:
            return "NONE";
        case OTF2_TYPE_UINT8:
            return "UINT8";
        case OTF2_TYPE_UINT16:
            return "UINT16";
        case OTF2_TYPE_UINT32:
            return "UINT32";
        case OTF2_TYPE_UINT64:
            return "UINT64";
        case OTF2_TYPE_INT8:
            return "INT8";
        case OTF2_TYPE_INT16:
            return "INT16";
        case OTF2_TYPE_INT32:
            return "INT32";
        case OTF2_TYPE_INT64:
            return "INT64";
        case OTF2_TYPE_FLOAT:
            return "FLOAT";
        case OTF2_TYPE_DOUBLE:
            return "DOUBLE";
        case OTF2_TYPE_STRING:
            return "STRING";
        case OTF2_TYPE_ATTRIBUTE:
            return "ATTRIBUTE";
        case OTF2_TYPE_LOCATION:
            return "LOCATION";
        case OTF2_TYPE_REGION:
            return "REGION";
        case OTF2_TYPE_GROUP:
            return "GROUP";
        case OTF2_TYPE_METRIC:
            return "METRIC";
        case OTF2_TYPE_COMM:
            return "COMM";
        case OTF2_TYPE_PARAMETER:
            return "PARAMETER";
        case OTF2_TYPE_RMA_WIN:
            return "RMA_WIN";
        case OTF2_TYPE_SOURCE_CODE_LOCATION:
            return "SOURCE_CODE_LOCATION";
        case OTF2_TYPE_CALLING_CONTEXT:
            return "CALLING_CONTEXT";
        case OTF2_TYPE_INTERRUPT_GENERATOR:
            return "INTERRUPT_GENERATOR";
        case OTF2_TYPE_IO_FILE:
            return "IO_FILE";
        case OTF2_TYPE_IO_HANDLE:
            return "IO_HANDLE";
        case OTF2_TYPE_LOCATION_GROUP:
            return "LOCATION_GROUP";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_type( OTF2_Type type )
{
    const char* result = otf2_print_get_raw_type( type );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( type );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_paradigm( OTF2_Paradigm paradigm )
{
    switch ( paradigm )
    {
        case OTF2_PARADIGM_UNKNOWN:
            return "UNKNOWN";
        case OTF2_PARADIGM_USER:
            return "USER";
        case OTF2_PARADIGM_COMPILER:
            return "COMPILER";
        case OTF2_PARADIGM_OPENMP:
            return "OPENMP";
        case OTF2_PARADIGM_MPI:
            return "MPI";
        case OTF2_PARADIGM_CUDA:
            return "CUDA";
        case OTF2_PARADIGM_MEASUREMENT_SYSTEM:
            return "MEASUREMENT_SYSTEM";
        case OTF2_PARADIGM_PTHREAD:
            return "PTHREAD";
        case OTF2_PARADIGM_HMPP:
            return "HMPP";
        case OTF2_PARADIGM_OMPSS:
            return "OMPSS";
        case OTF2_PARADIGM_HARDWARE:
            return "HARDWARE";
        case OTF2_PARADIGM_GASPI:
            return "GASPI";
        case OTF2_PARADIGM_UPC:
            return "UPC";
        case OTF2_PARADIGM_SHMEM:
            return "SHMEM";
        case OTF2_PARADIGM_WINTHREAD:
            return "WINTHREAD";
        case OTF2_PARADIGM_QTTHREAD:
            return "QTTHREAD";
        case OTF2_PARADIGM_ACETHREAD:
            return "ACETHREAD";
        case OTF2_PARADIGM_TBBTHREAD:
            return "TBBTHREAD";
        case OTF2_PARADIGM_OPENACC:
            return "OPENACC";
        case OTF2_PARADIGM_OPENCL:
            return "OPENCL";
        case OTF2_PARADIGM_MTAPI:
            return "MTAPI";
        case OTF2_PARADIGM_SAMPLING:
            return "SAMPLING";
        case OTF2_PARADIGM_NONE:
            return "NONE";
        case OTF2_PARADIGM_HIP:
            return "HIP";
        case OTF2_PARADIGM_KOKKOS:
            return "KOKKOS";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_paradigm( OTF2_Paradigm paradigm )
{
    const char* result = otf2_print_get_raw_paradigm( paradigm );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( paradigm );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_paradigm_class( OTF2_ParadigmClass paradigmClass )
{
    switch ( paradigmClass )
    {
        case OTF2_PARADIGM_CLASS_PROCESS:
            return "PROCESS";
        case OTF2_PARADIGM_CLASS_THREAD_FORK_JOIN:
            return "THREAD_FORK_JOIN";
        case OTF2_PARADIGM_CLASS_THREAD_CREATE_WAIT:
            return "THREAD_CREATE_WAIT";
        case OTF2_PARADIGM_CLASS_ACCELERATOR:
            return "ACCELERATOR";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_paradigm_class( OTF2_ParadigmClass paradigmClass )
{
    const char* result = otf2_print_get_raw_paradigm_class( paradigmClass );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( paradigmClass );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_paradigm_property( OTF2_ParadigmProperty paradigmProperty )
{
    switch ( paradigmProperty )
    {
        case OTF2_PARADIGM_PROPERTY_COMM_NAME_TEMPLATE:
            return "COMM_NAME_TEMPLATE";
        case OTF2_PARADIGM_PROPERTY_RMA_WIN_NAME_TEMPLATE:
            return "RMA_WIN_NAME_TEMPLATE";
        case OTF2_PARADIGM_PROPERTY_RMA_ONLY:
            return "RMA_ONLY";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_paradigm_property( OTF2_ParadigmProperty paradigmProperty )
{
    const char* result = otf2_print_get_raw_paradigm_property( paradigmProperty );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( paradigmProperty );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_thumbnail_type( OTF2_ThumbnailType thumbnailType )
{
    switch ( thumbnailType )
    {
        case OTF2_THUMBNAIL_TYPE_REGION:
            return "REGION";
        case OTF2_THUMBNAIL_TYPE_METRIC:
            return "METRIC";
        case OTF2_THUMBNAIL_TYPE_ATTRIBUTES:
            return "ATTRIBUTES";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_thumbnail_type( OTF2_ThumbnailType thumbnailType )
{
    const char* result = otf2_print_get_raw_thumbnail_type( thumbnailType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( thumbnailType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_system_tree_domain( OTF2_SystemTreeDomain systemTreeDomain )
{
    switch ( systemTreeDomain )
    {
        case OTF2_SYSTEM_TREE_DOMAIN_MACHINE:
            return "MACHINE";
        case OTF2_SYSTEM_TREE_DOMAIN_SHARED_MEMORY:
            return "SHARED_MEMORY";
        case OTF2_SYSTEM_TREE_DOMAIN_NUMA:
            return "NUMA";
        case OTF2_SYSTEM_TREE_DOMAIN_SOCKET:
            return "SOCKET";
        case OTF2_SYSTEM_TREE_DOMAIN_CACHE:
            return "CACHE";
        case OTF2_SYSTEM_TREE_DOMAIN_CORE:
            return "CORE";
        case OTF2_SYSTEM_TREE_DOMAIN_PU:
            return "PU";
        case OTF2_SYSTEM_TREE_DOMAIN_ACCELERATOR_DEVICE:
            return "ACCELERATOR_DEVICE";
        case OTF2_SYSTEM_TREE_DOMAIN_NETWORKING_DEVICE:
            return "NETWORKING_DEVICE";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_system_tree_domain( OTF2_SystemTreeDomain systemTreeDomain )
{
    const char* result = otf2_print_get_raw_system_tree_domain( systemTreeDomain );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( systemTreeDomain );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_location_group_type( OTF2_LocationGroupType locationGroupType )
{
    switch ( locationGroupType )
    {
        case OTF2_LOCATION_GROUP_TYPE_UNKNOWN:
            return "UNKNOWN";
        case OTF2_LOCATION_GROUP_TYPE_PROCESS:
            return "PROCESS";
        case OTF2_LOCATION_GROUP_TYPE_ACCELERATOR:
            return "ACCELERATOR";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_location_group_type( OTF2_LocationGroupType locationGroupType )
{
    const char* result = otf2_print_get_raw_location_group_type( locationGroupType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( locationGroupType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_location_type( OTF2_LocationType locationType )
{
    switch ( locationType )
    {
        case OTF2_LOCATION_TYPE_UNKNOWN:
            return "UNKNOWN";
        case OTF2_LOCATION_TYPE_CPU_THREAD:
            return "CPU_THREAD";
        case OTF2_LOCATION_TYPE_ACCELERATOR_STREAM:
            return "ACCELERATOR_STREAM";
        case OTF2_LOCATION_TYPE_METRIC:
            return "METRIC";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_location_type( OTF2_LocationType locationType )
{
    const char* result = otf2_print_get_raw_location_type( locationType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( locationType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_region_role( OTF2_RegionRole regionRole )
{
    switch ( regionRole )
    {
        case OTF2_REGION_ROLE_UNKNOWN:
            return "UNKNOWN";
        case OTF2_REGION_ROLE_FUNCTION:
            return "FUNCTION";
        case OTF2_REGION_ROLE_WRAPPER:
            return "WRAPPER";
        case OTF2_REGION_ROLE_LOOP:
            return "LOOP";
        case OTF2_REGION_ROLE_CODE:
            return "CODE";
        case OTF2_REGION_ROLE_PARALLEL:
            return "PARALLEL";
        case OTF2_REGION_ROLE_SECTIONS:
            return "SECTIONS";
        case OTF2_REGION_ROLE_SECTION:
            return "SECTION";
        case OTF2_REGION_ROLE_WORKSHARE:
            return "WORKSHARE";
        case OTF2_REGION_ROLE_SINGLE:
            return "SINGLE";
        case OTF2_REGION_ROLE_SINGLE_SBLOCK:
            return "SINGLE_SBLOCK";
        case OTF2_REGION_ROLE_MASTER:
            return "MASTER";
        case OTF2_REGION_ROLE_CRITICAL:
            return "CRITICAL";
        case OTF2_REGION_ROLE_CRITICAL_SBLOCK:
            return "CRITICAL_SBLOCK";
        case OTF2_REGION_ROLE_ATOMIC:
            return "ATOMIC";
        case OTF2_REGION_ROLE_BARRIER:
            return "BARRIER";
        case OTF2_REGION_ROLE_IMPLICIT_BARRIER:
            return "IMPLICIT_BARRIER";
        case OTF2_REGION_ROLE_FLUSH:
            return "FLUSH";
        case OTF2_REGION_ROLE_ORDERED:
            return "ORDERED";
        case OTF2_REGION_ROLE_ORDERED_SBLOCK:
            return "ORDERED_SBLOCK";
        case OTF2_REGION_ROLE_TASK:
            return "TASK";
        case OTF2_REGION_ROLE_TASK_CREATE:
            return "TASK_CREATE";
        case OTF2_REGION_ROLE_TASK_WAIT:
            return "TASK_WAIT";
        case OTF2_REGION_ROLE_COLL_ONE2ALL:
            return "COLL_ONE2ALL";
        case OTF2_REGION_ROLE_COLL_ALL2ONE:
            return "COLL_ALL2ONE";
        case OTF2_REGION_ROLE_COLL_ALL2ALL:
            return "COLL_ALL2ALL";
        case OTF2_REGION_ROLE_COLL_OTHER:
            return "COLL_OTHER";
        case OTF2_REGION_ROLE_FILE_IO:
            return "FILE_IO";
        case OTF2_REGION_ROLE_POINT2POINT:
            return "POINT2POINT";
        case OTF2_REGION_ROLE_RMA:
            return "RMA";
        case OTF2_REGION_ROLE_DATA_TRANSFER:
            return "DATA_TRANSFER";
        case OTF2_REGION_ROLE_ARTIFICIAL:
            return "ARTIFICIAL";
        case OTF2_REGION_ROLE_THREAD_CREATE:
            return "THREAD_CREATE";
        case OTF2_REGION_ROLE_THREAD_WAIT:
            return "THREAD_WAIT";
        case OTF2_REGION_ROLE_TASK_UNTIED:
            return "TASK_UNTIED";
        case OTF2_REGION_ROLE_ALLOCATE:
            return "ALLOCATE";
        case OTF2_REGION_ROLE_DEALLOCATE:
            return "DEALLOCATE";
        case OTF2_REGION_ROLE_REALLOCATE:
            return "REALLOCATE";
        case OTF2_REGION_ROLE_FILE_IO_METADATA:
            return "FILE_IO_METADATA";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_region_role( OTF2_RegionRole regionRole )
{
    const char* result = otf2_print_get_raw_region_role( regionRole );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( regionRole );
    }

    return result;
}


static inline const char*
otf2_print_get_region_flag( OTF2_RegionFlag regionFlag )
{
    size_t buffer_size =
        2 + ( 2 * 3 )
        + sizeof( "NONE" )
        + sizeof( "DYNAMIC" )
        + sizeof( "PHASE" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( regionFlag == OTF2_REGION_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( regionFlag & OTF2_REGION_FLAG_DYNAMIC )
    {
        strcat( buffer, sep );
        strcat( buffer, "DYNAMIC" );
        sep         = ", ";
        regionFlag &= ~OTF2_REGION_FLAG_DYNAMIC;
    }
    if ( regionFlag & OTF2_REGION_FLAG_PHASE )
    {
        strcat( buffer, sep );
        strcat( buffer, "PHASE" );
        sep         = ", ";
        regionFlag &= ~OTF2_REGION_FLAG_PHASE;
    }
    if ( regionFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, regionFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_raw_group_type( OTF2_GroupType groupType )
{
    switch ( groupType )
    {
        case OTF2_GROUP_TYPE_UNKNOWN:
            return "UNKNOWN";
        case OTF2_GROUP_TYPE_LOCATIONS:
            return "LOCATIONS";
        case OTF2_GROUP_TYPE_REGIONS:
            return "REGIONS";
        case OTF2_GROUP_TYPE_METRIC:
            return "METRIC";
        case OTF2_GROUP_TYPE_COMM_LOCATIONS:
            return "COMM_LOCATIONS";
        case OTF2_GROUP_TYPE_COMM_GROUP:
            return "COMM_GROUP";
        case OTF2_GROUP_TYPE_COMM_SELF:
            return "COMM_SELF";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_group_type( OTF2_GroupType groupType )
{
    const char* result = otf2_print_get_raw_group_type( groupType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( groupType );
    }

    return result;
}


static inline const char*
otf2_print_get_group_flag( OTF2_GroupFlag groupFlag )
{
    size_t buffer_size =
        2 + ( 2 * 2 )
        + sizeof( "NONE" )
        + sizeof( "GLOBAL_MEMBERS" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( groupFlag == OTF2_GROUP_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( groupFlag & OTF2_GROUP_FLAG_GLOBAL_MEMBERS )
    {
        strcat( buffer, sep );
        strcat( buffer, "GLOBAL_MEMBERS" );
        sep        = ", ";
        groupFlag &= ~OTF2_GROUP_FLAG_GLOBAL_MEMBERS;
    }
    if ( groupFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, groupFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_raw_base( OTF2_Base base )
{
    switch ( base )
    {
        case OTF2_BASE_BINARY:
            return "BINARY";
        case OTF2_BASE_DECIMAL:
            return "DECIMAL";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_base( OTF2_Base base )
{
    const char* result = otf2_print_get_raw_base( base );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( base );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_metric_occurrence( OTF2_MetricOccurrence metricOccurrence )
{
    switch ( metricOccurrence )
    {
        case OTF2_METRIC_SYNCHRONOUS_STRICT:
            return "SYNCHRONOUS_STRICT";
        case OTF2_METRIC_SYNCHRONOUS:
            return "SYNCHRONOUS";
        case OTF2_METRIC_ASYNCHRONOUS:
            return "ASYNCHRONOUS";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_metric_occurrence( OTF2_MetricOccurrence metricOccurrence )
{
    const char* result = otf2_print_get_raw_metric_occurrence( metricOccurrence );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( metricOccurrence );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_metric_type( OTF2_MetricType metricType )
{
    switch ( metricType )
    {
        case OTF2_METRIC_TYPE_OTHER:
            return "OTHER";
        case OTF2_METRIC_TYPE_PAPI:
            return "PAPI";
        case OTF2_METRIC_TYPE_RUSAGE:
            return "RUSAGE";
        case OTF2_METRIC_TYPE_USER:
            return "USER";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_metric_type( OTF2_MetricType metricType )
{
    const char* result = otf2_print_get_raw_metric_type( metricType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( metricType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_metric_value_property( OTF2_MetricValueProperty metricValueProperty )
{
    switch ( metricValueProperty )
    {
        case OTF2_METRIC_VALUE_ACCUMULATED:
            return "ACCUMULATED";
        case OTF2_METRIC_VALUE_ABSOLUTE:
            return "ABSOLUTE";
        case OTF2_METRIC_VALUE_RELATIVE:
            return "RELATIVE";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_metric_value_property( OTF2_MetricValueProperty metricValueProperty )
{
    const char* result = otf2_print_get_raw_metric_value_property( metricValueProperty );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( metricValueProperty );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_metric_timing( OTF2_MetricTiming metricTiming )
{
    switch ( metricTiming )
    {
        case OTF2_METRIC_TIMING_START:
            return "START";
        case OTF2_METRIC_TIMING_POINT:
            return "POINT";
        case OTF2_METRIC_TIMING_LAST:
            return "LAST";
        case OTF2_METRIC_TIMING_NEXT:
            return "NEXT";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_metric_timing( OTF2_MetricTiming metricTiming )
{
    const char* result = otf2_print_get_raw_metric_timing( metricTiming );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( metricTiming );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_metric_mode( OTF2_MetricMode metricMode )
{
    switch ( metricMode )
    {
        case OTF2_METRIC_ACCUMULATED_START:
            return "ACCUMULATED_START";
        case OTF2_METRIC_ACCUMULATED_POINT:
            return "ACCUMULATED_POINT";
        case OTF2_METRIC_ACCUMULATED_LAST:
            return "ACCUMULATED_LAST";
        case OTF2_METRIC_ACCUMULATED_NEXT:
            return "ACCUMULATED_NEXT";
        case OTF2_METRIC_ABSOLUTE_POINT:
            return "ABSOLUTE_POINT";
        case OTF2_METRIC_ABSOLUTE_LAST:
            return "ABSOLUTE_LAST";
        case OTF2_METRIC_ABSOLUTE_NEXT:
            return "ABSOLUTE_NEXT";
        case OTF2_METRIC_RELATIVE_POINT:
            return "RELATIVE_POINT";
        case OTF2_METRIC_RELATIVE_LAST:
            return "RELATIVE_LAST";
        case OTF2_METRIC_RELATIVE_NEXT:
            return "RELATIVE_NEXT";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_metric_mode( OTF2_MetricMode metricMode )
{
    const char* result = otf2_print_get_raw_metric_mode( metricMode );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( metricMode );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_metric_scope( OTF2_MetricScope metricScope )
{
    switch ( metricScope )
    {
        case OTF2_SCOPE_LOCATION:
            return "LOCATION";
        case OTF2_SCOPE_LOCATION_GROUP:
            return "LOCATION_GROUP";
        case OTF2_SCOPE_SYSTEM_TREE_NODE:
            return "SYSTEM_TREE_NODE";
        case OTF2_SCOPE_GROUP:
            return "GROUP";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_metric_scope( OTF2_MetricScope metricScope )
{
    const char* result = otf2_print_get_raw_metric_scope( metricScope );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( metricScope );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_recorder_kind( OTF2_RecorderKind recorderKind )
{
    switch ( recorderKind )
    {
        case OTF2_RECORDER_KIND_UNKNOWN:
            return "UNKNOWN";
        case OTF2_RECORDER_KIND_ABSTRACT:
            return "ABSTRACT";
        case OTF2_RECORDER_KIND_CPU:
            return "CPU";
        case OTF2_RECORDER_KIND_GPU:
            return "GPU";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_recorder_kind( OTF2_RecorderKind recorderKind )
{
    const char* result = otf2_print_get_raw_recorder_kind( recorderKind );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( recorderKind );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_parameter_type( OTF2_ParameterType parameterType )
{
    switch ( parameterType )
    {
        case OTF2_PARAMETER_TYPE_STRING:
            return "STRING";
        case OTF2_PARAMETER_TYPE_INT64:
            return "INT64";
        case OTF2_PARAMETER_TYPE_UINT64:
            return "UINT64";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_parameter_type( OTF2_ParameterType parameterType )
{
    const char* result = otf2_print_get_raw_parameter_type( parameterType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( parameterType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_cart_periodicity( OTF2_CartPeriodicity cartPeriodicity )
{
    switch ( cartPeriodicity )
    {
        case OTF2_CART_PERIODIC_FALSE:
            return "FALSE";
        case OTF2_CART_PERIODIC_TRUE:
            return "TRUE";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_cart_periodicity( OTF2_CartPeriodicity cartPeriodicity )
{
    const char* result = otf2_print_get_raw_cart_periodicity( cartPeriodicity );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( cartPeriodicity );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_interrupt_generator_mode( OTF2_InterruptGeneratorMode interruptGeneratorMode )
{
    switch ( interruptGeneratorMode )
    {
        case OTF2_INTERRUPT_GENERATOR_MODE_TIME:
            return "TIME";
        case OTF2_INTERRUPT_GENERATOR_MODE_COUNT:
            return "COUNT";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_interrupt_generator_mode( OTF2_InterruptGeneratorMode interruptGeneratorMode )
{
    const char* result = otf2_print_get_raw_interrupt_generator_mode( interruptGeneratorMode );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( interruptGeneratorMode );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_measurement_mode( OTF2_MeasurementMode measurementMode )
{
    switch ( measurementMode )
    {
        case OTF2_MEASUREMENT_ON:
            return "ON";
        case OTF2_MEASUREMENT_OFF:
            return "OFF";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_measurement_mode( OTF2_MeasurementMode measurementMode )
{
    const char* result = otf2_print_get_raw_measurement_mode( measurementMode );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( measurementMode );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_collective_op( OTF2_CollectiveOp collectiveOp )
{
    switch ( collectiveOp )
    {
        case OTF2_COLLECTIVE_OP_BARRIER:
            return "BARRIER";
        case OTF2_COLLECTIVE_OP_BCAST:
            return "BCAST";
        case OTF2_COLLECTIVE_OP_GATHER:
            return "GATHER";
        case OTF2_COLLECTIVE_OP_GATHERV:
            return "GATHERV";
        case OTF2_COLLECTIVE_OP_SCATTER:
            return "SCATTER";
        case OTF2_COLLECTIVE_OP_SCATTERV:
            return "SCATTERV";
        case OTF2_COLLECTIVE_OP_ALLGATHER:
            return "ALLGATHER";
        case OTF2_COLLECTIVE_OP_ALLGATHERV:
            return "ALLGATHERV";
        case OTF2_COLLECTIVE_OP_ALLTOALL:
            return "ALLTOALL";
        case OTF2_COLLECTIVE_OP_ALLTOALLV:
            return "ALLTOALLV";
        case OTF2_COLLECTIVE_OP_ALLTOALLW:
            return "ALLTOALLW";
        case OTF2_COLLECTIVE_OP_ALLREDUCE:
            return "ALLREDUCE";
        case OTF2_COLLECTIVE_OP_REDUCE:
            return "REDUCE";
        case OTF2_COLLECTIVE_OP_REDUCE_SCATTER:
            return "REDUCE_SCATTER";
        case OTF2_COLLECTIVE_OP_SCAN:
            return "SCAN";
        case OTF2_COLLECTIVE_OP_EXSCAN:
            return "EXSCAN";
        case OTF2_COLLECTIVE_OP_REDUCE_SCATTER_BLOCK:
            return "REDUCE_SCATTER_BLOCK";
        case OTF2_COLLECTIVE_OP_CREATE_HANDLE:
            return "CREATE_HANDLE";
        case OTF2_COLLECTIVE_OP_DESTROY_HANDLE:
            return "DESTROY_HANDLE";
        case OTF2_COLLECTIVE_OP_ALLOCATE:
            return "ALLOCATE";
        case OTF2_COLLECTIVE_OP_DEALLOCATE:
            return "DEALLOCATE";
        case OTF2_COLLECTIVE_OP_CREATE_HANDLE_AND_ALLOCATE:
            return "CREATE_HANDLE_AND_ALLOCATE";
        case OTF2_COLLECTIVE_OP_DESTROY_HANDLE_AND_DEALLOCATE:
            return "DESTROY_HANDLE_AND_DEALLOCATE";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_collective_op( OTF2_CollectiveOp collectiveOp )
{
    const char* result = otf2_print_get_raw_collective_op( collectiveOp );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( collectiveOp );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_rma_sync_type( OTF2_RmaSyncType rmaSyncType )
{
    switch ( rmaSyncType )
    {
        case OTF2_RMA_SYNC_TYPE_MEMORY:
            return "MEMORY";
        case OTF2_RMA_SYNC_TYPE_NOTIFY_IN:
            return "NOTIFY_IN";
        case OTF2_RMA_SYNC_TYPE_NOTIFY_OUT:
            return "NOTIFY_OUT";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_rma_sync_type( OTF2_RmaSyncType rmaSyncType )
{
    const char* result = otf2_print_get_raw_rma_sync_type( rmaSyncType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( rmaSyncType );
    }

    return result;
}


static inline const char*
otf2_print_get_rma_sync_level( OTF2_RmaSyncLevel rmaSyncLevel )
{
    size_t buffer_size =
        2 + ( 2 * 3 )
        + sizeof( "NONE" )
        + sizeof( "PROCESS" )
        + sizeof( "MEMORY" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( rmaSyncLevel == OTF2_RMA_SYNC_LEVEL_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( rmaSyncLevel & OTF2_RMA_SYNC_LEVEL_PROCESS )
    {
        strcat( buffer, sep );
        strcat( buffer, "PROCESS" );
        sep           = ", ";
        rmaSyncLevel &= ~OTF2_RMA_SYNC_LEVEL_PROCESS;
    }
    if ( rmaSyncLevel & OTF2_RMA_SYNC_LEVEL_MEMORY )
    {
        strcat( buffer, sep );
        strcat( buffer, "MEMORY" );
        sep           = ", ";
        rmaSyncLevel &= ~OTF2_RMA_SYNC_LEVEL_MEMORY;
    }
    if ( rmaSyncLevel )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, rmaSyncLevel );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_raw_lock_type( OTF2_LockType lockType )
{
    switch ( lockType )
    {
        case OTF2_LOCK_EXCLUSIVE:
            return "EXCLUSIVE";
        case OTF2_LOCK_SHARED:
            return "SHARED";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_lock_type( OTF2_LockType lockType )
{
    const char* result = otf2_print_get_raw_lock_type( lockType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( lockType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_rma_atomic_type( OTF2_RmaAtomicType rmaAtomicType )
{
    switch ( rmaAtomicType )
    {
        case OTF2_RMA_ATOMIC_TYPE_ACCUMULATE:
            return "ACCUMULATE";
        case OTF2_RMA_ATOMIC_TYPE_INCREMENT:
            return "INCREMENT";
        case OTF2_RMA_ATOMIC_TYPE_TEST_AND_SET:
            return "TEST_AND_SET";
        case OTF2_RMA_ATOMIC_TYPE_COMPARE_AND_SWAP:
            return "COMPARE_AND_SWAP";
        case OTF2_RMA_ATOMIC_TYPE_SWAP:
            return "SWAP";
        case OTF2_RMA_ATOMIC_TYPE_FETCH_AND_ADD:
            return "FETCH_AND_ADD";
        case OTF2_RMA_ATOMIC_TYPE_FETCH_AND_INCREMENT:
            return "FETCH_AND_INCREMENT";
        case OTF2_RMA_ATOMIC_TYPE_FETCH_AND_ACCUMULATE:
            return "FETCH_AND_ACCUMULATE";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_rma_atomic_type( OTF2_RmaAtomicType rmaAtomicType )
{
    const char* result = otf2_print_get_raw_rma_atomic_type( rmaAtomicType );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( rmaAtomicType );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_io_paradigm_class( OTF2_IoParadigmClass ioParadigmClass )
{
    switch ( ioParadigmClass )
    {
        case OTF2_IO_PARADIGM_CLASS_SERIAL:
            return "SERIAL";
        case OTF2_IO_PARADIGM_CLASS_PARALLEL:
            return "PARALLEL";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_io_paradigm_class( OTF2_IoParadigmClass ioParadigmClass )
{
    const char* result = otf2_print_get_raw_io_paradigm_class( ioParadigmClass );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( ioParadigmClass );
    }

    return result;
}


static inline const char*
otf2_print_get_io_paradigm_flag( OTF2_IoParadigmFlag ioParadigmFlag )
{
    size_t buffer_size =
        2 + ( 2 * 2 )
        + sizeof( "NONE" )
        + sizeof( "OS" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( ioParadigmFlag == OTF2_IO_PARADIGM_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( ioParadigmFlag & OTF2_IO_PARADIGM_FLAG_OS )
    {
        strcat( buffer, sep );
        strcat( buffer, "OS" );
        sep             = ", ";
        ioParadigmFlag &= ~OTF2_IO_PARADIGM_FLAG_OS;
    }
    if ( ioParadigmFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, ioParadigmFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_raw_io_paradigm_property( OTF2_IoParadigmProperty ioParadigmProperty )
{
    switch ( ioParadigmProperty )
    {
        case OTF2_IO_PARADIGM_PROPERTY_VERSION:
            return "VERSION";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_io_paradigm_property( OTF2_IoParadigmProperty ioParadigmProperty )
{
    const char* result = otf2_print_get_raw_io_paradigm_property( ioParadigmProperty );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( ioParadigmProperty );
    }

    return result;
}


static inline const char*
otf2_print_get_io_handle_flag( OTF2_IoHandleFlag ioHandleFlag )
{
    size_t buffer_size =
        2 + ( 2 * 3 )
        + sizeof( "NONE" )
        + sizeof( "PRE_CREATED" )
        + sizeof( "ALL_PROXY" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( ioHandleFlag == OTF2_IO_HANDLE_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( ioHandleFlag & OTF2_IO_HANDLE_FLAG_PRE_CREATED )
    {
        strcat( buffer, sep );
        strcat( buffer, "PRE_CREATED" );
        sep           = ", ";
        ioHandleFlag &= ~OTF2_IO_HANDLE_FLAG_PRE_CREATED;
    }
    if ( ioHandleFlag & OTF2_IO_HANDLE_FLAG_ALL_PROXY )
    {
        strcat( buffer, sep );
        strcat( buffer, "ALL_PROXY" );
        sep           = ", ";
        ioHandleFlag &= ~OTF2_IO_HANDLE_FLAG_ALL_PROXY;
    }
    if ( ioHandleFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, ioHandleFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_raw_io_access_mode( OTF2_IoAccessMode ioAccessMode )
{
    switch ( ioAccessMode )
    {
        case OTF2_IO_ACCESS_MODE_READ_ONLY:
            return "READ_ONLY";
        case OTF2_IO_ACCESS_MODE_WRITE_ONLY:
            return "WRITE_ONLY";
        case OTF2_IO_ACCESS_MODE_READ_WRITE:
            return "READ_WRITE";
        case OTF2_IO_ACCESS_MODE_EXECUTE_ONLY:
            return "EXECUTE_ONLY";
        case OTF2_IO_ACCESS_MODE_SEARCH_ONLY:
            return "SEARCH_ONLY";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_io_access_mode( OTF2_IoAccessMode ioAccessMode )
{
    const char* result = otf2_print_get_raw_io_access_mode( ioAccessMode );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( ioAccessMode );
    }

    return result;
}


static inline const char*
otf2_print_get_io_creation_flag( OTF2_IoCreationFlag ioCreationFlag )
{
    size_t buffer_size =
        2 + ( 2 * 12 )
        + sizeof( "NONE" )
        + sizeof( "CREATE" )
        + sizeof( "TRUNCATE" )
        + sizeof( "DIRECTORY" )
        + sizeof( "EXCLUSIVE" )
        + sizeof( "NO_CONTROLLING_TERMINAL" )
        + sizeof( "NO_FOLLOW" )
        + sizeof( "PATH" )
        + sizeof( "TEMPORARY_FILE" )
        + sizeof( "LARGEFILE" )
        + sizeof( "NO_SEEK" )
        + sizeof( "UNIQUE" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( ioCreationFlag == OTF2_IO_CREATION_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_CREATE )
    {
        strcat( buffer, sep );
        strcat( buffer, "CREATE" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_CREATE;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_TRUNCATE )
    {
        strcat( buffer, sep );
        strcat( buffer, "TRUNCATE" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_TRUNCATE;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_DIRECTORY )
    {
        strcat( buffer, sep );
        strcat( buffer, "DIRECTORY" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_DIRECTORY;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_EXCLUSIVE )
    {
        strcat( buffer, sep );
        strcat( buffer, "EXCLUSIVE" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_EXCLUSIVE;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_NO_CONTROLLING_TERMINAL )
    {
        strcat( buffer, sep );
        strcat( buffer, "NO_CONTROLLING_TERMINAL" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_NO_CONTROLLING_TERMINAL;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_NO_FOLLOW )
    {
        strcat( buffer, sep );
        strcat( buffer, "NO_FOLLOW" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_NO_FOLLOW;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_PATH )
    {
        strcat( buffer, sep );
        strcat( buffer, "PATH" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_PATH;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_TEMPORARY_FILE )
    {
        strcat( buffer, sep );
        strcat( buffer, "TEMPORARY_FILE" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_TEMPORARY_FILE;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_LARGEFILE )
    {
        strcat( buffer, sep );
        strcat( buffer, "LARGEFILE" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_LARGEFILE;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_NO_SEEK )
    {
        strcat( buffer, sep );
        strcat( buffer, "NO_SEEK" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_NO_SEEK;
    }
    if ( ioCreationFlag & OTF2_IO_CREATION_FLAG_UNIQUE )
    {
        strcat( buffer, sep );
        strcat( buffer, "UNIQUE" );
        sep             = ", ";
        ioCreationFlag &= ~OTF2_IO_CREATION_FLAG_UNIQUE;
    }
    if ( ioCreationFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, ioCreationFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_io_status_flag( OTF2_IoStatusFlag ioStatusFlag )
{
    size_t buffer_size =
        2 + ( 2 * 10 )
        + sizeof( "NONE" )
        + sizeof( "CLOSE_ON_EXEC" )
        + sizeof( "APPEND" )
        + sizeof( "NON_BLOCKING" )
        + sizeof( "ASYNC" )
        + sizeof( "SYNC" )
        + sizeof( "DATA_SYNC" )
        + sizeof( "AVOID_CACHING" )
        + sizeof( "NO_ACCESS_TIME" )
        + sizeof( "DELETE_ON_CLOSE" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( ioStatusFlag == OTF2_IO_STATUS_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_CLOSE_ON_EXEC )
    {
        strcat( buffer, sep );
        strcat( buffer, "CLOSE_ON_EXEC" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_CLOSE_ON_EXEC;
    }
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_APPEND )
    {
        strcat( buffer, sep );
        strcat( buffer, "APPEND" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_APPEND;
    }
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_NON_BLOCKING )
    {
        strcat( buffer, sep );
        strcat( buffer, "NON_BLOCKING" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_NON_BLOCKING;
    }
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_ASYNC )
    {
        strcat( buffer, sep );
        strcat( buffer, "ASYNC" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_ASYNC;
    }
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_SYNC )
    {
        strcat( buffer, sep );
        strcat( buffer, "SYNC" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_SYNC;
    }
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_DATA_SYNC )
    {
        strcat( buffer, sep );
        strcat( buffer, "DATA_SYNC" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_DATA_SYNC;
    }
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_AVOID_CACHING )
    {
        strcat( buffer, sep );
        strcat( buffer, "AVOID_CACHING" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_AVOID_CACHING;
    }
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_NO_ACCESS_TIME )
    {
        strcat( buffer, sep );
        strcat( buffer, "NO_ACCESS_TIME" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_NO_ACCESS_TIME;
    }
    if ( ioStatusFlag & OTF2_IO_STATUS_FLAG_DELETE_ON_CLOSE )
    {
        strcat( buffer, sep );
        strcat( buffer, "DELETE_ON_CLOSE" );
        sep           = ", ";
        ioStatusFlag &= ~OTF2_IO_STATUS_FLAG_DELETE_ON_CLOSE;
    }
    if ( ioStatusFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, ioStatusFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_raw_io_seek_option( OTF2_IoSeekOption ioSeekOption )
{
    switch ( ioSeekOption )
    {
        case OTF2_IO_SEEK_FROM_START:
            return "FROM_START";
        case OTF2_IO_SEEK_FROM_CURRENT:
            return "FROM_CURRENT";
        case OTF2_IO_SEEK_FROM_END:
            return "FROM_END";
        case OTF2_IO_SEEK_DATA:
            return "DATA";
        case OTF2_IO_SEEK_HOLE:
            return "HOLE";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_io_seek_option( OTF2_IoSeekOption ioSeekOption )
{
    const char* result = otf2_print_get_raw_io_seek_option( ioSeekOption );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( ioSeekOption );
    }

    return result;
}


static inline const char*
otf2_print_get_raw_io_operation_mode( OTF2_IoOperationMode ioOperationMode )
{
    switch ( ioOperationMode )
    {
        case OTF2_IO_OPERATION_MODE_READ:
            return "READ";
        case OTF2_IO_OPERATION_MODE_WRITE:
            return "WRITE";
        case OTF2_IO_OPERATION_MODE_FLUSH:
            return "FLUSH";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_io_operation_mode( OTF2_IoOperationMode ioOperationMode )
{
    const char* result = otf2_print_get_raw_io_operation_mode( ioOperationMode );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( ioOperationMode );
    }

    return result;
}


static inline const char*
otf2_print_get_io_operation_flag( OTF2_IoOperationFlag ioOperationFlag )
{
    size_t buffer_size =
        2 + ( 2 * 3 )
        + sizeof( "NONE" )
        + sizeof( "NON_BLOCKING" )
        + sizeof( "COLLECTIVE" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( ioOperationFlag == OTF2_IO_OPERATION_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( ioOperationFlag & OTF2_IO_OPERATION_FLAG_NON_BLOCKING )
    {
        strcat( buffer, sep );
        strcat( buffer, "NON_BLOCKING" );
        sep              = ", ";
        ioOperationFlag &= ~OTF2_IO_OPERATION_FLAG_NON_BLOCKING;
    }
    if ( ioOperationFlag & OTF2_IO_OPERATION_FLAG_COLLECTIVE )
    {
        strcat( buffer, sep );
        strcat( buffer, "COLLECTIVE" );
        sep              = ", ";
        ioOperationFlag &= ~OTF2_IO_OPERATION_FLAG_COLLECTIVE;
    }
    if ( ioOperationFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, ioOperationFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_comm_flag( OTF2_CommFlag commFlag )
{
    size_t buffer_size =
        2 + ( 2 * 2 )
        + sizeof( "NONE" )
        + sizeof( "CREATE_DESTROY_EVENTS" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( commFlag == OTF2_COMM_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( commFlag & OTF2_COMM_FLAG_CREATE_DESTROY_EVENTS )
    {
        strcat( buffer, sep );
        strcat( buffer, "CREATE_DESTROY_EVENTS" );
        sep       = ", ";
        commFlag &= ~OTF2_COMM_FLAG_CREATE_DESTROY_EVENTS;
    }
    if ( commFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, commFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_rma_win_flag( OTF2_RmaWinFlag rmaWinFlag )
{
    size_t buffer_size =
        2 + ( 2 * 2 )
        + sizeof( "NONE" )
        + sizeof( "CREATE_DESTROY_EVENTS" )
        + 1 + sizeof( "INVALID <0x00000000>" );
    char* buffer = otf2_print_get_buffer( buffer_size );

    buffer[ 0 ] = '\0';
    if ( rmaWinFlag == OTF2_RMA_WIN_FLAG_NONE )
    {
        strcat( buffer, "NONE" );
        return buffer;
    }

    const char* sep = "";
    strcat( buffer, "{" );
    if ( rmaWinFlag & OTF2_RMA_WIN_FLAG_CREATE_DESTROY_EVENTS )
    {
        strcat( buffer, sep );
        strcat( buffer, "CREATE_DESTROY_EVENTS" );
        sep         = ", ";
        rmaWinFlag &= ~OTF2_RMA_WIN_FLAG_CREATE_DESTROY_EVENTS;
    }
    if ( rmaWinFlag )
    {
        snprintf( buffer + strlen( buffer ),
                  2 + sizeof( "INVALID <0x00000000>" ),
                  "%sINVALID <0x%" PRIx32 ">",
                  sep, rmaWinFlag );
    }
    strcat( buffer, "}" );

    return buffer;
}


static inline const char*
otf2_print_get_raw_collective_root( OTF2_CollectiveRoot collectiveRoot )
{
    switch ( collectiveRoot )
    {
        case OTF2_COLLECTIVE_ROOT_NONE:
            return "NONE";
        case OTF2_COLLECTIVE_ROOT_SELF:
            return "SELF";
        case OTF2_COLLECTIVE_ROOT_THIS_GROUP:
            return "THIS_GROUP";

        default:
            return NULL;
    }
}


static inline const char*
otf2_print_get_collective_root( OTF2_CollectiveRoot collectiveRoot )
{
    const char* result = otf2_print_get_raw_collective_root( collectiveRoot );
    if ( result == NULL )
    {
        result = otf2_print_get_invalid( collectiveRoot );
    }

    return result;
}
