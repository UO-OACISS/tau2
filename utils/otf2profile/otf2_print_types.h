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

/** @internal
 *  @brief width of the column with the definition names. */
static int otf2_DEF_COLUMN_WIDTH = 25;

/** @internal
 *  @brief width of the column with the event names. */
static int otf2_EVENT_COLUMN_WIDTH = 28;

/** @internal
 *  @brief width of the column with the snap names. */
static int otf2_SNAPSHOT_COLUMN_WIDTH = 22;

/** @internal
 *  @brief max value of an OTF2_Paradigm entry + 1. */
static int otf2_max_known_paradigm = 22 + 1;

static inline const char*
otf2_print_get_boolean( OTF2_Boolean boolean )
{
    switch ( boolean )
    {
        case OTF2_FALSE:
            return "FALSE";
        case OTF2_TRUE:
            return "TRUE";

        default:
            return otf2_print_get_invalid( boolean );
    }
}


static inline const char*
otf2_print_get_file_type( OTF2_FileType fileType )
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
            return otf2_print_get_invalid( fileType );
    }
}


static inline const char*
otf2_print_get_file_substrate( OTF2_FileSubstrate fileSubstrate )
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
            return otf2_print_get_invalid( fileSubstrate );
    }
}


static inline const char*
otf2_print_get_mapping_type( OTF2_MappingType mappingType )
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

        default:
            return otf2_print_get_invalid( mappingType );
    }
}


static inline const char*
otf2_print_get_type( OTF2_Type type )
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

        default:
            return otf2_print_get_invalid( type );
    }
}


static inline const char*
otf2_print_get_paradigm( OTF2_Paradigm paradigm )
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

        default:
            return otf2_print_get_invalid( paradigm );
    }
}


static inline const char*
otf2_print_get_paradigm_class( OTF2_ParadigmClass paradigmClass )
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
            return otf2_print_get_invalid( paradigmClass );
    }
}


static inline const char*
otf2_print_get_paradigm_property( OTF2_ParadigmProperty paradigmProperty )
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
            return otf2_print_get_invalid( paradigmProperty );
    }
}


static inline const char*
otf2_print_get_thumbnail_type( OTF2_ThumbnailType thumbnailType )
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
            return otf2_print_get_invalid( thumbnailType );
    }
}


static inline const char*
otf2_print_get_system_tree_domain( OTF2_SystemTreeDomain systemTreeDomain )
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

        default:
            return otf2_print_get_invalid( systemTreeDomain );
    }
}


static inline const char*
otf2_print_get_location_group_type( OTF2_LocationGroupType locationGroupType )
{
    switch ( locationGroupType )
    {
        case OTF2_LOCATION_GROUP_TYPE_UNKNOWN:
            return "UNKNOWN";
        case OTF2_LOCATION_GROUP_TYPE_PROCESS:
            return "PROCESS";

        default:
            return otf2_print_get_invalid( locationGroupType );
    }
}


static inline const char*
otf2_print_get_location_type( OTF2_LocationType locationType )
{
    switch ( locationType )
    {
        case OTF2_LOCATION_TYPE_UNKNOWN:
            return "UNKNOWN";
        case OTF2_LOCATION_TYPE_CPU_THREAD:
            return "CPU_THREAD";
        case OTF2_LOCATION_TYPE_GPU:
            return "GPU";
        case OTF2_LOCATION_TYPE_METRIC:
            return "METRIC";

        default:
            return otf2_print_get_invalid( locationType );
    }
}


static inline const char*
otf2_print_get_region_role( OTF2_RegionRole regionRole )
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

        default:
            return otf2_print_get_invalid( regionRole );
    }
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
otf2_print_get_group_type( OTF2_GroupType groupType )
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
            return otf2_print_get_invalid( groupType );
    }
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
otf2_print_get_base( OTF2_Base base )
{
    switch ( base )
    {
        case OTF2_BASE_BINARY:
            return "BINARY";
        case OTF2_BASE_DECIMAL:
            return "DECIMAL";

        default:
            return otf2_print_get_invalid( base );
    }
}


static inline const char*
otf2_print_get_metric_occurrence( OTF2_MetricOccurrence metricOccurrence )
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
            return otf2_print_get_invalid( metricOccurrence );
    }
}


static inline const char*
otf2_print_get_metric_type( OTF2_MetricType metricType )
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
            return otf2_print_get_invalid( metricType );
    }
}


static inline const char*
otf2_print_get_metric_value_property( OTF2_MetricValueProperty metricValueProperty )
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
            return otf2_print_get_invalid( metricValueProperty );
    }
}


static inline const char*
otf2_print_get_metric_timing( OTF2_MetricTiming metricTiming )
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
            return otf2_print_get_invalid( metricTiming );
    }
}


static inline const char*
otf2_print_get_metric_mode( OTF2_MetricMode metricMode )
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
            return otf2_print_get_invalid( metricMode );
    }
}


static inline const char*
otf2_print_get_metric_scope( OTF2_MetricScope metricScope )
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
            return otf2_print_get_invalid( metricScope );
    }
}


static inline const char*
otf2_print_get_recorder_kind( OTF2_RecorderKind recorderKind )
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
            return otf2_print_get_invalid( recorderKind );
    }
}


static inline const char*
otf2_print_get_parameter_type( OTF2_ParameterType parameterType )
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
            return otf2_print_get_invalid( parameterType );
    }
}


static inline const char*
otf2_print_get_cart_periodicity( OTF2_CartPeriodicity cartPeriodicity )
{
    switch ( cartPeriodicity )
    {
        case OTF2_CART_PERIODIC_FALSE:
            return "FALSE";
        case OTF2_CART_PERIODIC_TRUE:
            return "TRUE";

        default:
            return otf2_print_get_invalid( cartPeriodicity );
    }
}


static inline const char*
otf2_print_get_interrupt_generator_mode( OTF2_InterruptGeneratorMode interruptGeneratorMode )
{
    switch ( interruptGeneratorMode )
    {
        case OTF2_INTERRUPT_GENERATOR_MODE_TIME:
            return "TIME";
        case OTF2_INTERRUPT_GENERATOR_MODE_COUNT:
            return "COUNT";

        default:
            return otf2_print_get_invalid( interruptGeneratorMode );
    }
}


static inline const char*
otf2_print_get_measurement_mode( OTF2_MeasurementMode measurementMode )
{
    switch ( measurementMode )
    {
        case OTF2_MEASUREMENT_ON:
            return "ON";
        case OTF2_MEASUREMENT_OFF:
            return "OFF";

        default:
            return otf2_print_get_invalid( measurementMode );
    }
}


static inline const char*
otf2_print_get_collective_op( OTF2_CollectiveOp collectiveOp )
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
            return otf2_print_get_invalid( collectiveOp );
    }
}


static inline const char*
otf2_print_get_rma_sync_type( OTF2_RmaSyncType rmaSyncType )
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
            return otf2_print_get_invalid( rmaSyncType );
    }
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
otf2_print_get_lock_type( OTF2_LockType lockType )
{
    switch ( lockType )
    {
        case OTF2_LOCK_EXCLUSIVE:
            return "EXCLUSIVE";
        case OTF2_LOCK_SHARED:
            return "SHARED";

        default:
            return otf2_print_get_invalid( lockType );
    }
}


static inline const char*
otf2_print_get_rma_atomic_type( OTF2_RmaAtomicType rmaAtomicType )
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

        default:
            return otf2_print_get_invalid( rmaAtomicType );
    }
}
