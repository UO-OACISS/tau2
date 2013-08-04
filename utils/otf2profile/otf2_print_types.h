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

/** @internal
 *  @brief width of the column with the definition names. */
static int otf2_DEF_COLUMN_WIDTH = 16;

/** @internal
 *  @brief width of the column with the event names. */
static int otf2_EVENT_COLUMN_WIDTH = 22;

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
        case OTF2_MAPPING_MPI_COMM:
            return "MPI_COMM";
        case OTF2_MAPPING_PARAMETER:
            return "PARAMETER";

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
        case OTF2_TYPE_MPI_COMM:
            return "MPI_COMM";
        case OTF2_TYPE_PARAMETER:
            return "PARAMETER";

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

        default:
            return otf2_print_get_invalid( paradigm );
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

        default:
            return otf2_print_get_invalid( regionRole );
    }
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
        case OTF2_GROUP_TYPE_MPI_GROUP:
            return "MPI_GROUP";
        case OTF2_GROUP_TYPE_MPI_COMM_SELF:
            return "MPI_COMM_SELF";
        case OTF2_GROUP_TYPE_MPI_LOCATIONS:
            return "MPI_LOCATIONS";

        default:
            return otf2_print_get_invalid( groupType );
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
otf2_print_get_metric_base( OTF2_MetricBase metricBase )
{
    switch ( metricBase )
    {
        case OTF2_BASE_BINARY:
            return "BINARY";
        case OTF2_BASE_DECIMAL:
            return "DECIMAL";

        default:
            return otf2_print_get_invalid( metricBase );
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
otf2_print_get_mpi_collective_type( OTF2_MpiCollectiveType mpiCollectiveType )
{
    switch ( mpiCollectiveType )
    {
        case OTF2_MPI_BARRIER:
            return "BARRIER";
        case OTF2_MPI_BCAST:
            return "BCAST";
        case OTF2_MPI_GATHER:
            return "GATHER";
        case OTF2_MPI_GATHERV:
            return "GATHERV";
        case OTF2_MPI_SCATTER:
            return "SCATTER";
        case OTF2_MPI_SCATTERV:
            return "SCATTERV";
        case OTF2_MPI_ALLGATHER:
            return "ALLGATHER";
        case OTF2_MPI_ALLGATHERV:
            return "ALLGATHERV";
        case OTF2_MPI_ALLTOALL:
            return "ALLTOALL";
        case OTF2_MPI_ALLTOALLV:
            return "ALLTOALLV";
        case OTF2_MPI_ALLTOALLW:
            return "ALLTOALLW";
        case OTF2_MPI_ALLREDUCE:
            return "ALLREDUCE";
        case OTF2_MPI_REDUCE:
            return "REDUCE";
        case OTF2_MPI_REDUCE_SCATTER:
            return "REDUCE_SCATTER";
        case OTF2_MPI_SCAN:
            return "SCAN";
        case OTF2_MPI_EXSCAN:
            return "EXSCAN";
        case OTF2_MPI_REDUCE_SCATTER_BLOCK:
            return "REDUCE_SCATTER_BLOCK";

        default:
            return otf2_print_get_invalid( mpiCollectiveType );
    }
}
