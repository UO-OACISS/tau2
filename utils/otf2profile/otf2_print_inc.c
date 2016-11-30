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


struct otf2_print_defs
{
    otf2_hash_table* paradigms;
    otf2_hash_table* strings;
    otf2_hash_table* attributes;
    otf2_hash_table* system_tree_nodes;
    otf2_hash_table* location_groups;
    otf2_hash_table* locations;
    otf2_hash_table* regions;
    otf2_hash_table* callsites;
    otf2_hash_table* callpaths;
    otf2_hash_table* groups;
    otf2_hash_table* metric_members;
    otf2_hash_table* metrics;
    otf2_hash_table* comms;
    otf2_hash_table* parameters;
    otf2_hash_table* rma_wins;
    otf2_hash_table* cart_dimensions;
    otf2_hash_table* cart_topologys;
    otf2_hash_table* source_code_locations;
    otf2_hash_table* calling_contexts;
    otf2_hash_table* interrupt_generators;
};


static void
otf2_print_def_create_hash_tables( struct otf2_print_defs* defs )
{
    defs->paradigms = otf2_hash_table_create_size(
        16,
        otf2_hash_table_hash_uint8,
        otf2_hash_table_compare_uint8 );
    defs->strings = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->attributes = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->system_tree_nodes = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->location_groups = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->locations = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->regions = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->callsites = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->callpaths = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->groups = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->metric_members = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->metrics = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->comms = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->parameters = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->rma_wins = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->cart_dimensions = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->cart_topologys = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->source_code_locations = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->calling_contexts = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
    defs->interrupt_generators = otf2_hash_table_create_size(
        256,
        otf2_hash_table_hash_uint64,
        otf2_hash_table_compare_uint64 );
}


static void
otf2_print_def_destroy_hash_tables( struct otf2_print_defs* defs )
{
    otf2_hash_table_free_all( defs->paradigms, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->strings, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->attributes, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->system_tree_nodes, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->location_groups, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->locations, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->regions, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->callsites, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->callpaths, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->groups, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->metric_members, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->metrics, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->comms, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->parameters, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->rma_wins, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->cart_dimensions, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->cart_topologys, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->source_code_locations, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->calling_contexts, otf2_hash_table_delete_none, free );
    otf2_hash_table_free_all( defs->interrupt_generators, otf2_hash_table_delete_none, free );
}


const char*
otf2_print_get_attribute_value( struct otf2_print_defs* defs,
                                OTF2_Type               type,
                                OTF2_AttributeValue     value )
{
    char* value_buffer = otf2_print_get_buffer( BUFFER_SIZE );

    switch ( type )
    {
        case OTF2_TYPE_UINT8:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIUint8, value.uint8 );
            break;

        case OTF2_TYPE_UINT16:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIUint16, value.uint16 );
            break;

        case OTF2_TYPE_UINT32:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIUint32, value.uint32 );
            break;

        case OTF2_TYPE_UINT64:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIUint64, value.uint64 );
            break;

        case OTF2_TYPE_INT8:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIInt8, value.int8 );
            break;

        case OTF2_TYPE_INT16:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIInt16, value.int16 );
            break;

        case OTF2_TYPE_INT32:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIInt32, value.int32 );
            break;

        case OTF2_TYPE_INT64:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIInt64, value.int64 );
            break;

        case OTF2_TYPE_FLOAT:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIFloat, value.float32 );
            break;

        case OTF2_TYPE_DOUBLE:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIDouble, value.float64 );
            break;

        case OTF2_TYPE_STRING:
            value_buffer = ( char* )otf2_print_get_def_name( defs->strings, value.stringRef );
            break;

        case OTF2_TYPE_ATTRIBUTE:
            value_buffer = ( char* )otf2_print_get_def_name( defs->attributes, value.attributeRef );
            break;

        case OTF2_TYPE_LOCATION:
            value_buffer = ( char* )otf2_print_get_def64_name( defs->locations, value.locationRef );
            break;

        case OTF2_TYPE_REGION:
            value_buffer = ( char* )otf2_print_get_def_name( defs->regions, value.regionRef );
            break;

        case OTF2_TYPE_GROUP:
            value_buffer = ( char* )otf2_print_get_def_name( defs->groups, value.groupRef );
            break;

        case OTF2_TYPE_METRIC:
            value_buffer = ( char* )otf2_print_get_def_name( defs->metrics, value.metricRef );
            break;

        case OTF2_TYPE_COMM:
            value_buffer = ( char* )otf2_print_get_def_name( defs->comms, value.commRef );
            break;

        case OTF2_TYPE_PARAMETER:
            value_buffer = ( char* )otf2_print_get_def_name( defs->parameters, value.parameterRef );
            break;

        case OTF2_TYPE_RMA_WIN:
            value_buffer = ( char* )otf2_print_get_def_name( defs->rma_wins, value.rmaWinRef );
            break;

        case OTF2_TYPE_SOURCE_CODE_LOCATION:
            value_buffer = ( char* )otf2_print_get_def_name( defs->source_code_locations, value.sourceCodeLocationRef );
            break;

        case OTF2_TYPE_CALLING_CONTEXT:
            value_buffer = ( char* )otf2_print_get_def_name( defs->calling_contexts, value.callingContextRef );
            break;

        case OTF2_TYPE_INTERRUPT_GENERATOR:
            value_buffer = ( char* )otf2_print_get_def_name( defs->interrupt_generators, value.interruptGeneratorRef );
            break;

        default:
            snprintf( value_buffer, BUFFER_SIZE,
                      "%" PRIu64, value.uint64 );
            break;
    }

    return value_buffer;
}


static OTF2_CallbackCode
print_global_def_clock_properties( void*    userData,
                                   uint64_t timerResolution,
                                   uint64_t globalOffset,
                                   uint64_t traceLength );


static OTF2_CallbackCode
print_global_def_paradigm( void*              userData,
                           OTF2_Paradigm      paradigm,
                           OTF2_StringRef     name,
                           OTF2_ParadigmClass paradigmClass );


static OTF2_CallbackCode
print_global_def_paradigm_property( void*                 userData,
                                    OTF2_Paradigm         paradigm,
                                    OTF2_ParadigmProperty property,
                                    OTF2_Type             type,
                                    OTF2_AttributeValue   value );


static OTF2_CallbackCode
print_global_def_string( void*          userData,
                         OTF2_StringRef self,
                         const char*    string );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "String",
                             defs->strings,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "string: %" PRIString ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "STRING",
            self,
            string,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_attribute( void*             userData,
                            OTF2_AttributeRef self,
                            OTF2_StringRef    name,
                            OTF2_StringRef    description,
                            OTF2_Type         type );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Attribute",
                             defs->attributes,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "description: %s, "
            "type: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "ATTRIBUTE",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->strings, description ),
            otf2_print_get_type( type ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_system_tree_node( void*                  userData,
                                   OTF2_SystemTreeNodeRef self,
                                   OTF2_StringRef         name,
                                   OTF2_StringRef         className,
                                   OTF2_SystemTreeNodeRef parent );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "SystemTreeNode",
                             defs->system_tree_nodes,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "className: %s, "
            "parent: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "SYSTEM_TREE_NODE",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->strings, className ),
            otf2_print_get_def_name( defs->system_tree_nodes, parent ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_location_group( void*                  userData,
                                 OTF2_LocationGroupRef  self,
                                 OTF2_StringRef         name,
                                 OTF2_LocationGroupType locationGroupType,
                                 OTF2_SystemTreeNodeRef systemTreeParent );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "LocationGroup",
                             defs->location_groups,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "locationGroupType: %s, "
            "systemTreeParent: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "LOCATION_GROUP",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_location_group_type( locationGroupType ),
            otf2_print_get_def_name( defs->system_tree_nodes, systemTreeParent ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_location( void*                 userData,
                           OTF2_LocationRef      self,
                           OTF2_StringRef        name,
                           OTF2_LocationType     locationType,
                           uint64_t              numberOfEvents,
                           OTF2_LocationGroupRef locationGroup );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def64_name( "Location",
                               defs->locations,
                               defs->strings,
                               self,
                               OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint64
            "  "
            "name: %s, "
            "locationType: %s, "
            "numberOfEvents: %" PRIUint64 ", "
            "locationGroup: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "LOCATION",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_location_type( locationType ),
            numberOfEvents,
            otf2_print_get_def_name( defs->location_groups, locationGroup ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
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
                         uint32_t        endLineNumber );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Region",
                             defs->regions,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "canonicalName: %s, "
            "description: %s, "
            "regionRole: %s, "
            "paradigm: %s, "
            "regionFlags: %s, "
            "sourceFile: %s, "
            "beginLineNumber: %" PRIUint32 ", "
            "endLineNumber: %" PRIUint32 ", "
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

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_callsite( void*            userData,
                           OTF2_CallsiteRef self,
                           OTF2_StringRef   sourceFile,
                           uint32_t         lineNumber,
                           OTF2_RegionRef   enteredRegion,
                           OTF2_RegionRef   leftRegion );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Callsite",
                             defs->callsites,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "sourceFile: %s, "
            "lineNumber: %" PRIUint32 ", "
            "enteredRegion: %s, "
            "leftRegion: %s, "
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
#endif


static OTF2_CallbackCode
print_global_def_callpath( void*            userData,
                           OTF2_CallpathRef self,
                           OTF2_CallpathRef parent,
                           OTF2_RegionRef   region );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Callpath",
                             defs->callpaths,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "parent: %s, "
            "region: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CALLPATH",
            self,
            otf2_print_get_def_name( defs->callpaths, parent ),
            otf2_print_get_def_name( defs->regions, region ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_group( void*           userData,
                        OTF2_GroupRef   self,
                        OTF2_StringRef  name,
                        OTF2_GroupType  groupType,
                        OTF2_Paradigm   paradigm,
                        OTF2_GroupFlag  groupFlags,
                        uint32_t        numberOfMembers,
                        const uint64_t* members );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Group",
                             defs->groups,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "groupType: %s, "
            "paradigm: %s, "
            "groupFlags: %s, "
            "numberOfMembers: %" PRIUint32 ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "GROUP",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_group_type( groupType ),
            otf2_print_get_paradigm_name( defs->paradigms, paradigm ),
            otf2_print_get_group_flag( groupFlags ),
            numberOfMembers,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_metric_member( void*                userData,
                                OTF2_MetricMemberRef self,
                                OTF2_StringRef       name,
                                OTF2_StringRef       description,
                                OTF2_MetricType      metricType,
                                OTF2_MetricMode      metricMode,
                                OTF2_Type            valueType,
                                OTF2_Base            base,
                                int64_t              exponent,
                                OTF2_StringRef       unit );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "MetricMember",
                             defs->metric_members,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "description: %s, "
            "metricType: %s, "
            "metricMode: %s, "
            "valueType: %s, "
            "base: %s, "
            "exponent: %" PRIInt64 ", "
            "unit: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "METRIC_MEMBER",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->strings, description ),
            otf2_print_get_metric_type( metricType ),
            otf2_print_get_metric_mode( metricMode ),
            otf2_print_get_type( valueType ),
            otf2_print_get_base( base ),
            exponent,
            otf2_print_get_def_name( defs->strings, unit ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_metric_class( void*                       userData,
                               OTF2_MetricRef              self,
                               uint8_t                     numberOfMetrics,
                               const OTF2_MetricMemberRef* metricMembers,
                               OTF2_MetricOccurrence       metricOccurrence,
                               OTF2_RecorderKind           recorderKind );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "MetricClass",
                             defs->metrics,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "numberOfMetrics: %" PRIUint8 ", "
            "metricOccurrence: %s, "
            "recorderKind: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "METRIC_CLASS",
            self,
            numberOfMetrics,
            otf2_print_get_metric_occurrence( metricOccurrence ),
            otf2_print_get_recorder_kind( recorderKind ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_metric_instance( void*            userData,
                                  OTF2_MetricRef   self,
                                  OTF2_MetricRef   metricClass,
                                  OTF2_LocationRef recorder,
                                  OTF2_MetricScope metricScope,
                                  uint64_t         scope );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "MetricInstance",
                             defs->metrics,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "metricClass: %s, "
            "recorder: %s, "
            "metricScope: %s, "
            "scope: %" PRIUint64 ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "METRIC_INSTANCE",
            self,
            otf2_print_get_def_name( defs->metrics, metricClass ),
            otf2_print_get_def_name( defs->locations, recorder ),
            otf2_print_get_metric_scope( metricScope ),
            scope,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_comm( void*          userData,
                       OTF2_CommRef   self,
                       OTF2_StringRef name,
                       OTF2_GroupRef  group,
                       OTF2_CommRef   parent );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Comm",
                             defs->comms,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "group: %s, "
            "parent: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "COMM",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->groups, group ),
            otf2_print_get_def_name( defs->comms, parent ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_parameter( void*              userData,
                            OTF2_ParameterRef  self,
                            OTF2_StringRef     name,
                            OTF2_ParameterType parameterType );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "Parameter",
                             defs->parameters,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "parameterType: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "PARAMETER",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_parameter_type( parameterType ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_rma_win( void*          userData,
                          OTF2_RmaWinRef self,
                          OTF2_StringRef name,
                          OTF2_CommRef   comm );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "RmaWin",
                             defs->rma_wins,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "comm: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "RMA_WIN",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->comms, comm ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_metric_class_recorder( void*            userData,
                                        OTF2_MetricRef   metricClass,
                                        OTF2_LocationRef recorder );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "metricClass: %s, "
            "recorder: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "METRIC_CLASS_RECORDER",
            "",
            otf2_print_get_def_name( defs->metrics, metricClass ),
            otf2_print_get_def_name( defs->locations, recorder ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_system_tree_node_property( void*                  userData,
                                            OTF2_SystemTreeNodeRef systemTreeNode,
                                            OTF2_StringRef         name,
                                            OTF2_Type              type,
                                            OTF2_AttributeValue    value );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "systemTreeNode: %s, "
            "name: %s, "
            "type: %s, "
            "value: %" PRIAttributeValue ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "SYSTEM_TREE_NODE_PROPERTY",
            "",
            otf2_print_get_def_name( defs->system_tree_nodes, systemTreeNode ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_type( type ),
            value,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_system_tree_node_domain( void*                  userData,
                                          OTF2_SystemTreeNodeRef systemTreeNode,
                                          OTF2_SystemTreeDomain  systemTreeDomain );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "systemTreeNode: %s, "
            "systemTreeDomain: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "SYSTEM_TREE_NODE_DOMAIN",
            "",
            otf2_print_get_def_name( defs->system_tree_nodes, systemTreeNode ),
            otf2_print_get_system_tree_domain( systemTreeDomain ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_location_group_property( void*                 userData,
                                          OTF2_LocationGroupRef locationGroup,
                                          OTF2_StringRef        name,
                                          OTF2_Type             type,
                                          OTF2_AttributeValue   value );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "locationGroup: %s, "
            "name: %s, "
            "type: %s, "
            "value: %" PRIAttributeValue ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "LOCATION_GROUP_PROPERTY",
            "",
            otf2_print_get_def_name( defs->location_groups, locationGroup ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_type( type ),
            value,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_location_property( void*               userData,
                                    OTF2_LocationRef    location,
                                    OTF2_StringRef      name,
                                    OTF2_Type           type,
                                    OTF2_AttributeValue value );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "location: %s, "
            "name: %s, "
            "type: %s, "
            "value: %" PRIAttributeValue ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "LOCATION_PROPERTY",
            "",
            otf2_print_get_def_name( defs->locations, location ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_type( type ),
            value,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_cart_dimension( void*                 userData,
                                 OTF2_CartDimensionRef self,
                                 OTF2_StringRef        name,
                                 uint32_t              size,
                                 OTF2_CartPeriodicity  cartPeriodicity );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "CartDimension",
                             defs->cart_dimensions,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "size: %" PRIUint32 ", "
            "cartPeriodicity: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CART_DIMENSION",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            size,
            otf2_print_get_cart_periodicity( cartPeriodicity ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_cart_topology( void*                        userData,
                                OTF2_CartTopologyRef         self,
                                OTF2_StringRef               name,
                                OTF2_CommRef                 communicator,
                                uint8_t                      numberOfDimensions,
                                const OTF2_CartDimensionRef* cartDimensions );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "CartTopology",
                             defs->cart_topologys,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "communicator: %s, "
            "numberOfDimensions: %" PRIUint8 ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CART_TOPOLOGY",
            self,
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_def_name( defs->comms, communicator ),
            numberOfDimensions,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_cart_coordinate( void*                userData,
                                  OTF2_CartTopologyRef cartTopology,
                                  uint32_t             rank,
                                  uint8_t              numberOfDimensions,
                                  const uint32_t*      coordinates );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "cartTopology: %s, "
            "rank: %" PRIUint32 ", "
            "numberOfDimensions: %" PRIUint8 ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CART_COORDINATE",
            "",
            otf2_print_get_def_name( defs->cart_topologys, cartTopology ),
            rank,
            numberOfDimensions,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_source_code_location( void*                      userData,
                                       OTF2_SourceCodeLocationRef self,
                                       OTF2_StringRef             file,
                                       uint32_t                   lineNumber );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "SourceCodeLocation",
                             defs->source_code_locations,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "file: %s, "
            "lineNumber: %" PRIUint32 ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "SOURCE_CODE_LOCATION",
            self,
            otf2_print_get_def_name( defs->strings, file ),
            lineNumber,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_calling_context( void*                      userData,
                                  OTF2_CallingContextRef     self,
                                  OTF2_RegionRef             region,
                                  OTF2_SourceCodeLocationRef sourceCodeLocation,
                                  OTF2_CallingContextRef     parent );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "CallingContext",
                             defs->calling_contexts,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "region: %s, "
            "sourceCodeLocation: %s, "
            "parent: %s, "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CALLING_CONTEXT",
            self,
            otf2_print_get_def_name( defs->regions, region ),
            otf2_print_get_def_name( defs->source_code_locations, sourceCodeLocation ),
            otf2_print_get_def_name( defs->calling_contexts, parent ),
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_calling_context_property( void*                  userData,
                                           OTF2_CallingContextRef callingContext,
                                           OTF2_StringRef         name,
                                           OTF2_Type              type,
                                           OTF2_AttributeValue    value );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12s"
            "  "
            "callingContext: %s, "
            "name: %s, "
            "type: %s, "
            "value: %" PRIAttributeValue ", "
            "%s",
            otf2_DEF_COLUMN_WIDTH, "CALLING_CONTEXT_PROPERTY",
            "",
            otf2_print_get_def_name( defs->calling_contexts, callingContext ),
            otf2_print_get_def_name( defs->strings, name ),
            otf2_print_get_type( type ),
            value,
            "\n" );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_global_def_interrupt_generator( void*                       userData,
                                      OTF2_InterruptGeneratorRef  self,
                                      OTF2_StringRef              name,
                                      OTF2_InterruptGeneratorMode interruptGeneratorMode,
                                      OTF2_Base                   base,
                                      int64_t                     exponent,
                                      uint64_t                    period );

#if 0
{
    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    otf2_print_add_def_name( "InterruptGenerator",
                             defs->interrupt_generators,
                             defs->strings,
                             self,
                             OTF2_UNDEFINED_STRING /* name attribute */ );

    /* Print definition if selected. */
    if ( !otf2_GLOBDEFS )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    printf( "%-*s "
            "%12" PRIUint32
            "  "
            "name: %s, "
            "interruptGeneratorMode: %s, "
            "base: %s, "
            "exponent: %" PRIInt64 ", "
            "period: %" PRIUint64 ", "
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
#endif


static OTF2_GlobalDefReaderCallbacks*
otf2_print_create_global_def_callbacks( void )
{
    OTF2_GlobalDefReaderCallbacks* def_callbacks = OTF2_GlobalDefReaderCallbacks_New();
    check_pointer( def_callbacks, "Create global definition callback handle." );
    OTF2_GlobalDefReaderCallbacks_SetUnknownCallback( def_callbacks, print_global_def_unknown );
    OTF2_GlobalDefReaderCallbacks_SetClockPropertiesCallback( def_callbacks, print_global_def_clock_properties );
    OTF2_GlobalDefReaderCallbacks_SetParadigmCallback( def_callbacks, print_global_def_paradigm );
    OTF2_GlobalDefReaderCallbacks_SetParadigmPropertyCallback( def_callbacks, print_global_def_paradigm_property );
    OTF2_GlobalDefReaderCallbacks_SetStringCallback( def_callbacks, print_global_def_string );
    OTF2_GlobalDefReaderCallbacks_SetAttributeCallback( def_callbacks, print_global_def_attribute );
    OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeCallback( def_callbacks, print_global_def_system_tree_node );
    OTF2_GlobalDefReaderCallbacks_SetLocationGroupCallback( def_callbacks, print_global_def_location_group );
    OTF2_GlobalDefReaderCallbacks_SetLocationCallback( def_callbacks, print_global_def_location );
    OTF2_GlobalDefReaderCallbacks_SetRegionCallback( def_callbacks, print_global_def_region );
    OTF2_GlobalDefReaderCallbacks_SetCallsiteCallback( def_callbacks, print_global_def_callsite );
    OTF2_GlobalDefReaderCallbacks_SetCallpathCallback( def_callbacks, print_global_def_callpath );
    OTF2_GlobalDefReaderCallbacks_SetGroupCallback( def_callbacks, print_global_def_group );
    OTF2_GlobalDefReaderCallbacks_SetMetricMemberCallback( def_callbacks, print_global_def_metric_member );
    OTF2_GlobalDefReaderCallbacks_SetMetricClassCallback( def_callbacks, print_global_def_metric_class );
    OTF2_GlobalDefReaderCallbacks_SetMetricInstanceCallback( def_callbacks, print_global_def_metric_instance );
    OTF2_GlobalDefReaderCallbacks_SetCommCallback( def_callbacks, print_global_def_comm );
    OTF2_GlobalDefReaderCallbacks_SetParameterCallback( def_callbacks, print_global_def_parameter );
    OTF2_GlobalDefReaderCallbacks_SetRmaWinCallback( def_callbacks, print_global_def_rma_win );
    OTF2_GlobalDefReaderCallbacks_SetMetricClassRecorderCallback( def_callbacks, print_global_def_metric_class_recorder );
    OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodePropertyCallback( def_callbacks, print_global_def_system_tree_node_property );
    OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeDomainCallback( def_callbacks, print_global_def_system_tree_node_domain );
    OTF2_GlobalDefReaderCallbacks_SetLocationGroupPropertyCallback( def_callbacks, print_global_def_location_group_property );
    OTF2_GlobalDefReaderCallbacks_SetLocationPropertyCallback( def_callbacks, print_global_def_location_property );
    OTF2_GlobalDefReaderCallbacks_SetCartDimensionCallback( def_callbacks, print_global_def_cart_dimension );
    OTF2_GlobalDefReaderCallbacks_SetCartTopologyCallback( def_callbacks, print_global_def_cart_topology );
    OTF2_GlobalDefReaderCallbacks_SetCartCoordinateCallback( def_callbacks, print_global_def_cart_coordinate );
    OTF2_GlobalDefReaderCallbacks_SetSourceCodeLocationCallback( def_callbacks, print_global_def_source_code_location );
    OTF2_GlobalDefReaderCallbacks_SetCallingContextCallback( def_callbacks, print_global_def_calling_context );
    OTF2_GlobalDefReaderCallbacks_SetCallingContextPropertyCallback( def_callbacks, print_global_def_calling_context_property );
    OTF2_GlobalDefReaderCallbacks_SetInterruptGeneratorCallback( def_callbacks, print_global_def_interrupt_generator );

    return def_callbacks;
}


static OTF2_CallbackCode
print_buffer_flush( OTF2_LocationRef    location,
                    OTF2_TimeStamp      time,
                    void*               userData,
                    OTF2_AttributeList* attributes,
                    OTF2_TimeStamp      stopTime );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "stopTime: %" PRIUint64Full ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "BUFFER_FLUSH",
            location,
            otf2_print_get_timestamp( data, time ),
            stopTime,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_measurement_on_off( OTF2_LocationRef     location,
                          OTF2_TimeStamp       time,
                          void*                userData,
                          OTF2_AttributeList*  attributes,
                          OTF2_MeasurementMode measurementMode );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "measurementMode: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MEASUREMENT_ON_OFF",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_measurement_mode( measurementMode ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_enter( OTF2_LocationRef    location,
             OTF2_TimeStamp      time,
             void*               userData,
             OTF2_AttributeList* attributes,
             OTF2_RegionRef      region );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "region: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "ENTER",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->regions, region ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_leave( OTF2_LocationRef    location,
             OTF2_TimeStamp      time,
             void*               userData,
             OTF2_AttributeList* attributes,
             OTF2_RegionRef      region );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "region: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "LEAVE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->regions, region ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_send( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            receiver,
                OTF2_CommRef        communicator,
                uint32_t            msgTag,
                uint64_t            msgLength );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "receiver: %" PRIUint32 ", "
            "communicator: %s, "
            "msgTag: %" PRIUint32 ", "
            "msgLength: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_SEND",
            location,
            otf2_print_get_timestamp( data, time ),
            receiver,
            otf2_print_get_def_name( defs->comms, communicator ),
            msgTag,
            msgLength,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_isend( OTF2_LocationRef    location,
                 OTF2_TimeStamp      time,
                 void*               userData,
                 OTF2_AttributeList* attributes,
                 uint32_t            receiver,
                 OTF2_CommRef        communicator,
                 uint32_t            msgTag,
                 uint64_t            msgLength,
                 uint64_t            requestID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "receiver: %" PRIUint32 ", "
            "communicator: %s, "
            "msgTag: %" PRIUint32 ", "
            "msgLength: %" PRIUint64 ", "
            "requestID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_ISEND",
            location,
            otf2_print_get_timestamp( data, time ),
            receiver,
            otf2_print_get_def_name( defs->comms, communicator ),
            msgTag,
            msgLength,
            requestID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_isend_complete( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          uint64_t            requestID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "requestID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_ISEND_COMPLETE",
            location,
            otf2_print_get_timestamp( data, time ),
            requestID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_irecv_request( OTF2_LocationRef    location,
                         OTF2_TimeStamp      time,
                         void*               userData,
                         OTF2_AttributeList* attributes,
                         uint64_t            requestID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "requestID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_IRECV_REQUEST",
            location,
            otf2_print_get_timestamp( data, time ),
            requestID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_recv( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            sender,
                OTF2_CommRef        communicator,
                uint32_t            msgTag,
                uint64_t            msgLength );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "sender: %" PRIUint32 ", "
            "communicator: %s, "
            "msgTag: %" PRIUint32 ", "
            "msgLength: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_RECV",
            location,
            otf2_print_get_timestamp( data, time ),
            sender,
            otf2_print_get_def_name( defs->comms, communicator ),
            msgTag,
            msgLength,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_irecv( OTF2_LocationRef    location,
                 OTF2_TimeStamp      time,
                 void*               userData,
                 OTF2_AttributeList* attributes,
                 uint32_t            sender,
                 OTF2_CommRef        communicator,
                 uint32_t            msgTag,
                 uint64_t            msgLength,
                 uint64_t            requestID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "sender: %" PRIUint32 ", "
            "communicator: %s, "
            "msgTag: %" PRIUint32 ", "
            "msgLength: %" PRIUint64 ", "
            "requestID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_IRECV",
            location,
            otf2_print_get_timestamp( data, time ),
            sender,
            otf2_print_get_def_name( defs->comms, communicator ),
            msgTag,
            msgLength,
            requestID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_request_test( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        uint64_t            requestID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "requestID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_REQUEST_TEST",
            location,
            otf2_print_get_timestamp( data, time ),
            requestID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_request_cancelled( OTF2_LocationRef    location,
                             OTF2_TimeStamp      time,
                             void*               userData,
                             OTF2_AttributeList* attributes,
                             uint64_t            requestID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "requestID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_REQUEST_CANCELLED",
            location,
            otf2_print_get_timestamp( data, time ),
            requestID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_collective_begin( OTF2_LocationRef    location,
                            OTF2_TimeStamp      time,
                            void*               userData,
                            OTF2_AttributeList* attributes );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_COLLECTIVE_BEGIN",
            location,
            otf2_print_get_timestamp( data, time ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_mpi_collective_end( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_CollectiveOp   collectiveOp,
                          OTF2_CommRef        communicator,
                          uint32_t            root,
                          uint64_t            sizeSent,
                          uint64_t            sizeReceived );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "collectiveOp: %s, "
            "communicator: %s, "
            "root: %" PRIUint32 ", "
            "sizeSent: %" PRIUint64 ", "
            "sizeReceived: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "MPI_COLLECTIVE_END",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_collective_op( collectiveOp ),
            otf2_print_get_def_name( defs->comms, communicator ),
            root,
            sizeSent,
            sizeReceived,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_omp_fork( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes,
                uint32_t            numberOfRequestedThreads );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "numberOfRequestedThreads: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_FORK",
            location,
            otf2_print_get_timestamp( data, time ),
            numberOfRequestedThreads,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_omp_join( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_JOIN",
            location,
            otf2_print_get_timestamp( data, time ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_omp_acquire_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        uint32_t            lockID,
                        uint32_t            acquisitionOrder );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "lockID: %" PRIUint32 ", "
            "acquisitionOrder: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_ACQUIRE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            lockID,
            acquisitionOrder,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_omp_release_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        uint32_t            lockID,
                        uint32_t            acquisitionOrder );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "lockID: %" PRIUint32 ", "
            "acquisitionOrder: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_RELEASE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            lockID,
            acquisitionOrder,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_omp_task_create( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint64_t            taskID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "taskID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_TASK_CREATE",
            location,
            otf2_print_get_timestamp( data, time ),
            taskID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_omp_task_switch( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       uint64_t            taskID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "taskID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_TASK_SWITCH",
            location,
            otf2_print_get_timestamp( data, time ),
            taskID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_omp_task_complete( OTF2_LocationRef    location,
                         OTF2_TimeStamp      time,
                         void*               userData,
                         OTF2_AttributeList* attributes,
                         uint64_t            taskID );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "taskID: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "OMP_TASK_COMPLETE",
            location,
            otf2_print_get_timestamp( data, time ),
            taskID,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_metric( OTF2_LocationRef        location,
              OTF2_TimeStamp          time,
              void*                   userData,
              OTF2_AttributeList*     attributes,
              OTF2_MetricRef          metric,
              uint8_t                 numberOfMetrics,
              const OTF2_Type*        typeIDs,
              const OTF2_MetricValue* metricValues );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "metric: %s, "
            "numberOfMetrics: %" PRIUint8 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "METRIC",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->metrics, metric ),
            numberOfMetrics,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_parameter_string( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        OTF2_ParameterRef   parameter,
                        OTF2_StringRef      string );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "parameter: %s, "
            "string: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_STRING",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->parameters, parameter ),
            otf2_print_get_def_name( defs->strings, string ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_parameter_int( OTF2_LocationRef    location,
                     OTF2_TimeStamp      time,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     OTF2_ParameterRef   parameter,
                     int64_t             value );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "parameter: %s, "
            "value: %" PRIInt64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_INT",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->parameters, parameter ),
            value,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_parameter_unsigned_int( OTF2_LocationRef    location,
                              OTF2_TimeStamp      time,
                              void*               userData,
                              OTF2_AttributeList* attributes,
                              OTF2_ParameterRef   parameter,
                              uint64_t            value );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "parameter: %s, "
            "value: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "PARAMETER_UNSIGNED_INT",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->parameters, parameter ),
            value,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_win_create( OTF2_LocationRef    location,
                      OTF2_TimeStamp      time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      OTF2_RmaWinRef      win );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_WIN_CREATE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_win_destroy( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       OTF2_RmaWinRef      win );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_WIN_DESTROY",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_collective_begin( OTF2_LocationRef    location,
                            OTF2_TimeStamp      time,
                            void*               userData,
                            OTF2_AttributeList* attributes );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_COLLECTIVE_BEGIN",
            location,
            otf2_print_get_timestamp( data, time ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_collective_end( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_CollectiveOp   collectiveOp,
                          OTF2_RmaSyncLevel   syncLevel,
                          OTF2_RmaWinRef      win,
                          uint32_t            root,
                          uint64_t            bytesSent,
                          uint64_t            bytesReceived );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "collectiveOp: %s, "
            "syncLevel: %s, "
            "win: %s, "
            "root: %" PRIUint32 ", "
            "bytesSent: %" PRIUint64 ", "
            "bytesReceived: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_COLLECTIVE_END",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_collective_op( collectiveOp ),
            otf2_print_get_rma_sync_level( syncLevel ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            root,
            bytesSent,
            bytesReceived,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_group_sync( OTF2_LocationRef    location,
                      OTF2_TimeStamp      time,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      OTF2_RmaSyncLevel   syncLevel,
                      OTF2_RmaWinRef      win,
                      OTF2_GroupRef       group );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "syncLevel: %s, "
            "win: %s, "
            "group: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_GROUP_SYNC",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_rma_sync_level( syncLevel ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            otf2_print_get_def_name( defs->groups, group ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_request_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        OTF2_RmaWinRef      win,
                        uint32_t            remote,
                        uint64_t            lockId,
                        OTF2_LockType       lockType );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "remote: %" PRIUint32 ", "
            "lockId: %" PRIUint64 ", "
            "lockType: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_REQUEST_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            remote,
            lockId,
            otf2_print_get_lock_type( lockType ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_acquire_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        OTF2_RmaWinRef      win,
                        uint32_t            remote,
                        uint64_t            lockId,
                        OTF2_LockType       lockType );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "remote: %" PRIUint32 ", "
            "lockId: %" PRIUint64 ", "
            "lockType: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_ACQUIRE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            remote,
            lockId,
            otf2_print_get_lock_type( lockType ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_try_lock( OTF2_LocationRef    location,
                    OTF2_TimeStamp      time,
                    void*               userData,
                    OTF2_AttributeList* attributes,
                    OTF2_RmaWinRef      win,
                    uint32_t            remote,
                    uint64_t            lockId,
                    OTF2_LockType       lockType );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "remote: %" PRIUint32 ", "
            "lockId: %" PRIUint64 ", "
            "lockType: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_TRY_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            remote,
            lockId,
            otf2_print_get_lock_type( lockType ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_release_lock( OTF2_LocationRef    location,
                        OTF2_TimeStamp      time,
                        void*               userData,
                        OTF2_AttributeList* attributes,
                        OTF2_RmaWinRef      win,
                        uint32_t            remote,
                        uint64_t            lockId );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "remote: %" PRIUint32 ", "
            "lockId: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_RELEASE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            remote,
            lockId,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_sync( OTF2_LocationRef    location,
                OTF2_TimeStamp      time,
                void*               userData,
                OTF2_AttributeList* attributes,
                OTF2_RmaWinRef      win,
                uint32_t            remote,
                OTF2_RmaSyncType    syncType );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "remote: %" PRIUint32 ", "
            "syncType: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_SYNC",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            remote,
            otf2_print_get_rma_sync_type( syncType ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_wait_change( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       OTF2_RmaWinRef      win );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_WAIT_CHANGE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_put( OTF2_LocationRef    location,
               OTF2_TimeStamp      time,
               void*               userData,
               OTF2_AttributeList* attributes,
               OTF2_RmaWinRef      win,
               uint32_t            remote,
               uint64_t            bytes,
               uint64_t            matchingId );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "remote: %" PRIUint32 ", "
            "bytes: %" PRIUint64 ", "
            "matchingId: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_PUT",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            remote,
            bytes,
            matchingId,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_get( OTF2_LocationRef    location,
               OTF2_TimeStamp      time,
               void*               userData,
               OTF2_AttributeList* attributes,
               OTF2_RmaWinRef      win,
               uint32_t            remote,
               uint64_t            bytes,
               uint64_t            matchingId );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "remote: %" PRIUint32 ", "
            "bytes: %" PRIUint64 ", "
            "matchingId: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_GET",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            remote,
            bytes,
            matchingId,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_atomic( OTF2_LocationRef    location,
                  OTF2_TimeStamp      time,
                  void*               userData,
                  OTF2_AttributeList* attributes,
                  OTF2_RmaWinRef      win,
                  uint32_t            remote,
                  OTF2_RmaAtomicType  type,
                  uint64_t            bytesSent,
                  uint64_t            bytesReceived,
                  uint64_t            matchingId );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "remote: %" PRIUint32 ", "
            "type: %s, "
            "bytesSent: %" PRIUint64 ", "
            "bytesReceived: %" PRIUint64 ", "
            "matchingId: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_ATOMIC",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            remote,
            otf2_print_get_rma_atomic_type( type ),
            bytesSent,
            bytesReceived,
            matchingId,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_op_complete_blocking( OTF2_LocationRef    location,
                                OTF2_TimeStamp      time,
                                void*               userData,
                                OTF2_AttributeList* attributes,
                                OTF2_RmaWinRef      win,
                                uint64_t            matchingId );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "matchingId: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_OP_COMPLETE_BLOCKING",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            matchingId,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_op_complete_non_blocking( OTF2_LocationRef    location,
                                    OTF2_TimeStamp      time,
                                    void*               userData,
                                    OTF2_AttributeList* attributes,
                                    OTF2_RmaWinRef      win,
                                    uint64_t            matchingId );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "matchingId: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_OP_COMPLETE_NON_BLOCKING",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            matchingId,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_op_test( OTF2_LocationRef    location,
                   OTF2_TimeStamp      time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_RmaWinRef      win,
                   uint64_t            matchingId );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "matchingId: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_OP_TEST",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            matchingId,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_rma_op_complete_remote( OTF2_LocationRef    location,
                              OTF2_TimeStamp      time,
                              void*               userData,
                              OTF2_AttributeList* attributes,
                              OTF2_RmaWinRef      win,
                              uint64_t            matchingId );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "win: %s, "
            "matchingId: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "RMA_OP_COMPLETE_REMOTE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->rma_wins, win ),
            matchingId,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_fork( OTF2_LocationRef    location,
                   OTF2_TimeStamp      time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_Paradigm       model,
                   uint32_t            numberOfRequestedThreads );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "model: %s, "
            "numberOfRequestedThreads: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_FORK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_paradigm_name( defs->paradigms, model ),
            numberOfRequestedThreads,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_join( OTF2_LocationRef    location,
                   OTF2_TimeStamp      time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_Paradigm       model );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "model: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_JOIN",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_paradigm_name( defs->paradigms, model ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_team_begin( OTF2_LocationRef    location,
                         OTF2_TimeStamp      time,
                         void*               userData,
                         OTF2_AttributeList* attributes,
                         OTF2_CommRef        threadTeam );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadTeam: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TEAM_BEGIN",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_team_end( OTF2_LocationRef    location,
                       OTF2_TimeStamp      time,
                       void*               userData,
                       OTF2_AttributeList* attributes,
                       OTF2_CommRef        threadTeam );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadTeam: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TEAM_END",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_acquire_lock( OTF2_LocationRef    location,
                           OTF2_TimeStamp      time,
                           void*               userData,
                           OTF2_AttributeList* attributes,
                           OTF2_Paradigm       model,
                           uint32_t            lockID,
                           uint32_t            acquisitionOrder );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "model: %s, "
            "lockID: %" PRIUint32 ", "
            "acquisitionOrder: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_ACQUIRE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_paradigm_name( defs->paradigms, model ),
            lockID,
            acquisitionOrder,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_release_lock( OTF2_LocationRef    location,
                           OTF2_TimeStamp      time,
                           void*               userData,
                           OTF2_AttributeList* attributes,
                           OTF2_Paradigm       model,
                           uint32_t            lockID,
                           uint32_t            acquisitionOrder );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "model: %s, "
            "lockID: %" PRIUint32 ", "
            "acquisitionOrder: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_RELEASE_LOCK",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_paradigm_name( defs->paradigms, model ),
            lockID,
            acquisitionOrder,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_task_create( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_CommRef        threadTeam,
                          uint32_t            creatingThread,
                          uint32_t            generationNumber );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadTeam: %s, "
            "creatingThread: %" PRIUint32 ", "
            "generationNumber: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TASK_CREATE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            creatingThread,
            generationNumber,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_task_switch( OTF2_LocationRef    location,
                          OTF2_TimeStamp      time,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_CommRef        threadTeam,
                          uint32_t            creatingThread,
                          uint32_t            generationNumber );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadTeam: %s, "
            "creatingThread: %" PRIUint32 ", "
            "generationNumber: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TASK_SWITCH",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            creatingThread,
            generationNumber,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_task_complete( OTF2_LocationRef    location,
                            OTF2_TimeStamp      time,
                            void*               userData,
                            OTF2_AttributeList* attributes,
                            OTF2_CommRef        threadTeam,
                            uint32_t            creatingThread,
                            uint32_t            generationNumber );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadTeam: %s, "
            "creatingThread: %" PRIUint32 ", "
            "generationNumber: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_TASK_COMPLETE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadTeam ),
            creatingThread,
            generationNumber,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_create( OTF2_LocationRef    location,
                     OTF2_TimeStamp      time,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     OTF2_CommRef        threadContingent,
                     uint64_t            sequenceCount );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadContingent: %s, "
            "sequenceCount: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_CREATE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadContingent ),
            sequenceCount,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_begin( OTF2_LocationRef    location,
                    OTF2_TimeStamp      time,
                    void*               userData,
                    OTF2_AttributeList* attributes,
                    OTF2_CommRef        threadContingent,
                    uint64_t            sequenceCount );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadContingent: %s, "
            "sequenceCount: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_BEGIN",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadContingent ),
            sequenceCount,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_wait( OTF2_LocationRef    location,
                   OTF2_TimeStamp      time,
                   void*               userData,
                   OTF2_AttributeList* attributes,
                   OTF2_CommRef        threadContingent,
                   uint64_t            sequenceCount );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadContingent: %s, "
            "sequenceCount: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_WAIT",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadContingent ),
            sequenceCount,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_thread_end( OTF2_LocationRef    location,
                  OTF2_TimeStamp      time,
                  void*               userData,
                  OTF2_AttributeList* attributes,
                  OTF2_CommRef        threadContingent,
                  uint64_t            sequenceCount );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "threadContingent: %s, "
            "sequenceCount: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "THREAD_END",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->comms, threadContingent ),
            sequenceCount,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_calling_context_enter( OTF2_LocationRef       location,
                             OTF2_TimeStamp         time,
                             void*                  userData,
                             OTF2_AttributeList*    attributes,
                             OTF2_CallingContextRef callingContext,
                             uint32_t               unwindDistance );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "callingContext: %s, "
            "unwindDistance: %" PRIUint32 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "CALLING_CONTEXT_ENTER",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->calling_contexts, callingContext ),
            unwindDistance,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_calling_context_leave( OTF2_LocationRef       location,
                             OTF2_TimeStamp         time,
                             void*                  userData,
                             OTF2_AttributeList*    attributes,
                             OTF2_CallingContextRef callingContext );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "callingContext: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "CALLING_CONTEXT_LEAVE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->calling_contexts, callingContext ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_calling_context_sample( OTF2_LocationRef           location,
                              OTF2_TimeStamp             time,
                              void*                      userData,
                              OTF2_AttributeList*        attributes,
                              OTF2_CallingContextRef     callingContext,
                              uint32_t                   unwindDistance,
                              OTF2_InterruptGeneratorRef interruptGenerator );

#if 0
{
    if ( time < otf2_MINTIME || time > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "callingContext: %s, "
            "unwindDistance: %" PRIUint32 ", "
            "interruptGenerator: %s, "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "CALLING_CONTEXT_SAMPLE",
            location,
            otf2_print_get_timestamp( data, time ),
            otf2_print_get_def_name( defs->calling_contexts, callingContext ),
            unwindDistance,
            otf2_print_get_def_name( defs->interrupt_generators, interruptGenerator ),
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif



static OTF2_CallbackCode
print_snap_snapshot_start( OTF2_LocationRef    location,
                           OTF2_TimeStamp      snapTime,
                           void*               userData,
                           OTF2_AttributeList* attributes,
                           uint64_t            numberOfRecords );

#if 0
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "numberOfRecords: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "SNAPSHOT_START",
            location,
            otf2_print_get_timestamp( data, snapTime ),
            numberOfRecords,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_snap_snapshot_end( OTF2_LocationRef    location,
                         OTF2_TimeStamp      snapTime,
                         void*               userData,
                         OTF2_AttributeList* attributes,
                         uint64_t            contReadPos );

#if 0
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    struct otf2_print_data* data = userData;
    struct otf2_print_defs* defs = data->defs;

    printf( "%-*s %15" PRIu64 " %20s  "
            "contReadPos: %" PRIUint64 ", "
            "%s",
            otf2_EVENT_COLUMN_WIDTH, "SNAPSHOT_END",
            location,
            otf2_print_get_timestamp( data, snapTime ),
            contReadPos,
            "\n" );

    otf2_print_attribute_list( data, attributes );

    return OTF2_CALLBACK_SUCCESS;
}
#endif


static OTF2_CallbackCode
print_snap_measurement_on_off( OTF2_LocationRef     location,
                               OTF2_TimeStamp       snapTime,
                               void*                userData,
                               OTF2_AttributeList*  attributes,
                               OTF2_TimeStamp       origEventTime,
                               OTF2_MeasurementMode measurementMode )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_measurement_on_off( location,
                                     origEventTime,
                                     userData,
                                     attributes,
                                     measurementMode );
}


static OTF2_CallbackCode
print_snap_enter( OTF2_LocationRef    location,
                  OTF2_TimeStamp      snapTime,
                  void*               userData,
                  OTF2_AttributeList* attributes,
                  OTF2_TimeStamp      origEventTime,
                  OTF2_RegionRef      region )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_enter( location,
                        origEventTime,
                        userData,
                        attributes,
                        region );
}


static OTF2_CallbackCode
print_snap_mpi_send( OTF2_LocationRef    location,
                     OTF2_TimeStamp      snapTime,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     OTF2_TimeStamp      origEventTime,
                     uint32_t            receiver,
                     OTF2_CommRef        communicator,
                     uint32_t            msgTag,
                     uint64_t            msgLength )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_mpi_send( location,
                           origEventTime,
                           userData,
                           attributes,
                           receiver,
                           communicator,
                           msgTag,
                           msgLength );
}


static OTF2_CallbackCode
print_snap_mpi_isend( OTF2_LocationRef    location,
                      OTF2_TimeStamp      snapTime,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      OTF2_TimeStamp      origEventTime,
                      uint32_t            receiver,
                      OTF2_CommRef        communicator,
                      uint32_t            msgTag,
                      uint64_t            msgLength,
                      uint64_t            requestID )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_mpi_isend( location,
                            origEventTime,
                            userData,
                            attributes,
                            receiver,
                            communicator,
                            msgTag,
                            msgLength,
                            requestID );
}


static OTF2_CallbackCode
print_snap_mpi_isend_complete( OTF2_LocationRef    location,
                               OTF2_TimeStamp      snapTime,
                               void*               userData,
                               OTF2_AttributeList* attributes,
                               OTF2_TimeStamp      origEventTime,
                               uint64_t            requestID )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_mpi_isend_complete( location,
                                     origEventTime,
                                     userData,
                                     attributes,
                                     requestID );
}


static OTF2_CallbackCode
print_snap_mpi_recv( OTF2_LocationRef    location,
                     OTF2_TimeStamp      snapTime,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     OTF2_TimeStamp      origEventTime,
                     uint32_t            sender,
                     OTF2_CommRef        communicator,
                     uint32_t            msgTag,
                     uint64_t            msgLength )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_mpi_recv( location,
                           origEventTime,
                           userData,
                           attributes,
                           sender,
                           communicator,
                           msgTag,
                           msgLength );
}


static OTF2_CallbackCode
print_snap_mpi_irecv_request( OTF2_LocationRef    location,
                              OTF2_TimeStamp      snapTime,
                              void*               userData,
                              OTF2_AttributeList* attributes,
                              OTF2_TimeStamp      origEventTime,
                              uint64_t            requestID )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_mpi_irecv_request( location,
                                    origEventTime,
                                    userData,
                                    attributes,
                                    requestID );
}


static OTF2_CallbackCode
print_snap_mpi_irecv( OTF2_LocationRef    location,
                      OTF2_TimeStamp      snapTime,
                      void*               userData,
                      OTF2_AttributeList* attributes,
                      OTF2_TimeStamp      origEventTime,
                      uint32_t            sender,
                      OTF2_CommRef        communicator,
                      uint32_t            msgTag,
                      uint64_t            msgLength,
                      uint64_t            requestID )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_mpi_irecv( location,
                            origEventTime,
                            userData,
                            attributes,
                            sender,
                            communicator,
                            msgTag,
                            msgLength,
                            requestID );
}


static OTF2_CallbackCode
print_snap_mpi_collective_begin( OTF2_LocationRef    location,
                                 OTF2_TimeStamp      snapTime,
                                 void*               userData,
                                 OTF2_AttributeList* attributes,
                                 OTF2_TimeStamp      origEventTime )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_mpi_collective_begin( location,
                                       origEventTime,
                                       userData,
                                       attributes );
}


static OTF2_CallbackCode
print_snap_mpi_collective_end( OTF2_LocationRef    location,
                               OTF2_TimeStamp      snapTime,
                               void*               userData,
                               OTF2_AttributeList* attributes,
                               OTF2_TimeStamp      origEventTime,
                               OTF2_CollectiveOp   collectiveOp,
                               OTF2_CommRef        communicator,
                               uint32_t            root,
                               uint64_t            sizeSent,
                               uint64_t            sizeReceived )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_mpi_collective_end( location,
                                     origEventTime,
                                     userData,
                                     attributes,
                                     collectiveOp,
                                     communicator,
                                     root,
                                     sizeSent,
                                     sizeReceived );
}


static OTF2_CallbackCode
print_snap_omp_fork( OTF2_LocationRef    location,
                     OTF2_TimeStamp      snapTime,
                     void*               userData,
                     OTF2_AttributeList* attributes,
                     OTF2_TimeStamp      origEventTime,
                     uint32_t            numberOfRequestedThreads )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_omp_fork( location,
                           origEventTime,
                           userData,
                           attributes,
                           numberOfRequestedThreads );
}


static OTF2_CallbackCode
print_snap_omp_acquire_lock( OTF2_LocationRef    location,
                             OTF2_TimeStamp      snapTime,
                             void*               userData,
                             OTF2_AttributeList* attributes,
                             OTF2_TimeStamp      origEventTime,
                             uint32_t            lockID,
                             uint32_t            acquisitionOrder )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_omp_acquire_lock( location,
                                   origEventTime,
                                   userData,
                                   attributes,
                                   lockID,
                                   acquisitionOrder );
}


static OTF2_CallbackCode
print_snap_omp_task_create( OTF2_LocationRef    location,
                            OTF2_TimeStamp      snapTime,
                            void*               userData,
                            OTF2_AttributeList* attributes,
                            OTF2_TimeStamp      origEventTime,
                            uint64_t            taskID )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_omp_task_create( location,
                                  origEventTime,
                                  userData,
                                  attributes,
                                  taskID );
}


static OTF2_CallbackCode
print_snap_omp_task_switch( OTF2_LocationRef    location,
                            OTF2_TimeStamp      snapTime,
                            void*               userData,
                            OTF2_AttributeList* attributes,
                            OTF2_TimeStamp      origEventTime,
                            uint64_t            taskID )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_omp_task_switch( location,
                                  origEventTime,
                                  userData,
                                  attributes,
                                  taskID );
}


static OTF2_CallbackCode
print_snap_metric( OTF2_LocationRef        location,
                   OTF2_TimeStamp          snapTime,
                   void*                   userData,
                   OTF2_AttributeList*     attributes,
                   OTF2_TimeStamp          origEventTime,
                   OTF2_MetricRef          metric,
                   uint8_t                 numberOfMetrics,
                   const OTF2_Type*        typeIDs,
                   const OTF2_MetricValue* metricValues )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_metric( location,
                         origEventTime,
                         userData,
                         attributes,
                         metric,
                         numberOfMetrics,
                         typeIDs,
                         metricValues );
}


static OTF2_CallbackCode
print_snap_parameter_string( OTF2_LocationRef    location,
                             OTF2_TimeStamp      snapTime,
                             void*               userData,
                             OTF2_AttributeList* attributes,
                             OTF2_TimeStamp      origEventTime,
                             OTF2_ParameterRef   parameter,
                             OTF2_StringRef      string )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_parameter_string( location,
                                   origEventTime,
                                   userData,
                                   attributes,
                                   parameter,
                                   string );
}


static OTF2_CallbackCode
print_snap_parameter_int( OTF2_LocationRef    location,
                          OTF2_TimeStamp      snapTime,
                          void*               userData,
                          OTF2_AttributeList* attributes,
                          OTF2_TimeStamp      origEventTime,
                          OTF2_ParameterRef   parameter,
                          int64_t             value )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_parameter_int( location,
                                origEventTime,
                                userData,
                                attributes,
                                parameter,
                                value );
}


static OTF2_CallbackCode
print_snap_parameter_unsigned_int( OTF2_LocationRef    location,
                                   OTF2_TimeStamp      snapTime,
                                   void*               userData,
                                   OTF2_AttributeList* attributes,
                                   OTF2_TimeStamp      origEventTime,
                                   OTF2_ParameterRef   parameter,
                                   uint64_t            value )
{
    if ( snapTime < otf2_MINTIME || snapTime > otf2_MAXTIME )
    {
        return OTF2_CALLBACK_SUCCESS;
    }

    return print_parameter_unsigned_int( location,
                                         origEventTime,
                                         userData,
                                         attributes,
                                         parameter,
                                         value );
}


static OTF2_GlobalEvtReaderCallbacks*
otf2_print_create_global_evt_callbacks( void )
{
    OTF2_GlobalEvtReaderCallbacks* evt_callbacks = OTF2_GlobalEvtReaderCallbacks_New();
    check_pointer( evt_callbacks, "Create event reader callbacks." );
    OTF2_GlobalEvtReaderCallbacks_SetUnknownCallback( evt_callbacks, print_unknown );
    OTF2_GlobalEvtReaderCallbacks_SetBufferFlushCallback( evt_callbacks, print_buffer_flush );
    OTF2_GlobalEvtReaderCallbacks_SetMeasurementOnOffCallback( evt_callbacks, print_measurement_on_off );
    OTF2_GlobalEvtReaderCallbacks_SetEnterCallback( evt_callbacks, print_enter );
    OTF2_GlobalEvtReaderCallbacks_SetLeaveCallback( evt_callbacks, print_leave );
    OTF2_GlobalEvtReaderCallbacks_SetMpiSendCallback( evt_callbacks, print_mpi_send );
    OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCallback( evt_callbacks, print_mpi_isend );
    OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCompleteCallback( evt_callbacks, print_mpi_isend_complete );
    OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvRequestCallback( evt_callbacks, print_mpi_irecv_request );
    OTF2_GlobalEvtReaderCallbacks_SetMpiRecvCallback( evt_callbacks, print_mpi_recv );
    OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvCallback( evt_callbacks, print_mpi_irecv );
    OTF2_GlobalEvtReaderCallbacks_SetMpiRequestTestCallback( evt_callbacks, print_mpi_request_test );
    OTF2_GlobalEvtReaderCallbacks_SetMpiRequestCancelledCallback( evt_callbacks, print_mpi_request_cancelled );
    OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveBeginCallback( evt_callbacks, print_mpi_collective_begin );
    OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveEndCallback( evt_callbacks, print_mpi_collective_end );
    OTF2_GlobalEvtReaderCallbacks_SetOmpForkCallback( evt_callbacks, print_omp_fork );
    OTF2_GlobalEvtReaderCallbacks_SetOmpJoinCallback( evt_callbacks, print_omp_join );
    OTF2_GlobalEvtReaderCallbacks_SetOmpAcquireLockCallback( evt_callbacks, print_omp_acquire_lock );
    OTF2_GlobalEvtReaderCallbacks_SetOmpReleaseLockCallback( evt_callbacks, print_omp_release_lock );
    OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCreateCallback( evt_callbacks, print_omp_task_create );
    OTF2_GlobalEvtReaderCallbacks_SetOmpTaskSwitchCallback( evt_callbacks, print_omp_task_switch );
    OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCompleteCallback( evt_callbacks, print_omp_task_complete );
    OTF2_GlobalEvtReaderCallbacks_SetMetricCallback( evt_callbacks, print_metric );
    OTF2_GlobalEvtReaderCallbacks_SetParameterStringCallback( evt_callbacks, print_parameter_string );
    OTF2_GlobalEvtReaderCallbacks_SetParameterIntCallback( evt_callbacks, print_parameter_int );
    OTF2_GlobalEvtReaderCallbacks_SetParameterUnsignedIntCallback( evt_callbacks, print_parameter_unsigned_int );
    OTF2_GlobalEvtReaderCallbacks_SetRmaWinCreateCallback( evt_callbacks, print_rma_win_create );
    OTF2_GlobalEvtReaderCallbacks_SetRmaWinDestroyCallback( evt_callbacks, print_rma_win_destroy );
    OTF2_GlobalEvtReaderCallbacks_SetRmaCollectiveBeginCallback( evt_callbacks, print_rma_collective_begin );
    OTF2_GlobalEvtReaderCallbacks_SetRmaCollectiveEndCallback( evt_callbacks, print_rma_collective_end );
    OTF2_GlobalEvtReaderCallbacks_SetRmaGroupSyncCallback( evt_callbacks, print_rma_group_sync );
    OTF2_GlobalEvtReaderCallbacks_SetRmaRequestLockCallback( evt_callbacks, print_rma_request_lock );
    OTF2_GlobalEvtReaderCallbacks_SetRmaAcquireLockCallback( evt_callbacks, print_rma_acquire_lock );
    OTF2_GlobalEvtReaderCallbacks_SetRmaTryLockCallback( evt_callbacks, print_rma_try_lock );
    OTF2_GlobalEvtReaderCallbacks_SetRmaReleaseLockCallback( evt_callbacks, print_rma_release_lock );
    OTF2_GlobalEvtReaderCallbacks_SetRmaSyncCallback( evt_callbacks, print_rma_sync );
    OTF2_GlobalEvtReaderCallbacks_SetRmaWaitChangeCallback( evt_callbacks, print_rma_wait_change );
    OTF2_GlobalEvtReaderCallbacks_SetRmaPutCallback( evt_callbacks, print_rma_put );
    OTF2_GlobalEvtReaderCallbacks_SetRmaGetCallback( evt_callbacks, print_rma_get );
    OTF2_GlobalEvtReaderCallbacks_SetRmaAtomicCallback( evt_callbacks, print_rma_atomic );
    OTF2_GlobalEvtReaderCallbacks_SetRmaOpCompleteBlockingCallback( evt_callbacks, print_rma_op_complete_blocking );
    OTF2_GlobalEvtReaderCallbacks_SetRmaOpCompleteNonBlockingCallback( evt_callbacks, print_rma_op_complete_non_blocking );
    OTF2_GlobalEvtReaderCallbacks_SetRmaOpTestCallback( evt_callbacks, print_rma_op_test );
    OTF2_GlobalEvtReaderCallbacks_SetRmaOpCompleteRemoteCallback( evt_callbacks, print_rma_op_complete_remote );
    OTF2_GlobalEvtReaderCallbacks_SetThreadForkCallback( evt_callbacks, print_thread_fork );
    OTF2_GlobalEvtReaderCallbacks_SetThreadJoinCallback( evt_callbacks, print_thread_join );
    OTF2_GlobalEvtReaderCallbacks_SetThreadTeamBeginCallback( evt_callbacks, print_thread_team_begin );
    OTF2_GlobalEvtReaderCallbacks_SetThreadTeamEndCallback( evt_callbacks, print_thread_team_end );
    OTF2_GlobalEvtReaderCallbacks_SetThreadAcquireLockCallback( evt_callbacks, print_thread_acquire_lock );
    OTF2_GlobalEvtReaderCallbacks_SetThreadReleaseLockCallback( evt_callbacks, print_thread_release_lock );
    OTF2_GlobalEvtReaderCallbacks_SetThreadTaskCreateCallback( evt_callbacks, print_thread_task_create );
    OTF2_GlobalEvtReaderCallbacks_SetThreadTaskSwitchCallback( evt_callbacks, print_thread_task_switch );
    OTF2_GlobalEvtReaderCallbacks_SetThreadTaskCompleteCallback( evt_callbacks, print_thread_task_complete );
    OTF2_GlobalEvtReaderCallbacks_SetThreadCreateCallback( evt_callbacks, print_thread_create );
    OTF2_GlobalEvtReaderCallbacks_SetThreadBeginCallback( evt_callbacks, print_thread_begin );
    OTF2_GlobalEvtReaderCallbacks_SetThreadWaitCallback( evt_callbacks, print_thread_wait );
    OTF2_GlobalEvtReaderCallbacks_SetThreadEndCallback( evt_callbacks, print_thread_end );
    OTF2_GlobalEvtReaderCallbacks_SetCallingContextEnterCallback( evt_callbacks, print_calling_context_enter );
    OTF2_GlobalEvtReaderCallbacks_SetCallingContextLeaveCallback( evt_callbacks, print_calling_context_leave );
    OTF2_GlobalEvtReaderCallbacks_SetCallingContextSampleCallback( evt_callbacks, print_calling_context_sample );

    return evt_callbacks;
}

static OTF2_GlobalSnapReaderCallbacks*
otf2_print_create_global_snap_callbacks( void )
{
    OTF2_GlobalSnapReaderCallbacks* snap_callbacks = OTF2_GlobalSnapReaderCallbacks_New();
    check_pointer( snap_callbacks, "Create snapshot reader callbacks." );
    OTF2_GlobalSnapReaderCallbacks_SetUnknownCallback( snap_callbacks, print_unknown );
    OTF2_GlobalSnapReaderCallbacks_SetSnapshotStartCallback( snap_callbacks, print_snap_snapshot_start );
    OTF2_GlobalSnapReaderCallbacks_SetSnapshotEndCallback( snap_callbacks, print_snap_snapshot_end );
    OTF2_GlobalSnapReaderCallbacks_SetMeasurementOnOffCallback( snap_callbacks, print_snap_measurement_on_off );
    OTF2_GlobalSnapReaderCallbacks_SetEnterCallback( snap_callbacks, print_snap_enter );
    OTF2_GlobalSnapReaderCallbacks_SetMpiSendCallback( snap_callbacks, print_snap_mpi_send );
    OTF2_GlobalSnapReaderCallbacks_SetMpiIsendCallback( snap_callbacks, print_snap_mpi_isend );
    OTF2_GlobalSnapReaderCallbacks_SetMpiIsendCompleteCallback( snap_callbacks, print_snap_mpi_isend_complete );
    OTF2_GlobalSnapReaderCallbacks_SetMpiRecvCallback( snap_callbacks, print_snap_mpi_recv );
    OTF2_GlobalSnapReaderCallbacks_SetMpiIrecvRequestCallback( snap_callbacks, print_snap_mpi_irecv_request );
    OTF2_GlobalSnapReaderCallbacks_SetMpiIrecvCallback( snap_callbacks, print_snap_mpi_irecv );
    OTF2_GlobalSnapReaderCallbacks_SetMpiCollectiveBeginCallback( snap_callbacks, print_snap_mpi_collective_begin );
    OTF2_GlobalSnapReaderCallbacks_SetMpiCollectiveEndCallback( snap_callbacks, print_snap_mpi_collective_end );
    OTF2_GlobalSnapReaderCallbacks_SetOmpForkCallback( snap_callbacks, print_snap_omp_fork );
    OTF2_GlobalSnapReaderCallbacks_SetOmpAcquireLockCallback( snap_callbacks, print_snap_omp_acquire_lock );
    OTF2_GlobalSnapReaderCallbacks_SetOmpTaskCreateCallback( snap_callbacks, print_snap_omp_task_create );
    OTF2_GlobalSnapReaderCallbacks_SetOmpTaskSwitchCallback( snap_callbacks, print_snap_omp_task_switch );
    OTF2_GlobalSnapReaderCallbacks_SetMetricCallback( snap_callbacks, print_snap_metric );
    OTF2_GlobalSnapReaderCallbacks_SetParameterStringCallback( snap_callbacks, print_snap_parameter_string );
    OTF2_GlobalSnapReaderCallbacks_SetParameterIntCallback( snap_callbacks, print_snap_parameter_int );
    OTF2_GlobalSnapReaderCallbacks_SetParameterUnsignedIntCallback( snap_callbacks, print_snap_parameter_unsigned_int );

    return snap_callbacks;
}
