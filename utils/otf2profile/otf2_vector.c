/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2012,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2012,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2012,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2012,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2012,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2012,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2012,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

/**
 * @file
 * @ingroup         otf2_vector_module
 *
 * @brief           Implementation of a STL-like vector.
 */

#include <config.h>

#include <stdlib.h>
#include <string.h>

#include <UTILS_Error.h>
#include <UTILS_Debug.h>

#include "otf2_vector.h"

/*
 * ---------------------------------------------------------------------------
 * otf2_vector
 * ---------------------------------------------------------------------------
 */

#define DEFAULT_CAPACITY     16


/*--- Type definitions ----------------------------------------------------*/

/* Actual dynamic array data type */
struct otf2_vector_struct
{
    void** items;      /* Field elements */
    size_t capacity;   /* Total field size */
    size_t size;       /* Number of entries used */
};


/*--- Construction & destruction ------------------------------------------*/

/* Construction & destruction */

otf2_vector*
otf2_vector_create()
{
    otf2_vector* instance;

    /* Create data structure */
    instance = calloc( 1, sizeof( *instance ) );
    if ( !instance )
    {
        UTILS_ERROR_POSIX();
        return NULL;
    }

    return instance;
}

otf2_vector*
otf2_vector_create_size( size_t capacity )
{
    otf2_vector* instance;

    /* Create data structure */
    instance = otf2_vector_create();
    if ( !instance )
    {
        return NULL;
    }

    /* Try to reserve space */
    if ( otf2_vector_reserve( instance, capacity ) )
    {
        return instance;
    }

    /* If the memory could not be allocated, free instance and return NULL */
    free( instance );
    return NULL;
}

void
otf2_vector_free( otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    /* Release data structure */
    free( instance->items );
    free( instance );
}

/*--- Size operations -----------------------------------------------------*/

size_t
otf2_vector_size( const otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    return instance->size;
}

int
otf2_vector_empty( const otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    return instance->size == 0;
}


/*--- Capacity operations -------------------------------------------------*/

size_t
otf2_vector_capacity( const otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    return instance->capacity;
}

int32_t
otf2_vector_reserve( otf2_vector* instance,
                     size_t       capacity )
{
    void** newItems;

    /* Validate arguments */
    UTILS_ASSERT( instance );
    if ( capacity < instance->capacity )
    {
        return 1;
    }

    /* Try to allocate the new amount of memory */
    newItems = ( void** )realloc( instance->items,
                                  capacity * sizeof( void* ) );

    /* Handle failure in memory allocation */
    if ( !newItems )
    {
        UTILS_ERROR_POSIX();
        return 0;
    }

    /* Store new data location and capacity */
    instance->items    = newItems;
    instance->capacity = capacity;

    return 1;
}

int32_t
otf2_vector_resize( otf2_vector* instance,
                    size_t       size )
{
    size_t old_size;
    void** newItems;

    /* Validate arguments */
    UTILS_ASSERT( instance );
    if ( size < instance->size )
    {
        return 1;
    }

    /* Save old size */
    old_size = instance->size;

    /* Eventually resize dynamic array */
    if ( instance->capacity < size )
    {
        newItems = ( void** )realloc( instance->items,
                                      size * sizeof( void* ) );

        /* Handle failure in memory allocation */
        if ( !newItems )
        {
            UTILS_ERROR_POSIX();
            return 0;
        }

        /* Store new capacity and memory pointer */
        instance->capacity = size;
        instance->items    = newItems;
    }

    /* Update size & initialize new entries */
    instance->size = size;
    if ( old_size < size )
    {
        memset( &instance->items[ old_size ], 0,
                ( size - old_size ) * sizeof( void* ) );
    }
    return 1;
}


/*--- Element access ------------------------------------------------------*/

void*
otf2_vector_at( const otf2_vector* instance,
                size_t             index )
{
    /* Validate arguments */
    UTILS_ASSERT( instance && index < instance->size );

    return instance->items[ index ];
}

void
otf2_vector_set( otf2_vector* instance,
                 size_t       index,
                 void*        item )
{
    /* Validate arguments */
    UTILS_ASSERT( instance && index < instance->size );

    instance->items[ index ] = item;
}

void*
otf2_vector_front( const otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance && instance->size > 0 );

    return instance->items[ 0 ];
}

void*
otf2_vector_back( const otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance && instance->size > 0 );

    return instance->items[ instance->size - 1 ];
}


/*--- Iterator functions --------------------------------------------------*/

void**
otf2_vector_begin( const otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    return instance->items;
}

void**
otf2_vector_end( const otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    return instance->items + instance->size;
}


/*--- Inserting & removing elements ---------------------------------------*/

int32_t
otf2_vector_push_back( otf2_vector* instance,
                       void*        item )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    /* Eventually resize dynamic array */
    if ( instance->size == instance->capacity )
    {
        if ( !otf2_vector_reserve( instance, ( instance->capacity == 0 )
                                   ? DEFAULT_CAPACITY
                                   : ( instance->capacity * 2 ) ) )
        {
            return 0;
        }
    }

    /* Append item */
    instance->items[ instance->size ] = item;
    instance->size++;
    return 1;
}

int32_t
otf2_vector_insert( otf2_vector* instance,
                    size_t       index,
                    void*        item )
{
    /* Validate arguments */
    UTILS_ASSERT( instance && index <= instance->size );

    /* Eventually resize dynamic array */
    if ( instance->size == instance->capacity )
    {
        if ( !otf2_vector_reserve( instance, ( instance->capacity == 0 )
                                   ? DEFAULT_CAPACITY
                                   : ( instance->capacity * 2 ) ) )
        {
            return 0;
        }
    }

    /* Insert item */
    memmove( &instance->items[ index + 1 ], &instance->items[ index ],
             ( instance->size - index ) * sizeof( void* ) );
    instance->items[ index ] = item;
    instance->size++;

    return 1;
}

void
otf2_vector_pop_back( otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    /* Remove last item */
    instance->size--;
}

void
otf2_vector_erase( otf2_vector* instance,
                   size_t       index )
{
    /* Validate arguments */
    UTILS_ASSERT( instance && index < instance->size );

    /* Remove element */
    instance->size--;
    memmove( &instance->items[ index ], &instance->items[ index + 1 ],
             ( instance->size - index ) * sizeof( void* ) );
}

void
otf2_vector_clear( otf2_vector* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    /* Forget everything... */
    instance->size = 0;
}


/*--- Algorithms ----------------------------------------------------------*/

int32_t
otf2_vector_find( const otf2_vector*       instance,
                  const void*              value,
                  otf2_vector_compare_func compareFunc,
                  size_t*                  index )
{
    size_t i;

    /* Validate arguments */
    UTILS_ASSERT( instance && compareFunc );

    /* Search item using comparison function */
    for ( i = 0; i < instance->size; ++i )
    {
        if ( 0 == compareFunc( value, instance->items[ i ] ) )
        {
            if ( index )
            {
                *index = i;
            }
            return 1;
        }
    }

    return 0;
}

int32_t
otf2_vector_lower_bound( const otf2_vector*       instance,
                         const void*              value,
                         otf2_vector_compare_func compareFunc,
                         size_t*                  index )
{
    size_t left;
    size_t size;

    /* Validate arguments */
    UTILS_ASSERT( instance && compareFunc );

    /* Perform lower bound search */
    left = 0;
    size = instance->size;
    while ( size > 0 )
    {
        size_t half = size / 2;

        if ( compareFunc( value, instance->items[ left + half ] ) > 0 )
        {
            left += half + 1;
            size -= half + 1;
        }
        else
        {
            size = half;
        }
    }

    /* Store lower bound */
    if ( index )
    {
        *index = left;
    }

    return left < instance->size && 0 ==
           compareFunc( value, instance->items[ left ] );
}

int32_t
otf2_vector_upper_bound( const otf2_vector*       instance,
                         const void*              value,
                         otf2_vector_compare_func compareFunc,
                         size_t*                  index )
{
    size_t left;
    size_t size;

    /* Validate arguments */
    UTILS_ASSERT( instance && compareFunc );

    /* Perform upper bound search */
    left = 0;
    size = instance->size;
    while ( size > 0 )
    {
        size_t half = size / 2;

        if ( compareFunc( value, instance->items[ left + half ] ) < 0 )
        {
            size = half;
        }
        else
        {
            left += half + 1;
            size -= half + 1;
        }
    }

    /* Store upper bound */
    if ( index )
    {
        *index = left;
    }

    return left > 0 && 0 == compareFunc( value, instance->items[ left - 1 ] );
}

void
otf2_vector_foreach( const otf2_vector*          instance,
                     otf2_vector_processing_func procFunc )
{
    size_t index;

    /* Validate arguments */
    UTILS_ASSERT( instance && procFunc );

    /* Execute function for each entry */
    for ( index = 0; index < instance->size; ++index )
    {
        procFunc( instance->items[ index ] );
    }
}
