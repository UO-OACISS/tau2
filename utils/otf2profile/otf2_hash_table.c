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
 * @ingroup         otf2_hash_table_module
 *
 * @brief           Implementation of a STL-like hash table.
 */

#include <config.h>
#include <stdlib.h>
#include <string.h>

#include <UTILS_Error.h>
#include <UTILS_Debug.h>

#include "otf2_hash_table.h"

/*
 * --------------------------------------------------------------------------
 * otf2_hash_table
 * --------------------------------------------------------------------------
 */

/*--- Type definitions ----------------------------------------------------*/

typedef struct otf2_hash_table_list_item_struct otf2_hash_table_list_item;

/* Collision list entry */
struct otf2_hash_table_list_item_struct
{
    otf2_hash_table_entry      entry;      /* Table entry (key, value) */
    size_t                     hash_value; /* hash value for entry.key, for fast comparison */
    otf2_hash_table_list_item* next;       /* Pointer to next entry */
};

/* Actual hash table data type */
struct otf2_hash_table_struct
{
    otf2_hash_table_list_item**  table;   /* Field elements */
    size_t                       tabsize; /* Number of field elements */
    size_t                       size;    /* Number of items stored */
    otf2_hash_table_hash_func    hash;    /* Hashing function */
    otf2_hash_table_compare_func kcmp;    /* Comparison function */
};


/*--- Construction & destruction ------------------------------------------*/

otf2_hash_table*
otf2_hash_table_create_size( size_t                       size,
                             otf2_hash_table_hash_func    hashfunc,
                             otf2_hash_table_compare_func kcmpfunc )
{
    otf2_hash_table* instance;

    /* Validate arguments */
    UTILS_ASSERT( size > 0 && hashfunc && kcmpfunc );

    /* Create hash table data structure */
    instance = calloc( 1, sizeof( *instance ) );
    if ( !instance )
    {
        UTILS_ERROR_POSIX();
        return NULL;
    }

    instance->table = calloc( size, sizeof( otf2_hash_table_list_item* ) );
    if ( !instance->table )
    {
        UTILS_ERROR_POSIX();
        return NULL;
    }

    /* Initialization */
    instance->tabsize = size;
    instance->size    = 0;
    instance->hash    = hashfunc;
    instance->kcmp    = kcmpfunc;

    return instance;
}


void
otf2_hash_table_free( otf2_hash_table* instance )
{
    size_t index;

    /* Validate arguments */
    UTILS_ASSERT( instance );

    /* Release data structure */
    for ( index = 0; index < instance->tabsize; ++index )
    {
        otf2_hash_table_list_item* item = instance->table[ index ];
        while ( item )
        {
            otf2_hash_table_list_item* next = item->next;
            free( item );
            item = next;
        }
    }
    free( instance->table );
    free( instance );
}


void
otf2_hash_table_free_all( otf2_hash_table*            instance,
                          otf2_hash_table_delete_func deleteKey,
                          otf2_hash_table_delete_func deleteValue )
{
    otf2_hash_table_iterator* iter;
    otf2_hash_table_entry*    entry;

    /* Validate arguments */
    UTILS_ASSERT( instance && deleteKey && deleteValue );

    /* Execute function for each entry */
    iter  = otf2_hash_table_iterator_create( instance );
    entry = otf2_hash_table_iterator_first( iter );
    while ( entry )
    {
        deleteKey( entry->key );
        deleteValue( entry->value );
        entry = otf2_hash_table_iterator_next( iter );
    }
    otf2_hash_table_iterator_free( iter );
    otf2_hash_table_free( instance );
}


/*--- Size operations -----------------------------------------------------*/


size_t
otf2_hash_table_size( const otf2_hash_table* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    return instance->size;
}

int32_t
otf2_hash_table_empty( const otf2_hash_table* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    return instance->size == 0;
}


/*--- Inserting & removing elements ---------------------------------------*/

void
otf2_hash_table_insert( otf2_hash_table* instance,
                        void*            key,
                        void*            value,
                        size_t*          hashValPtr )
{
    otf2_hash_table_list_item* item;
    size_t                     hashval;
    size_t                     index;

    /* Validate arguments */
    UTILS_ASSERT( instance && key );

    /* Eventually calculate hash value */
    if ( hashValPtr )
    {
        hashval = *hashValPtr;
    }
    else
    {
        hashval = instance->hash( key );
    }
    index = hashval % instance->tabsize;

    /* Create new item */
    item = calloc( 1, sizeof( *item ) );
    if ( !item )
    {
        UTILS_ERROR_POSIX();
        return;
    }

    /* Initialize item */
    item->entry.key   = key;
    item->entry.value = value;
    item->hash_value  = hashval;
    item->next        = instance->table[ index ];

    /* Add item to hash table */
    instance->table[ index ] = item;
    instance->size++;
}

void
otf2_hash_table_remove( const otf2_hash_table*      instance,
                        const void*                 key,
                        otf2_hash_table_delete_func deleteKey,
                        otf2_hash_table_delete_func deleteValue,
                        size_t*                     hashValPtr )
{
    otf2_hash_table_list_item* item;
    otf2_hash_table_list_item* last = NULL;
    size_t                     hashval;
    size_t                     index;

    /* Validate arguments */
    UTILS_ASSERT( instance && key );

    /* Eventually calculate hash value */
    if ( hashValPtr )
    {
        hashval = *hashValPtr;
    }
    else
    {
        hashval = instance->hash( key );
    }
    index = hashval % instance->tabsize;

    /* Search element */
    item = instance->table[ index ];
    while ( item )
    {
        if ( hashval == item->hash_value &&
             0 == instance->kcmp( key, item->entry.key ) )
        {
            if ( last == NULL )
            {
                instance->table[ index ] = item->next;
            }
            else
            {
                last->next = item->next;
            }

            deleteKey( item->entry.key );
            deleteValue( item->entry.value );
            free( item );
            return;
        }

        last = item;
        item = item->next;
    }
}



/*--- Algorithms ----------------------------------------------------------*/

otf2_hash_table_entry*
otf2_hash_table_find( const otf2_hash_table* instance,
                      const void*            key,
                      size_t*                hashValPtr )
{
    otf2_hash_table_list_item* item;
    size_t                     hashval;
    size_t                     index;

    /* Validate arguments */
    UTILS_ASSERT( instance && key );

    /* Calculate hash value */
    hashval = instance->hash( key );
    if ( hashValPtr )
    {
        *hashValPtr = hashval;
    }
    index = hashval % instance->tabsize;

    /* Search element */
    item = instance->table[ index ];
    while ( item )
    {
        if ( hashval == item->hash_value &&
             0 == instance->kcmp( key, item->entry.key ) )
        {
            return &item->entry;
        }

        item = item->next;
    }

    return NULL;
}

void
otf2_hash_table_foreach( const otf2_hash_table*       instance,
                         otf2_hash_table_process_func procfunc )
{
    otf2_hash_table_iterator* iter;
    otf2_hash_table_entry*    entry;

    /* Validate arguments */
    UTILS_ASSERT( instance && procfunc );

    /* Execute function for each entry */
    iter  = otf2_hash_table_iterator_create( instance );
    entry = otf2_hash_table_iterator_first( iter );
    while ( entry )
    {
        procfunc( entry );
        entry = otf2_hash_table_iterator_next( iter );
    }
    otf2_hash_table_iterator_free( iter );
}


/*
 * --------------------------------------------------------------------------
 * otf2_hash_table_iterator
 * --------------------------------------------------------------------------
 */

/*--- Type definitions ----------------------------------------------------*/

/* Actual hash table iterator data structure */
struct otf2_hash_table_iter_struct
{
    const otf2_hash_table*     hashtab;    /* corresponding hash table */
    size_t                     index;      /* current field index      */
    otf2_hash_table_list_item* item;       /* current item             */
};


/*--- Construction & destruction ------------------------------------------*/

otf2_hash_table_iterator*
otf2_hash_table_iterator_create( const otf2_hash_table* hashtab )
{
    otf2_hash_table_iterator* instance;

    /* Validate arguments */
    UTILS_ASSERT( hashtab );

    /* Create iterator */
    instance = calloc( 1, sizeof( *instance ) );
    if ( !instance )
    {
        UTILS_ERROR_POSIX();
        return NULL;
    }

    /* Initialization */
    instance->hashtab = hashtab;
    instance->index   = 0;
    instance->item    = NULL;

    return instance;
}

void
otf2_hash_table_iterator_free( otf2_hash_table_iterator* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    /* Release iterator */
    free( instance );
}


/*--- Element access ------------------------------------------------------*/

otf2_hash_table_entry*
otf2_hash_table_iterator_first( otf2_hash_table_iterator* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    /* Hash table empty? */
    if ( 0 == instance->hashtab->size )
    {
        return NULL;
    }

    /* Reset iterator */
    instance->index = 0;
    instance->item  = NULL;

    /* Search first list entry */
    while ( instance->hashtab->table[ instance->index ] == NULL &&
            instance->index < instance->hashtab->tabsize )
    {
        instance->index++;
    }
    instance->item = instance->hashtab->table[ instance->index ];

    return &instance->item->entry;
}

otf2_hash_table_entry*
otf2_hash_table_iterator_next( otf2_hash_table_iterator* instance )
{
    /* Validate arguments */
    UTILS_ASSERT( instance );

    /* No more entries? */
    if ( instance->item == NULL )
    {
        return NULL;
    }

    /* Search next entry */
    instance->item = instance->item->next;
    while ( instance->item == NULL )
    {
        instance->index++;
        if ( instance->index == instance->hashtab->tabsize )
        {
            return NULL;
        }

        instance->item = instance->hashtab->table[ instance->index ];
    }

    return &instance->item->entry;
}

int32_t
otf2_hash_table_compare_strings( const void* key,
                                 const void* item_key )
{
    return strcmp( ( const char* )key, ( const char* )item_key );
}

int32_t
otf2_hash_table_compare_int8( const void* key,
                              const void* item_key )
{
    return *( const int8_t* )key == *( const int8_t* )item_key ? 0 : 1;
}

int32_t
otf2_hash_table_compare_int16( const void* key,
                               const void* item_key )
{
    return *( const int16_t* )key == *( const int16_t* )item_key ? 0 : 1;
}

int32_t
otf2_hash_table_compare_int32( const void* key,
                               const void* item_key )
{
    return *( const int32_t* )key == *( const int32_t* )item_key ? 0 : 1;
}

int32_t
otf2_hash_table_compare_int64( const void* key,
                               const void* item_key )
{
    return *( const int64_t* )key == *( const int64_t* )item_key ? 0 : 1;
}

int32_t
otf2_hash_table_compare_uint8( const void* key,
                               const void* item_key )
{
    return *( const uint8_t* )key == *( const uint8_t* )item_key ? 0 : 1;
}

int32_t
otf2_hash_table_compare_uint16( const void* key,
                                const void* item_key )
{
    return *( const uint16_t* )key == *( const uint16_t* )item_key ? 0 : 1;
}

int32_t
otf2_hash_table_compare_uint32( const void* key,
                                const void* item_key )
{
    return *( const uint32_t* )key != *( const uint32_t* )item_key;
}

int32_t
otf2_hash_table_compare_uint64( const void* key,
                                const void* item_key )
{
    return *( const uint64_t* )key != *( const uint64_t* )item_key;
}

int32_t
otf2_hash_table_compare_pointer( const void* key,
                                 const void* item_key )
{
    return key == item_key ? 0 : 1;
}

size_t
otf2_hash_table_hash_string( const void* key )
{
    const unsigned char* str  = ( const unsigned char* )key;
    size_t               hash = 0;

    /* one-at-a-time hashing */
    while ( *str != 0 )
    {
        hash += *str++;
        hash += hash << 10;
        hash ^= hash >>  6;
    }

    hash += hash <<  3;
    hash ^= hash >> 11;
    hash += hash << 15;

    return hash;
}

size_t
otf2_hash_table_hash_uint8( const void* key )
{
    return ( uint32_t )( *( uint8_t* )key ) * UINT32_C( 2654435761l );
}

size_t
otf2_hash_table_hash_uint16( const void* key )
{
    return ( uint32_t )( *( uint16_t* )key ) * UINT32_C( 2654435761l );
}

size_t
otf2_hash_table_hash_uint32( const void* key )
{
    return *( uint32_t* )key * UINT32_C( 2654435761l );
}

size_t
otf2_hash_table_hash_uint64( const void* key )
{
    return *( uint64_t* )key * UINT64_C( 11400714819323199488 );
}

size_t
otf2_hash_table_hash_pointer( const void* key )
{
    if ( sizeof( size_t ) == sizeof( uint64_t ) )
    {
        return ( ( ( size_t )key ) / 8 ) * UINT64_C( 11400714819323199488 );
    }
    else
    {
        return ( ( ( size_t )key ) / 8 ) * UINT32_C( 2654435761l );
    }
}

void
otf2_hash_table_delete_free( void* item )
{
    free( item );
}

void
otf2_hash_table_delete_none( void* item )
{
    ( void )item;
}

void
otf2_hash_table_delete_pointer( void* item )
{
    free( ( void** )item );
}
