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

#ifndef OTF2_HASH_TABLE_H
#define OTF2_HASH_TABLE_H

/*--- Header file documentation -------------------------------------------*/
/**
 * @file
 * @ingroup         otf2_hash_table_module
 *
 * @brief           This file provides the function header and type definiions for
 *                  a STL-like hash table.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#  define EXTERN extern "C"
#else
#  define EXTERN extern
#endif

/*--- Module documentation for the hash table -----------------------------*/
/**
 * @defgroup  otf2_hash_table_module SCOREP Hash Table
 *
 * This module provides type definitions and function prototypes for a generic
 * hash table data structure. The elements of such a hash table are stored in
 * no particular order as a key/value pair, where both items are represented
 * by @c void pointers (see otf2_hash_table_entry). Both pointers must point to
 * a valid memory location as long as the hash table entry is not deleted. If the
 * content of that memory location is changed, the content of the key/value will
 * change, too.
 *
 * In addition, this module defines an associated iterator data type
 * otf2_hash_table_iterator, which allows convenient traversal of all entries stored in
 * the hash table.
 *
 * The scorep_hashtab module follows an object-oriented style. That is, each
 * function (except the two @a create functions) takes a pointer to an
 * instance of either type @ref otf2_hash_table or @ref otf2_hash_table_iterator as their first
 * argument, which is the object (i.e., the data structure) they operate on.
 *
 * @note This module uses the @c UTILS_ASSERT() macro to check various conditions
 *       (especially the values of given parameters) at runtime, which can
 *       cause a performance penalty.
 *
 * @{
 */

/*--- Type definitions ----------------------------------------------------*/

/** Data structure representing key/value pairs stored in the hash table. */
typedef struct
{
    void* key;     /**< Unique key */
    void* value;   /**< Stored value */
} otf2_hash_table_entry;

/** Opaque data structure representing a hash table. */
typedef struct otf2_hash_table_struct otf2_hash_table;

/** Opaque data structure representing a hash table iterator. */
typedef struct otf2_hash_table_iter_struct otf2_hash_table_iterator;

/**
 * Pointer-to-function type describing hashing functions. The function has to
 * compute an integral index value for the given @a key. If the computed index
 * is larger than the size of the hash table, it will be automatically adjusted
 * using modulo arithmetics.
 *
 * @param key Key value
 *
 * @return entry Computed hash table index
 */
typedef size_t ( * otf2_hash_table_hash_func )( const void* key );

/**
 * Pointer-to-function type describing key comparison functions. It has to
 * return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 *
 * @param key      Key value to compare
 * @param item_key Key value of the current item
 * @return It has to
 * return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
typedef int32_t ( * otf2_hash_table_compare_func )( const void* key,
                                                    const void* item_key );

/**
 * Pointer-to-function type describing unary processing functions that can
 * be used with otf2_hash_table_foreach(). Here, the current key/value pair will
 * be passed as parameter @a entry.
 *
 * @param entry Hash table entry
 */
typedef void ( * otf2_hash_table_process_func )( otf2_hash_table_entry* entry );

/**
 * Pointer-to-functions type which frees the memory for data item. It is
 * used in otf2_hash_table_free_all() to free the memory of key and value items.
 *
 * @param item The data item which should be deleted.
 */
typedef void ( * otf2_hash_table_delete_func )( void* item );

/*
 * --------------------------------------------------------------------------
 * otf2_hash_table
 * --------------------------------------------------------------------------
 */

/**
 * Creates and returns an instance of otf2_hash_table. Besides the @a size of the
 * hash table, pointers to a hashing function as well as a key comparison
 * function have to be provided. If the memory allocation request cannot be
 * fulfilled, an error message is printed and the program is aborted.
 *
 * @param size     Size of the hash table
 * @param hashfunc Hashing function
 * @param kcmpfunc Key comparison function
 *
 * @return Pointer to new instance
 */
EXTERN otf2_hash_table*
otf2_hash_table_create_size( size_t                       size,
                             otf2_hash_table_hash_func    hashfunc,
                             otf2_hash_table_compare_func kcmpfunc );

/**
 * Destroys the given @a instance of otf2_hash_table and releases the allocated
 * memory.
 *
 * @note Similar to the @ref SCOREP_Vector data structure, this call does not
 *       free the memory occupied by the elements, i.e., keys and values,
 *       since only pointers are stored. This has to be done separately.
 *
 * @param instance Object to be freed
 */
EXTERN void
otf2_hash_table_free( otf2_hash_table* instance );

/* Size operations */

/**
 * Returns the actual number of elements stored in the given otf2_hash_table
 * @a instance.
 *
 * @param instance Queried object
 *
 * @return Number of elements stored
 */
EXTERN size_t
otf2_hash_table_size( const otf2_hash_table* instance );

/**
 * Returns whether the given otf2_hash_table @a instance is empty.
 *
 * @param instance Queried object
 *
 * @return Non-zero value if instance if empty; zero otherwise
 */
EXTERN int32_t
otf2_hash_table_empty( const otf2_hash_table* instance );

/* Inserting & removing elements */

/**
 * Inserts the given (@a key,@a value) pair into the otf2_hash_table @a instance.
 * In addition, the hash value (e.g., returned by a preceeding call of
 * otf2_hash_table_find()) can be specified. If the hash value should be
 * (re-)calculated instead, @c NULL should be passed.
 *
 * This function also has to allocate memory for internal data structures. If
 * this memory allocation request cannot be fulfilled, an error message is
 * printed and the program is aborted.
 *
 * @param instance   Object in which the key/value pair should be inserted
 * @param key        Unique key. The memory location to which it points must be
 *                   valid as long as the hash entry exists.
 * @param value      Associated value. The memory location to which it points must be
 *                   valid as long as the hash entry exists.
 * @param hashValPtr Optional storage where an hash value was previously stored
 *                   by a call to @a otf2_hash_table_find() with the same key @a
 *                   key. (ignored if @c NULL)
 */
EXTERN void
otf2_hash_table_insert( otf2_hash_table* instance,
                        void*            key,
                        void*            value,
                        size_t*          hashValPtr );

/* Algorithms */

/**
 * Searches for an hash table entry with the specified @a key in the
 * given otf2_hash_table @a instance, using the binary key comparison function
 * @a cmpfunc (see @ref otf2_hash_table_compare_func for implementation details). If a
 * matching item could be found, a pointer to the key/value pair is returned.
 * In addition, if @a hashValPtr is non-@c NULL, the hash value is stored in
 * the memory location pointed to by @a hashValPtr, which can be used in an
 * subsequent call to otf2_hash_table_insert(). Otherwise, i.e., if no
 * matching item could be found, this function returns @c NULL.
 *
 * @param instance   Object in which the item is searched
 * @param key        Unique key to search for
 * @param hashValPtr Storage where hash value of the key will be is stored. For
 *                   use to a later call to @a otf2_hash_table_insert() with the
 *                   same key @a key. (ignored if @c NULL)
 *
 * @return Pointer to hash table entry if matching item cound be found;
 *         @c NULL otherwise
 */
EXTERN otf2_hash_table_entry*
otf2_hash_table_find( const otf2_hash_table* instance,
                      const void*            key,
                      size_t*                hashValPtr );

/**
 * Calls the unary processing function @a procfunc for each element of
 * the given otf2_hash_table @a instance.
 *
 * @param instance Object whose entries should be processed
 * @param procfunc Unary processing function
 */
EXTERN void
otf2_hash_table_foreach( const otf2_hash_table*       instance,
                         otf2_hash_table_process_func procfunc );

/**
 * Removes the entry specified by @a key from the hash table @a instance.
 * The user must provide appropriate deletion funtion to free the memory allocated
 * for @a key and value.
 *
 * @param instance    Object in which the item is searched
 * @param key         Unique key to search for
 * @param hashValPtr  Storage where hash value of the key will be is stored. For
 *                    use to a later call to @a otf2_hash_table_insert() with the
 *                    same key @a key. (ignored if @c NULL)
 * @param deleteKey   Function pointer to a function which deletes the key
 *                    objects.
 * @param deleteValue Function pointer to a function which deletes the value
 *                    objects.
 */
EXTERN void
otf2_hash_table_remove( const otf2_hash_table*      instance,
                        const void*                 key,
                        otf2_hash_table_delete_func deleteKey,
                        otf2_hash_table_delete_func deleteValue,
                        size_t*                     hashValPtr );


/*
 * --------------------------------------------------------------------------
 * otf2_hash_table_iterator
 * --------------------------------------------------------------------------
 */

/* Construction & destruction */

/**
 * Creates and returns an iterator for the given otf2_hash_table @a instance. If
 * the memory allocation request cannot be fulfilled, an error message is
 * printed and the program is aborted.
 *
 * @return Pointer to new instance
 */
EXTERN otf2_hash_table_iterator*
otf2_hash_table_iterator_create( const otf2_hash_table* hashtab );

/**
 * Destroys the given @a instance of otf2_hash_table_iterator and releases the allocated
 * memory.
 *
 * @param instance Object to be freed
 */
EXTERN void
otf2_hash_table_iterator_free( otf2_hash_table_iterator* instance );

/* Element access */

/**
 * Returns a pointer to the first entry of the hash table which is
 * associated to the given iterator @a instance. If the hash table is
 * empty, @c NULL is returned.
 *
 * @param instance Iterator object
 *
 * @return Pointer to first entry of the hash table or @c NULL if empty
 */
EXTERN otf2_hash_table_entry*
otf2_hash_table_iterator_first( otf2_hash_table_iterator* instance );

/**
 * Returns a pointer to the next entry of the hash table which is
 * associated to the given iterator @a instance. If no next entry is
 * available, @c NULL is returned.
 *
 * @param instance Iterator object
 *
 * @return Pointer to next entry of the hash table or @c NULL if unavailable
 */
EXTERN otf2_hash_table_entry*
otf2_hash_table_iterator_next( otf2_hash_table_iterator* instance );

/*
 * --------------------------------------------------------------------------
 * Extensions
 * --------------------------------------------------------------------------
 */

/**
 * Destroys the given @a instance of otf2_hash_table and releases the allocated
 * memory.
 *
 * @note In difference to otf2_hash_table_free, this call does
 *       free the memory occupied by the elements, i.e., keys and values.
 *       For freeing keys and values the function pointers @a deleteKey and
 *       @a deleteValue are used
 *
 * @param instance    Object to be freed
 * @param deleteKey   Function pointer to a function which deletes the key
 *                    objects.
 * @param deleteValue Function pointer to a function which deletes the value
 *                    objects.
 */

EXTERN void
otf2_hash_table_free_all( otf2_hash_table*            instance,
                          otf2_hash_table_delete_func deleteKey,
                          otf2_hash_table_delete_func deleteValue );

/*
 * --------------------------------------------------------------------------
 * Default Comparison functions
 * --------------------------------------------------------------------------
 */

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to NULL-terminated strings. It uses strcmp() for comparison.
 *
 * @param key      Pointer to a NULL-terminated string.
 * @param item_key Pointer to a NULL-terminated string that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_strings( const void* key,
                                 const void* item_key );

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to 8-bit integers.
 *
 * @param key      Pointer to a 8-bit integer.
 * @param item_key Pointer to a 8-bit integer that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_int8( const void* key,
                              const void* item_key );

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to 16-bit integers.
 *
 * @param key      Pointer to a 16-bit integer.
 * @param item_key Pointer to a 16-bit integer that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_int16( const void* key,
                               const void* item_key );

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to 32-bit integers.
 *
 * @param key      Pointer to a 32-bit integer.
 * @param item_key Pointer to a 32-bit integer that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_int32( const void* key,
                               const void* item_key );

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to 64-bit integers.
 *
 * @param key      Pointer to a 64-bit integer.
 * @param item_key Pointer to a 64-bit integer that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_int64( const void* key,
                               const void* item_key );

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to unsigned 8-bit integers.
 *
 * @param key      Pointer to a 8-bit integer.
 * @param item_key Pointer to a 8-bit integer that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_uint8( const void* key,
                               const void* item_key );

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to unsigned 16-bit integers.
 *
 * @param key      Pointer to a 16-bit integer.
 * @param item_key Pointer to a 16-bit integer that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_uint16( const void* key,
                                const void* item_key );

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to unsigned 32-bit integers.
 *
 * @param key      Pointer to a unsigned 32-bit integer.
 * @param item_key Pointer to a unsigned 32-bit integer that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_uint32( const void* key,
                                const void* item_key );

/**
 * A comparison function for the hash table which interpretes the keys as pointers
 * to unsigned 64-bit integers.
 *
 * @param key      Pointer to a unsigned 64-bit integer.
 * @param item_key Pointer to a unsigned 64-bit integer that is compared to @a key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_uint64( const void* key,
                                const void* item_key );

/**
 * A comparison function for the hash table which compares the pointers.
 *
 * @param key      Comparison pointer.
 * @param item_key Pointer of the key.
 * @return 0 if the given @a key equals the key of the current item (@a item_key)
 * or a non-zero value otherwise.
 */
EXTERN int32_t
otf2_hash_table_compare_pointer( const void* key,
                                 const void* item_key );

/*
 * --------------------------------------------------------------------------
 * Default hash functions
 * --------------------------------------------------------------------------
 */

/**
 * Computes a hash value from a NULL-terminated string. It interprets the key as a
 * pointer to a NULL-terminated string and sums up all letters of the string.
 *
 * @param key A pointer to a NULL-terminated string which serves as key.
 * @return A hashvalue which is computed as sum of the letters of the string.
 */
EXTERN size_t
otf2_hash_table_hash_string( const void* key );

size_t
otf2_hash_table_hash_uint8( const void* key );

size_t
otf2_hash_table_hash_uint16( const void* key );

/**
 * Default hash function for 32-integer values. It just returns the integer value.
 *
 * @param key A pointer to a 32-bit integer.
 * @return The value of the key.
 */
EXTERN size_t
otf2_hash_table_hash_uint32( const void* key );

/**
 * Default hash function for 64-integer values. It just returns the integer value.
 *
 * @param key A pointer to a 64-bit integer.
 * @return The value of the key.
 */
EXTERN size_t
otf2_hash_table_hash_uint64( const void* key );

/**
 * Default hash function. It uses the adress/8 as hash value. The division by
 * 8 is done to avoid gaps because of wordsize alignments.
 *
 * @param key A pointer.
 * @return The value of the key.
 */
EXTERN size_t
otf2_hash_table_hash_pointer( const void* key );

/*
 * --------------------------------------------------------------------------
 * Default deletion functions
 * --------------------------------------------------------------------------
 */

/**
 * Default deletion function as requested otf2_hash_table_free_all() which frees
 * the item.
 *
 * @param item Pointer to the item which is freed.
 */
EXTERN void
otf2_hash_table_delete_free( void* item );

/**
 * Default deletion function as requested otf2_hash_table_free_all() which
 * Does nothing. It can be used in case that only one item from the key/value
 * pair must be freed. The one which should not be freed is handled with this
 * function.
 *
 * @param item Pointer to the item which should not be freed.
 */
EXTERN void
otf2_hash_table_delete_none( void* item );

/**
 * Default deletion function for pointers to pointers as requested for
 * otf2_hash_table_free_all().
 *
 * @param item Pointer to the item which is freed.
 */
EXTERN void
otf2_hash_table_delete_pointer( void* item );

/** @} */

#endif   /* !OTF2_HASH_TABLE_H */
