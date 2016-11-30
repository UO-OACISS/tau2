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

#ifndef OTF2_VECTOR_H
#define OTF2_VECTOR_H

/*--- Header file documentation -------------------------------------------*/
/**
 * @file
 * @ingroup         otf2_vector_module
 *
 * @brief           A STL-like C-implementation of a vector.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#  define EXTERN extern "C"
#else
#  define EXTERN extern
#endif

/*--- Module documentation for the vector ---------------------------------*/

/**
 *  @defgroup otf2_vector_module SCOREP Vector
 *
 *  This module provides type definitions and function prototypes for a generic
 *  dynamic array data structure. Although its programming interface tries to
 *  resemble the interface of the C++ STL class template @c Vector, this
 *  implementation can only store @c void pointers as its elements due to the
 *  lacking template support in C.
 *
 *  Nevertheless, the otf2_vector module follows an object-oriented style. That
 *  is, each function (except the two @c create functions) takes a pointer to
 *  an instance of type @ref otf2_vector as their first argument, which is the
 *  object (i.e., the data structure) they operate on.
 *
 *  @note This module uses the @ref UTILS_ASSERT() macro to check various conditions
 *        (especially the values of given parameters) at runtime, which can
 *        cause a performance penalty.
 *
 *   @{
 */

/*--- Type definitions ----------------------------------------------------*/

/** Opaque data structure representing an dynamic array. */
typedef struct otf2_vector_struct otf2_vector;

/**
 * Pointer-to-function type describing comparison functions that can be used
 * with otf2_vector_find() or otf2_vector_lower_bound(). It has to return 0 if
 * @a value equals @a item, a negative value if @a value is less than @a item,
 * or a positive value if @a value is greater than @a item.
 *
 * @param value Searched value
 * @param item  Current array element
 *
 * @return Zero if @a value is equal to @a item, a negative value if @a value
 *         @< @a item, and a positive value if @a value @> @a item.
 */
typedef int8_t ( *otf2_vector_compare_func )( const void* value,
                                              const void* item );

/**
 * Pointer-to-function type describing unary processing functions that can
 * be used with otf2_vector_foreach(). Here, the current array element will
 * be passed as parameter @a item.
 *
 * @param item Current array element
 */
typedef void ( *otf2_vector_processing_func )( void* item );


/*
 * ---------------------------------------------------------------------------
 *  otf2_vector
 * ---------------------------------------------------------------------------
 */

/**
 * Creates and returns an empty instance of otf2_vector with an initial
 * capacity of zero elements. If the memory allocation request cannot
 * be fulfilled, an error message is printed and NULL is returned.
 *
 * @return Pointer to new instance or NULL if the memory could not be allocated.
 */
EXTERN otf2_vector*
otf2_vector_create();

/**
 * Creates and returns an instance of otf2_vector with the given initial
 * @a capacity.  If the memory allocation request cannot be fulfilled,
 * an error message is printed and NULL is returned.
 *
 * @param capacity Initial capacity
 *
 * @return Pointer to new instance or NULL if the memory could not be allocated.
 */
EXTERN otf2_vector*
otf2_vector_create_size( size_t capacity );

/**
 * Destroys the given @a instance of otf2_vector and releases the allocated
 * memory.
 *
 * @note Similar to the destructor of C++ STL Vector's, this call does not
 *       free the memory occupied by the elements since only pointers are
 *       stored. This has to be done separately.
 *
 * @param instance Object to be freed
 */
EXTERN void
otf2_vector_free( otf2_vector* instance );

/*--- Size operations -----------------------------------------------------*/

/**
 * Returns the actual number of elements stored in the given otf2_vector
 * @a instance.
 *
 * @param instance Queried object
 *
 * @return Number of elements stored
 */
EXTERN size_t
otf2_vector_size( const otf2_vector* instance );

/**
 * Returns whether the given otf2_vector @a instance is empty.
 *
 * @param instance Queried object
 *
 * @return Non-zero value if instance if empty; zero otherwise
 */
EXTERN int
otf2_vector_empty( const otf2_vector* instance );

/*--- Capacity operations -------------------------------------------------*/

/**
 * Returns the current capacity of the given otf2_vector @a instance.
 *
 * @param instance Queried object
 *
 * @return Current capacity
 */
EXTERN size_t
otf2_vector_capacity( const otf2_vector* instance );

/**
 * Resizes the otf2_vector @a instance to provide the given @a capacity. If
 * the memory reallocation request cannot be fulfilled, an error message
 * is printed.
 *
 * @note It is only possible to increase the size of an dynamic array.
 *       If the current size is equal to or larger than the requested
 *       size, the function call has no effect.
 *
 * @param instance Object to be resized
 * @param capacity New capacity
 *
 * @return Zero is returned in case that the allocation of memory failed. Else, a
 *         non-zero value is returned.
 */
EXTERN int32_t
otf2_vector_reserve( otf2_vector* instance,
                     size_t       capacity );

/**
 * Resizes the otf2_vector @a instance to provide the given @a size. Newly
 * created entries will be initialized with @c NULL. If the memory
 * reallocation request cannot be fulfilled, an error message is printed.
 *
 * @note It is only possible to increase the size of an dynamic array.
 *       If the current size is equal to or larger than the requested
 *       size, the function call has no effect.
 *
 * @param instance Object to be resized
 * @param size     New size
 *
 * @return  Zero is returned in case that the allocation of memory failed. Else, a
 *          non-zero value is returned.
 */
EXTERN int32_t
otf2_vector_resize( otf2_vector* instance,
                    size_t       size );

/*--- Element access ------------------------------------------------------*/

/**
 * Returns the element stored at entry @a index in the given otf2_vector
 * @a instance.
 *
 * @param instance Queried object
 * @param index    Index of the element
 *
 * @return Requested element
 */
EXTERN void*
otf2_vector_at( const otf2_vector* instance,
                size_t             index );

/**
 * Stores the given @a item at position @a index in the otf2_vector @a instance.
 *
 * @param instance Object to be modified
 * @param index    Index where the item should be stored
 * @param item     Item to be stored
 */
EXTERN void
otf2_vector_set( otf2_vector* instance,
                 size_t       index,
                 void*        item );


/**
 * Returns the first element stored in the given otf2_vector @a instance.
 *
 * @param instance Queried object
 *
 * @return First element
 */
EXTERN void*
otf2_vector_front( const otf2_vector* instance );

/**
 * Returns the last element stored in the given otf2_vector @a instance.
 *
 * @param instance Queried object
 *
 * @return Last element
 */
EXTERN void*
otf2_vector_back( const otf2_vector* instance );

/*--- Iterator functions --------------------------------------------------*/

/**
 * Returns an "iterator" for (i.e., an pointer to) the first element stored
 * in the given otf2_vector @a instance.
 *
 * @param instance Queried object
 *
 * @return Iterator for first element
 */
EXTERN void**
otf2_vector_begin( const otf2_vector* instance );

/**
 * Returns an "iterator" for (i.e., an pointer to) the position after the
 * last element stored in the given otf2_vector @a instance.
 *
 * @param instance Queried object
 *
 * @return Iterator for the position after the last element
 */
EXTERN void**
otf2_vector_end( const otf2_vector* instance );

/*--- Inserting & removing elements ---------------------------------------*/

/**
 * Appends the given @a item (which can also be a @c NULL pointer) at the
 * end of the otf2_vector @a instance. If the current capacity does not suffice,
 * the data structure is automatically resized. If this memory reallocation
 * request cannot be fulfilled, an error message is printed.
 *
 * @param instance Object to which the item should be appended
 * @param item     Item to append
 *
 * @return  Zero is returned in case that the allocation of memory failed. on Success, a
 *          non-zero value is returned.
 */
EXTERN int32_t
otf2_vector_push_back( otf2_vector* instance,
                       void*        item );

/**
 * Removes the last element stored in the given otf2_vector @a instance.
 * However, it does not release the memory occupied by the item itself.
 *
 * @param instance Object from which the item should be removed
 */
EXTERN void
otf2_vector_pop_back( otf2_vector* instance );

/**
 * Inserts the given @a item (which can also be a @c NULL pointer) at the
 * given position @a index in the otf2_vector @a instance. If the current
 * capacity does not suffice, the data structure is automatically resized.
 * If this memory reallocation request cannot be fulfilled, an error message
 * is printed and the program is aborted.
 *
 * @param instance Object to which the item should be inserted
 * @param index    Index where item should be inserted
 * @param item     Item to insert
 *
 * @return  Zero is returned in case that the allocation of memory failed. on Success, a
 *          non-zero value is returned.
 */
EXTERN int32_t
otf2_vector_insert( otf2_vector* instance,
                    size_t       index,
                    void*        item );

/**
 * Removes the element stored at entry @a index in the given otf2_vector
 * @a instance. However, it does not release the memory occupied by the
 * item itself.
 *
 * @param instance Object from which the item should be removed
 * @param index    Index of item to remove
 */
EXTERN void
otf2_vector_erase( otf2_vector* instance,
                   size_t       index );

/**
 * Removes all elements stored in the given otf2_vector @a instance. However,
 * it does not release the memory occupied by the items themselves.
 *
 * @param instance Object to remove items from
 */
EXTERN void
otf2_vector_clear( otf2_vector* instance );

/*--- Algorithms ----------------------------------------------------------*/

/**
 * Searches for the first occurence of @a value in the given otf2_vector
 * @a instance, using the binary comparison function @a compareFunc (see
 * @ref otf2_vector_compare_func for implementation details). If a matching
 * item could
 * be found, a non-zero value is returned. In addition, if @a index is
 * non-@c NULL, the index of the corresponding entry is stored in the memory
 * location pointed to by @a index. Otherwise, i.e., if no matching item
 * could be found, this function returns zero.
 *
 * @param instance    Object in which the item is searched
 * @param value       Item to search for
 * @param compareFunc Binary comparison function
 * @param index       Memory location where index of matching item is stored
 *                    (ignored if @c NULL)
 *
 * @return Non-zero if matching item cound be found; zero otherwise
 */
EXTERN int32_t
otf2_vector_find( const otf2_vector*       instance,
                  const void*              value,
                  otf2_vector_compare_func cmpfunc,
                  size_t*                  index );

/**
 * Determines the index of the first element that has a value greater than or
 * equal to the given @a value in the otf2_vector @a instance, using the binary
 * comparison function @a compareFunc (see @ref otf2_vector_compare_func for implementation
 * details). If a matching item could be found, a non-zero value is returned.
 * In addition, if @a index is non-@c NULL, the index of the corresponding
 * entry is stored in the memory location pointed to by @a index. Otherwise,
 * i.e., if no matching item could be found, this function returns zero.
 *
 * This function can be used to maintain sorted dynamic arrays. If a new item shall be
 * inserted, the new item's position can be determined with @ref otf2_vector_lower_bound
 * or @ref otf2_vector_upper_bound and afterwards inserted at this position.
 *
 * @note For this algorithm to work correctly, the elements in the given
 *       otf2_vector instance have to be sorted in ascending order with respect
 *       to the binary comparison function @a compareFunc.
 *
 * @param instance     Object in which the item is searched
 * @param value        Item to search for
 * @param compareFunc  Binary comparison function
 * @param index        Memory location where index of matching item is stored
 *                     (ignored if @c NULL)
 *
 * @return Non-zero if matching item cound be found; zero otherwise
 */
EXTERN int32_t
otf2_vector_lower_bound( const otf2_vector*       instance,
                         const void*              value,
                         otf2_vector_compare_func cmpfunc,
                         size_t*                  index );

/**
 * Determines the index of the first element that has a value greater than the
 * given @a value in the otf2_vector @a instance, using the binary comparison
 * function @a compareFunc (see @ref otf2_vector_compare_func for implementation details). If
 * a matching item could be found, a non-zero value is returned. In addition,
 * if @a index is non-@c NULL, the index of the corresponding entry is stored
 * in the memory location pointed to by @a index. Otherwise, i.e., if no
 * matching item could be found, this function returns zero.
 *
 * This function can be used to maintain sorted dynamic arrays. If a new item shall be
 * inserted, the new item's position can be determined with @ref otf2_vector_lower_bound
 * or @ref otf2_vector_upper_bound and afterwards inserted at this position.
 *
 * @note For this algorithm to work correctly, the elements in the given
 *       otf2_vector instance have to be sorted in ascending order with respect
 *       to the binary comparison function @a compareFunc.
 *
 * @param instance     Object in which the item is searched
 * @param value        Item to search for
 * @param compareFunc  Binary comparison function
 * @param index        Memory location where index of matching item is stored
 *                    (ignored if @c NULL)
 *
 * @return Non-zero if matching item cound be found; zero otherwise
 */
EXTERN int32_t
otf2_vector_upper_bound( const otf2_vector*       instance,
                         const void*              value,
                         otf2_vector_compare_func cmpfunc,
                         size_t*                  index );

/**
 * Calls the unary processing function @a procfunc for each element of
 * the given otf2_vector @a instance.
 *
 * @param instance Object whose entries should be processed
 * @param procfunc Unary processing function
 */
EXTERN void
otf2_vector_foreach( const otf2_vector*          instance,
                     otf2_vector_processing_func procfunc );

/** @} */


#endif   /* !OTF2_VECTOR_H */
