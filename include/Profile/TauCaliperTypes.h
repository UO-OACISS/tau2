/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.cs.uoregon.edu/research/tau             **
 * *****************************************************************************
 * **    Copyright 1997-2017                                                  **
 * **    Department of Computer and Information Science, University of Oregon **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauCaliperTypes.h                                **
 * **      Description     : Type definitions for TAU-CALIPER integration     **
 * **      Contact         : sramesh@cs.uoregon.edu                           **
 * **      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
 * ***************************************************************************/

#ifndef _TAU_CALIPER_TYPES_H_
#define _TAU_CALIPER_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t cali_id_t;

#define CALI_INV_ID 0xFFFFFFFFFFFFFFFF

/**
 * \brief Data type of an attribute.
 */
typedef enum {
  CALI_TYPE_INV    = 0, /**< Invalid type               */
  CALI_TYPE_USR    = 1, /**< User-defined type (pointer to binary data) */
  CALI_TYPE_INT    = 2, /**< 64-bit signed integer      */
  CALI_TYPE_UINT   = 3, /**< 64-bit unsigned integer    */
  CALI_TYPE_STRING = 4, /**< String (\a char*)          */
  CALI_TYPE_ADDR   = 5, /**< 64-bit address             */
  CALI_TYPE_DOUBLE = 6, /**< Double-precision floating point type */
  CALI_TYPE_BOOL   = 7, /**< C or C++ boolean           */
  CALI_TYPE_TYPE   = 8  /**< Instance of cali_attr_type */
} cali_attr_type;

#define CALI_MAXTYPE CALI_TYPE_TYPE

/**
 * \brief 
 */
const char* 
cali_type2string(cali_attr_type type);

cali_attr_type 
cali_string2type(const char* str);

/**
 * \brief Attribute property flags.
 *
 * These flags control how the caliper runtime system handles the
 * associated attributes. Flags can be combined with a bitwise OR
 * (however, the scope flags are mutually exclusive).
 */
typedef enum {
  /** \brief Default value */
  CALI_ATTR_DEFAULT       =  0,

  /** 
   * \brief Store directly as key:value pair, not in the context tree.
   *
   * Attributes with this property will be not be put into the context
   * tree, but stored directly as key:value pairs on the blackboard
   * and in snapshot records. ASVALUE attributes cannot be
   * nested. Only applicable to scalar data types.
   */
  CALI_ATTR_ASVALUE       =  1,
  /** \brief Create a separate context tree root node for this attribute. */
  CALI_ATTR_NOMERGE       =  2,
  /** \brief Process-scope attribute. Shared between all threads. */
  CALI_ATTR_SCOPE_PROCESS = 12, /* make scope flags mutually exclusive when &'ed with SCOPE_MASK */
  /** \brief Thread-scope attribute. */
  CALI_ATTR_SCOPE_THREAD  = 20, 
  /** \brief Task-scope attribute. Currently unused. */
  CALI_ATTR_SCOPE_TASK    = 24,

  /** \brief Skip event callbacks for blackboard updates with this attribute */
  CALI_ATTR_SKIP_EVENTS   = 64,

  /** \brief Do not include this attribute in snapshots */
  CALI_ATTR_HIDDEN        = 128,

  /** \brief Begin/end calls are properly aligned with the call stack.
   * 
   * Indicates that \a begin/end calls for this attribute are
   * correctly nested with the call stack and other NESTED attributes.
   * That is, an active region of a NESTED attribute does not
   * partially overlap function calls or other NESTED attribute
   * regions.
   */
  CALI_ATTR_NESTED        = 256
} cali_attr_properties;

#define CALI_ATTR_SCOPE_MASK 60

/**
 * \brief  Provides descriptive string of given attribute property flags, separated with ':'
 * \param  prop Attribute property flag
 * \param  buf  Buffer to write string to
 * \param  len  Length of string buffer
 * \return      -1 if provided buffer is too short; length of written string otherwise
 */  
int
cali_prop2string(int prop, char* buf, size_t len);

int
cali_string2prop(const char*);
  
typedef enum {
  CALI_OP_SUM = 1,
  CALI_OP_MIN = 2,
  CALI_OP_MAX = 3
} cali_op;

typedef enum {
  CALI_SUCCESS = 0,
  CALI_EBUSY,
  CALI_ELOCKED,
  CALI_EINV,
  CALI_ETYPE,
  CALI_ESTACK
} cali_err;

/** The variant struct manages values of different types in Caliper.    
 *  Types with fixed size (i.e., numeric types) are stored in the variant directly.
 *  Variable-length types (strings and blobs) are stored as unmanaged pointers. 
 */    
typedef struct {
    /** Least significant bytes encode the type.
     *  Remaining bytes encode the size of variable-length types (strings and blobs (usr)).
     */
    uint64_t type_and_size;
    
    /** Value in various type representations
     */
    union {
        bool           v_bool;
        double         v_double;
        int            v_int;
        uint64_t       v_uint;
        cali_attr_type v_type;
        const void*    unmanaged_ptr;
    }        value;
} cali_variant_t;

#define CALI_VARIANT_TYPE_MASK 0xFF

inline cali_variant_t
cali_make_empty_variant()
{
    cali_variant_t v = { 0, { .v_uint = 0 } };
    return v;
}
    
/** \brief Test if variant is empty
 */
inline bool
cali_variant_is_empty(cali_variant_t v)
{
    return 0 == v.type_and_size;
}
  
/** \brief Return type of a variant  
 */
cali_attr_type
cali_variant_get_type(cali_variant_t v);

/** \brief Return size of the variant's value
 */
size_t
cali_variant_get_size(cali_variant_t v);

/** \brief Get a pointer to the variant's data
 */
const void*
cali_variant_get_data(const cali_variant_t* v);

/** \brief Construct variant from type, pointer, and size 
 */
cali_variant_t
cali_make_variant(cali_attr_type type, const void* ptr, size_t size);


inline cali_variant_t
cali_make_variant_from_bool(bool value)
{
    cali_variant_t v = { CALI_TYPE_BOOL, { .v_uint = 0 } };  /* set to zero */
    v.value.v_bool = value;    
    return v;
}
    
inline cali_variant_t
cali_make_variant_from_int(int value)
{
    cali_variant_t v = { CALI_TYPE_INT, { .v_uint = 0 } };  /* set to zero */
    v.value.v_int = value;    
    return v;
}

inline cali_variant_t
cali_make_variant_from_uint(uint64_t value)
{
    cali_variant_t v = { CALI_TYPE_UINT, { .v_uint = value } };
    return v;
}

inline cali_variant_t
cali_make_variant_from_double(double value)
{
    cali_variant_t v = { CALI_TYPE_DOUBLE, { .v_double = value } };
    return v;
}

inline cali_variant_t
cali_make_variant_from_type(cali_attr_type value)
{
    cali_variant_t v = { CALI_TYPE_TYPE, { .v_uint = 0 } }; /* set to zero */
    v.value.v_type = value;
    return v;
}

/********** End CALIPER Type Definitions****************/

enum Type {INTEGER, DOUBLE, STRING};
union Data {
  int as_integer;
  double as_double;
  char str[100];
};

struct StackValue {
  Type type;
  Data data;
};



#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_CALIPER_TYPES_H_ */

