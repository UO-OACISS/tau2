/****************************************************************************
**			TAU Portable Profiling Package			                       **
**			http://www.cs.uoregon.edu/research/tau	                       **
*****************************************************************************
**    Copyright 2010  						   	                           **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		    : TauMetaData.h                                        **
**	Description 	: TAU Profiling Package                                **
**	Contact		    : tau-bugs@cs.uoregon.edu                              **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau           **
**                                                                         **
**  Description     : This file contains metadata related structures       **
**                                                                         **
****************************************************************************/


#ifndef _TAU_METADATA_TYPES_H_
#define _TAU_METADATA_TYPES_H_

typedef enum {
TAU_METADATA_TYPE_STRING, 
TAU_METADATA_TYPE_INTEGER, 
TAU_METADATA_TYPE_DOUBLE, 
TAU_METADATA_TYPE_OBJECT, 
TAU_METADATA_TYPE_ARRAY, 
TAU_METADATA_TYPE_TRUE, 
TAU_METADATA_TYPE_FALSE, 
TAU_METADATA_TYPE_NULL } Tau_metadata_type_t;

// forward declare the metadata structures
struct tau_metadata_object;
struct tau_metadata_array;
struct tau_metadata_value;

// the actual metadata structure, can be nested.
// The object is an unordered list of name, value pairs
typedef struct tau_metadata_object {
  int count; // number of pairs
  char** names; // array of names
  struct tau_metadata_value** values; // array of values
} Tau_metadata_object_t;

// an array to store array values
// the array is an ordered list of values
typedef struct tau_metadata_array {
  int length; // array length
  struct tau_metadata_value** values; // array of pointers to values
} Tau_metadata_array_t;

// a struct to store the value
typedef union tau_metadata_union {
  char*  cval;                       // string
  int    ival;                       // integer number
  double dval;                       // floating point number
  struct tau_metadata_object* oval;  // object
  struct tau_metadata_array* aval;   // array 
  // true, false, null are handled by the type enumeration.
} Tau_metadata_union_t;

// a struct to store the value
typedef struct tau_metadata_value {
  Tau_metadata_type_t type;
  Tau_metadata_union_t data;
} Tau_metadata_value_t;

/* struct GpuThread */
/* { */
/*   // int device_id;     // gpu num */
/*   unsigned int sys_tid;     // pthread */
/*   int gpu_tid;     // size of map_cudathread at time of insert */
/*   unsigned int parent_tid; */
/*   int node_id; */
/* }; */

#endif /* _TAU_METADATA_TYPES_H_ */
