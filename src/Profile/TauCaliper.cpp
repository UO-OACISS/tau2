/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.cs.uoregon.edu/research/tau             **
 * *****************************************************************************
 * **    Copyright 1997-2017                                                  **
 * **    Department of Computer and Information Science, University of Oregon **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauCaliper.cpp                                   **
 * **      Description     : Wrapper library for CALIPER calls                **
 * **      Contact         : sramesh@cs.uoregon.edu                           **
 * **      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
 * ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <stack>
#include <TAU.h>

#include <Profile/TauCaliperTypes.h>
#include <Profile/TAU_CALIPER.h>

//Global data structures
std::map<std::string, cali_id_t> _attribute_name_map_;
std::map<std::string, cali_attr_type> _attribute_type_map_name_key;
std::map<cali_id_t, cali_attr_type> _attribute_type_map_id_key;
std::map<cali_id_t, std::string> _attribute_id_map_;

cali_id_t current_id;
std::map<std::string, std::stack<StackValue> > attribute_stack;

//Externs
extern "C" int Tau_init_initializeTAU();
extern "C" void Tau_trigger_userevent(const char *name, double data);

/**
  * \brief Initialize Caliper.
  *
  * Typically, it is not necessary to initialize Caliper explicitly.
  * Caliper will lazily initialize itself on the first Caliper API call.
  * This function is used primarily by the Caliper annotation macros,
  * to ensure that Caliper's pre-defined annotation attributes are 
  * initialized.
  * It can also be used to avoid high initialization costs in the first
  * Caliper API call.
  *
  * TAU Wrapper: Calls TAU initialize, and initializes the "id" generation variable to 0
  */

int cali_tau_initialized = 0;

extern "C" void cali_init() {

  TAU_VERBOSE("TAU: CALIPER init invoked.\n");

 
  RtsLayer::LockEnv(); 

  //Initialize the ID field to 0. This is the dummy id generation mechanism that we 
  current_id = 0;

  if(Tau_init_initializeTAU()) {

    fprintf(stderr, "TAU: Initialization from within Caliper wrapper failed\n");
  }

  cali_tau_initialized = 1;

  RtsLayer::UnLockEnv();
}

/*
 * --- Instrumentation API -----------------------------------
 */

/**
 * \addtogroup AnnotationAPI
 * \{
 * \name Low-level source-code annotation API
 * \{
 */

/**
 * \brief Put attribute attr on the blackboard. 
 * Parameters:
 * \param attr An attribute of type CALI_TYPE_BOOL
 */
/* TAU Wrapper: Create and start a timer provided  that the attribute with the given ID exists*/
cali_err cali_begin(cali_id_t  attr) {

  std::map<cali_id_t, std::string>::iterator it = _attribute_id_map_.find(attr);
  if(it == _attribute_id_map_.end()) {
    fprintf(stderr, "TAU: CALIPER: Not a valid attribute ID. Please use cali_create_attribute to generate an attribute of type STRING, and then pass the generate ID to %s.\n", cali_begin);
    return CALI_EINV;
  }

  RtsLayer::LockEnv();
  
  //Sanity check for the type of the attribute
  if(_attribute_type_map_id_key[attr] != CALI_TYPE_STRING) {
    return CALI_ETYPE;
  }

  if(!cali_tau_initialized)
    cali_init();

  const char *name = it->second.c_str();

  TAU_VERBOSE("TAU: CALIPER create and start a TAU static timer with name: %s\n", name);
  TAU_START(name);

  RtsLayer::UnLockEnv();

  return CALI_SUCCESS;

}


/**
 * Add \a val for attribute \a attr to the blackboard.
 * The new value is nested under the current value of \a attr. 
 */

cali_err cali_begin_double(cali_id_t attr, double val) {

  std::map<cali_id_t, std::string>::iterator it = _attribute_id_map_.find(attr);
  if(it == _attribute_id_map_.end()) {
    fprintf(stderr, "TAU: CALIPER: Not a valid attribute ID. Please use cali_create_attribute to generate an attribute of type DOUBLE, and then pass the generate ID to %s.\n", cali_begin_double);
    return CALI_EINV;
  }

  /*We do not support "stacking" semantics for UserEvents of double and integer types*/
  RtsLayer::LockEnv();
  const char* attr_name = it->second.c_str();

  if(!attribute_stack[std::string(attr_name)].empty()) {
    fprintf(stderr, "TAU: CALIPER operation: %s not supported for this attribute type. TAU UserEvent has already been created for %s. Use cali_set_double instead to update the value\n", cali_begin_double, attr_name);

    RtsLayer::UnLockEnv();

    return CALI_EINV;
  }

  //Create the attribute if it hasn't already been created explicitly
  RtsLayer::UnLockEnv();

  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_DOUBLE) {
    return CALI_ETYPE;
  }

  RtsLayer::LockEnv();


  TAU_VERBOSE("TAU: CALIPER create a TAU UserEvent named %s\n of double type\n", attr_name);
  Tau_trigger_userevent(attr_name, val);

  StackValue value;
  value.type = DOUBLE;
  value.data.as_double = val;
  attribute_stack[std::string(attr_name)].push(value);

  RtsLayer::UnLockEnv();

  return CALI_SUCCESS;
}

cali_err cali_begin_int(cali_id_t attr, int val) {

  std::map<cali_id_t, std::string>::iterator it = _attribute_id_map_.find(attr);
  if(it == _attribute_id_map_.end()) {
    fprintf(stderr, "TAU: CALIPER: Not a valid attribute ID. Please use cali_create_attribute to generate an attribute of type INTEGER, and then pass the generate ID to %s.\n", cali_begin_int);
    return CALI_EINV;
  }

  RtsLayer::LockEnv();
  const char* attr_name = it->second.c_str();

  if(!attribute_stack[std::string(attr_name)].empty()) {
    fprintf(stderr, "TAU: CALIPER operation: %s not supported for this attribute type. TAU UserEvent has already been created for %s. Use cali_set_int instead to update the value.\n", cali_begin_int, attr_name);

    RtsLayer::UnLockEnv();

    return CALI_EINV;
  }

  //Create the attribute if it hasn't already been created explicitly
  RtsLayer::UnLockEnv();

  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_INT) {
    return CALI_ETYPE;
  }

  RtsLayer::LockEnv();

  TAU_VERBOSE("TAU: CALIPER create a TAU UserEvent named %s\n of integer type\n", attr_name);
  Tau_trigger_userevent(attr_name, val);

  StackValue value;
  value.type = INTEGER;
  value.data.as_integer = val;
  attribute_stack[std::string(attr_name)].push(value);

  RtsLayer::UnLockEnv();

  return CALI_SUCCESS;
}

cali_err cali_begin_string(cali_id_t attr, const char* val) {

  
  std::map<cali_id_t, std::string>::iterator it = _attribute_id_map_.find(attr);
  if(it == _attribute_id_map_.end()) {
    fprintf(stderr, "TAU: CALIPER: Not a valid attribute ID. Please use cali_create_attribute to generate an attribute of type STRING, and then pass the generate ID to %s.\n", cali_begin_string);
    return CALI_EINV;
  }

  const char* attr_name = it->second.c_str();

  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_STRING) {
    return CALI_ETYPE;
  }

  RtsLayer::LockEnv();

  if(!cali_tau_initialized)
    cali_init();
  
  StackValue value;
  value.type = STRING;
  strcpy(value.data.str, val);
  TAU_VERBOSE("TAU: CALIPER create and start nested timers with names: %s %s\n", val, attr_name);

  /* Start the top level timer with name \a attr_name if it hasn't already been started*/
  if(attribute_stack[std::string(attr_name)].empty()) {
    TAU_START(attr_name);
  }

  attribute_stack[std::string(attr_name)].push(value);

  //Start timer with name \a val
  TAU_START(val);

  RtsLayer::UnLockEnv(); 

  return CALI_SUCCESS;
}

/**
 * Remove innermost value for attribute `attr` from the blackboard.
 */

cali_err cali_end  (cali_id_t   attr);

/**
 * \brief Remove innermost value for attribute \a attr from the blackboard.
 *
 * Creates a mismatch warning if the current value does not match \a val.
 * This function is primarily used by the high-level annotation API.
 *
 * \param attr Attribute ID
 * \param val  Expected value
 */

cali_err
cali_safe_end_string(cali_id_t attr, const char* val);

/**
 * \brief Change current innermost value on the blackboard for attribute \a attr 
 * to value taken from \a value with size \a size
 */

cali_err  
cali_set  (cali_id_t   attr, 
           const void* value,
           size_t      size);

cali_err  
cali_set_double(cali_id_t attr, double val);
cali_err  
cali_set_int(cali_id_t attr, int val);
cali_err  
cali_set_string(cali_id_t attr, const char* val);

/**
 * Put attribute with name \a attr_name on the blackboard.
 * TAU Wrapper:
 *  1. For strings: Begins a timer with the same name, and add string name to stack
 *  2. For int, double: Create a user event, and add the int/double value to stack
 */

cali_err cali_begin_double_byname(const char* attr_name, double val) {

  /*We do not support "stacking" semantics for UserEvents of double and integer types*/
  RtsLayer::LockEnv();

  if(!attribute_stack[std::string(attr_name)].empty()) {
    fprintf(stderr, "TAU: CALIPER operation: %s not supported for this attribute type. TAU UserEvent has already been created for %s. Use cali_set_double_byname instead to update the value\n", cali_begin_double_byname, attr_name);

    RtsLayer::UnLockEnv();

    return CALI_EINV;
  }

  //Create the attribute if it hasn't already been created explicitly
  RtsLayer::UnLockEnv();
  cali_create_attribute(attr_name, CALI_TYPE_DOUBLE, CALI_ATTR_DEFAULT);

  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_DOUBLE) {
    return CALI_ETYPE;
  }

  RtsLayer::LockEnv();


  TAU_VERBOSE("TAU: CALIPER create a TAU UserEvent named %s\n of double type\n", attr_name);
  Tau_trigger_userevent(attr_name, val);

  StackValue value;
  value.type = DOUBLE;
  value.data.as_double = val;
  attribute_stack[std::string(attr_name)].push(value);

  RtsLayer::UnLockEnv();

  return CALI_SUCCESS;
}

cali_err cali_begin_int_byname(const char* attr_name, int val) {

  RtsLayer::LockEnv();

  if(!attribute_stack[std::string(attr_name)].empty()) {
    fprintf(stderr, "TAU: CALIPER operation: %s not supported for this attribute type. TAU UserEvent has already been created for %s. Use cali_set_int_byname instead to update the value.\n", cali_begin_int_byname, attr_name);

    RtsLayer::UnLockEnv();

    return CALI_EINV;
  }

  //Create the attribute if it hasn't already been created explicitly
  RtsLayer::UnLockEnv();
  cali_create_attribute(attr_name, CALI_TYPE_INT, CALI_ATTR_DEFAULT);

  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_INT) {
    return CALI_ETYPE;
  }

  RtsLayer::LockEnv();

  TAU_VERBOSE("TAU: CALIPER create a TAU UserEvent named %s\n of integer type\n", attr_name);
  Tau_trigger_userevent(attr_name, val);

  StackValue value;
  value.type = INTEGER;
  value.data.as_integer = val;
  attribute_stack[std::string(attr_name)].push(value);

  RtsLayer::UnLockEnv();

  return CALI_SUCCESS;
}

/* TAU Wrapper: Create and start a timer with a given name*/
extern "C" cali_err cali_begin_byname(const char* attr_name) {
  
  //Create the attribute if it hasn't already been created explicitly
  cali_create_attribute(attr_name, CALI_TYPE_STRING, CALI_ATTR_DEFAULT);

  RtsLayer::LockEnv();
  
  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_STRING) {
    return CALI_ETYPE;
  }


  if(!cali_tau_initialized)
    cali_init();

  TAU_VERBOSE("TAU: CALIPER create and start a TAU static timer with name: %s\n", attr_name);
  TAU_START(attr_name);

  RtsLayer::UnLockEnv();

  return CALI_SUCCESS;
}

/* TAU Wrapper: Start a nested timer with \a val name*/
cali_err cali_begin_string_byname(const char* attr_name, const char* val) {

  //Create the attribute if it hasn't already been created explicitly
  cali_create_attribute(attr_name, CALI_TYPE_STRING, CALI_ATTR_DEFAULT);
  
  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_STRING) {
    return CALI_ETYPE;
  }

  RtsLayer::LockEnv();

  if(!cali_tau_initialized)
    cali_init();
  
  StackValue value;
  value.type = STRING;
  strcpy(value.data.str, val);
  TAU_VERBOSE("TAU: CALIPER create and start nested timers with names: %s %s\n", val, attr_name);

  /* Start the top level timer with name \a attr_name if it hasn't already been starte*/
  if(attribute_stack[std::string(attr_name)].empty()) {
    TAU_START(attr_name);
  }

  attribute_stack[std::string(attr_name)].push(value);

  //Start timer with name \a val
  TAU_START(val);

  RtsLayer::UnLockEnv(); 

  return CALI_SUCCESS;
} 

/* TAU Wrapper: 
 * 1. Replace value at the top of the stack.
 * 2. Trigger a TAU UserEvent for int/double types.
 * 3. String types: Operation currently not supported
 */
cali_err cali_set_double_byname(const char* attr_name, double val) {

  //Create the attribute if it hasn't already been created explicitly
  cali_create_attribute(attr_name, CALI_TYPE_DOUBLE, CALI_ATTR_DEFAULT);

  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_DOUBLE) {
    return CALI_ETYPE;
  }

  RtsLayer::LockEnv();

  if(!cali_tau_initialized)
    cali_init();

  TAU_VERBOSE("TAU: CALIPER trigger TAU UserEvent with name: %s with value %f\n", attr_name, val);
  if(!attribute_stack[std::string(attr_name)].empty()) {
    attribute_stack[std::string(attr_name)].pop();
  }

  Tau_trigger_userevent(attr_name, val);
  StackValue value;
  value.type = DOUBLE;
  value.data.as_double = val;

  attribute_stack[std::string(attr_name)].push(value);

  RtsLayer::UnLockEnv();

  return CALI_SUCCESS;
}

cali_err cali_set_int_byname(const char* attr_name, int val) {
  //Create the attribute if it hasn't already been created explicitly
  cali_create_attribute(attr_name, CALI_TYPE_INT, CALI_ATTR_DEFAULT);

  //Sanity check for the type of the attribute
  if(_attribute_type_map_name_key[attr_name] != CALI_TYPE_INT) {
    return CALI_ETYPE;
  }

  RtsLayer::LockEnv();
 
  if(!cali_tau_initialized)
    cali_init();

  TAU_VERBOSE("TAU: CALIPER trigger TAU UserEvent with name: %s with value %d\n", attr_name, val);
  if(!attribute_stack[std::string(attr_name)].empty()) {
    attribute_stack[std::string(attr_name)].pop();
  }

  Tau_trigger_userevent(attr_name, val);
  StackValue value;
  value.type = INTEGER;
  value.data.as_integer = val;

  attribute_stack[std::string(attr_name)].push(value);

  RtsLayer::UnLockEnv();

  return CALI_SUCCESS;
}

/* Interesting question: What do we do here?*/
cali_err cali_set_string_byname(const char* attr_name, const char* val) {
  fprintf(stderr, "TAU: CALIPER operation: %s is not supported\n", cali_set_string_byname);
  return CALI_EINV;
}

/**
 * \brief Remove \a value for the attribute with the name \a attr_name to the 
 * blackboard.
 * TAU Wrapper: If stack is not empty, pop the stack. Further, if the popped value is a string, then stop the corresponding timer. 
 *              If stack is empty, stop the top level timer with name \a attr_name
 */
cali_err cali_end_byname(const char* attr_name) {
  RtsLayer::LockEnv(); 

  if(!cali_tau_initialized)
    cali_init();

  if(!attribute_stack[std::string(attr_name)].empty()) {

    StackValue value = attribute_stack[std::string(attr_name)].top();
    attribute_stack[std::string(attr_name)].pop();
   

    if(value.type == STRING) {
      TAU_VERBOSE("TAU: CALIPER stop timer with name: %s\n", attr_name);
      TAU_STOP(value.data.str);
    } else {
       //Nothing to do for integer, double types
    }
  } else {
    TAU_VERBOSE("TAU: CALIPER stop top level timer with name %s\n", attr_name);
    TAU_STOP(attr_name);
  }
  
  RtsLayer::UnLockEnv(); 
  return CALI_SUCCESS;
} 
/*
 * --- Attributes ------------------------------------------------------
 */

/**
 * \name Attribute management
 * \{
 */

/**
 * \brief Create an attribute
 * \param name Name of the attribute
 * \param type Type of the attribute
 * \param properties Attribute properties
 * \return Attribute id
 * TAU Wrapper: Maintain a local key:value mapping of name:id and id:name and name:type mapping
 * 		We generate our own ID (dumb counter), but we don't pass through
 * 		any calls to CALIPER.
 * 		As of the moment, we do not support properties while creating the caliper attributes.
 * 		We may support add more support in the future.
 */

cali_id_t cali_create_attribute(const char*     name,
                      cali_attr_type  type,
                      int             properties) {

  if(!cali_tau_initialized)
    cali_init();

  RtsLayer::LockEnv(); 

  std::map<std::string, cali_id_t>::iterator it = _attribute_name_map_.find(name);
  if(it != _attribute_name_map_.end()) {
    cali_id_t ID = _attribute_name_map_[name];
    TAU_VERBOSE("TAU: CALIPER attribute with the name %s already exists. Returning the already created ID: %d\n", name, ID);
    RtsLayer::UnLockEnv();
    return ID;
  }

  //Critical section
  ++current_id;
  //Maintain a map of name <=> id for "fast" lookups
  _attribute_name_map_[name] = current_id;
  _attribute_id_map_[current_id] = name;

   //Also maintain a map of name:type and id:type to ensure user doesn't invoke wrong functionality for attributes defined with a certain type
  _attribute_type_map_name_key[name] = type;
  _attribute_type_map_id_key[current_id] = type;
  RtsLayer::UnLockEnv();

  if(properties != CALI_ATTR_DEFAULT) {
    fprintf(stderr, "TAU: CALIPER: Property combination for attribute not supported. CALI_ATTR_SCOPE_PROCESS is assumed as default\n");
  }

  return current_id;
}  

/**
 * \brief Create an attribute with additional metadata. 
 *
 * Metadata is provided via (meta-attribute id, pointer-to-data, size) in
 * the \a meta_attr_list, \a meta_val_list, and \a meta_size_list.
 * \param name Name of the attribute
 * \param type Type of the attribute
 * \param properties Attribute properties
 * \param n Number of metadata entries
 * \param meta_attr_list Attribute IDs of the metadata entries
 * \param meta_val_list  Pointers to values of the metadata entries
 * \param meta_size_list Sizes (in bytes) of the metadata values
 * \return Attribute id
 * \sa cali_create_attribute
 * TAU Wrapper: We do not support this version of create_attribute yet.
 * 		Call cali_create_attribute and print appropriate warning message
 */

cali_id_t cali_create_attribute_with_metadata(const char*     name,
                                    cali_attr_type  type,
                                    int             properties,
                                    int             n,
                                    const cali_id_t meta_attr_list[],
                                    const void*     meta_val_list[],
                                    const size_t    meta_size_list[]) {

  fprintf(stderr, "TAU: CALIPER: creating attribute with metadata is currently not supported. Using default create_attribute method\n");
  return cali_create_attribute(name, type, properties);
}
  
/**
 * \brief Find attribute by name 
 * \param name Name of attribute
 * \return Attribute ID, or CALI_INV_ID if attribute was not found
 * TAU Wrapper: Do exactly as caliper does.
 */

cali_id_t cali_find_attribute(const char* name) {
  std::map<std::string, cali_id_t>::iterator it = _attribute_name_map_.find(name);

  if(it == _attribute_name_map_.end()) {
    return CALI_INV_ID;
  }
  
  return it->second;
}
/**
 * \brief  Return name of attribute with given ID
 * \param  attr_id Attribute id
 * \return Attribute name, or NULL if `attr_id` is not a valid attribute ID
 * TAU Wrapper: Do exactly as caliper does.
 */
const char* cali_attribute_name(cali_id_t attr_id) {
  std::map<cali_id_t, std::string>::iterator it = _attribute_id_map_.find(attr_id);

  if(it == _attribute_id_map_.end()) {
    return NULL;
  }
   
  return (it->second).c_str();
}
/**
 * \brief Return the type of the attribute with given ID
 * \param attr_id Attribute id
 * \return Attribute type, or CALI_TYPE_INV if `attr_id` is not a valid attribute ID
 */
cali_attr_type cali_attribute_type(cali_id_t attr_id) {
  std::map<cali_id_t, cali_attr_type>::iterator it = _attribute_type_map_id_key.find(attr_id);

  if(it == _attribute_type_map_id_key.end()) {
    return CALI_TYPE_INV;
  }
   
  return it->second;
}

/*
 * --- Snapshot ---------------------------------------------------------
 */

/**
 * \name Taking snapshots
 * \{
 */

/**
 * \brief Take a snapshot and push it into the processing queue.
 * \param scope Indicates which scopes (process, thread, or task) the 
 *   snapshot should span
 * \param n Number of event info entries
 * \param trigger_info_attr_list Attribute IDs of event info entries
 * \param trigger_info_val_list  Pointers to values of event info entries
 * \param trigger_info_size_list Sizes (in bytes) of event info entries
 */
void cali_push_snapshot(int scope, int n,
                   const cali_id_t trigger_info_attr_list[],
                   const void*     trigger_info_val_list[],
                   const size_t    trigger_info_size_list[]) {
  fprintf(stderr, "TAU: CALIPER operation: %s is not supported\n", cali_set_string_byname);
}

/**
 * \brief Take a snapshot and write it into the user-provided buffer.
 *
 * This function can be safely called from a signal handler. However,
 * it is not guaranteed to succeed. Specifically, the function will
 * fail if the signal handler interrupts already running Caliper
 * code.
 * 
 * The snapshot representation returned in \a buf is valid only on the
 * local process, while Caliper is active (which is up until Caliper's 
 * `finish_evt` callback is invoked).
 * It can be parsed with cali_unpack_snapshot().
 *
 * \param scope Indicates which scopes (process, thread, or task) the
 *   snapshot should span
 * \param len   Length of the provided snapshot buffer.
 * \param buf   User-provided snapshot storage buffer.
 * \return Actual size of the snapshot representation. 
 *   If this is larger than `len`, the provided buffer was too small and 
 *   not all of the snapshot was returned.
 *   If this is zero, no snapshot was taken.
 */
size_t cali_pull_snapshot(int scope, size_t len, unsigned char* buf) {

  fprintf(stderr, "TAU: CALIPER operation: %s is not supported\n", cali_pull_snapshot);
  return 0;
}

/**
 * \}
 * \name Processing snapshot contents
 * \{
 */

/**
 * \brief Unpack a snapshot buffer.
 *
 * Unpack a snapshot that was previously obtained on the same process
 * and examine its attribute:value entries with the given \a proc_fn 
 * callback function.
 *
 * The function will invoke \a proc_fn repeatedly, once for each
 * unpacked entry. \a proc_fn should return a non-zero value if it
 * wants to continue processing, otherwise processing will stop. Note
 * that snapshot processing cannot be re-started from a partially read
 * snapshot buffer position: the buffer has to be read again from the
 * beginning.
 *
 * Hierarchical values will be given to \a proc_fn in top-down order.
 *
 * \note This function is async-signal safe if \a proc_fn is
 *   async-signal safe.
 *
 * \param buf Snapshot buffer
 * \param bytes_read Number of bytes read from the buffer
 *   (i.e., length of the snapshot)
 * \param proc_fn Callback function to process individidual entries
 * \param user_arg User-defined parameter passed through to \a proc_fn
 *
 * \sa cali_pull_snapshot, cali_entry_proc_fn
 */    
void cali_unpack_snapshot(const unsigned char* buf,
                     size_t*              bytes_read,
                     cali_entry_proc_fn   proc_fn,
                     void*                user_arg) {

  fprintf(stderr, "TAU: CALIPER operation: %s is not supported\n", cali_unpack_snapshot);
}

/**
 * Return top-most value for attribute ID \a attr_id from snapshot \a buf.
 * The snapshot must have previously been obtained on the same process with
 * cali_pull_snapshot().
 *
 * \note This function is async-signal safe
 *
 * \param buf Snapshot buffer
 * \param attr_id Attribute id
 * \param bytes_read Number of bytes read from the buffer
 *   (i.e., length of the snapshot)
 * \return The top-most stacked value for the given attribute ID, or an empty
 *   variant if none was found
 */    

cali_variant_t cali_find_first_in_snapshot(const unsigned char* buf,
                            cali_id_t            attr_id,
                            size_t*              bytes_read) {

  fprintf(stderr, "TAU: CALIPER operation: %s is not supported\n", cali_find_first_in_snapshot);
  return cali_make_variant_from_int(0);
}

/**
 * Run all entries with attribute `attr_id` in a snapshot that was previously 
 * obtained on the same process through the given `proc_fn` callback function.
 *
 * \note This function is async-signal safe if `proc_fn` is async-signal safe.
 *
 * \param buf Snapshot buffer
 * \param attr_id Attribute to read from snapshot
 * \param bytes_read Number of bytes read from the buffer
 *   (i.e., length of the snapshot)
 * \param proc_fn Callback function to process individidual entries
 * \param userdata User-defined parameter passed to `proc_fn`  
 */    

void
cali_find_all_in_snapshot(const unsigned char* buf,
                          cali_id_t            attr_id,
                          size_t*              bytes_read,
                          cali_entry_proc_fn   proc_fn,
                          void*                userdata) {

  fprintf(stderr, "TAU: CALIPER operation: %s is not supported\n", cali_find_all_in_snapshot);
}

/*
 * --- Blackboard access API ---------------------------------
 */

/**
 * \name Blackboard access
 * \{
 */

/**
 * \brief Return top-most value for attribute \a attr_id from the blackboard.
 *
 * \note This function is async-signal safe.
 *
 * \param attr_id Attribute ID to find
 * \return The top-most stacked value on the blackboard for the given
 *    attribute ID, or an empty variant if it was not found
 */
cali_variant_t cali_get(cali_id_t attr_id) {

  std::map<cali_id_t, std::string>::iterator it = _attribute_id_map_.find(attr_id);
  if(it == _attribute_id_map_.end()) {
    fprintf(stderr, "TAU: CALIPER: Attribute with id: %d doesn't exist\n", attr_id);
    return cali_make_empty_variant();
  }

  if(attribute_stack[it->second].empty()) {
    fprintf(stderr, "TAU: CALIPER: Attribute with id: %d doesn't have any values on the blackboard\n", attr_id);
    return cali_make_empty_variant();
  }

  StackValue value = attribute_stack[it->second].top();

  switch(value.type) {
    case STRING:
      return cali_make_variant_from_string(value.data.str);
      break;
    case INTEGER:
      return cali_make_variant_from_int(value.data.as_integer);
      break;
    case DOUBLE:
      return cali_make_variant_from_double(value.data.as_double);
      break;
    default:
      return cali_make_empty_variant();
  }

}

  
