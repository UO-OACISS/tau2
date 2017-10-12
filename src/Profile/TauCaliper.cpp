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
#include <caliper/cali.h>
#include <TAU.h>


#include <Profile/TauCaliperTypes.h>

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
    printf("TAU: CALIPER operation not supported! TAU UserEvent has already been created for %s. Use cali_set_double_byname instead to update the value\n", attr_name);

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
    printf("TAU: CALIPER operation not supported! TAU UserEvent has already been created for %s. Use cali_set_int_byname instead to update the value.\n", attr_name);

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
  TAU_VERBOSE("TAU: CALIPER operation: %s is not supported\n", cali_set_string_byname);
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

  auto it = _attribute_name_map_.find(name);
  if(it != _attribute_name_map_.end()) {
    auto ID = _attribute_name_map_[name];
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
    fprintf(stdout, "TAU: CALIPER Warning: Property combination for attribute not supported. CALI_ATTR_SCOPE_PROCESS is assumed as default\n");
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

  printf("TAU: CALIPER creating attribute with metadata is currently not supported. Using default create_attribute method\n");
  return cali_create_attribute(name, type, properties);
}
  
/**
 * \brief Find attribute by name 
 * \param name Name of attribute
 * \return Attribute ID, or CALI_INV_ID if attribute was not found
 * TAU Wrapper: Do exactly as caliper does.
 */

cali_id_t cali_find_attribute(const char* name) {
  auto it = _attribute_name_map_.find(name);

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
  auto it = _attribute_id_map_.find(attr_id);

  if(it == _attribute_id_map_.end()) {
    return NULL;
  }
   
  return it->second;
}
/**
 * \brief Return the type of the attribute with given ID
 * \param attr_id Attribute id
 * \return Attribute type, or CALI_TYPE_INV if `attr_id` is not a valid attribute ID
 */
cali_attr_type cali_attribute_type(cali_id_t attr_id) {
  auto it = _attribute_type_map_id_key.find(attr_id);

  if(it == _attribute_type_map_id_key.end()) {
    return CALI_TYPE_INV;
  }
   
  return it->second;
}
