#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <stack>
#include <mutex>
#include <caliper/cali.h>
#include <TAU.h>

#include <Profile/TauCaliperTypes.h>

std::map<std::string, cali_id_t> _attribute_name_map_;
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

  
  //Initialize the ID field to 0. This is the dummy id generation mechanism that we 
  RtsLayer::LockDB();
  current_id = 0;
  RtsLayer::UnLockDB();

  if(Tau_init_initializeTAU()) {

    fprintf(stderr, "TAU: Initialization from within Caliper wrapper failed\n");
  }

  cali_tau_initialized = 1;
}

/**
 * Put attribute with name \a attr_name on the blackboard.
 * TAU Wrapper:
 *  1. For strings: Begins a timer with the same name, and add string name to stack
 *  2. For int, double: Create a user event, and add the int/double value to stack
 */

cali_err cali_begin_double_byname(const char* attr_name, double val) {

  /*We do not supporting "stacking" semantics for UserEvents*/
  if(!attribute_stack[std::string(attr_name)].empty()) {
    printf("TAU: CALIPER operation not supported! TAU UserEvent has already been created for %s. Use cali_set_double_byname instead to update the value\n", attr_name);
    return CALI_EINV;
  }

  TAU_VERBOSE("TAU: CALIPER create a TAU UserEvent named %s\n of double type\n", attr_name);
  Tau_trigger_userevent(attr_name, val);

  StackValue value;
  value.type = DOUBLE;
  value.data.as_double = val;
  attribute_stack[std::string(attr_name)].push(value);
  return CALI_SUCCESS;
}

cali_err cali_begin_int_byname(const char* attr_name, int val) {
  if(!attribute_stack[std::string(attr_name)].empty()) {
    printf("TAU: CALIPER operation not supported! TAU UserEvent has already been created for %s. Use cali_set_int_byname instead to update the value.\n", attr_name);
    return CALI_EINV;
  }

  TAU_VERBOSE("TAU: CALIPER create a TAU UserEvent named %s\n of integer type\n", attr_name);
  Tau_trigger_userevent(attr_name, val);

  StackValue value;
  value.type = INTEGER;
  value.data.as_integer = val;
  attribute_stack[std::string(attr_name)].push(value);

  return CALI_SUCCESS;
}

/* TAU Wrapper: Create and start a timer with a given name*/
extern "C" cali_err cali_begin_byname(const char* attr_name) {
  if(!cali_tau_initialized)
    cali_init();

  TAU_VERBOSE("TAU: CALIPER create and start a TAU static timer with name: %s\n", attr_name);
  TAU_START(attr_name);

  return CALI_SUCCESS;
}

/* TAU Wrapper: Start a nested timer with \a val name*/
cali_err cali_begin_string_byname(const char* attr_name, const char* val) {
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
  return CALI_SUCCESS;
} 

/* TAU Wrapper: 
 * 1. Replace value at the top of the stack.
 * 2. Trigger a TAU UserEvent for int/double types.
 * 3. String types: Operation currently not supported
 */
cali_err cali_set_double_byname(const char* attr_name, double val) {
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

  return CALI_SUCCESS;
}

cali_err cali_set_int_byname(const char* attr_name, int val) {
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
 */

cali_id_t cali_create_attribute(const char*     name,
                      cali_attr_type  type,
                      int             properties) {

  if(!cali_tau_initialized)
    cali_init();

  RtsLayer::LockDB();

  //Critical section?    
  ++current_id;
  _attribute_name_map_[name] = current_id;
  _attribute_id_map_[current_id] = name;

  RtsLayer::UnLockDB();

  return current_id;
}  
   
