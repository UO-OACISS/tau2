#include <stdio.h>
#include <stdlib.h>

#include <caliper/cali.h>
#include <TAU.h>

//Externs
extern "C" int Tau_init_initializeTAU();

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
  * TAU Wrapper: Calls TAU initialize
  */

extern "C" void cali_init() {

  TAU_VERBOSE("TAU: CALIPER init invoked.\n");

  if(Tau_init_initializeTAU()) {
    fprintf(stderr, "TAU: Initialization from within Caliper wrapper failed\n");
  }

}

/**
 * Put attribute with name \a attr_name on the blackboard.
 * TAU Wrapper: Begins a timer with the same name
 *   */

extern "C" cali_err cali_begin_byname(const char* attr_name) {

  TAU_VERBOSE("TAU: CALIPER begin an attribute by name: %s \n", attr_name);
  TAU_START(attr_name);

}

/**
 * Remove innermost value for attribute \a attr from the blackboard.
 * TAU Wrapper: Ends a timer with a given name
 */

extern "C" cali_err cali_end_byname(const char* attr_name) {

  TAU_VERBOSE("TAU: CALIPER end an attribute by name: %s \n", attr_name);
  TAU_STOP(attr_name);

}

