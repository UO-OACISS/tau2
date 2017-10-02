#include <stdio.h>
#include <stdlib.h>

#include <caliper/cali.h>

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
 */
extern "C" void cali_init() {
  printf("TAU: CALIPER init invoked.\n");
}

/**
 *  * Put attribute with name \a attr_name on the blackboard.
 *   */

extern "C" cali_err cali_begin_byname(const char* attr_name) {
  printf("TAU: CALIPER begin an attribute by name: %s \n", attr_name);
}

/**
 *  * Remove innermost value for attribute \a attr from the blackboard.
 *   */

extern "C" cali_err cali_end_byname(const char* attr_name) {
  printf("TAU: CALIPER end an attribute by name: %s \n", attr_name);

}

