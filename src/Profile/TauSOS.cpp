
#include "Profile/Profiler.h"
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <stdexcept>
#include "sos.h"

// the set of keys (strings) that we have registered so far
std::set<std::string> TAU_SOS_key_set;
// the map of keys (strings) to SOS keys (integers)
std::map<std::string,int> TAU_SOS_key_map;

void TAU_SOS_init(void) {
    //SOS_init(SOS_APP);
}

void TAU_SOS_finalize(void) {
    //SOS_finalize();
}

void TAU_SOS_send_data(void) {
    // get the list of functions
    const char **functionList;
    int numOfFunctions;
    TAU_GET_FUNC_NAMES(functionList, numOfFunctions);
    // get the list of counters
    // todo...
    // register keys (if necessary)
    // send list of keys
    // send list of values
    double **counterExclusiveValues;
    double **counterInclusiveValues;
    int *numOfCalls;
    int *numOfSubRoutines;
    const char **counterNames;
    int numOfCounters;

    TAU_GET_FUNC_VALS(inFuncs, size, counterExclusiveValues, 
            counterInclusiveValues, numOfCalls, numOfSubRoutines, 
            counterNames, numOfCounters);
}
