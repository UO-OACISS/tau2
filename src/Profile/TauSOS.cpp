
#include "Profile/Profiler.h"
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <stdexcept>
#include "sos.h"
#include "stdio.h"

// the set of keys (strings) that we have registered so far
std::set<std::string> TAU_SOS_key_set;
// the map of keys (strings) to SOS keys (integers)
std::map<std::string,unsigned long> TAU_SOS_key_map;

void TAU_SOS_init(void) {
    //SOS_init(SOS_APP);
}

void TAU_SOS_finalize(void) {
    //SOS_finalize();
}

extern "C" void TAU_SOS_send_data(void) {
    // get the list of functions
    const char **functionList;
    int numOfFunctions;
    TAU_GET_FUNC_NAMES(functionList, numOfFunctions);
    // get the list of counters
    // todo: get list of counters
    // register keys (if necessary)
    int index;
    std::set<std::string>::iterator finder;
    for (index = 0 ; index < numOfFunctions ; index++) {
        std::string tmp(functionList[index]);
        finder = TAU_SOS_key_set.find(tmp);
        // if not found, register it and add it to the set
        if (finder == TAU_SOS_key_set.end()) {
            // todo: register
            TAU_SOS_key_map[tmp] = TAU_SOS_key_set.size();
            TAU_SOS_key_set.insert(tmp);
        }
    }
    // send list of keys
    std::map<std::string,unsigned long>::iterator it;
    for (it = TAU_SOS_key_map.begin() ; it != TAU_SOS_key_map.end(); it++) {
        std::cout << it->second << " " << it->first << std::endl;
    }

    // send list of values
    double **counterExclusiveValues;
    double **counterInclusiveValues;
    int *numOfCalls;
    int *numOfSubRoutines;
    const char **counterNames;
    int numOfCounters;

    TAU_GET_FUNC_VALS(functionList, numOfFunctions, counterExclusiveValues, 
            counterInclusiveValues, numOfCalls, numOfSubRoutines, 
            counterNames, numOfCounters);
    for (index = 0 ; index < numOfFunctions ; index++) {
        printf("%.*s...[%lu]: %d %10.1f %10.1f\n", 10,
                functionList[index], 
                TAU_SOS_key_map[std::string(functionList[index])],
                numOfCalls[index], 
                counterInclusiveValues[index][0], 
                counterExclusiveValues[index][0]);
    }
}
