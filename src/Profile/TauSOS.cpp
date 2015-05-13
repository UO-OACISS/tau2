
#include "Profile/Profiler.h"
#include "TauMetrics.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "sos.h"
#include "stdio.h"

SOS_pub_handle *pub;

extern "C" void TAU_SOS_init(int argc, char ** argv) {
    SOS_init(&argc, &argv, SOS_APP);
    SOS_comm_split();
    pub = SOS_new_pub((char *)"TAU Application");
}

extern "C" void TAU_SOS_finalize(void) {
    SOS_finalize();
}

extern "C" int TauProfiler_updateAllIntermediateStatistics(void);

extern "C" void TAU_SOS_send_data(void) {
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();

    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::iterator it;
  const char **counterNames;
  int numCounters;
  TauMetrics_getCounterList(&counterNames, &numCounters);
  RtsLayer::LockDB();
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    // get the number of calls
    int tid = 0; // todo: get ALL thread data.
    SOS_val calls, inclusive, exclusive;
    calls.d_val = fi->GetCalls(tid);
    std::stringstream calls_str;
    calls_str << "TAU::calls::" << fi->GetName();
    const std::string& tmpcalls = calls_str.str();
    SOS_pack(pub, tmpcalls.c_str(), SOS_DOUBLE, calls);
    // todo - subroutines
    // iterate over metrics 
    std::stringstream incl_str;
    std::stringstream excl_str;
    for (int m = 0; m < Tau_Global_numCounters; m++) {
        inclusive.d_val = fi->getDumpInclusiveValues(tid)[m];
        incl_str.clear();
        incl_str << "TAU::inclusive::" << counterNames[m] << "::" << fi->GetName();
        const std::string& tmpincl = incl_str.str();
        SOS_pack(pub, tmpincl.c_str(), SOS_DOUBLE, inclusive);
        exclusive.d_val = fi->getDumpInclusiveValues(tid)[m];
        excl_str.clear();
        excl_str << "TAU::exclusive::" << counterNames[m] << "::" << fi->GetName();
        const std::string& tmpexcl = excl_str.str();
        SOS_pack(pub, tmpexcl.c_str(), SOS_DOUBLE, exclusive);
    }
  }
  RtsLayer::UnLockDB();
}
