//TauRocProfilerSDK.h
#ifndef _TAU_ROCMSDK_H_
#define _TAU_ROCMSDK_H_

#include <Profile/TauBfd.h>  // for name demangling
#include <Profile/TauRocm.h>

#include "Profile/Profiler.h"

//Need to check, are they all needed? 
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>


//Need to check, are they all needed? 
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <unistd.h>



// This is for the windows buffer, similar to the one in TauRocm.cpp
//re-implemented here, as the SDK and Rocm will be separated in the future
#ifndef TAU_ROCMSDK_LOOK_AHEAD
#define TAU_ROCMSDK_LOOK_AHEAD 256
#endif /* TAU_ROCMSDK_LOOK_AHEAD */

//Enum to enable or disable metric profiling
typedef enum profile_metrics {
	NO_METRICS = 1,
	WRONG_NAME = 2,
	PROFILE_METRICS = 3
};


struct TauSDKEvent {

    rocprofiler_timestamp_t entry;
    rocprofiler_timestamp_t exit;
    std::string name;
    int taskid;

    TauSDKEvent(): taskid(0) {}
    TauSDKEvent(string event_name, rocprofiler_timestamp_t begin, rocprofiler_timestamp_t end, int t) : name(event_name), taskid(t)
    {
        entry = begin;
        exit  = end;
    }
    void printEvent() {
        std::cout <<name<<" Task: "<<taskid<<", \t\tEntry: "<<entry<<" , Exit = "<<exit;
    }
    bool appearsBefore(struct TauSDKEvent other_event) {
        if ((taskid == other_event.taskid) &&
            (entry < other_event.entry) &&
            (exit < other_event.entry))  {
            // both entry and exit of my event is before the entry of the other event.
            return true;
        } else
            return false;
    }

    bool operator < (struct TauSDKEvent two) {
        if (entry < two.entry) 
            return true;
        else 
            return false;
    }
  
};

#ifndef ROCPROFILER_CALL
#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg << " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }
#endif



#endif // _TAU_ROCMSDK_H_