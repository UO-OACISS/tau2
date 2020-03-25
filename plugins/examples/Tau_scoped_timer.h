
#pragma once

#include <stdlib.h> 
#include <string.h> 
#include <Profile/Profiler.h>
#include <Profile/TauAPI.h>

namespace tau {

    namespace plugins {

        class ScopedTimer {
            public:
                ScopedTimer(const char * name) {
                    _name = strdup(name);
                    Tau_pure_start(_name);
                }
                ~ScopedTimer() {
                    Tau_pure_stop(_name);
                    free(_name);
                }
                char * _name;
        };
    }
}

