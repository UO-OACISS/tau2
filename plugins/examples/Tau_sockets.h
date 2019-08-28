
#pragma once

#include <stdlib.h> 
#include <string.h> 
#include <thread>
#include <Profile/Profiler.h>
#include <Profile/TauAPI.h>

namespace tau {

    namespace plugins {

        class HostInfo {
            public:
                HostInfo() {};
                HostInfo(const char * hostname, const char * address, int portnum) :
                    _hostname(hostname), _address(address), _port(portnum) {
                }
                std::string _hostname;
                std::string _address;
                int _port;
        };

        typedef char * CallbackFunctionType(char * request);

        class Sockets {
            public:
                Sockets(int rank, CallbackFunctionType * cb) : _rank(rank),
                    _server(&tau::plugins::Sockets::Run, rank, cb) { }
                ~Sockets() {
                    // tell the server to exit
                    send_message(_rank, "");
                    _server.join();
                    //ClearHostInfo();
                }
                static void Run(int rank, CallbackFunctionType * cb);
                static void GetHostInfo(int port);
                //void ClearHostInfo();
                char * send_message(int rank, const char * message);
            private:
                int _rank;
                std::thread _server;
        };

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

