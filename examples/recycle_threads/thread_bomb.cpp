/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2020  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/

#include <iostream>
#include <thread>
#include <unistd.h>
#include <Profile/Profiler.h>

 
void foo() 
{
    TAU_REGISTER_THREAD();
    TAU_PROFILE("foo()", "void ()", TAU_USER);
    // do stuff...
    usleep(1);
}

int main(int argc, char* argv[])
{
    TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
    TAU_PROFILE_INIT(argc,argv);
    TAU_PROFILE_SET_NODE(0);
    /* Get the number of cores on this machine */
    unsigned int cores = std::thread::hardware_concurrency();
    cores = cores > 0 ? cores : sysconf(_SC_NPROCESSORS_ONLN);
    cores = cores > 32 ? cores : 34;

    for (unsigned int i = 0; i < (cores * 4) ; i++) {
        std::thread first (foo);     // spawn new thread that calls foo()
        // synchronize threads:
        first.join();                // pauses until first finishes
    }
}


