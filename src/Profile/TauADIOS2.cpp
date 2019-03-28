/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997-2017	          			   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **	File 		: TauKokkos.cpp					  **
 **	Description 	: TAU Profiling Interface for Kokkos. Use the env **
 **                       var KOKKOS_PROFILE_LIBRARY to point to libTAU.so**
 **	Contact		: tau-bugs@cs.uoregon.edu 		 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#ifdef TAU_DOT_H_LESS_HEADERS
#include <cstdio>
#include <inttypes.h>
#include <vector>
#include <map>
#include <string>
#include <stack>
#include <iostream>
using namespace std; 
#endif /* TAU_DOT_H_LESS_HEADERS */
#include <stdlib.h>

#include <TAU.h>

/* Function pointers */

extern "C" void Adiost_init(void) {
#ifndef TAU_MPI
    int _argc = 1;
    const char *_dummy = "";
    char *_argv[1];
    _argv[0] = (char *)(_dummy);
    Tau_init(_argc, _argv);
    Tau_set_node(0);
    Tau_create_top_level_timer_if_necessary();
#endif
}

extern "C" void Adiost_register_thread(void) {
    Tau_register_thread();
    Tau_create_top_level_timer_if_necessary();
}

extern "C" void Tau_profile_exit_all_threads();

extern "C" void Adiost_exit(void) {
    Tau_destructor_trigger();
    Tau_profile_exit_all_threads();
    Tau_exit("stub exiting");
}

extern "C" void (*Adiost_start)(const char *) = &Tau_pure_start;
extern "C" void (*Adiost_stop)(const char *) = &Tau_pure_stop;
extern "C" void (*Adiost_trigger_context_event)(const char *, double) = &Tau_trigger_context_event;
extern "C" void (*Adiost_metadata)(const char *, const char *) = &Tau_metadata;

