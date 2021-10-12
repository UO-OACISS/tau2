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
#include <Profile/TauBfd.h>

map<int,FunctionInfo*> KokkosFunctionInfoDB;

///////////////////////////////////////////////////////////
//// prints indents for the Kokkos regions on the callstack
///////////////////////////////////////////////////////////
void kokkosp_print_region_stack_indent(const int level) {
}

///////////////////////////////////////////////////////////
//// prints Kokkos regions on the callstack
///////////////////////////////////////////////////////////
int kokkosp_print_region_stack() {
  return 0;
}

///////////////////////////////////////////////////////////
//// initialize the library
///////////////////////////////////////////////////////////
extern "C" void kokkosp_init_library(const int loadSeq,
	const uint64_t interfaceVer,
	const uint32_t devInfoCount,
	void* deviceInfo) {

	TAU_VERBOSE("TAU: Example Library Initialized (sequence is %d, version: %llu)\n", loadSeq, interfaceVer);

}

///////////////////////////////////////////////////////////
//// finalize the library
///////////////////////////////////////////////////////////
extern "C" void kokkosp_finalize_library() {
	TAU_VERBOSE("TAU: Kokkos library finalization called.\n");
}


///////////////////////////////////////////////////////////
//// start Kokkos timer with a string (operation) and a name
///////////////////////////////////////////////////////////
extern "C" void Tau_start_kokkos_timer(string operation, const char* name, const uint32_t devID, uint64_t* kID) {
    char *dem_name = Tau_demangle_name(name);
	char buf[256]; sprintf(buf," [device=%d]", devID);
	//string region_name(std::string("Kokkos::parallel_for ")+dem_name+buf);
	string region_name(operation+" "+dem_name+buf);

	void *fiptr;
	TAU_PROFILER_CREATE(fiptr, region_name.c_str(), "", TAU_KOKKOS);
	TAU_PROFILER_START(fiptr);
	FunctionInfo *fi = (FunctionInfo *)fiptr;
  	*kID=fi->GetFunctionId();

        KokkosFunctionInfoDB[*kID] = fi;

/*
	TAU_VERBOSE("TAU: Start : %s kernel id=%llu on device %d\n", fi->GetName(), *kID, devID);
        TAU_VERBOSE("TAU: Start: KokkosFunctionInfoDB[%d]->GetName() is %s, addr = %p\n",
	  (*(kID)), KokkosFunctionInfoDB[*kID]->GetName(), KokkosFunctionInfoDB[*kID]);
*/
	//cout <<"Region: "<<region_name<<" id = "<<*kID<<endl;
	//printf("Kokkos::parallel_for %s [device=%d]\n", dem_name, devID);
    free(dem_name);
}

extern "C" void kokkosp_begin_parallel_for(const char* name, const uint32_t devID, uint64_t* kID) {
  	Tau_start_kokkos_timer(string("Kokkos::parallel_for"), name, devID, kID);
}
///////////////////////////////////////////////////////////
//// end parallel for
///////////////////////////////////////////////////////////
extern "C" void Tau_stop_kokkos_timer(const uint64_t kID) {
	//FunctionInfo *fiptr = TheFunctionDB()[kID-1];
	FunctionInfo *fiptr = KokkosFunctionInfoDB[kID];
	TAU_PROFILER_STOP(fiptr);
/*
	TAU_VERBOSE("TAU: Stop:  %s kernel id=%d is complete.\n",
		fiptr->GetName(), kID);
*/
}

extern "C" void kokkosp_end_parallel_for(const uint64_t kID) {
      	Tau_stop_kokkos_timer(kID);
}
///////////////////////////////////////////////////////////
//// begin parallel scan
///////////////////////////////////////////////////////////
extern "C" void kokkosp_begin_parallel_scan(const char* name, const uint32_t devID, uint64_t* kID) {
  	Tau_start_kokkos_timer(string("Kokkos::parallel_scan"), name, devID, kID);

}

///////////////////////////////////////////////////////////
//// end parallel scan
///////////////////////////////////////////////////////////
extern "C" void kokkosp_end_parallel_scan(const uint64_t kID) {
      	Tau_stop_kokkos_timer(kID);
}

///////////////////////////////////////////////////////////
//// begin parallel reduce
///////////////////////////////////////////////////////////
extern "C" void kokkosp_begin_parallel_reduce(const char* name, const uint32_t devID, uint64_t* kID) {
  	Tau_start_kokkos_timer(string("Kokkos::parallel_reduce"), name, devID, kID);
}

///////////////////////////////////////////////////////////
//// end parallel reduce
///////////////////////////////////////////////////////////
extern "C" void kokkosp_end_parallel_reduce(const uint64_t kID) {
      	Tau_stop_kokkos_timer(kID);
}


stack <string>  Tau_kokkos_stack;
///////////////////////////////////////////////////////////
//// push parallel region to the callstack
///////////////////////////////////////////////////////////
extern "C" void kokkosp_push_profile_region(char* regionName) {
  Tau_kokkos_stack.push(regionName);
  //TAU_VERBOSE("TAU: kokkosp_push_profile_region: %s\n", regionName);
  TAU_STATIC_PHASE_START(regionName);
}

///////////////////////////////////////////////////////////
//// pop parallel region
///////////////////////////////////////////////////////////
extern "C" void kokkosp_pop_profile_region() {
  TAU_STATIC_PHASE_STOP(Tau_kokkos_stack.top().c_str());
  //TAU_VERBOSE("TAU: kokkosp_pop_profile_region: %s\n", Tau_kokkos_stack.top().c_str());
  Tau_kokkos_stack.pop();
}
/***************************************************************************
 * $RCSfile: TauKokkos.cpp,v $   $Author: sameer $
 * $Revision: 1.0 $   $Date: 2017/02/01 22:16:23 $
 * POOMA_VERSION_ID: $Id: TauKokkos.cpp,v 1.46 2017/02/01 22:16:23 sameer Exp $
 ***************************************************************************/
