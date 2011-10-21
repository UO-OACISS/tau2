// *CWL* This code is used for two (rather different) purposes.
//
//    1. When a shared library for BFD is not available and -optShared
//       is required as an option to build the application.
//
//    2. When BFD is not available at all (the interface is still required).

/* Not exactly empty. Support for address maps are independent of BFD */
#include <TAU.h>
#include <Profile/TauBfd.h>

#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
using namespace std;

#define TAU_BFD_UNIMPLEMENTED_HANDLE -1337
#define TAU_BFD_UNIMPLEMENTED_MODULE_HANDLE -11337

// Empty interface functions
static void Tau_bfd_internal_issueBfdWarningIfNecessary() {
  static bool warningIssued = false;
  if (!warningIssued) {
    fprintf(stderr,"TAU Warning: BFD is not available in at least one part of this TAU-instrumented application! Please check to see if BFD is not shared or not present. Expect some missing BFD functionality.\n");
    warningIssued = true;
  }
}

void Tau_bfd_initializeBfdIfNecessary() {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
}

tau_bfd_handle_t Tau_bfd_registerUnit(int flag) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  addressMaps = new vector<TauBfdAddrMap>();
  return TAU_BFD_UNIMPLEMENTED_HANDLE;
}

bool Tau_bfd_checkHandle(tau_bfd_handle_t handle) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  if (handle == TAU_BFD_UNIMPLEMENTED_HANDLE) {
    return true;
  }
  return false;
}

void Tau_bfd_updateAddressMaps(tau_bfd_handle_t handle) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  if (!Tau_bfd_checkHandle(handle)) {
    return;
  }
}

vector<TauBfdAddrMap> *Tau_bfd_getAddressMaps(tau_bfd_handle_t handle) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  if (!Tau_bfd_checkHandle(handle)) {
    return NULL;
  }
  return addressMaps;
}

tau_bfd_module_handle_t Tau_bfd_getModuleHandle(tau_bfd_handle_t handle,
		uintptr_t probeAddr) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  if (!Tau_bfd_checkHandle(handle)) {
    return TAU_BFD_NULL_MODULE_HANDLE;
  }
  return TAU_BFD_UNIMPLEMENTED_MODULE_HANDLE;
}

bool Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
		uintptr_t probe_addr, TauBfdInfo & info)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	return false;
}

bool Tau_bfd_resolveBfdExecInfo(tau_bfd_handle_t handle,
		uintptr_t probe_addr, TauBfdInfo & info) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return false;
}

int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
		tau_bfd_module_handle_t moduleHandle, TauBfdIterFn fn)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	}
	return TAU_BFD_SYMTAB_LOAD_UNRESOLVED;
}

int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, TauBfdIterFn fn)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	}
	return TAU_BFD_SYMTAB_LOAD_UNRESOLVED;
}
