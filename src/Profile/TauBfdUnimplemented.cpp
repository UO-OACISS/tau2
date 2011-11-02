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

static vector<TauBfdAddrMap*> addressMaps;

// Empty interface functions
static void Tau_bfd_internal_issueBfdWarningIfNecessary() {
  static bool warningIssued = false;
  if (!warningIssued) {
    fprintf(stderr,"TAU Warning: BFD is not available in at least one part "
    		"of this TAU-instrumented application! Please check to see if "
    		"BFD is not shared or not present. Expect some missing BFD "
    		"functionality.\n");
    warningIssued = true;
  }
}

void Tau_bfd_initializeBfdIfNecessary() {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
}

tau_bfd_handle_t Tau_bfd_registerUnit() {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return TAU_BFD_UNIMPLEMENTED_HANDLE;
}

bool Tau_bfd_checkHandle(tau_bfd_handle_t handle) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return (handle == TAU_BFD_UNIMPLEMENTED_HANDLE);
}

void Tau_bfd_updateAddressMaps(tau_bfd_handle_t handle) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  Tau_bfd_checkHandle(handle);
}

std::vector<TauBfdAddrMap*> const &
Tau_bfd_getAddressMaps(tau_bfd_handle_t handle) {
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return addressMaps;
}

TauBfdAddrMap const *
Tau_bfd_getAddressMap(tau_bfd_handle_t handle, unsigned long probe_addr)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	return NULL;
}

tau_bfd_module_handle_t Tau_bfd_getModuleHandle(tau_bfd_handle_t handle,
		unsigned long probeAddr)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_NULL_MODULE_HANDLE;
	}
	return TAU_BFD_UNIMPLEMENTED_MODULE_HANDLE;
}

bool Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
		unsigned long probeAddr, TauBfdInfo & info)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	info.secure(probeAddr);
	return false;
}

int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
		tau_bfd_module_handle_t moduleHandle,TauBfdIterFn fn)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	if (Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_SUCCESS;
	}
	return TAU_BFD_SYMTAB_LOAD_FAILED;
}

int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, TauBfdIterFn fn)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	if (Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_SUCCESS;
	}
	return TAU_BFD_SYMTAB_LOAD_FAILED;
}


//
// Deprecated interface functions maintained for backwards compatibility.
// These should be phased out soon since they do unnecessary work and
// have lead to memory leaks.
//

tau_bfd_handle_t Tau_bfd_registerUnit(int flag) {
	return Tau_bfd_registerUnit();
}

TauBfdInfo * Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
		unsigned long probe_addr)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();

	TauBfdInfo * info = new TauBfdInfo;
	Tau_bfd_resolveBfdInfo(handle, probe_addr, *info);
	return info;
}

TauBfdInfo * Tau_bfd_resolveBfdExecInfo(tau_bfd_handle_t handle,
		unsigned long probe_addr)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	return NULL;
}

int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
		tau_bfd_module_handle_t moduleHandle, int maxProbe,
		DeprecatedTauBfdIterFn fn)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	if (Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_SUCCESS;
	}
	return TAU_BFD_SYMTAB_LOAD_FAILED;
}

int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, int maxProbe,
		DeprecatedTauBfdIterFn fn)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	if (Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_SUCCESS;
	}
	return TAU_BFD_SYMTAB_LOAD_FAILED;
}

//
// Deprecated query functions maintained for backwards compatibility.
// These should be phased out soon since they do unnecessary work and
// have lead to memory leaks.
//

int Tau_bfd_getAddressMap(tau_bfd_handle_t handle, unsigned long probe_addr,
		TauBfdAddrMap * mapInfo)
{
	Tau_bfd_internal_issueBfdWarningIfNecessary();
	return 0;
}
