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

typedef int tau_bfd_handle_t;
typedef int tau_bfd_module_handle_t;

static vector<TauBfdAddrMap*> addressMaps;

// Empty interface functions
static void Tau_bfd_internal_issueBfdWarningIfNecessary()
{
  static bool warningIssued = false;
  if (!warningIssued) {
#ifndef __APPLE__
    fprintf(stderr, "TAU Warning: BFD is not available in at least one part "
        "of this TAU-instrumented application! Please check to see if "
        "BFD is not shared or not present. Expect some missing BFD "
        "functionality.\n");
#endif
    warningIssued = true;
  }
}

void Tau_bfd_initializeBfd()
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
}

tau_bfd_handle_t Tau_bfd_registerUnit()
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return TAU_BFD_UNIMPLEMENTED_HANDLE;
}

bool Tau_bfd_checkHandle(tau_bfd_handle_t handle)
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return (handle == TAU_BFD_UNIMPLEMENTED_HANDLE);
}

void Tau_bfd_updateAddressMaps(tau_bfd_handle_t handle)
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  Tau_bfd_checkHandle(handle);
}

bool Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle, unsigned long probeAddr, TauBfdInfo & info)
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  info.secure(probeAddr);
  return false;
}

int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, TauBfdIterFn fn)
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return TAU_BFD_SYMTAB_LOAD_FAILED;
}

int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle, tau_bfd_module_handle_t moduleHandle, TauBfdIterFn fn)
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return TAU_BFD_SYMTAB_LOAD_FAILED;
}

std::vector<TauBfdAddrMap*> const & Tau_bfd_getAddressMaps(tau_bfd_handle_t handle)
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return addressMaps;
}

TauBfdAddrMap const * Tau_bfd_getAddressMap(tau_bfd_handle_t handle, unsigned long probe_addr)
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  return NULL;
}

tau_bfd_module_handle_t Tau_bfd_getModuleHandle(tau_bfd_handle_t handle, unsigned long probeAddr)
{
  Tau_bfd_internal_issueBfdWarningIfNecessary();
  if (!Tau_bfd_checkHandle(handle)) {
    return TAU_BFD_NULL_MODULE_HANDLE;
  }
  return TAU_BFD_UNIMPLEMENTED_MODULE_HANDLE;
}

void Tau_delete_bfd_units(void) {
  return;
}

/* Dummy instantiation for tau_exec -io -memory support when Binutils aren't included */

typedef int * (*objopen_counter_t)(void);
extern "C" void Tau_bfd_register_objopen_counter(objopen_counter_t handle) {
  return;
}

/* If no bfd + demangle, and we have C++ support, use it */
#if defined(__GNUC__)
#include <cxxabi.h>
char * Tau_demangle_name(const char * name) {
    int status;
    char * dem_name = abi::__cxa_demangle(name, 0, 0, &status);
    if (status != 0 || dem_name == nullptr) {
        switch (status) {
            case 0:
                TAU_VERBOSE("The demangling operation succeeded, but realname is NULL\n");
                break;
            case -1:
                TAU_VERBOSE("The demangling operation failed:");
                TAU_VERBOSE(" A memory allocation failiure occurred.\n");
                break;
            case -2:
                TAU_VERBOSE("The demangling operation failed:");
                TAU_VERBOSE(" '%s' is not a valid", name);
                TAU_VERBOSE(" name under the C++ ABI mangling rules.\n");
                break;
            case -3:
                TAU_VERBOSE("The demangling operation failed: One of the");
                TAU_VERBOSE(" arguments is invalid.\n");
                break;
            default:
                TAU_VERBOSE("The demangling operation failed: Unknown error.\n");
                break;
        }
		dem_name = strdup(name);
    }
    TAU_VERBOSE("Demangled: '%s'\n", dem_name);
    return dem_name;
}
/* No support for either, just return the name */
#else
char * Tau_demangle_name(const char * name) {
    TAU_VERBOSE("Warning: No demangling support provided...\n");
    dem_name = strdup(name);
    return dem_name;
}
#endif // #if defined(__GNUC__)
