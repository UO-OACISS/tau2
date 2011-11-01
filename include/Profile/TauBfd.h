#ifndef _TAU_BFD_H
#define _TAU_BFD_H

#include <vector>
#include <stdint.h>
/* *CWL* - you will need these headers for portability if you 
   have code in this header that depends on external modules.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define TAU_BFD_SYMTAB_LOAD_FAILED		(0)
#define TAU_BFD_SYMTAB_LOAD_SUCCESS		(1)
#define TAU_BFD_SYMTAB_NOT_LOADED		(3)

#define TAU_BFD_NULL_HANDLE				(-1)
#define TAU_BFD_NULL_MODULE_HANDLE		(-1)
#define TAU_BFD_INVALID_MODULE			(-2)

// Deprecated flag maintained for backwards compatibility.
// In past, TauBfd could only resolve up to 3000 symbols
// and this flag would indicate that some symbols were not
// resolved.  This has since been fixed so all relevant
// symbols are resolved efficiently. Hence, this flag no
// longer has any meaning.
#define TAU_BFD_SYMTAB_LOAD_UNRESOLVED 2

// Deprecated flags maintained for backwards compatibility.
// In past, these flags were used to try to avoid duplicating
// common data.  TauBfd.cpp has since been updated so that
// there is no duplication of data in any case, so these
// flags no longer have any meaning.
#define TAU_BFD_KEEP_GLOBALS 0
#define TAU_BFD_REUSE_GLOBALS 1

// Iterator function type.  Accepts a symbol address and name.
// The name should be an approximation of the full name, for example,
// the contents of asymbol::name from BFD.  TauBfd.cpp will
// discover the correct name as needed.
typedef void (*TauBfdIterFn)(unsigned long, const char *);

// Deprecated iterator function type.  Accepts a symbol addresss,
// name, filename, and line number.  In past, this was used to
// iterate over all module symbol tables as the modules were
// "touched" by a function call and set the full demangled
// function name, filename, line and line number for every symbol
// in the module (up to 3000).  That approach was not scalable.
// In the new approach, the symbol name is quickly approximated.
// The expensive BFD search for the full function name, filename
// and line number is only performed as needed.  For large
// applications, the new approach is orders of magnitude faster
// than the old. Hence, it is recommended that this not be used.
typedef void (*DeprecatedTauBfdIterFn)(unsigned long, const char*, const char*, int);

typedef int tau_bfd_handle_t;
typedef int tau_bfd_module_handle_t;

struct TauBfdAddrMap
{
	TauBfdAddrMap() :
		start(0), end(0), offset(0)
	{ }

	TauBfdAddrMap(unsigned long _start, unsigned long _end,
			unsigned long _offset, char const * _name) :
		start(_start), end(_end), offset(_offset)
	{
		// Safely copy the name string and always
		// end with a NUL char.
		int end = 1;
		if(_name != NULL) {
			strncpy(name, _name, sizeof(name));
			end = sizeof(name);
		}
		name[end-1] = '\0';
	}

	unsigned long start;
	unsigned long end;
	unsigned long offset;
	char name[512];
};

struct TauBfdInfo
{
	TauBfdInfo() :
		probeAddr(0), filename(NULL), funcname(NULL), lineno(-1)
	{ }

	TauBfdInfo(unsigned long _probeAddr, char const * _filename,
			char const * _funcname, int _lineno) :
		probeAddr(_probeAddr), filename(_filename),
		funcname(_funcname), lineno(_lineno)
	{ }

	// Makes all fields safe to query
	void secure(unsigned long addr) {
		probeAddr = addr;
		if(funcname == NULL) {
			funcname = (char*)malloc(64);
			sprintf((char*)funcname, "addr=<%p>", addr);
		}
		if(filename == NULL) filename = "(unknown)";
		if(lineno < 0) lineno = 0;
	}

	unsigned long probeAddr;
	char const * filename;
	char const * funcname;
	int lineno;
};


//
// Main interface functions
//

// Initialize TauBFD
void Tau_bfd_initializeBfdIfNecessary();

// Register a BFD unit (i.e. an executable and its shared libraries)
tau_bfd_handle_t Tau_bfd_registerUnit();

// Return true if the given handle is valid
bool Tau_bfd_checkHandle(tau_bfd_handle_t handle);

// Scan the BFD unit for address maps
void Tau_bfd_updateAddressMaps(tau_bfd_handle_t handle);

// Forward lookup of symbol information for a given address.
// Searches the appropriate shared libraries and/or the executable
// for information.
bool Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
		unsigned long probeAddr, TauBfdInfo & info);

// Fast scan of the executable symbol table.
int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, TauBfdIterFn fn);

// Fast scan of a module symbol table.
// Note that it's usually not worth doing this since it is unlikely
// that all symbols in the module will be required in a single
// application (e.g. a shared library in a large application).
// Instead, use Tau_bfd_resolveBfdInfo as needed.
int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
		tau_bfd_module_handle_t moduleHandle,TauBfdIterFn fn);

//
// Query functions
//

// Get the address maps in the specified BFD unit
std::vector<TauBfdAddrMap*> const &
Tau_bfd_getAddressMaps(tau_bfd_handle_t handle);

// Find the address map that probably contains the given address
TauBfdAddrMap const *
Tau_bfd_getAddressMap(tau_bfd_handle_t handle, unsigned long probeAddr);

// Get the module that possibly defines the given address
tau_bfd_module_handle_t
Tau_bfd_getModuleHandle(tau_bfd_handle_t handle, unsigned long probeAddr);

//
// Deprecated interface functions maintained for backwards compatibility.
// These should be phased out soon since they do unnecessary work and
// have lead to memory leaks.
//

tau_bfd_handle_t Tau_bfd_registerUnit(int flag);
TauBfdInfo * Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
		unsigned long probe_addr);
TauBfdInfo * Tau_bfd_resolveBfdExecInfo(tau_bfd_handle_t handle,
		unsigned long probe_addr);
int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
		tau_bfd_module_handle_t moduleHandle, int maxProbe,
		DeprecatedTauBfdIterFn fn);
int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, int maxProbe,
		DeprecatedTauBfdIterFn fn);
int Tau_bfd_getAddressMap(tau_bfd_handle_t handle, unsigned long probe_addr,
		TauBfdAddrMap * mapInfo);

#endif /* _TAU_BFD_H */
