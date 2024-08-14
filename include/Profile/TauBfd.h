#ifndef _TAU_BFD_H
#define _TAU_BFD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <vector>

#define TAU_BFD_SYMTAB_LOAD_FAILED		(0)
#define TAU_BFD_SYMTAB_LOAD_SUCCESS		(1)
#define TAU_BFD_SYMTAB_NOT_LOADED		(3)

#define TAU_BFD_NULL_HANDLE				(-1)
#define TAU_BFD_NULL_MODULE_HANDLE		(-1)
#define TAU_BFD_INVALID_MODULE			(-2)

// Iterator function type.  Accepts a symbol address and name.
// The name should be an approximation of the full name, for example,
// the contents of asymbol::name from BFD.  TauBfd.cpp will
// discover the correct name as needed.
typedef void (*TauBfdIterFn)(unsigned long, const char *);

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
		probeAddr(0), filename(NULL), funcname(NULL),
                lineno(-1), discriminator(0)
	{ }

	// Makes all fields safe to query
	void secure(unsigned long addr) {
		probeAddr = addr;
		if(!funcname) {
			char * tmp = (char*)malloc(256);
			snprintf(tmp, 256,  "addr=<%p>", (void*)(addr));
			funcname = tmp;
		}
		if(!filename) filename = "(unknown)";
		if(lineno < 0) lineno = 0;
	}

	unsigned long probeAddr;
	char const * filename;
	char const * funcname;
	int lineno;
        unsigned int discriminator;
};


//
// Main interface functions
//

// Initialize TauBFD
void Tau_bfd_initializeBfd();

// Register a BFD unit (i.e. an executable and its shared libraries)
tau_bfd_handle_t Tau_bfd_registerUnit();

// free the unit vector
void Tau_delete_bfd_units(void);

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
int Tau_get_lineno_for_function(tau_bfd_handle_t handle, const char *funcname);

// Name demangling, used everywhere
char * Tau_demangle_name(const char * name);


#endif /* _TAU_BFD_H */
