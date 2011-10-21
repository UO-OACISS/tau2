#ifndef _TAU_BFD_H
#define _TAU_BFD_H

#include <vector>
#include <stdint.h>

#define TAU_BFD_SYMTAB_LOAD_FAILED		(0)
#define TAU_BFD_SYMTAB_LOAD_SUCCESS		(1)
#define TAU_BFD_SYMTAB_NOT_LOADED		(3)

#define TAU_BFD_NULL_HANDLE				(-1)
#define TAU_BFD_NULL_MODULE_HANDLE		(-1)
#define TAU_BFD_INVALID_MODULE			(-2)

typedef int tau_bfd_handle_t;
typedef int tau_bfd_module_handle_t;

/* An iterator function that will take in values generated: */
/*   function name, file name, line number */
/*   by BFD functionality and do some user-defined work on them. */
typedef struct HashNode * (*TauBfdIterFn)(
		unsigned long, const char *, const char *, int);

struct TauBfdAddrMap
{
	TauBfdAddrMap() :
		start(0), end(0), offset(0)
	{ }

	TauBfdAddrMap(unsigned long _start, unsigned long _end,
			unsigned long _offset, char const * _name) :
		start(_start), end(_end), offset(_offset)
	{
		strncpy(name, _name, sizeof(name));
		name[sizeof(name)-1] = '\0';
	}

	unsigned long start;
	unsigned long end;
	unsigned long offset;
	char name[512];
};

struct TauBfdInfo
{
	TauBfdInfo() :
		probeAddr(0), filename(NULL), funcname(NULL), lno(-1)
	{ }

	TauBfdInfo(unsigned long _probeAddr, char const * _filename,
			char const * _funcname, int _lno) :
		probeAddr(_probeAddr), filename(_filename),
		funcname(_funcname), lno(_lno)
	{ }

	uintptr_t probeAddr;
	char const * filename;
	char const * funcname;
	unsigned int lno;
};


//
// Main interface functions
//

void Tau_bfd_initializeBfdIfNecessary();
tau_bfd_handle_t Tau_bfd_registerUnit();
bool Tau_bfd_checkHandle(tau_bfd_handle_t handle);
void Tau_bfd_updateAddressMaps(tau_bfd_handle_t handle);
bool Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
		uintptr_t probeAddr, TauBfdInfo & info);
bool Tau_bfd_resolveBfdExecInfo(tau_bfd_handle_t handle,
		uintptr_t probeAddr, TauBfdInfo & info);
//int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
//		tau_bfd_module_handle_t moduleHandle, TauBfdIterFn fn);
int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, TauBfdIterFn fn);

//
// Query functions
//

std::vector<TauBfdAddrMap*> & Tau_bfd_getAddressMaps(tau_bfd_handle_t handle);

TauBfdAddrMap const * Tau_bfd_getAddressMap(
		tau_bfd_handle_t handle, uintptr_t probeAddr);

tau_bfd_module_handle_t Tau_bfd_getModuleHandle(
		tau_bfd_handle_t handle, uintptr_t probeAddr);

#endif /* _TAU_BFD_H */
