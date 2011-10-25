/* *CWL* Temporary fix. An appropriate reconciliation of the semantics of the two versions
   will need to be explored and implemented at some point */

/***********************************
 * ******* TAU MINGW VERSION *******
 ***********************************/
#ifdef TAU_MINGW

#ifdef TAU_BFD

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>

#include <TAU.h>
#include <Profile/TauBfd.h>
#include <bfd.h>
#include <dirent.h>
#include <stdint.h>

#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#define HAVE_DECL_BASENAME 1
#include <demangle.h>
#define DEMANGLE_FLAGS (DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE | DMGL_TYPES)
#endif /* HAVE_GNU_DEMANGLE */

#ifdef TAU_BGP
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* _GNU_SOURCE */
#include <link.h>
#endif /* TAU_BGP */

#if defined(TAU_WINDOWS) && defined(TAU_MINGW)
#include <windows.h>
#include <psapi.h>
#endif

using namespace std;

static char const * Tau_bfd_internal_getExecutablePath();

struct TauBfdModule
{
	TauBfdModule() :
		bfdImage(NULL), syms(NULL), nr_all_syms(0), bfdOpen(false),
		processCode(TAU_BFD_SYMTAB_NOT_LOADED)
	{ }

	~TauBfdModule() {
		free(syms);
		delete bfdImage;
	}

	bool loadSymbolTable(char const * path) {

		// Executable symbol table is already loaded.
		if (bfdOpen) return true;

		Tau_bfd_initializeBfdIfNecessary();

		if (!(bfdImage = bfd_openr(path, 0))) {
			TAU_VERBOSE("loadSymbolTable: Failed to open [%s]\n", path);
			return (bfdOpen = false);
		}

		if (!bfd_check_format(bfdImage, bfd_object)) {
			TAU_VERBOSE("loadSymbolTable: bfd format check failed [%s]\n", path);
			return (bfdOpen = false);
		}

		if (!(bfd_get_file_flags(bfdImage) & HAS_SYMS)) {
			TAU_VERBOSE("loadSymbolTable: bfd has no symbols [%s]\n", path);
			return (bfdOpen = false);
		}

		size_t size = bfd_get_symtab_upper_bound(bfdImage);
		if (size < 1) {
			TAU_VERBOSE("loadSymbolTable: bfd_get_symtab_upper_bound() < 1 [%s]\n", path);
			return (bfdOpen = false);
		}

		syms = (asymbol **) malloc(size);
		nr_all_syms = bfd_canonicalize_symtab(bfdImage, syms);
		bfdOpen = nr_all_syms > 0;

		return bfdOpen;
	}

	bfd *bfdImage;
	asymbol **syms;
	size_t nr_all_syms;

	// For EBS book-keeping
	bool bfdOpen; // once open, symtabs are loaded and never released

	// Remember the result of the last process to avoid reprocessing
	int processCode;
};


struct TauBfdUnit
{
	TauBfdUnit() {
		executablePath = Tau_bfd_internal_getExecutablePath();
		executableModule = new TauBfdModule;
	}

	void ClearMaps() {
		for(size_t i=0; i<addressMaps.size(); ++i) {
			delete addressMaps[i];
		}
		addressMaps.clear();
	}

	void ClearModules() {
		for(size_t i=0; i<modules.size(); ++i) {
			delete modules[i];
		}
		modules.clear();
	}

	char const * executablePath;
	TauBfdModule * executableModule;
	vector<TauBfdAddrMap*> addressMaps;
	vector<TauBfdModule*> modules;
};


struct LocateAddressData
{
	LocateAddressData(TauBfdModule * _module) :
		found(false), module(_module)
	{ }

	bool found;
	TauBfdModule * module;
	TauBfdInfo info;
};

// Internal function prototypes
static bool Tau_bfd_internal_loadSymTab(TauBfdUnit *unit, int moduleIndex);
static bool Tau_bfd_internal_loadExecSymTab(TauBfdUnit *unit);
static int Tau_bfd_internal_getModuleIndex(
		TauBfdUnit *unit, unsigned long probe_addr);
static TauBfdModule * Tau_bfd_internal_getModuleFromIdx(
		TauBfdUnit *unit, int moduleIndex);
static unsigned long Tau_bfd_internal_getOffsetAddress(
		TauBfdUnit *unit, int moduleIndex, unsigned long probe_addr);
static void Tau_bfd_internal_addExeAddressMap();
static void Tau_bfd_internal_locateAddress(
		bfd *bfdptr, asection *section, void *data ATTRIBUTE_UNUSED);
static int Tau_bfd_internal_getBGPJobID(const char *path, char *name);
static void Tau_bfd_internal_updateProcSelfMaps(TauBfdUnit *unit);
static void Tau_bfd_internal_updateBGPMaps(TauBfdUnit *unit);

// BFD units (e.g. executables and their dynamic libraries)
vector<TauBfdUnit*> bfdUnits;

//
// Main interface functions
//

void Tau_bfd_initializeBfdIfNecessary() {
  static bool bfdInitialized = false;
  if (!bfdInitialized) {
    bfd_init();
    bfdInitialized = true;
  }
}

tau_bfd_handle_t Tau_bfd_registerUnit()
{
	tau_bfd_handle_t ret = bfdUnits.size();
	bfdUnits.push_back(new TauBfdUnit);

	TAU_VERBOSE("Tau_bfd_registerUnit: Unit %d registered and initialized\n", ret);

	// Initialize the first address maps for the unit.
	Tau_bfd_updateAddressMaps(ret);

	return ret;
}

bool Tau_bfd_checkHandle(tau_bfd_handle_t handle)
{
  if (handle == TAU_BFD_NULL_HANDLE) {
    TAU_VERBOSE("TauBfd: Warning - attempt to use uninitialized BFD handle\n");
    return false;
  }
  if (handle >= bfdUnits.size()) {
    TAU_VERBOSE("TauBfd: Warning - invalid BFD unit handle %d, max value %d\n",
    		handle, bfdUnits.size());
    return false;
  }
  //  TAU_VERBOSE("TauBfd: Valid BFD Handle\n");
  return true;
}

static void Tau_bfd_internal_updateProcSelfMaps(TauBfdUnit *unit)
{
	// *CWL* - This is important! We DO NOT want to use /proc/self/maps on
	//         the BGP because the information acquired comes from the I/O nodes
	//         and not the compute nodes. You could end up with an overlapping
	//         range for address resolution if used!
#ifndef TAU_BGP
	// *JCL* - Windows has no /proc filesystem, so don't try to use it
#ifndef TAU_WINDOWS

	// Note: Linux systems only.
	FILE *mapsfile = fopen("/proc/self/maps", "r");

	// *JCL* - Check that mapsfile actually opened to prevent future problems
	if(mapsfile == NULL) {
		TAU_VERBOSE("Tau_bfd_internal_updateProcSelfMaps: Warning - "
				"/proc/self/maps could not be opened.\n");
		return;
	}

	char line[4096];
	// count is used for TAU_VERBOSE only
	int count = 0;
	while (!feof(mapsfile)) {
		fgets(line, 4096, mapsfile);
		unsigned long start, end, offset;
		char module[4096];
		char perms[5];
		module[0] = 0;

		sscanf(line, "%lx-%lx %s %lx %*s %*u %[^\n]",
				&start, &end, perms, &offset, module);
		if (*module && ((strcmp(perms, "r-xp") == 0) ||
						(strcmp(perms, "rwxp") == 0)))
		{
			TAU_VERBOSE("[%d] Module: %s, %p-%p (%d)\n",
					count++, module, start, end, offset);
			unit->addressMaps.push_back(new TauBfdAddrMap(start, end, offset, module));
			unit->modules.push_back(new TauBfdModule);
		}
	}
	fclose(mapsfile);
#endif /* TAU_WINDOWS */
#endif /* TAU_BGP */
}

#ifdef TAU_BGP
static int Tau_bfd_internal_BGP_dl_iter_callback(
		struct dl_phdr_info *info, size_t size, void *data)
{
	if (strlen(info->dlpi_name) == 0) {
		TAU_VERBOSE("Tau_bfd_internal_BGP_dl_iter_callback: Nameless module. Ignored.\n");
		return 0;
	}

	TauBfdUnit *unit = (TauBfdUnit *)data;
	TauBfdAddrMap * map = new TauBfdAddrMap;

	map->start = (unsigned long)info->dlpi_addr;
	// assumption - the max of the physical addresses of each segment added
	//              to the memory size yields the end of the address range.
	//              It is unclear if any of the other factors affect it.
	unsigned long max_addr = 0x0;
	for (int j=0; j<info->dlpi_phnum; j++) {
		unsigned long local_max =
				(unsigned long)info->dlpi_phdr[j].p_paddr +
				(unsigned long)info->dlpi_phdr[j].p_memsz;
		if (local_max > max_addr) {
			max_addr = local_max;
		}
	}
	map->end = map->start + max_addr;
	map->offset = 0; // assume.
	sprintf(map->name, "%s", info->dlpi_name);
	TAU_VERBOSE("BGP Module: %s, %p-%p (%d)\n",
			map->name, map->start, map->end, map->offset);
	unit->addressMaps.push_back(map);
	unit->modules.push_back(new TauBfdModule);
	return 0;
}
#endif /* TAU_BGP */

static void Tau_bfd_internal_updateBGPMaps(TauBfdUnit *unit) {
#ifdef TAU_BGP
    dl_iterate_phdr(Tau_bfd_internal_BGP_dl_iter_callback, (void *)unit);
#endif /* TAU_BGP */
}


// *JCL* - Executables compiled by MinGW are strange beasts in that
//         they use GNU debugger symbols, but are Windows executables.
//         BFD support for windows is incomplete (e.g. dl_iterate_phdr
//         is not implemented and probably never will be), so we must
//         use the Windows API to walk through the PE imports directory
//         to discover our external modules (e.g. DLLs).  However, we
//         still need BFD to parse the GNU debugger symbols.  In fact,
//         the DEBUG PE header of an executable produced by MinGW is
//         just an empty table.
static void Tau_bfd_internal_updateWindowsMaps(TauBfdUnit *unit)
{
#if defined(TAU_WINDOWS) && defined(TAU_MINGW)

	// Use Windows Process API to find modules
	// This is preferable to walking the PE file headers with
	// Windows API calls because it provides a more complete
	// and accurate picture of the process memory layout, and
	// memory addresses aren't truncated on 64-bit Windows.

	HMODULE hMod[1024];		// Handles for each module
	HANDLE hProc;			// A handle on the current process
	DWORD cbNeeded;			// Bytes needed to store all handles
	MODULEINFO modInfo;		// Information about a module
	int count = 0;			// for TAU_VERBOSE only

	// Get the process handle
	hProc = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
			FALSE, GetCurrentProcessId());
	if (hProc == NULL) {
		TAU_VERBOSE("Tau_bfd_internal_updateWindowsMaps: "
				"Cannot get process handle.\n");
		return;
	}

	// Get handles on all modules in this process
	if (EnumProcessModules(hProc, hMod, sizeof(hMod), &cbNeeded) == 0) {
		TAU_VERBOSE("Tau_bfd_internal_updateWindowsMaps: "
				"Cannot enumerate process modules.\n");
		return;
	}

	// Calculate number of handles enumerated
	size_t const nModHandle = cbNeeded / sizeof(HMODULE);

	// Iterate over module handles
	for(size_t i=0; i<nModHandle; ++i) {

		// Get the module information structure
		if(GetModuleInformation(hProc, hMod[i], &modInfo, sizeof(modInfo)) == 0) {
			TAU_VERBOSE("Tau_bfd_internal_updateWindowsMaps: "
					"Cannot get module info (handle 0x%x).\n", hMod[i]);
			continue;
		}

		// Create a new BFD map for this module
		TauBfdAddrMap * map = new TauBfdAddrMap;
		map->start = Tau_convert_ptr_to_unsigned_long(modInfo.lpBaseOfDll);
		map->end = map->start + modInfo.SizeOfImage;
		map->offset = 0; // TODO: What's this supposed to be?

		// Get the full module path name for the map
		if(GetModuleFileNameEx(hProc, hMod[i], map->name, sizeof(map->name)) == 0) {
			TAU_VERBOSE("Tau_bfd_internal_updateWindowsMaps: "
					"Cannot get absolute path to module (handle 0x%x).\n", hMod[i]);
			continue;
		}

		TAU_VERBOSE("[%d] Module: %s, %p-%p (%d)\n",
				count++, map->name, map->start, map->end, map->offset);

		unit->addressMaps.push_back(map);
		unit->modules.push_back(new TauBfdModule);
	}

	// Release the process handle
	CloseHandle(hProc);

#endif /* TAU_WINDOWS && TAU_MINGW */
}

void Tau_bfd_updateAddressMaps(tau_bfd_handle_t handle)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return;
	}

	TAU_VERBOSE("Tau_bfd_updateAddressMaps: Updating object maps\n");
	TauBfdUnit * unit = bfdUnits[handle];

	unit->ClearMaps();
	unit->ClearModules();

#if defined(TAU_BGP)
	Tau_bfd_internal_updateBGPMaps(unit);
#elif defined(TAU_WINDOWS) && defined(TAU_MINGW)
	Tau_bfd_internal_updateWindowsMaps(unit);
#else
	Tau_bfd_internal_updateProcSelfMaps(unit);
#endif

	TAU_VERBOSE("Tau_bfd_updateAddressMaps: %d modules discovered\n",
			unit->modules.size());
}

vector<TauBfdAddrMap*> & Tau_bfd_getAddressMaps(tau_bfd_handle_t handle)
{
	Tau_bfd_checkHandle(handle);
	return bfdUnits[handle]->addressMaps;
}

tau_bfd_module_handle_t Tau_bfd_getModuleHandle(tau_bfd_handle_t handle,
		uintptr_t probeAddr)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_INVALID_MODULE;
	}
	TauBfdUnit *unit = bfdUnits[handle];

	int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probeAddr);
	if (matchingIdx != -1) {
		return (tau_bfd_module_handle_t) matchingIdx;
	}
	return TAU_BFD_NULL_MODULE_HANDLE;
}

TauBfdAddrMap const *
Tau_bfd_getAddressMap(tau_bfd_handle_t handle, unsigned long probe_addr)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return NULL;
	}

	TauBfdUnit *unit = bfdUnits[handle];
	int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probe_addr);
	if (matchingIdx == -1) {
		return NULL;
	}

	return unit->addressMaps[matchingIdx];
}

// Probe for BFD information given a single address.
bool Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
		uintptr_t probeAddr, TauBfdInfo & info)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return false;
	}
	TauBfdUnit *unit = bfdUnits[handle];

	int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probeAddr);
	if (matchingIdx == -1) {
		return false;
	}

	if (!Tau_bfd_internal_loadSymTab(unit, matchingIdx)) {
		return false;
	}

	TauBfdModule *module = Tau_bfd_internal_getModuleFromIdx(unit, matchingIdx);

	// Calculate first and second search addresses
#if defined(TAU_WINDOWS) && defined(TAU_MINGW)
	unsigned long firstAddr = probeAddr;
	unsigned long secondAddr = Tau_bfd_internal_getOffsetAddress(
			unit, matchingIdx, probeAddr);
#else
	unsigned long firstAddr = Tau_bfd_internal_getOffsetAddress(
			unit, matchingIdx, probeAddr);
	unsigned long secondAddr = probeAddr;
#endif

	// Convert address to something bfd can use.
	char hex_pc_string[100];
	sprintf(hex_pc_string, "%p", firstAddr);

	// Search BFD sections for address
	LocateAddressData data(module);
	data.info.probeAddr = bfd_scan_vma(hex_pc_string, NULL, 16);
	bfd_map_over_sections(module->bfdImage, Tau_bfd_internal_locateAddress, &data);

	if (!data.found) {
		// if not initially found, try again with second address
		if (firstAddr != secondAddr) {
//			TAU_VERBOSE("Tau_bfd_resolveBfdInfo: "
//					"Trying alternate address [%p] from address [%p]\n",
//					secondAddr, firstAddr);
			sprintf(hex_pc_string, "%p", secondAddr);
			data.info.probeAddr = bfd_scan_vma(hex_pc_string, NULL, 16);
			bfd_map_over_sections(module->bfdImage, Tau_bfd_internal_locateAddress, &data);
		}
	}

	bool resolved = data.found && (data.info.funcname != NULL);
	if (resolved) {
		info = data.info;
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
		char const * demangled = bfd_demangle(module->bfdImage,
						info.funcname, DEMANGLE_FLAGS);
		if (demangled) {
			info.funcname = demangled;
		}
#endif
	}
	return resolved;
}

bool Tau_bfd_resolveBfdExecInfo(tau_bfd_handle_t handle,
		uintptr_t probeAddr, TauBfdInfo & info)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return false;
	}
	TauBfdUnit * unit = bfdUnits[handle];
	TauBfdModule * module = unit->executableModule;

	if (!Tau_bfd_internal_loadExecSymTab(unit)) {
		return false;
	}

	// Convert address to something bfd can use.
	char hex_pc_string[100];
	sprintf(hex_pc_string, "%p", probeAddr);

	// Search BFD sections for address
	// If we can only fail once for the executable
	LocateAddressData data(module);
	data.info.probeAddr = bfd_scan_vma(hex_pc_string, NULL, 16);
	bfd_map_over_sections(module->bfdImage, Tau_bfd_internal_locateAddress, &data);

	bool resolved = data.found && (data.info.funcname != NULL);
	if (resolved) {
		info = data.info;
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
		char const * demangled = bfd_demangle(module->bfdImage,
						info.funcname, DEMANGLE_FLAGS);
		if (demangled) {
			info.funcname = demangled;
		}
#endif
	}
	return resolved;
}

static void Tau_bfd_internal_iterateOverSymTab(
		TauBfdModule * module, TauBfdIterFn fn)
{
	for(asymbol ** s=module->syms; *s; s++)
	{
		asymbol const & asym = **s;

		// Skip useless symbols (e.g. line numbers)
		// It would be easier to use BFD_FLAGS, but those aren't reliable
		// since the debug symbol format is unpredictable
		if (!asym.name || !asym.section->size) {
			continue;
		}

		// Calculate symbol address
		unsigned long addr = asym.section->vma + asym.value;

		// Get symbol name
		char const * name = asym.name;
		if(name[0] == '.') {
			char const * mark = strchr((char*)name, '$');
			if(mark) name = mark + 1;
		}

		// Apply the iterator function
		// Names will be resolved and demangled later
		fn(addr, name, NULL, 0);
	}

	module->processCode = TAU_BFD_SYMTAB_LOAD_SUCCESS;
}

#if 0
// Run a unit-defined iterator through symbols discovered in a Bfd module.
//   If there are too many symbols in the module, we will avoid resolving
//   the symbols but allow the iterator to perform its work.
int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
		tau_bfd_module_handle_t moduleHandle, TauBfdIterFn fn)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	}
	TauBfdUnit * unit = bfdUnits[handle];

	int moduleIndex = (int)moduleHandle;
	char const * moduleName = unit->addressMaps[moduleIndex]->name;
	TauBfdModule * module = Tau_bfd_internal_getModuleFromIdx(unit, moduleIndex);

	// Only process each module once.
	if(module->processCode != TAU_BFD_SYMTAB_NOT_LOADED) {
		TAU_VERBOSE("Tau_bfd_processBfdModuleInfo: "
				"%s was already processed (code %d).  Will not reprocess.\n",
				moduleName, module->processCode);
		return module->processCode;
	}
	TAU_VERBOSE("Tau_bfd_processBfdModuleInfo: processing %s\n", moduleName);

	// Make sure module symbol table is loaded
	if (!Tau_bfd_internal_loadSymTab(unit, moduleIndex)) {
		module->processCode = TAU_BFD_SYMTAB_LOAD_FAILED;
		return module->processCode;
	}

	// Apply the iterator function to all symbols in the table
	Tau_bfd_internal_iterateOverSymTab(module, fn);
	return module->processCode;
}
#endif

int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, TauBfdIterFn fn)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	}
	TauBfdUnit * unit = bfdUnits[handle];

	char const * execName = unit->executablePath;
	TauBfdModule * module = unit->executableModule;

	// Only process the executable once.
	if(module->processCode != TAU_BFD_SYMTAB_NOT_LOADED) {
		TAU_VERBOSE("Tau_bfd_processBfdExecInfo: "
				"%s already processed (code %d).  Will not reprocess.\n",
				execName, module->processCode);
		return module->processCode;
	}
	TAU_VERBOSE("Tau_bfd_processBfdExecInfo: processing executable %s\n", execName);

	// Make sure executable symbol table is loaded
	if (!Tau_bfd_internal_loadExecSymTab(unit)) {
		module->processCode = TAU_BFD_SYMTAB_LOAD_FAILED;
		return module->processCode;
	}

	// Apply the iterator function to all symbols in the table
	Tau_bfd_internal_iterateOverSymTab(module, fn);
	return module->processCode;
}

static bool Tau_bfd_internal_loadSymTab(TauBfdUnit *unit, int moduleIndex)
{
	if ((moduleIndex == TAU_BFD_NULL_MODULE_HANDLE) ||
		(moduleIndex == TAU_BFD_INVALID_MODULE)) {
		return false;
	}

	char const * name = unit->addressMaps[moduleIndex]->name;
	TauBfdModule * module = Tau_bfd_internal_getModuleFromIdx(unit, moduleIndex);

	return module->loadSymbolTable(name);
}

static bool Tau_bfd_internal_loadExecSymTab(TauBfdUnit *unit)
{
	char const * name = unit->executablePath;
	TauBfdModule * module = unit->executableModule;

	return module->loadSymbolTable(name);
}

// Internal BFD helper functions
static int Tau_bfd_internal_getModuleIndex(
		TauBfdUnit *unit, unsigned long probe_addr)
{
	vector<TauBfdAddrMap*> & addressMaps = unit->addressMaps;
	for (int i = 0; i < addressMaps.size(); i++) {
		if (probe_addr >= addressMaps[i]->start &&
			probe_addr <= addressMaps[i]->end)
			return i;
	}
	return -1;
}

static unsigned long Tau_bfd_internal_getOffsetAddress(
		TauBfdUnit *unit, int moduleIndex, unsigned long probe_addr)
{
	if (moduleIndex == TAU_BFD_NULL_MODULE_HANDLE) {
		return probe_addr;
	}
	return (probe_addr - unit->addressMaps[moduleIndex]->start);
}

static TauBfdModule *Tau_bfd_internal_getModuleFromIdx(
		TauBfdUnit *unit, int moduleIndex)
{
	if (moduleIndex == -1) {
		return unit->executableModule;
	}
	return unit->modules[moduleIndex];
}

static int Tau_bfd_internal_getBGPJobID(const char *path, char *name) {
  DIR *pdir = NULL;
  pdir = opendir(path);
  if (pdir == NULL) {
    return -1;
  }

  struct dirent *pent = NULL;
  int i;
  for (i=0; i < 3; i++) {
    pent = readdir(pdir);
    if (pent == NULL) {
      return -1;
    }
  }

  strcpy(name, pent->d_name);
  closedir(pdir);
  return 0;
}

static int Tau_bfd_internal_getBGPExePath(char *path) {
  int rc;
  char jobid[256];
  rc = Tau_bfd_internal_getBGPJobID("/jobs", jobid);
  if (rc != 0) {
    return -1;
  }
  sprintf (path, "/jobs/%s/exe", jobid);
  return 0;
}

static char const * Tau_bfd_internal_getExecutablePath()
{
	static char path[4096];

	/* System dependent methods to find the executable path */

#if defined(TAU_AIX)
	// AIX
	sprintf(path, "/proc/%d/object/a.out", getpid());
#elif defined(TAU_BGP)
	// BlueGene
	if (Tau_bfd_internal_getBGPExePath(path) != 0) {
		fprintf(stderr, "Tau_bfd_internal_getExecutablePath: "
				"Warning! Cannot find BG/P executable path [%s], "
				"symbols will not be resolved\n", path);
		return NULL;
	}
#elif defined(__APPLE__)
	// MacOS
	uint32_t size = sizeof(path);
	_NSGetExecutablePath(path, &size);
#elif defined(TAU_WINDOWS) && defined(TAU_MINGW)
	// MinGW for Windows
	GetModuleFileName(NULL, path, sizeof(path));
#else
	// Default: Linux systems
	sprintf(path, "%s", "/proc/self/exe");
#endif

	return path;
}


// This is the mapping function to be applied across sections for random
//   probing addresses to be supplied in some global variable. In this
//   module, the global variables are compartmentalized by unit and pointed
//   to by the variable currentModule.
static void Tau_bfd_internal_locateAddress(bfd * bfdptr,
		asection * section, void * dataPtr)
{
	// Assume dataPtr != NULL because if that parameter is
	// NULL then we've got bigger problems elsewhere in the code
	LocateAddressData & data = *(LocateAddressData*)dataPtr;

	// Skip this section if we've already resolved the address data
	if (data.found) return;

	bfd_vma vma;
	bfd_size_type size;

	// Skip this section if it isn't a debug info section
	if ((bfd_get_section_flags(bfdptr, section) & SEC_ALLOC) == 0) return;

	// Skip this section if the address is before the section start
	vma = bfd_get_section_vma(bfdptr, section);
	if (data.info.probeAddr < vma) return;

	// Skip this section if the address is after the section end
	size = bfd_get_section_size(section);
	if (data.info.probeAddr >= vma + size) return;

	// The section contains this address, so try to resolve info
	data.found = bfd_find_nearest_line(bfdptr, section,
			data.module->syms, (data.info.probeAddr - vma),
			&data.info.filename, &data.info.funcname, &data.info.lno);
}

#endif /* TAU_BFD */

/******************************************
 * ******* END OF TAU MINGW VERSION *******
 ******************************************/

#else /* TAU_MINGW */

/**************************************
 * ******* ORIGINAL BFD VERSION *******
 **************************************/

#ifdef TAU_BFD

#include <TAU.h>
#include <Profile/TauBfd.h>
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#define HAVE_DECL_BASENAME 1
#include <demangle.h>
#endif /* HAVE_GNU_DEMANGLE */
#include <bfd.h>

#ifdef TAU_BGP
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* _GNU_SOURCE */
#include <link.h>
#endif /* TAU_BGP */

#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
using namespace std;

class TauBfdModule {
public:
  bfd *bfdImage;
  asymbol **syms;
  int nr_all_syms;

  // per-pc global mapping variables
  const char *located_filename; 
  const char *located_funcname;
  unsigned int located_lineno;
  bool symbol_found;
  bfd_vma curr_pc;

  // For EBS book-keeping
  bool bfdOpen; // once open, symtabs are loaded and never released
  char *demangled_funcname;
  
public:
  TauBfdModule() {
    bfdImage = NULL;
    syms = NULL;
    nr_all_syms = 0;

    located_lineno = 0;
    symbol_found = false;

    bfdOpen = false;
    demangled_funcname = NULL;
  };
  ~TauBfdModule() {
    free(bfdImage);
    free(syms);
  };
};

typedef struct {
  int type;
  char *executablePath;
  TauBfdModule *executableModule;
  vector<TauBfdAddrMap> *addressMaps;
  vector<TauBfdModule *> *modules;
} TauBfdUnit;

#define TAU_MAX_BFD_UNITS 64

// Global Variables
int bfdUnitCount = 0;
TauBfdUnit *bfdUnits[TAU_MAX_BFD_UNITS];
static TauBfdModule *currentModule = NULL;

// Internal function prototypes
static bool Tau_bfd_internal_loadSymTab(TauBfdUnit *unit, int moduleIndex);
static bool Tau_bfd_internal_loadExecSymTab(TauBfdUnit *unit);
static int Tau_bfd_internal_getModuleIndex(TauBfdUnit *unit,
					   unsigned long probe_addr);
static TauBfdModule *Tau_bfd_internal_getModuleFromIdx(TauBfdUnit *unit,
						       int moduleIndex);
static unsigned long Tau_bfd_internal_getOffsetAddress(TauBfdUnit *unit,
						       int moduleIndex,
						       unsigned long probe_addr);
static void Tau_bfd_internal_addExeAddressMap();
static void Tau_bfd_internal_locateAddress(bfd *bfdptr,
					   asection *section,
					   void *data ATTRIBUTE_UNUSED);
static int Tau_bfd_internal_getBGPJobID(const char *path, char *name);
static char *Tau_bfd_internal_getExecutablePath();
static void Tau_bfd_internal_updateProcSelfMaps(TauBfdUnit *unit);
static void Tau_bfd_internal_updateBGPMaps(TauBfdUnit *unit);

#ifdef TAU_BGP
static int Tau_bfd_internal_BGP_dl_iter_callback(struct dl_phdr_info *info,
						 size_t size,
						 void *data);
#endif /* TAU_BGP */

// Main interface functions
void Tau_bfd_initializeBfdIfNecessary() {
  static bool bfdInitialized = false;
  if (!bfdInitialized) {
    bfd_init();
    bfdInitialized = true;
  }
}

tau_bfd_handle_t Tau_bfd_registerUnit(int flag) {
  tau_bfd_handle_t ret = bfdUnitCount;
  bfdUnits[bfdUnitCount] = (TauBfdUnit*)malloc(sizeof(TauBfdUnit));
  bfdUnits[bfdUnitCount]->type = flag;

  bfdUnits[bfdUnitCount]->modules = new vector<TauBfdModule *>();
  bfdUnits[bfdUnitCount]->addressMaps = new vector<TauBfdAddrMap>();

  // These fields should always be available.
  bfdUnits[bfdUnitCount]->executablePath = 
    Tau_bfd_internal_getExecutablePath();
  bfdUnits[bfdUnitCount]->executableModule = new TauBfdModule();

  TAU_VERBOSE("Tau_bfd_registerUnit: Unit %d registered and initialized\n",
	      bfdUnitCount);
  // Initialize the first address maps for the unit.
  bfdUnitCount++;
  Tau_bfd_updateAddressMaps(ret);
  return ret;
}

bool Tau_bfd_checkHandle(tau_bfd_handle_t handle) {
  if (handle == TAU_BFD_NULL_HANDLE) {
    TAU_VERBOSE("TauBfd: Warning - attempt to use uninitialized BFD handle\n");
    return false;
  }
  if (handle >= bfdUnitCount) {
    TAU_VERBOSE("TauBfd: Warning - invalid BFD unit handle %d, max value %d\n", handle, bfdUnitCount);
    return false;
  }
  //  TAU_VERBOSE("TauBfd: Valid BFD Handle\n");
  return true;
}

static void Tau_bfd_internal_updateProcSelfMaps(TauBfdUnit *unit) {
// *CWL* - This is important! We DO NOT want to use /proc/self/maps on
//         the BGP because the information acquired comes from the I/O nodes
//         and not the compute nodes. You could end up with an overlapping
//         range for address resolution if used!
#ifndef TAU_BGP
  // Note: Linux systems only.
  FILE *mapsfile = fopen("/proc/self/maps", "r");
  char line[4096];
  // count is used for TAU_VERBOSE only
  int count = 0;
  while (!feof(mapsfile)) {
    fgets(line, 4096, mapsfile);
    unsigned long start, end, offset;
    char module[4096];
    char perms[5];
    module[0] = 0;
    
    sscanf(line, "%lx-%lx %s %lx %*s %*u %[^\n]",
	   &start, &end, perms, &offset, module);
    if (*module && ((strcmp(perms, "r-xp") == 0) ||
		    (strcmp(perms, "rwxp") == 0))) {
      TAU_VERBOSE("[%d] Module: %s, %p-%p (%d)\n", 
		  count++, module, start, end, offset);
      TauBfdAddrMap map;
      map.start = start;
      map.end = end;
      map.offset = offset;
      sprintf(map.name, "%s", module);
      unit->addressMaps->push_back(map);
      if (unit->type == TAU_BFD_KEEP_GLOBALS) {
	// create a mirror module structure for each address map
	TauBfdModule *bfdModule = new TauBfdModule();
	unit->modules->push_back(bfdModule);
      }
    }
  }
  fclose(mapsfile);
#endif /* TAU_BGP */
}

#ifdef TAU_BGP
static int Tau_bfd_internal_BGP_dl_iter_callback(struct dl_phdr_info *info,
						 size_t size,
						 void *data) {
    if (strcmp("", info->dlpi_name) == 0) {
	TAU_VERBOSE("Tau_bfd_internal_BGP_dl_iter_callback: Nameless module. Ignored.\n");
	return 0;
    }
    TauBfdUnit *unit = (TauBfdUnit *)data;
    TauBfdAddrMap map;
    map.start = (unsigned long)info->dlpi_addr;
    // assumption - the max of the physical addresses of each segment added
    //              to the memory size yields the end of the address range.
    //              It is unclear if any of the other factors affect it.
    unsigned long max_addr = 0x0;
    for (int j=0; j<info->dlpi_phnum; j++) {
	unsigned long local_max =
	    (unsigned long)info->dlpi_phdr[j].p_paddr +
	    (unsigned long)info->dlpi_phdr[j].p_memsz;
	if (local_max > max_addr) {
	    max_addr = local_max;
	}
    }
    map.end = map.start + max_addr;
    map.offset = 0; // assume.
    sprintf(map.name, "%s", info->dlpi_name);
    TAU_VERBOSE("BGP Module: %s, %p-%p (%d)\n", 
		map.name, map.start, map.end, map.offset);
    unit->addressMaps->push_back(map);
    if (unit->type == TAU_BFD_KEEP_GLOBALS) {
        // create a mirror module structure for each address map
	TauBfdModule *bfdModule = new TauBfdModule();
        unit->modules->push_back(bfdModule);
    }
    return 0;
}
#endif /* TAU_BGP */

static void Tau_bfd_internal_updateBGPMaps(TauBfdUnit *unit) {
#ifdef TAU_BGP
    dl_iterate_phdr(Tau_bfd_internal_BGP_dl_iter_callback, (void *)unit);
#endif /* TAU_BGP */
}

void Tau_bfd_updateAddressMaps(tau_bfd_handle_t handle) {
  if (!Tau_bfd_checkHandle(handle)) {
    return;
  }
  //  TAU_VERBOSE("Tau_bfd_updateAddressMaps: Updating object maps\n");
  TauBfdUnit *unit = bfdUnits[handle];

  // Clear old information from vector. This relies on the fact that
  //   addressMaps are non-pointer structures
  //   modules are class objects with properly written destructors
  unit->addressMaps->clear();
  if (unit->type != TAU_BFD_REUSE_GLOBALS) {
    unit->modules->clear();
  }

#ifdef TAU_BGP
  Tau_bfd_internal_updateBGPMaps(unit);
#else
  Tau_bfd_internal_updateProcSelfMaps(unit);
#endif /* TAU_BGP */

  // We must have at least one module structure to host globals for BFD
  //    operations. 
  // Note that TAU_BFD_KEEP_GLOBALS may still create 0 module entries if
  //    there are zero address maps (which is allowed).
  TAU_VERBOSE("Tau_bfd_updateAddressMaps: %d modules discovered\n",
	      unit->modules->size());
  if (unit->modules->size() == 0) {
    TauBfdModule *bfdModule = new TauBfdModule();
    unit->modules->push_back(bfdModule);
  }
}

vector<TauBfdAddrMap> *Tau_bfd_getAddressMaps(tau_bfd_handle_t handle) {
  if (!Tau_bfd_checkHandle(handle)) {
    return NULL;
  }

  // If we have a valid bfd handle, addressMaps is always available.
  return bfdUnits[handle]->addressMaps;
}

tau_bfd_module_handle_t Tau_bfd_getModuleHandle(tau_bfd_handle_t handle,
						unsigned long probe_addr) {
  if (!Tau_bfd_checkHandle(handle)) {
    return TAU_BFD_INVALID_MODULE;
  }
  TauBfdUnit *unit = bfdUnits[handle];
  int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probe_addr);
  if (matchingIdx == -1) {
    return TAU_BFD_NULL_MODULE_HANDLE;
  } else {
    return (tau_bfd_module_handle_t)matchingIdx;
  }
}

int Tau_bfd_getAddressMap(tau_bfd_handle_t handle, 
			  unsigned long probe_addr, TauBfdAddrMap *mapInfo) {
  if (!Tau_bfd_checkHandle(handle)) {
    return 0;
  }
  //  TAU_VERBOSE("Tau_bfd_updateAddressMaps: Updating object maps\n");
  TauBfdUnit *unit = bfdUnits[handle];
  int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probe_addr);
  if (matchingIdx == -1) {
    return TAU_BFD_NULL_MODULE_HANDLE;
  }
  mapInfo->start = (*unit->addressMaps)[matchingIdx].start;
  mapInfo->end = (*unit->addressMaps)[matchingIdx].end;
  mapInfo->offset = (*unit->addressMaps)[matchingIdx].offset;
  strcpy(mapInfo->name, (*unit->addressMaps)[matchingIdx].name);
}

// Probe for BFD information given a single address.
TauBfdInfo *Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
				   unsigned long probe_addr) {
  if (!Tau_bfd_checkHandle(handle)) {
    return NULL;
  }
  TauBfdUnit *unit = bfdUnits[handle];
  int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probe_addr);
  //  TAU_VERBOSE("Tau_bfd_resolveBfdInfo: Matching module index %d for addr [%p]\n", matchingIdx, probe_addr);

  if (matchingIdx == -1) {
      return NULL;
  }

  if (!Tau_bfd_internal_loadSymTab(unit, matchingIdx)) {
    return NULL;
  }

  // Convert address to something bfd can use.
  char hex_pc_string[100];
  sprintf(hex_pc_string, "%p", probe_addr);

  TauBfdModule *module = Tau_bfd_internal_getModuleFromIdx(unit, matchingIdx);

  module->curr_pc = bfd_scan_vma(hex_pc_string, NULL, 16);
  module->symbol_found = false;
  if (module->demangled_funcname != NULL) {
    free(module->demangled_funcname);
  }
  module->demangled_funcname = NULL;
  currentModule = module; // set up global variable for mapping over BFD
  bfd_map_over_sections(module->bfdImage,
                        Tau_bfd_internal_locateAddress, 0);
  if (!module->symbol_found) {
    // if not initially found, try again with offset address.
    unsigned long offsetAddr = Tau_bfd_internal_getOffsetAddress(unit,
								 matchingIdx,
								 probe_addr);
    TAU_VERBOSE("Tau_bfd_resolveBfdInfo: Trying alternate address [%p] from address [%p]\n", offsetAddr, probe_addr);
    if (offsetAddr != probe_addr) {
	// try only if they are not the same
	sprintf(hex_pc_string, "%p", offsetAddr);
	module->curr_pc = bfd_scan_vma(hex_pc_string, NULL, 16);
	bfd_map_over_sections(module->bfdImage,
			      Tau_bfd_internal_locateAddress, 0);
    }
  }

  // If we fail again, then there's nothing we can do about it.
  if (module->symbol_found &&
      (module->located_funcname != (char *)NULL)) {
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
    // could also use bfd_demangle
    module->demangled_funcname =
      cplus_demangle(module->located_funcname,
                     DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE | DMGL_TYPES);
    /*
    TAU_VERBOSE("Tau_bfd_resolveBfdInfo: [%s] demangles to [%s]\n",
                module->located_funcname,
		module->demangled_funcname);
    */
#endif /* HAVE_GNU_DEMANGLE */
    TauBfdInfo *info = (TauBfdInfo *)malloc(sizeof(TauBfdInfo));
    if (module->located_filename != NULL) {
      info->filename = strdup(module->located_filename);
    } else {
      info->filename = NULL;
    }
    if (module->demangled_funcname != NULL) {
      info->funcname = strdup(module->demangled_funcname);
    } else {
      info->funcname = strdup(module->located_funcname);
    }
    info->lineno = module->located_lineno;
    TAU_VERBOSE("Tau_bfd_resolveBfdInfo: Addr [%p] symbol found [%s|%s|%d]\n",
		probe_addr, info->filename, info->funcname, info->lineno);
    currentModule = NULL; // reset the global
    return info;
  }
  currentModule = NULL; // reset the global
  return NULL;
}

TauBfdInfo *Tau_bfd_resolveBfdExecInfo(tau_bfd_handle_t handle,
				       unsigned long probe_addr) {
  if (!Tau_bfd_checkHandle(handle)) {
    return NULL;
  }
  TauBfdUnit *unit = bfdUnits[handle];

  if (!Tau_bfd_internal_loadExecSymTab(unit)) {
    return NULL;
  }

  // Convert address to something bfd can use.
  char hex_pc_string[100];
  sprintf(hex_pc_string, "%p", probe_addr);

  TauBfdModule *module = unit->executableModule;

  module->curr_pc = bfd_scan_vma(hex_pc_string, NULL, 16);
  module->symbol_found = false;
  if (module->demangled_funcname != NULL) {
    free(module->demangled_funcname);
  }
  module->demangled_funcname = NULL;
  currentModule = module; // set up global variable for mapping over BFD
  bfd_map_over_sections(module->bfdImage,
                        Tau_bfd_internal_locateAddress, 0);
  // If we can only fail once for the executable
  if (module->symbol_found &&
      (module->located_funcname != (char *)NULL)) {
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
    // could also use bfd_demangle
    module->demangled_funcname =
      cplus_demangle(module->located_funcname,
                     DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE | DMGL_TYPES);
    TAU_VERBOSE("Tau_bfd_resolveBfdExecInfo: [%s] demangles to [%s]\n",
                module->located_funcname,
		module->demangled_funcname);
#endif /* HAVE_GNU_DEMANGLE */
    TauBfdInfo *info = (TauBfdInfo *)malloc(sizeof(TauBfdInfo));
    if (module->located_filename != NULL) {
      info->filename = strdup(module->located_filename);
    } else {
      info->filename = NULL;
    }
    if (module->demangled_funcname != NULL) {
      info->funcname = strdup(module->demangled_funcname);
    } else {
      info->funcname = strdup(module->located_funcname);
    }
    info->lineno = module->located_lineno;
    TAU_VERBOSE("Tau_bfd_resolveBfdExecInfo: Addr [%p] symbol found [%s|%s|%d]\n",
		probe_addr, info->filename, info->funcname, info->lineno);
    currentModule = NULL; // reset the global
    return info;
  }
  currentModule = NULL; // reset the global
  return NULL;
}

// Run a unit-defined iterator through symbols discovered in a Bfd module.
//   If there are too many symbols in the module, we will avoid resolving
//   the symbols but allow the iterator to perform its work.
int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
				 tau_bfd_module_handle_t moduleHandle,
				 int symbolLimit,
				 TauBfdIterFn fn) {
  bool do_getsrc;
  if (!Tau_bfd_checkHandle(handle)) {
    return TAU_BFD_SYMTAB_LOAD_FAILED;
  }
  TauBfdUnit *unit = bfdUnits[handle];

  int moduleIndex = (int)moduleHandle;
  char *moduleName = (*unit->addressMaps)[moduleIndex].name;
  unsigned long offset = (*unit->addressMaps)[moduleIndex].start;
  TauBfdModule *module = Tau_bfd_internal_getModuleFromIdx(unit, moduleIndex);

  TAU_VERBOSE("Tau_bfd_processModuleInfo: processing %s\n", moduleName);
  bool success = Tau_bfd_internal_loadSymTab(unit, moduleIndex);
  if (!success) {
    return TAU_BFD_SYMTAB_LOAD_FAILED;
  } else {
    asymbol **syms = module->syms;
    int nr_all_syms = module->nr_all_syms;

    do_getsrc = true;
    if (nr_all_syms > symbolLimit) {
      TAU_VERBOSE("Tau_bfd_processBfdModuleInfo: Too many [%d] symbols in module [%s]. Not resolving details.\n", nr_all_syms, moduleName);
      do_getsrc = false; 
    }
    for (int i=0; i<nr_all_syms; ++i) {
      char* dem_name = 0;
      unsigned long addr;
      const char* filename;
      const char* funcname;
      unsigned int lno;
      
      /* get filename and linenumber from debug info */
      /* needs -g */
      filename = NULL;
      lno = 0;
      if (do_getsrc) {
	bfd_find_nearest_line(module->bfdImage, 
			      bfd_get_section(syms[i]), syms,
			      syms[i]->value, &filename, &funcname, &lno);
      }
      
      /* calculate function address */
      addr = syms[i]->section->vma+syms[i]->value;

      /* use demangled name if possible */
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE 
      dem_name = cplus_demangle(syms[i]->name,
				DMGL_PARAMS | DMGL_ANSI 
				| DMGL_VERBOSE | DMGL_TYPES);
#endif /* HAVE_GNU_DEMANGLE */
      
      const char *name = syms[i]->name;
      if (dem_name) {
	name = dem_name;
      }
      fn(offset+addr, name, filename, lno);
    }
  }
  if (do_getsrc) {
    return TAU_BFD_SYMTAB_LOAD_SUCCESS;
  } else {
    return TAU_BFD_SYMTAB_LOAD_UNRESOLVED;
  }
}

int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle,
			       int symbolLimit,
			       TauBfdIterFn fn) {
  bool do_getsrc;

  if (!Tau_bfd_checkHandle(handle)) {
    return TAU_BFD_SYMTAB_LOAD_FAILED;
  }
  TauBfdUnit *unit = bfdUnits[handle];
  char *execName = unit->executablePath;
  unsigned long offset = 0;
  TauBfdModule *module = unit->executableModule;
  
  TAU_VERBOSE("Tau_bfd_processBfdExecInfo: processing executable %s\n",
	      execName);
  bool success = Tau_bfd_internal_loadExecSymTab(unit);
  if (!success) {
    return TAU_BFD_SYMTAB_LOAD_FAILED;
  } else {
    asymbol **syms = module->syms;
    int nr_all_syms = module->nr_all_syms;

    do_getsrc = true;
    if (nr_all_syms > symbolLimit) {
      TAU_VERBOSE("Tau_bfd_processBfdExecInfo: Too many [%d] symbols in executable [%s]. Not resolving.\n", nr_all_syms, execName);
      do_getsrc = false; 
    }
    for (int i=0; i<nr_all_syms; ++i) {
      char* dem_name = 0;
      unsigned long addr;
      const char* filename;
      const char* funcname;
      unsigned int lno;
      
      /* get filename and linenumber from debug info */
      /* needs -g */
      filename = NULL;
      lno = 0;
      if (do_getsrc) {
	bfd_find_nearest_line(module->bfdImage, 
			      bfd_get_section(syms[i]), syms,
			      syms[i]->value, &filename, &funcname, &lno);
      }
      
      /* calculate function address */
      addr = syms[i]->section->vma+syms[i]->value;

      /* use demangled name if possible */
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE 
      dem_name = cplus_demangle(syms[i]->name,
				DMGL_PARAMS | DMGL_ANSI 
				| DMGL_VERBOSE | DMGL_TYPES);
#endif /* HAVE_GNU_DEMANGLE */
      
      const char *name = syms[i]->name;
      if (dem_name) {
	name = dem_name;
      }
      fn(offset+addr, name, filename, lno);
    }
  }
  if (do_getsrc) {
    return TAU_BFD_SYMTAB_LOAD_SUCCESS;
  } else {
    return TAU_BFD_SYMTAB_LOAD_UNRESOLVED;
  }
}

bool Tau_bfd_internal_loadSymTab(TauBfdUnit *unit, int moduleIndex) {
  int size;

  // No such module.
  if ((moduleIndex == TAU_BFD_NULL_MODULE_HANDLE) ||
      (moduleIndex == TAU_BFD_INVALID_MODULE)) {
    return false;
  }
  char *name = ((*unit->addressMaps)[moduleIndex]).name;
  TauBfdModule *module = Tau_bfd_internal_getModuleFromIdx(unit, moduleIndex);

  // Symbol table is already open, safe to return true.
  if (module->bfdOpen) {
    return true;
  }

  Tau_bfd_initializeBfdIfNecessary();
  module->bfdImage = bfd_openr(name, 0);
  if (module->bfdImage == NULL) {
    TAU_VERBOSE("Tau_bfd_internal_loadSymTab: Failed to open module [%s]\n", name);
    return false;
  }

  if (!bfd_check_format(module->bfdImage, bfd_object)) {
    TAU_VERBOSE("Tau_bfd_internal_loadSymTab: bfd format check failed [%s]\n", name);
    return false;
  }
  if (!(bfd_get_file_flags(module->bfdImage) & HAS_SYMS)) {
    TAU_VERBOSE("Tau_bfd_internal_loadSymTab: bfd has no symbols [%s]\n", name);
    return false;
  }
  size = bfd_get_symtab_upper_bound(module->bfdImage);
  if (size < 1) {
    TAU_VERBOSE("Tau_bfd_internal_loadSymTab: bfd_get_symtab_upper_bound() < 1\n");
    return false;
  }

  module->syms = (asymbol **)malloc(size);
  module->nr_all_syms = bfd_canonicalize_symtab(module->bfdImage,
                                                module->syms);
  if (module->nr_all_syms < 1) {
    TAU_VERBOSE("Tau_bfd_internal_loadSymTab: No canonical symbols found in [%s]\n", 
		name);
    return false;
  }

  module->bfdOpen = true;
  return true;
}

bool Tau_bfd_internal_loadExecSymTab(TauBfdUnit *unit) {
  int size;

  char *execPath = unit->executablePath;
  TauBfdModule *module = unit->executableModule;

  // Executable symbol table is already loaded. Safe to return true.
  if (module->bfdOpen) {
    return true;
  }

  Tau_bfd_initializeBfdIfNecessary();
  module->bfdImage = bfd_openr(execPath, 0);
  if (module->bfdImage == NULL) {
    TAU_VERBOSE("Tau_bfd_internal_loadExecSymTab: Failed to open module [%s]\n", execPath);
    return false;
  }

  if (!bfd_check_format(module->bfdImage, bfd_object)) {
    TAU_VERBOSE("Tau_bfd_internal_loadExecSymTab: bfd format check failed [%s]\n", execPath);
    return false;
  }

  if (!(bfd_get_file_flags(module->bfdImage) & HAS_SYMS)) {
    TAU_VERBOSE("Tau_bfd_internal_loadExecSymTab: bfd has no symbols [%s]\n", execPath);
    return false;
  }

  size = bfd_get_symtab_upper_bound(module->bfdImage);
  if (size < 1) {
    TAU_VERBOSE("Tau_bfd_internal_loadExecSymTab: bfd_get_symtab_upper_bound() < 1\n");
    return false;
  }

  module->syms = (asymbol **)malloc(size);
  module->nr_all_syms = bfd_canonicalize_symtab(module->bfdImage,
                                                module->syms);
  if (module->nr_all_syms < 1) {
    TAU_VERBOSE("Tau_bfd_internal_loadExecSymTab: No canonical symbols found in [%s]\n", 
		execPath);
    return false;
  }

  module->bfdOpen = true;
  return true;
}

// Internal BFD helper functions
static int Tau_bfd_internal_getModuleIndex(TauBfdUnit *unit,
					   unsigned long probe_addr) {
  int ret = -1;
  vector<TauBfdAddrMap> *addressMaps = unit->addressMaps;
  for (int i=0;i<addressMaps->size();i++) {
    if (probe_addr >= (*addressMaps)[i].start && 
	probe_addr <= (*addressMaps)[i].end) {
      return i;
    }
  }
  return ret;
}

static unsigned long Tau_bfd_internal_getOffsetAddress(TauBfdUnit *unit,
						       int moduleIndex,
						       unsigned long probe_addr) {
  if (moduleIndex == TAU_BFD_NULL_MODULE_HANDLE) {
    return probe_addr;
  }
  vector<TauBfdAddrMap> *addressMaps = unit->addressMaps;
  return (probe_addr - (*addressMaps)[moduleIndex].start);
}

static TauBfdModule *Tau_bfd_internal_getModuleFromIdx(TauBfdUnit *unit,
						       int moduleIndex) {
  TauBfdModule *module = NULL;
  if (moduleIndex == -1) {
    return unit->executableModule;
  }
  if (unit->type == TAU_BFD_REUSE_GLOBALS) {
    module = (*(unit->modules))[0];
  } else {
    module = ((*unit->modules))[moduleIndex];
  }
  return module;
}

static int Tau_bfd_internal_getBGPJobID(const char *path, char *name) {
  DIR *pdir = NULL;
  pdir = opendir(path);
  if (pdir == NULL) {
    return -1;
  }

  struct dirent *pent = NULL;
  int i;
  for (i=0; i < 3; i++) {
    pent = readdir(pdir);
    if (pent == NULL) {
      return -1;
    }
  }

  strcpy(name, pent->d_name);
  closedir(pdir);
  return 0;
}

static int Tau_bfd_internal_getBGPExePath(char *path) {
  int rc;
  char jobid[256];
  rc = Tau_bfd_internal_getBGPJobID("/jobs", jobid);
  if (rc != 0) {
    return -1;
  }
  sprintf (path, "/jobs/%s/exe", jobid);
  return 0;
}

static char *Tau_bfd_internal_getExecutablePath() {
  char path[4096];
  int rc;

#ifndef TAU_BFD
  fprintf(stderr, "TAU: Warning! BFD not found, symbols will not be resolved\n");
  fprintf(stderr, "Please re-configure TAU with -bfd=download to support runtime symbol resolution using the BFD library.\n");
  return NULL;
#endif

  /* System dependent methods to find the executable */

  /* Default: Linux systems */
  sprintf(path, "%s", "/proc/self/exe");
  
#ifdef TAU_AIX
  sprintf(path, "/proc/%d/object/a.out", getpid());
#endif
  
#ifdef TAU_BGP
  rc = Tau_bfd_internal_getBGPExePath(path);
  if (rc != 0) {
    fprintf(stderr, "Tau_bfd_internal_getExecutablePath: Warning! Cannot find BG/P executable path [%s], symbols will not be resolved\n", path);
    return NULL;
  }
#endif
  
#ifdef __APPLE__
  uint32_t size = sizeof(path);
  _NSGetExecutablePath(path, &size);
#endif

  char *retPath = strdup(path);
  return retPath;
}

// This is the mapping function to be applied across sections for random
//   probing addresses to be supplied in some global variable. In this
//   module, the global variables are compartmentalized by unit and pointed
//   to by the variable currentModule.
static void Tau_bfd_internal_locateAddress(bfd *bfdptr,
					   asection *section,
					   void *data ATTRIBUTE_UNUSED) {
  if (currentModule == NULL) {
    TAU_VERBOSE("Tau_bfd_internal_locateAddress: No current module\n");
    return;
  }
  bfd_vma vma_addr;
  bfd_size_type size;
  
  if (currentModule->symbol_found) {
    return;
  }
  
  if ((bfd_get_section_flags(bfdptr, section) & SEC_ALLOC) == 0) {
    return;
  }
  
  vma_addr = bfd_get_section_vma(bfdptr, section);
  
  if (currentModule->curr_pc < vma_addr) {
    return;
  }
  
  size = bfd_get_section_size(section);
  if (currentModule->curr_pc >= vma_addr + size) {
    return;
  }
  
  currentModule->symbol_found =
    bfd_find_nearest_line(bfdptr, section,
			  currentModule->syms,
			  currentModule->curr_pc - vma_addr,
			  &(currentModule->located_filename),
			  &(currentModule->located_funcname),
			  &(currentModule->located_lineno));
}
#endif /* TAU_BFD */

/*********************************************
 * ******* END OF ORIGINAL BFD VERSION *******
 *********************************************/

#endif /* TAU_MINGW */
