#ifdef TAU_BFD

#if (defined(TAU_BGP) || defined(TAU_BGQ)) && defined(TAU_XLC)
// *CWL* - This is required to handle the different prototype for
//         asprintf and vasprintf between gnu and xlc compilers
//         on the BGP.
#define HAVE_DECL_VASPRINTF 1
#define HAVE_DECL_ASPRINTF 1
#endif

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

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

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

		TAU_VERBOSE("loadSymbolTable: %s contains %d canonical symbols\n",
				path, nr_all_syms);

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
	LocateAddressData(TauBfdModule * _module, TauBfdInfo & _info) :
		found(false), module(_module), info(_info)
	{ }

	bool found;
	TauBfdModule * module;
	TauBfdInfo & info;
};


// Internal function prototypes
static bool Tau_bfd_internal_loadSymTab(TauBfdUnit *unit, int moduleIndex);
static bool Tau_bfd_internal_loadExecSymTab(TauBfdUnit *unit);
static int Tau_bfd_internal_getModuleIndex(
		TauBfdUnit *unit, unsigned long probe_addr);
static TauBfdModule * Tau_bfd_internal_getModuleFromIdx(
		TauBfdUnit *unit, int moduleIndex);
static void Tau_bfd_internal_addExeAddressMap();
static void Tau_bfd_internal_locateAddress(
		bfd *bfdptr, asection *section, void *data ATTRIBUTE_UNUSED);
static void Tau_bfd_internal_updateProcSelfMaps(TauBfdUnit *unit);

#ifdef TAU_BGP
static int Tau_bfd_internal_getBGPJobID(const char *path, char *name);
static void Tau_bfd_internal_updateBGPMaps(TauBfdUnit *unit);
#endif /* TAU_BGP */

// BFD units (e.g. executables and their dynamic libraries)
//vector<TauBfdUnit*> bfdUnits;
//////////////////////////////////////////////////////////////////////
// Instead of using a global var., use static inside a function  to
// ensure that non-local static variables are initialised before being
// used (Ref: Scott Meyers, Item 47 Eff. C++).
//////////////////////////////////////////////////////////////////////
std::vector<TauBfdUnit*>& ThebfdUnits(void)
{ // FunctionDB contains pointers to each FunctionInfo static object

  // we now use the above FIvector, which subclasses vector
  //static vector<FunctionInfo*> FunctionDB;
  static std::vector<TauBfdUnit*> internal_bfd_units;

  static int flag = 1;
  if (flag) {
    flag = 0;
  }

  return internal_bfd_units;
}


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
	tau_bfd_handle_t ret = ThebfdUnits().size();
	ThebfdUnits().push_back(new TauBfdUnit);

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
  if (handle >= ThebfdUnits().size()) {
    TAU_VERBOSE("TauBfd: Warning - invalid BFD unit handle %d, max value %d\n",
    		handle, ThebfdUnits().size());
    return false;
  }
  //  TAU_VERBOSE("TauBfd: Valid BFD Handle\n");
  if (handle < 0) return false; 
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

#ifdef TAU_BGP
static void Tau_bfd_internal_updateBGPMaps(TauBfdUnit *unit) {
    dl_iterate_phdr(Tau_bfd_internal_BGP_dl_iter_callback, (void *)unit);
}
#endif /* TAU_BGP */


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
		map->offset = 0;

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
	TauBfdUnit * unit = ThebfdUnits()[handle];

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

vector<TauBfdAddrMap*> const &
Tau_bfd_getAddressMaps(tau_bfd_handle_t handle)
{
	Tau_bfd_checkHandle(handle);
	return ThebfdUnits()[handle]->addressMaps;
}

tau_bfd_module_handle_t Tau_bfd_getModuleHandle(tau_bfd_handle_t handle,
		unsigned long probeAddr)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_INVALID_MODULE;
	}
	TauBfdUnit *unit = ThebfdUnits()[handle];

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

	TauBfdUnit *unit = ThebfdUnits()[handle];
	int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probe_addr);
	if (matchingIdx == -1) {
		return NULL;
	}

	return unit->addressMaps[matchingIdx];
}

static char const *
Tau_bfd_internal_tryDemangle(bfd * bfdImage, char const * funcname)
{
	char const * demangled = NULL;
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
	if(funcname && bfdImage) {
		demangled = bfd_demangle(bfdImage, funcname, DEMANGLE_FLAGS);
	}
#endif
	if(demangled) return demangled;
	return funcname;
}

// Probe for BFD information given a single address.
bool Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle,
		unsigned long probeAddr, TauBfdInfo & info)
{
	if (!Tau_bfd_checkHandle(handle)) {
		info.secure(probeAddr);
		return false;
	}

	TauBfdUnit * unit = ThebfdUnits()[handle];
	TauBfdModule * module;
	unsigned long addr0;
	unsigned long addr1;

	// Discover if we are searching in the executable or a module
	int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probeAddr);
	if (matchingIdx != -1) {
		if (!Tau_bfd_internal_loadSymTab(unit, matchingIdx)) {
			info.secure(probeAddr);
			return false;
		}
		module = Tau_bfd_internal_getModuleFromIdx(unit, matchingIdx);

		// Calculate search addresses for module search
#if defined(TAU_WINDOWS) && defined(TAU_MINGW)
		addr0 = probeAddr;
		addr1 = probeAddr - unit->addressMaps[matchingIdx]->start;
#else
		addr0 = probeAddr - unit->addressMaps[matchingIdx]->start;
		addr1 = probeAddr;
#endif
	} else {
		if (!Tau_bfd_internal_loadExecSymTab(unit)) {
			info.secure(probeAddr);
			return false;
		}
		module = unit->executableModule;

		// Calculate search addresses for executable search
		// Only the first address is valid for the executable
		addr0 = probeAddr;
		addr1 = 0;
	}

	// Convert address to something bfd can use.
	char hex_pc_string[100];
	sprintf(hex_pc_string, "%p", addr0);

	// Search BFD sections for address
	LocateAddressData data(module, info);
	info.probeAddr = bfd_scan_vma(hex_pc_string, NULL, 16);
	bfd_map_over_sections(module->bfdImage, Tau_bfd_internal_locateAddress, &data);

	// If the data wasn't found where we expected and we are searching
	// in a module, try a few more addresses
	if (!data.found && (module != unit->executableModule)) {
		// Try the second address
		if (addr0 != addr1) {
			sprintf(hex_pc_string, "%p", addr1);
			info.probeAddr = bfd_scan_vma(hex_pc_string, NULL, 16);
			bfd_map_over_sections(module->bfdImage, Tau_bfd_internal_locateAddress, &data);
		}

		// Try the executable
		if(!data.found && Tau_bfd_internal_loadExecSymTab(unit)) {
			sprintf(hex_pc_string, "%p", probeAddr);
			info.probeAddr = bfd_scan_vma(hex_pc_string, NULL, 16);
			bfd_map_over_sections(unit->executableModule->bfdImage,
					Tau_bfd_internal_locateAddress, &data);
		}
	}

	bool resolved = data.found && (info.funcname != NULL);
	if (resolved) {
		info.funcname = Tau_bfd_internal_tryDemangle(
				module->bfdImage, info.funcname);
		if(info.filename == NULL) {
			info.filename = "(unknown)";
		}
	} else {
		// Couldn't resolve the address.
		// Fill in fields as best we can.
		if(info.funcname == NULL) {
			info.funcname = (char*)malloc(128);
			sprintf((char*)info.funcname, "addr=<%p>", probeAddr);
		}
		if(info.filename == NULL) {
		  if (matchingIdx != -1) {
			info.filename = unit->addressMaps[matchingIdx]->name;
		  } else {
		    info.filename = unit->executablePath;
		  }
		}
		info.probeAddr = probeAddr;
		info.lineno = 0;
	}
	return resolved;
}

static void Tau_bfd_internal_iterateOverSymtab(TauBfdModule * module,
		TauBfdIterFn fn, unsigned long offset)
{
	// Apply the iterator function to all symbols in the table
	for(asymbol ** s=module->syms; *s; s++) {
		asymbol const & asym = **s;

		// Skip useless symbols (e.g. line numbers)
		// It would be easier to use BFD_FLAGS, but those aren't reliable
		// since the debug symbol format is unpredictable
		if (!asym.name || !asym.section->size) {
			continue;
		}

		// Calculate symbol address
		unsigned long addr = asym.section->vma + asym.value;

		// Get apprixmate symbol name
		char const * name = asym.name;
		if(name[0] == '.') {
			char const * mark = strchr((char*)name, '$');
			if(mark) name = mark + 1;
		}

		// Apply the iterator function
		// Names will be resolved and demangled later
		fn(addr+offset, name);
	}
}

int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, TauBfdIterFn fn)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	}
	TauBfdUnit * unit = ThebfdUnits()[handle];

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

	// Process the symbol table
	Tau_bfd_internal_iterateOverSymtab(module, fn, 0);

	module->processCode = TAU_BFD_SYMTAB_LOAD_SUCCESS;
	return module->processCode;
}

int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
		tau_bfd_module_handle_t moduleHandle,TauBfdIterFn fn)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	}
	TauBfdUnit * unit = ThebfdUnits()[handle];

	unsigned int moduleIdx = (unsigned int)moduleHandle;
	TauBfdModule * module = Tau_bfd_internal_getModuleFromIdx(unit, moduleIdx);
	char const * name = unit->addressMaps[moduleIdx]->name;

	// Only process the module once.
	if(module->processCode != TAU_BFD_SYMTAB_NOT_LOADED) {
		TAU_VERBOSE("Tau_bfd_processBfdModuleInfo: "
				"%s already processed (code %d).  Will not reprocess.\n",
				name, module->processCode);
		return module->processCode;
	}
	TAU_VERBOSE("Tau_bfd_processBfdModuleInfo: processing module %s\n", name);

	// Make sure symbol table is loaded
	if (!Tau_bfd_internal_loadSymTab(unit, moduleHandle)) {
		module->processCode = TAU_BFD_SYMTAB_LOAD_FAILED;
		return module->processCode;
	}

	unsigned int offset;
#if defined(TAU_WINDOWS) && defined(TAU_MINGW)
	offset = 0;
#else
	offset = unit->addressMaps[moduleIdx]->start;
#endif

	// Process the symbol table
	Tau_bfd_internal_iterateOverSymtab(module, fn, offset);

	module->processCode = TAU_BFD_SYMTAB_LOAD_SUCCESS;
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
	vector<TauBfdAddrMap*> const & addressMaps = unit->addressMaps;
	for (int i = 0; i < addressMaps.size(); i++) {
		if (probe_addr >= addressMaps[i]->start &&
			probe_addr <= addressMaps[i]->end)
			return i;
	}
	return -1;
}

static TauBfdModule *
Tau_bfd_internal_getModuleFromIdx(TauBfdUnit * unit, int moduleIndex)
{
	if (moduleIndex == -1) {
		return unit->executableModule;
	}
	return unit->modules[moduleIndex];
}

#ifdef TAU_BGP
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
#endif /* TAU_BGP */

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
	// Note that data.info is a reference, so this call sets the
	// TauBfdInfo fields without an extra copy.  This also means
	// that the pointers in TauBfdInfo must never be deleted
	// since they point directly into the module's BFD.
	data.found = bfd_find_nearest_line(bfdptr, section,
			data.module->syms, (data.info.probeAddr - vma),
			&data.info.filename, &data.info.funcname,
			(unsigned int*)&data.info.lineno);
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
	TauBfdInfo * info = new TauBfdInfo;
	if(Tau_bfd_resolveBfdInfo(handle, probe_addr, *info)) {
		return info;
	}
	delete info;
	return NULL;
}

TauBfdInfo * Tau_bfd_resolveBfdExecInfo(tau_bfd_handle_t handle,
		unsigned long probe_addr)
{
	TauBfdInfo * info = new TauBfdInfo;

	// Tau_bfd_resolveBfdInfo works on both executables and modules
	if(Tau_bfd_resolveBfdInfo(handle, probe_addr, *info)) {
		return info;
	}
	delete info;
	return NULL;
}

int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
		tau_bfd_module_handle_t moduleHandle, int maxProbe,
		DeprecatedTauBfdIterFn fn)
{
	bool do_getsrc;
	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	}
	TauBfdUnit *unit = ThebfdUnits()[handle];

	int moduleIndex = (int) moduleHandle;
	char const * moduleName = unit->addressMaps[moduleIndex]->name;
	unsigned long offset = unit->addressMaps[moduleIndex]->start;
	TauBfdModule * module = Tau_bfd_internal_getModuleFromIdx(unit, moduleIndex);

	TAU_VERBOSE("Tau_bfd_processModuleInfo (deprecated): processing %s\n", moduleName);
	bool success = Tau_bfd_internal_loadSymTab(unit, moduleIndex);
	if (!success) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	} else {
		asymbol **syms = module->syms;
		int nr_all_syms = module->nr_all_syms;

		do_getsrc = true;
		if (nr_all_syms > maxProbe) {
			TAU_VERBOSE("Tau_bfd_processBfdModuleInfo (deprecated): "
					"Too many [%d] symbols in module [%s]. "
					"Not resolving details.\n",
					nr_all_syms, moduleName);
			do_getsrc = false;
		}
		for (int i = 0; i < nr_all_syms; ++i) {
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
			addr = syms[i]->section->vma + syms[i]->value;

			/* use demangled name if possible */
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
			dem_name = cplus_demangle(syms[i]->name, DMGL_PARAMS | DMGL_ANSI
					| DMGL_VERBOSE | DMGL_TYPES);
#endif /* HAVE_GNU_DEMANGLE */

			const char *name = syms[i]->name;
			if (dem_name) {
				name = dem_name;
			}
			fn(offset + addr, name, filename, lno);
		}
	}
	if (do_getsrc) {
		return TAU_BFD_SYMTAB_LOAD_SUCCESS;
	} else {
		return TAU_BFD_SYMTAB_LOAD_UNRESOLVED;
	}
}

int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle, int maxProbe,
		DeprecatedTauBfdIterFn fn)
{
	bool do_getsrc;

	if (!Tau_bfd_checkHandle(handle)) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	}
	TauBfdUnit *unit = ThebfdUnits()[handle];
	char const * execName = unit->executablePath;
	unsigned long offset = 0;
	TauBfdModule *module = unit->executableModule;

	TAU_VERBOSE("Tau_bfd_processBfdExecInfo (deprecated): processing executable %s\n",
			execName);
	bool success = Tau_bfd_internal_loadExecSymTab(unit);
	if (!success) {
		return TAU_BFD_SYMTAB_LOAD_FAILED;
	} else {
		asymbol **syms = module->syms;
		int nr_all_syms = module->nr_all_syms;

		do_getsrc = true;
		if (nr_all_syms > maxProbe) {
			TAU_VERBOSE("Tau_bfd_processBfdExecInfo (deprecated): "
					"Too many [%d] symbols in executable [%s]. Not resolving.\n",
					nr_all_syms, execName);
			do_getsrc = false;
		}
		for (int i = 0; i < nr_all_syms; ++i) {
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
			addr = syms[i]->section->vma + syms[i]->value;

			/* use demangled name if possible */
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
			dem_name = cplus_demangle(syms[i]->name, DMGL_PARAMS | DMGL_ANSI
					| DMGL_VERBOSE | DMGL_TYPES);
#endif /* HAVE_GNU_DEMANGLE */

			const char *name = syms[i]->name;
			if (dem_name) {
				name = dem_name;
			}
			fn(offset + addr, name, filename, lno);
		}
	}
	if (do_getsrc) {
		return TAU_BFD_SYMTAB_LOAD_SUCCESS;
	} else {
		return TAU_BFD_SYMTAB_LOAD_UNRESOLVED;
	}
}

int Tau_bfd_getAddressMap(tau_bfd_handle_t handle, unsigned long probe_addr,
		TauBfdAddrMap * mapInfo)
{
	if (!Tau_bfd_checkHandle(handle)) {
		return 0;
	}

	TauBfdUnit * unit = ThebfdUnits()[handle];
	int matchingIdx = Tau_bfd_internal_getModuleIndex(unit, probe_addr);
	if (matchingIdx == -1) {
		return TAU_BFD_NULL_MODULE_HANDLE;
	}

	mapInfo->start = unit->addressMaps[matchingIdx]->start;
	mapInfo->end = unit->addressMaps[matchingIdx]->end;
	mapInfo->offset = unit->addressMaps[matchingIdx]->offset;
	strcpy(mapInfo->name, unit->addressMaps[matchingIdx]->name);
	// *CWL* - This implementation is not 100% satisfactory. It is unclear
	//         what we should do with the return index of 0.
	return matchingIdx;
}

#endif /* TAU_BFD */
