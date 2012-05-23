#include <TAU.h>
#include <Profile/TauBfd.h>

#ifdef __GNUC__
#include <cxxabi.h>
#endif /* __GNUC__ */

static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;

static const char *tau_filename;
static const char *tau_funcname;
static unsigned int tau_line_no;
static int tau_symbol_found; 

extern "C" int Tau_get_backtrace_off_by_one_correction(void);

static void issueBfdWarningIfNecessary() {
  static bool warningIssued = false;
  if (!warningIssued) {
    fprintf(stderr,"TAU Warning: TauBackTrace - "
    		"BFD is not available during TAU build. Symbols may not be resolved!\n");
    warningIssued = true;
  }
}

static TauBfdAddrMap * getAddressMap(unsigned long addr)
{
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit();
  }

  // if Tau_bfd_registerUnit has been called, maps have been loaded
  vector<TauBfdAddrMap*> const & addressMap =
    Tau_bfd_getAddressMaps(bfdUnitHandle);
  for (size_t i = 0; i < addressMap.size(); i++) {
    if (addr >= addressMap[i]->start && addr <= addressMap[i]->end) {
      return addressMap[i];
    }
  }
  
  // Wasn't found in any ranges, try updating the maps.
  // NOTE: *CWL* - This simplified means of detecting epoch changes will
  //       suffer from pathological cases where a function's address in
  //       one dynamically loaded module can coincide with another
  //       function's address in another dynamically loaded module.
  //
  //       Sampling CANNOT take this approach to epoch changes. It must
  //       rely on traps to dlopen calls.
  
  Tau_bfd_updateAddressMaps(bfdUnitHandle);
  
  for (size_t i = 0; i < addressMap.size(); i++) {
    if (addr >= addressMap[i]->start && addr <= addressMap[i]->end) {
      return addressMap[i];
    }
  }
  
  TAU_VERBOSE("TauBackTrace: getAddressMap - "
	      "failed to find address [%p] after 2 tries\n", addr);
  // Still not found?  Give up
  return NULL;
}

bool tauGetFilenameAndLineNo(unsigned long addr)
{
  TAU_VERBOSE("TauBackTrace: tauGetFilenameAndLineNo: addr=%p\n", addr);
  
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit();
  }

  // Use BFD to resolve address info
  TauBfdInfo info;
  tau_symbol_found = Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, info);
  
  if (tau_symbol_found) {
    tau_line_no = info.lineno;
    if (info.funcname) {
      // TODO: Is this leaking memory?
      tau_funcname = strdup(info.funcname);
    } else {
      tau_funcname = NULL;
    }
    if (info.filename) {
      // TODO: Is this leaking memory?
      tau_filename = strdup(info.filename);
    } else {
      tau_filename = NULL;
    }
  } else {
    tau_line_no = 0;
    tau_funcname = NULL;
    tau_filename = NULL;
  }
  return tau_symbol_found;
}

int Tau_Backtrace_writeMetadata(int i, char *token, unsigned long addr) {
  static int flag = 0;
  if (flag == 0) {
    flag = 1;
  }
  char field[2048];
  char metadata[256];
  char *dem_name = NULL;
  char demangled_name[2048], line_info[2048];
  char cmd[2048];
  FILE *pipe_fp;
  TauBfdAddrMap *map = getAddressMap(addr);
  line_info[0]=0;

  /* Do we have a demangled name? */
  // Do not attempt to demangle via these means if using XLC. Use only BFD where
  //    available.
#ifndef TAU_XLC
  if (dem_name == (char *) NULL)  {
    char *subtoken=token;
    int i = 0;
    while (*subtoken!= '(' && i  < strlen(token)) {
      subtoken++; i++;
    }
    subtoken--; /* move the pointer to before the ( so we can use strtok */
    TAU_VERBOSE("Subtoken=%s\n", subtoken);
    char *subs=strtok(subtoken,"(+");
    subs = strtok(NULL,"+");
    if (subs == (char *) NULL) subs = token;
#ifndef __GNUC__
    sprintf(cmd, "c++filt %s", subs);
    TAU_VERBOSE("popen %s\n", cmd);
    pipe_fp = popen(cmd, "r");
    //fscanf(pipe_fp,"%s", demangled_name);
    int ret = fread(demangled_name, 1, 1024, pipe_fp);
    TAU_VERBOSE("name = %s, Demangled name = %s, ret = %d\n", token, demangled_name, ret);
    pclose(pipe_fp);
    dem_name = demangled_name;
#else /* __GNUC__ */
    std::size_t len=1024;
    int stat;
    char *out_buf= (char *) malloc (len);
    char *name = abi::__cxa_demangle(subs, out_buf, &len, &stat);
    if (stat == 0) dem_name = out_buf;
    else dem_name = subs;
    TAU_VERBOSE("DEM_NAME subs= %s dem_name= %s, name = %s, len = %d, stat=%d\n", subs, dem_name, name, len, stat);
#endif /* __GNUC__ */

  }
  if (dem_name == (char *) NULL) dem_name = token;
  TAU_VERBOSE("tauPrintAddr: final demangled name [%s]\n", dem_name);

#ifdef TAU_EXE
  if (map != NULL) {
    sprintf(cmd, "addr2line -e %s 0x%lx", map->name, addr);
    TAU_VERBOSE("popen %s\n", cmd);
    pipe_fp = popen(cmd, "r");
    fscanf(pipe_fp,"%s", line_info);
    TAU_VERBOSE("cmd = %s, line number = %s\n", cmd, line_info);
    pclose(pipe_fp);
    sprintf(field, "[%s] [%s] [%s]", dem_name, line_info, map->name);
  }
#endif /* TAU_EXE */
#endif /* TAU_XLC */

  /* The reason the TAU_BFD tag is still here is to allow for alternatives */
#ifdef TAU_BFD
  tauGetFilenameAndLineNo(addr);
  if (tau_symbol_found) {
    if (map != NULL) {
      TAU_VERBOSE("TauBackTrace: tauPrintAddr: Symbol found for [addr=%p] [name=%s] [file=%s] [line=%d] [map=%s]\n", addr, tau_funcname, tau_filename, tau_line_no, map->name);
      sprintf(field, "[%s] [%s:%d] [%s]", tau_funcname, tau_filename, tau_line_no, map->name);
    } else {
      TAU_VERBOSE("TauBackTrace: tauPrintAddr: Symbol found for [addr=%p] [name=%s] [file=%s] [line=%d] [map=%s]\n", addr, tau_funcname, tau_filename, tau_line_no, "unknown");
      sprintf(field, "[%s] [%s:%d] [%s]", tau_funcname, tau_filename, tau_line_no, "unknown");
    }
  } else {
    TAU_VERBOSE("TauBackTrace: tauPrintAddr: Symbol for [addr=%p] not found\n", addr);
    if (dem_name != NULL && map != NULL) {
      // Get address from gdb if possible
      TAU_VERBOSE("tauPrintAddr: Getting information from GDB instead\n");
      sprintf(field, "[%s] [Addr=%p] [%s]", dem_name,
	      addr+Tau_get_backtrace_off_by_one_correction(), map->name);
    } else {
      TAU_VERBOSE("tauPrintAddr: No Information Available\n");
      sprintf(field, "[%s] [addr=%p]", dem_name,
	      addr+Tau_get_backtrace_off_by_one_correction());
    }
  }
#else
  issueBfdWarningIfNecessary();
#endif /* TAU_BFD */
  sprintf(metadata, "BACKTRACE %3d", i-1);
  TAU_METADATA(metadata, field);
  return 0;
}

