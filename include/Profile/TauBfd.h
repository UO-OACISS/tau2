#ifndef _TAU_BFD_H
#define _TAU_BFD_H

#define TAU_BFD_SYMTAB_LOAD_FAILED 0
#define TAU_BFD_SYMTAB_LOAD_SUCCESS 1
#define TAU_BFD_SYMTAB_LOAD_UNRESOLVED 2

#define TAU_BFD_NULL_HANDLE -1
#define TAU_BFD_NULL_MODULE_HANDLE -1
#define TAU_BFD_INVALID_MODULE -2

#define TAU_BFD_KEEP_GLOBALS 0
#define TAU_BFD_REUSE_GLOBALS 1

typedef int tau_bfd_handle_t;
typedef int tau_bfd_module_handle_t;

/* An iterator function that will take in values generated: */
/*   function name, file name, line number */
/*   by BFD functionality and do some user-defined work on them. */
typedef void (*TauBfdIterFn)(unsigned long, const char *, const char *, int);

typedef struct {
  unsigned long start, end, offset;
  char name[512];
} TauBfdAddrMap;

typedef struct {
  unsigned long probeAddr;
  char *filename;
  char *funcname;
  int lineno;
} TauBfdInfo;

/* Main interface functions */
void Tau_bfd_initializeBfdIfNecessary();
tau_bfd_handle_t Tau_bfd_registerUnit(int flag);
bool Tau_bfd_checkHandle(tau_bfd_handle_t handle);
void Tau_bfd_updateAddressMaps(tau_bfd_handle_t handle);
TauBfdInfo *Tau_bfd_resolveBfdInfo(tau_bfd_handle_t handle, 
				   unsigned long probe_addr);
TauBfdInfo *Tau_bfd_resolveBfdExecInfo(tau_bfd_handle_t handle,
				       unsigned long probe_addr);
int Tau_bfd_processBfdModuleInfo(tau_bfd_handle_t handle,
				 tau_bfd_module_handle_t moduleHandle,
				 int maxProbe,
				 TauBfdIterFn fn);
int Tau_bfd_processBfdExecInfo(tau_bfd_handle_t handle,
			       int maxProbe,
			       TauBfdIterFn fn);

/* Query functions */
vector<TauBfdAddrMap> *Tau_bfd_getAddressMaps(tau_bfd_handle_t handle);
tau_bfd_module_handle_t Tau_bfd_getModuleHandle(tau_bfd_handle_t handle,
						unsigned long probe_addr);

#endif /* _TAU_BFD_H */
