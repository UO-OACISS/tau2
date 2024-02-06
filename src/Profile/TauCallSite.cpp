#ifdef __APPLE__
#include <dlfcn.h>
#define _XOPEN_SOURCE 600 /* Single UNIX Specification, Version 3 */
#endif /* __APPLE__ */

#include <stdlib.h>
#include <ctype.h>
#include <map>
#include <vector>

#include <Profile/TauSampling.h>
#include <Profile/Profiler.h>
#include <Profile/TauBfd.h>
#include <Profile/TauTrace.h>

#ifndef TAU_WINDOWS
// #ifndef _AIX

/* Android didn't provide <ucontext.h> so we make our own */
#ifdef TAU_ANDROID
#include "android_ucontext.h"
#else
#include <ucontext.h>
#endif

#if !defined(_AIX) && !defined(__sun) && !defined(TAU_WINDOWS) && !defined(TAU_ANDROID) && !defined(TAU_NEC_SX)
#include <execinfo.h>
#define TAU_EXECINFO 1
#endif /* _AIX */

using namespace std;
using namespace tau;

// For BFD-based name resolution
static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;

typedef struct TauCallSitePathElement
{
  bool isCallSite;
  unsigned long keyValue;
} tau_cs_path_element_t;

typedef struct TauCallSiteInfo
{
  bool resolved;
  unsigned long resolvedCallSite;
  bool hasName;
  string *resolvedName;
  unsigned long *key;
} tau_cs_info_t;

struct TauCsPath
{
  // Is v1 "less than" v2?
  bool operator()(const vector<tau_cs_path_element_t *> *v1, const vector<tau_cs_path_element_t *> *v2) const
  {
    int i;
    int l1, l2;
    l1 = v1->size();
    l2 = v2->size();
    if (l1 != l2) {
      return (l1 < l2);
    }

    for (i = 0; i < l1; i++) {
      // For each element
      if ((*v1)[i]->isCallSite ^ (*v2)[i]->isCallSite) {
        // We interpret callsites as "less than" paths
        return (*v1)[i]->isCallSite;
      } else {
        // We can compare the values.
        if ((*v1)[i]->keyValue != (*v2)[i]->keyValue) {
          return ((*v1)[i]->keyValue < (*v2)[i]->keyValue);
        }
      }
    }
    return false;
  }
};

struct TauCsULong
{
  bool operator()(const unsigned long *l1, const unsigned long *l2) const
  {
    unsigned int i;

    /* first check 0th index (size) */
    if (l1[0] != l2[0]) {
      //     printf("length differs %d %d\n", l1[0], l2[0]);
      return (l1[0] < l2[0]);
    }

    /* they're equal, see the size and iterate */
    for (i = 0; i < l1[0]; i++) {
      //     printf("[%p] [%p] | ", l1[i+1], l2[i+1]);
      if (l1[i + 1] != l2[i + 1]) {
        //       printf("Element %d different [%p] [%p]\n", i, l1[i+1], l2[i+1]);
        //       printf("\nNo Match!\n");
        return (l1[i + 1] < l2[i + 1]);
      }
    }

    //   return (l1[i] < l2[i]);
    // gone thru the list, must be equal, so reply false!
    //   printf("\nMatch\n");
    return false;
  }
};

//////////////////////////////////////////////////////////////////////
// Global variables (wrapped in routines for static initialization)
/////////////////////////////////////////////////////////////////////////
#define TAU_CALLSITE_KEY_ID_MAP_TYPE unsigned long *, unsigned long, TauCsULong
#define TAU_CALLSITE_FIRSTKEY_MAP_TYPE FunctionInfo *, FunctionInfo *
#define TAU_CALLSITE_PATH_MAP_TYPE vector<tau_cs_path_element_t *> *, FunctionInfo *, TauCsPath

/////////////////////////////////////////////////////////////////////////
// We use global maps to maintain callsite book-keeping information
/////////////////////////////////////////////////////////////////////////

extern "C" void finalizeCallSites_if_necessary();

struct callsiteKey2IdMap_t : public map<TAU_CALLSITE_KEY_ID_MAP_TYPE>
{
  callsiteKey2IdMap_t() {}
  virtual ~callsiteKey2IdMap_t() {
  //Wait! We might not be done! Unbelieveable as it may seem, this object
  //could (and does sometimes) get destroyed BEFORE we have resolved the addresses. Bummer.
    finalizeCallSites_if_necessary();
  }
};
static std::mutex KeyVectorMutex;
static callsiteKey2IdMap_t& TheCallSiteKey2IdMap(void)
{
  static vector<callsiteKey2IdMap_t*> callsiteKey2IdMap;//[TAU_MAX_THREADS];
  int tid = RtsLayer::myThread();
  if(callsiteKey2IdMap.size()<=tid){
      std::lock_guard<std::mutex> guard(KeyVectorMutex);
      while(callsiteKey2IdMap.size()<=tid){
        callsiteKey2IdMap.push_back(new callsiteKey2IdMap_t());
      }
    }
  return *callsiteKey2IdMap[tid];
}

struct callsiteId2KeyVec_t : public vector<tau_cs_info_t *>
{
  callsiteId2KeyVec_t() {}
  virtual ~callsiteId2KeyVec_t() {
  //Wait! We might not be done! Unbelieveable as it may seem, this object
  //could (and does sometimes) get destroyed BEFORE we have resolved the addresses. Bummer.
    finalizeCallSites_if_necessary();
  }
};

static std::mutex IDVectorMutex;
static callsiteId2KeyVec_t& TheCallSiteIdVector(void)
{
  static vector<callsiteId2KeyVec_t*> callsiteId2KeyVec;//[TAU_MAX_THREADS];
  int tid = RtsLayer::myThread();
  if(callsiteId2KeyVec.size()<=tid){
      std::lock_guard<std::mutex> guard(IDVectorMutex);
      while(callsiteId2KeyVec.size()<=tid){
        callsiteId2KeyVec.push_back(new callsiteId2KeyVec_t());
      }
    }
  return *callsiteId2KeyVec[tid];
}

struct callsiteFirstKeyMap_t : public map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE>
{
  callsiteFirstKeyMap_t() {}
  virtual ~callsiteFirstKeyMap_t() {
  //Wait! We might not be done! Unbelieveable as it may seem, this object
  //could (and does sometimes) get destroyed BEFORE we have resolved the addresses. Bummer.
    finalizeCallSites_if_necessary();
  }
};

/* Not used?
static callsiteFirstKeyMap_t& TheCallSiteFirstKeyMap(void)
{
  static callsiteFirstKeyMap_t callsiteFirstKeyMap[TAU_MAX_THREADS];
  int tid = RtsLayer::myThread();
  return callsiteFirstKeyMap[tid];
}
*/

struct callsitePathMap_t : public map<TAU_CALLSITE_PATH_MAP_TYPE>
{
  callsitePathMap_t() {}
  virtual ~callsitePathMap_t() {
  //Wait! We might not be done! Unbelieveable as it may seem, this object
  //could (and does sometimes) get destroyed BEFORE we have resolved the addresses. Bummer.
    finalizeCallSites_if_necessary();
  }
};

static std::mutex PathMapVectorMutex;
static callsitePathMap_t& TheCallSitePathMap(void)
{
  // to avoid initialization problems of non-local static variables
  static vector<callsitePathMap_t*> callsitePathMap;//[TAU_MAX_THREADS];
  int tid = RtsLayer::myThread();
  if(callsitePathMap.size()<=tid){
      std::lock_guard<std::mutex> guard(PathMapVectorMutex);
      while(callsitePathMap.size()<=tid){
        callsitePathMap.push_back(new callsitePathMap_t());
      }
    }
  return *callsitePathMap[tid];
}

static vector<unsigned long> callSiteId;//[TAU_MAX_THREADS];
static std::mutex CallSiteVectorMutex;
static inline void checkCallSiteVector(int tid){
    if(callSiteId.size()<=tid){
      std::lock_guard<std::mutex> guard(CallSiteVectorMutex);
      while(callSiteId.size()<=tid){
        callSiteId.push_back(0);
      }
    }
}
static inline unsigned long getCallSiteId(int tid){
    checkCallSiteVector(tid);
    return callSiteId[tid];
}
static inline void setCallSiteId(int tid, unsigned long value){
    checkCallSiteVector(tid);
    callSiteId[tid]=value;
}
static inline void incrementCallSiteId(int tid){
    checkCallSiteVector(tid);
    callSiteId[tid]++;
}

void initializeCallSiteDiscoveryIfNecessary()
{
  static bool initialized = false;
  if (!initialized) {
    int vecSize=callSiteId.size();
    for (int i = 0; i < vecSize; i++) {
      setCallSiteId(i, 0);
    }
    initialized = true;
  }
}

void Tau_callsite_issueFailureNotice_ifNecessary()
{
  static bool warningIssued = false;
  if (!warningIssued) {
    fprintf(stderr, "WARNING: At least one failure to acquire TAU callsite encountered.\n");
    warningIssued = true;
  }
}

char * Tau_callsite_resolveCallSite(unsigned long addr)
{
  // adjust for the fact that the return address is the next instruction.
  addr -= 1;

  // Get the address map name
  char const * mapName = "UNKNOWN";
  RtsLayer::LockDB();
  TauBfdAddrMap const * addressMap = Tau_bfd_getAddressMap(bfdUnitHandle, addr);
  if (addressMap) {
    mapName = addressMap->name;
  }

  // Use BFD to look up the callsite info
  TauBfdInfo resolvedInfo;
#if defined(__APPLE__)
  bool resolved;
      Dl_info info;
      int rc = dladdr((const void *)addr, &info);
      if (rc == 0) {
        resolved = false;
      } else {
        resolved = true;
        resolvedInfo.probeAddr = addr;
        resolvedInfo.filename = strdup(info.dli_fname);
        resolvedInfo.funcname = strdup(info.dli_sname);
        resolvedInfo.lineno = 0; // Apple doesn't give us line numbers.
      }
#else
  bool resolved = Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, resolvedInfo);
#endif
  RtsLayer::UnLockDB();

  // Prepare and return the callsite string
  char * resolvedBuffer = NULL;
  int length = 0;
  if (resolved) {
    // this should be enough...
    length = strlen(resolvedInfo.funcname) + strlen(resolvedInfo.filename) + 100;
    resolvedBuffer = (char*)malloc(length * sizeof(char));
#ifndef TAU_NEC_SX
    char *demangled_funcname = Tau_demangle_name(resolvedInfo.funcname);
#else
    char *demangled_funcname = strdup(resolvedInfo.funcname);
#endif
    snprintf(resolvedBuffer, length * sizeof(char),  "[%s] [{%s} {%d}]",
        demangled_funcname, resolvedInfo.filename, resolvedInfo.lineno);
    free(demangled_funcname);
  } else {
    // this should be enough...
    length = strlen(mapName) + 32;
    resolvedBuffer = (char*)malloc(length * sizeof(char));
    snprintf(resolvedBuffer, length * sizeof(char),  "[%s] UNRESOLVED ADDR", mapName);
  }

  return resolvedBuffer;
}

// *CWL* - This is really a character index search on a string with unsigned long
//         alphabets. The goal is to find the first different character.
//         This returns the callsite with respect to a1.
unsigned long determineCallSite(unsigned long *a1, unsigned long *a2)
{
  // a1 and a2 will not always have the same length
  /*
   printf("1) ");
   for (int i=0; i<a1[0]; i++) {
   printf("%p ", a1[i+1]);
   }
   printf("\n");

   printf("2) ");
   for (int i=0; i<a2[0]; i++) {
   printf("%p ", a2[i+1]);
   }
   printf("\n");
   */
  int minLength = 0;
  if (a1[0] < a2[0]) {
    minLength = a1[0];
  } else {
    minLength = a2[0];
  }

  // The TAU-specific run time prefix from the same event should always
  //   line-up for a straightforward comparison from the base address.
  for (int i = 0; i < minLength; i++) {
    if (a1[i + 1] != a2[i + 1]) {
      return a1[i + 1];
    }
  }
  return 0;
}

unsigned long determineCallSiteViaId(unsigned long a1, unsigned long a2)
{
  unsigned long *key1 = NULL;
  unsigned long *key2 = NULL;

  key1 = TheCallSiteIdVector()[a1]->key;
  key2 = TheCallSiteIdVector()[a2]->key;

  return determineCallSite(key1, key2);
}

void Profiler::CallSiteAddPath(long *callpath_path, int tid)
{
  path = NULL;
  // *CWL* Stub for a test for whether we wish to acquire callsites for this function.
  if (1) {
    if (callpath_path == NULL) {
      return;
    }
    long length = callpath_path[0];
    path = (long *)malloc((length + 1) * sizeof(long));
    for (int i = 0; i <= length; i++) {
      path[i] = callpath_path[i];
    }
  }
}

size_t trimwhitespace(char *out, size_t len, const char *str)
{
  if (len == 0) return 0;

  const char *end;
  size_t out_size;

  // Trim leading space
  while (isspace(*str))
    str++;

  if (*str == 0)    // All spaces?
  {
    *out = 0;
    return 1;
  }

  // Trim trailing space
  end = str + strlen(str) - 1;
  while (end > str && isspace(*end))
    end--;
  end++;

  // Set output size to minimum of trimmed string length and buffer size minus 1
  size_t diff = end - str;
  size_t len_minus_1 = len - 1;
  out_size = diff < len_minus_1 ? diff : len_minus_1;

  // Copy trimmed string and add null terminator
  memcpy(out, str, out_size);
  out[out_size] = 0;

  return out_size;
}

// *CWL* - Looking for the following pattern: "tau*/src/" where * has no "/".
//         Also look for "tau*/include/" where * has no "/".
bool nameInTau(const char *name)
{
  //if (strstr(name, "UNRESOLVED ADDR") != NULL)
    //return false;
  const char * tmp_name = strchr(name, '{');
  // no leading '{'?
  if (tmp_name == nullptr) {
    tmp_name = strchr(name, '[');
  }
  // no leading '[', either? Then return false.
  if (tmp_name == nullptr) {
    return false;
  }
  tmp_name = tmp_name+1;

  static char const * libprefix[] = {"libtau", "libTAU", NULL};
  static char const * libsuffix[] = {".a", ".so", ".dylib", NULL};
  int offset = 0;
  int length = 0;
  // Check libTAU and varients.
  char const * prefix;
  for (char const ** p=libprefix; (prefix = *p); ++p) {
    char const * head = strstr(tmp_name, prefix);
    if (!head) continue;
    char const * suffix;
    for (char const ** s=libsuffix; (suffix = *s); ++s) {
      char const * tail = strrchr(head, '.');
      if (tail && !strncmp(tail, suffix, strlen(suffix))) {
        return true;
      }
    }
  }
  // Pretty ugly hack, I foresee much trouble ahead.
  const char *strPtr = strstr(tmp_name, "tau");
  if (strPtr != NULL) {
    length = strlen(strPtr);
    offset = strcspn(strPtr, "/");
    if (offset != length) {
      strPtr += offset;
      const char *temp = strstr(strPtr, "src/");
      if (temp != NULL) {
        return true;
      } else {
        // Try again with "include".
        temp = strstr(strPtr, "include/");
        if (temp != NULL) {
          return true;
        }
        return false;
      }
    } else {
      // no directory follows "tau". Not it.
      return false;
    }
  }
  return false;
}

bool nameIsUnknown(const char *name)
{
  const char *strPtr = NULL;
  strPtr = strstr(name, "{(unknown)}");
  if (strPtr != NULL) {
    return true;
  }
  return false;
}

static bool nameInSHMEM(const char *name)
{
  name = strchr(name, '[') + 1;
  char buff[6];
  if (strlen(name) < sizeof(buff)) return false;
  for (size_t i=0; i<sizeof(buff); ++i) {
    buff[i] = tolower(name[i]);
  }
  return !strncmp("shmem_", buff, 6);
}

bool nameInMPI(const char *name)
{
  name = strchr(name, '[') + 1;
  char buff[4];
  if (strlen(name) < sizeof(buff)) return false;
  for (size_t i=0; i<sizeof(buff); ++i) {
    buff[i] = tolower(name[i]);
  }
  return !strncmp("mpi_", buff, 4);
}



void registerNewCallsiteInfo(char *name, unsigned long callsite, int id)
{
  TAU_VERBOSE("Found non-tau non-unknown callsite via string [%s]\n", name);
  // Register the newly discovered callsite
  TheCallSiteIdVector()[id]->resolved = true;
  TheCallSiteIdVector()[id]->resolvedCallSite = callsite;
  TheCallSiteIdVector()[id]->hasName = true;
  string *temp = new string("");
  *temp = *temp + string(" [@] ") + string(name);
  TheCallSiteIdVector()[id]->resolvedName = temp;
}

// callsite is an output parameter
bool determineCallSiteViaString(unsigned long *addresses)
{
  unsigned long length = addresses[0];
  char *name;

  map<TAU_CALLSITE_KEY_ID_MAP_TYPE>::iterator itCs = TheCallSiteKey2IdMap().find(addresses);
  if (itCs == TheCallSiteKey2IdMap().end()) {
    // Very bad. The address should have been encountered and registered before.
    return false;
  } else {
    unsigned long id = (*itCs).second;
    if (TheCallSiteIdVector()[id]->hasName) {
      return true;
    }

    // Was MPI in my unwind path at some point?
    bool hasMPI = false;
    bool hasSHMEM = false;

    for (unsigned int i = 0; i < length; i++) {
      name = Tau_callsite_resolveCallSite(addresses[i + 1]);
      if (nameInTau(name)) {
        hasMPI = hasMPI || nameInMPI(name);
        hasSHMEM = hasSHMEM || nameInSHMEM(name);
        free(name);
        continue;
      } else {
        // Not in TAU. Found a boundary candidate.
        //
        // *CWL* - We need a general solution for this. Right now it is a horrid hack.
        //         The ideal solution is a way to determine which of the following
        //         instrumentation classes we are dealing with:
        //         1. Implicit Instrumentation (wrappers, compInst, dyninst)
        //              => Take the immediate boundary as the callsite.
        //         2. Explicit Instrumentation representing functions (PDT, API)
        //              => Take the parent to the immediate boundary if possible.
        //         3. Explicit Instrumentation NOT representing functions (PDT loops)
        //              => Take the immediate boundary as the callsite.
        //
        // For now, We make a blanket correction for all non-MPI invocations.
        //
        unsigned long callsite;
        if (!hasMPI) {
          // This is not an MPI chain. We assume it is a function event. Skip one level.
          //   The callsite into a function probe is not the same as the callsite into
          //   the function itself.
          hasSHMEM = hasSHMEM || nameInSHMEM(name);
          free(name);
          // No idea why this works, or why the magical "2" (or 6 for __PGI) is required below.

          int offset = hasSHMEM ? 1 : TauEnv_get_callsite_offset();

          if (i + offset < length) {
            callsite = addresses[i + offset];
            name = Tau_callsite_resolveCallSite(addresses[i + offset]);
            if(strstr(name,"__wrap_") != NULL) {
              //if(i + 3 < length) {
              for(size_t j=3; j<length-i; j++) {
                unsigned long callsite_unwrapped = addresses[i + j];//3];
                char *name_unwrapped = Tau_callsite_resolveCallSite(addresses[i + j]);//3]);
                if (strstr(name_unwrapped,"UNRESOLVED ADDR") == NULL) {
                  callsite = callsite_unwrapped;
                  strcpy(name, name_unwrapped);
                }
                free(name_unwrapped);
              }
            }
            bool callsite_resolved = (strstr(name,"UNRESOLVED ADDR") == NULL);
            if(callsite_resolved)
              registerNewCallsiteInfo(name, callsite, id);
            free(name);
            if(callsite_resolved)
              return true;
          }
        } else {
          if (nameInMPI(name)) {
            // MPI could not possibly have invoked an MPI chain.
            //   Ignore and continue searching.
            free(name);
            continue;
          } else {
            free(name);
            // MPI invocations have immediate callsites.
            callsite = addresses[i + 1];
            name = Tau_callsite_resolveCallSite(addresses[i + 1]);
            registerNewCallsiteInfo(name, callsite, id);
            free(name);
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool Tau_unwind_unwindTauContext(int tid, unsigned long *addresses);
void Profiler::CallSiteStart(int tid, x_uint64 TraceTimeStamp)
{

  // We do not record callsites with the top level timer.
  if (ParentProfiler == NULL) {
    CallSiteFunction = NULL;
    return;
  }

  //Initialization
  CallSiteFunction = NULL;

  // *CWL* Stub for a test for whether we wish to acquire callsites for this function.
  if (0) {
    // We're not interested in this function's callsite.
    CallSiteFunction = NULL;
    return;
  }

  // *CWL* - It is EXTREMELY important that this be called at one and only one spot (here!)
  //         for the purposes of callsite discovery.
  bool retVal = false;
#ifdef TAU_UNWIND
  retVal = Tau_unwind_unwindTauContext(tid, callsites);
#else
  // No unwinder. We'll have to make do with backtrace. Unfortunately, backtrace will
  //   not allow us to mitigate the effects of deep direct recursion, so expect some
  //   strange results in that department.
#ifdef TAU_EXECINFO
  void* array[TAU_SAMP_NUM_ADDRESSES];
  size_t size;
  // get void*'s for all entries on the stack
  size = backtrace(array, TAU_SAMP_NUM_ADDRESSES);
  // *CWL* NOTE: backtrace_symbols() will work for __APPLE__. Since addr2line fails
  //       there, backup information using the "-->" format could be employed for
  //       Mac OS X instead of "unresolved".
  if (size > 0) {
    // construct the callsite structure from the buffer.
    callsites[0] = (unsigned long)size;
    for (unsigned int i = 0; i < size; i++) {
      callsites[i + 1] = (unsigned long)array[i];
    }
    retVal = true;
  } else {
    // backtrace failed, we surrender.
    retVal = false;
  }
#else
  // If no backtrace available, we raise our hands in surrender.
  retVal = false;
#endif /* TAU_EXECINFO = !(_AIX || sun || windows) */
#endif /* TAU_UNWIND */
  if (!retVal) {
    // Unwind failed. Issue warning if necessary. No Callsite information.
    Tau_callsite_issueFailureNotice_ifNecessary();
    CallSiteFunction = NULL;
    return;
  }

  // Make sure we don't walk off the top of the stack
  // First element of callsites is the number of callsites on the stack
  size_t callsite_depth = TauEnv_get_callsite_depth();
  if (callsite_depth > callsites[0]) {
    callsite_depth = callsites[0];
  }
  for (size_t depth=0; depth<callsite_depth; ++depth) {
    unsigned long callsiteKey[TAU_SAMP_NUM_ADDRESSES+1];
    memset(callsiteKey, 0, sizeof(callsiteKey));
    memcpy(callsiteKey+1, callsites+1+depth, sizeof(callsiteKey)-(depth+1)*sizeof(unsigned long));
    callsiteKey[0] = callsites[0] - depth;

    map<TAU_CALLSITE_KEY_ID_MAP_TYPE>::iterator itCs = TheCallSiteKey2IdMap().find(callsiteKey);

    if (itCs == TheCallSiteKey2IdMap().end()) {
      // *CWL* - It is important to make a copy of the callsiteKey for registration.
      unsigned long * callsiteKeyCopy = (unsigned long*)malloc(sizeof(callsiteKey));
      memcpy(callsiteKeyCopy, callsiteKey, sizeof(callsiteKey));
      callsiteKeyId = getCallSiteId(tid);
      TheCallSiteKey2IdMap().insert(map<TAU_CALLSITE_KEY_ID_MAP_TYPE>::value_type(callsiteKeyCopy, callsiteKeyId));
      tau_cs_info_t *callSiteInfo = (tau_cs_info_t *)malloc(sizeof(tau_cs_info_t));
      callSiteInfo->key = callsiteKeyCopy;
      callSiteInfo->resolved = false;
      callSiteInfo->resolvedCallSite = 0;
      callSiteInfo->hasName = false;
      callSiteInfo->resolvedName = NULL;
      TheCallSiteIdVector().push_back(callSiteInfo);
      incrementCallSiteId(tid);
    } else {
      // We've seen this callsite key before.
      callsiteKeyId = (*itCs).second;
      //	printf("Recalled CallSite Key %d\n", callsiteKeyId);
    }

    // Proceed to construct the key
    string *prefixPathName = new string("");
    string delimiter = string(" => ");
    vector<tau_cs_path_element_t *> *key = new vector<tau_cs_path_element_t *>();
    if (path == NULL) {
      // Flat profile. Record the current base FI.
      tau_cs_path_element_t *element = new tau_cs_path_element_t;
      element->isCallSite = false;
      element->keyValue = (unsigned long)ThisFunction;
      key->push_back(element);
    } else {
      // There's some call path up to and including the top FI.
      //      printf("Path Length = %d\n", path[0]);
      for (int i = 0; i < path[0]; i++) {
        // *CWL* TODO - This is a little silly. We should hash the call paths
        //       to an ID and use that in the context of a pathId x callsiteId
        //       tuple instead.
        //
        //       Also keep an eye out for that conversion from (long) to (FunctionInfo *)
        tau_cs_path_element_t *element = new tau_cs_path_element_t;
        element->isCallSite = false;
        element->keyValue = (unsigned long)path[i + 1];    // path[0] is the length
        // Note: The path is in reverse order
        // First element
        //	printf("%s\n", ((FunctionInfo *)path[i+1])->GetName());
        if (i == path[0] - 1) {
          *prefixPathName = *prefixPathName + string(((FunctionInfo *)path[i + 1])->GetName());
          //	  printf("%s\n", prefixPathName->c_str());
        } else if (i != 0) {
          // everything other than the last element (which is myself)
          *prefixPathName = string(((FunctionInfo *)path[i + 1])->GetName()) + delimiter + *prefixPathName;
          //	  printf("%s\n", prefixPathName->c_str());
        }
        key->push_back(element);
      }
      // We do not need the original path data anymore. Free it.
      free(path);
    }

    // Now distinguish this event with the callsite key.
    tau_cs_path_element_t *element = new tau_cs_path_element_t;
    element->isCallSite = true;
    element->keyValue = callsiteKeyId;
    key->push_back(element);

    // Create or pull up a CallSite object to record information into.
    map<TAU_CALLSITE_PATH_MAP_TYPE>::iterator itPath = TheCallSitePathMap().find(key);
    if (itPath == TheCallSitePathMap().end()) {
      RtsLayer::LockEnv();
      // This is a new callsite, create a new FI object for it.
      //   The name is the same as either the callpath or base function and will
      //     be enhanced later with a resolved entry.

      // Resolve via string. If successful, the resolved name is registered.

      // First step - trim the name of the base function for use.
      int nameLength = strlen(ThisFunction->GetName());
      int prefixLength = strcspn(ThisFunction->GetName(), "[");
      char *shortenedName = NULL;
      if (prefixLength < nameLength) {
        shortenedName = (char *)malloc((prefixLength + 1) * sizeof(char));
        strncpy(shortenedName, ThisFunction->GetName(), prefixLength);
        shortenedName[prefixLength] = '\0';
      } else {
        shortenedName = strdup(ThisFunction->GetName());
      }
      if (CallPathFunction != NULL) {
        string grname = string("TAU_CALLSITE | ") + RtsLayer::PrimaryGroup(CallPathFunction->GetAllGroups());
        string tempName = *prefixPathName + delimiter + string("[CALLSITE] ") + string(shortenedName);
        CallSiteFunction = new FunctionInfo(tempName.c_str(), "", CallPathFunction->GetProfileGroup(), grname.c_str(),
            true);
      } else {
        string grname = string("TAU_CALLSITE | ") + RtsLayer::PrimaryGroup(ThisFunction->GetAllGroups());
        string tempName = string("[CALLSITE] ") + string(shortenedName);
        CallSiteFunction = new FunctionInfo(tempName.c_str(), "", ThisFunction->GetProfileGroup(), grname.c_str(),
            true);
      }
      CallSiteFunction->isCallSite = true;
      CallSiteFunction->callSiteKeyId = callsiteKeyId;
      CallSiteFunction->callSiteResolved = false;

      CallSiteFunction->firstSpecializedFunction = NULL;    // non-base functions are always NULL
      string tempName = string(shortenedName);
      CallSiteFunction->SetShortName(tempName);
      TheCallSitePathMap().insert(map<TAU_CALLSITE_PATH_MAP_TYPE>::value_type(key, CallSiteFunction));
      RtsLayer::UnLockEnv();
    } else {
      CallSiteFunction = (*itPath).second;
      // sanity check
      if (CallSiteFunction != NULL) {
        if (CallSiteFunction->callSiteKeyId != callsiteKeyId) {
          fprintf(stderr, "WARNING: Something is wrong. FI has Id %lu from Unwind %lu\n", CallSiteFunction->callSiteKeyId,
              callsiteKeyId);
        }
      }
    }

    if (TraceTimeStamp && TauEnv_get_tracing()) {
      // Tweak time stamp to preserve event order in trace
      TauTraceEvent(CallSiteFunction->GetFunctionId(), 1 /* entry */, tid, TraceTimeStamp-1, 1, TAU_TRACE_EVENT_KIND_CALLSITE);
    }

    // Set up metrics. Increment number of calls and subrs
    CallSiteFunction->IncrNumCalls(tid);
  } // END for (depth)

} // END Profiler::CallSiteStart


// *CWL* - Perform the necessary time accounting for CallSites. Note that CallSites
//         are essentially specialized mirrors of their CallPath or Base counterparts.
//         This means that we need to make adjustments to any active callsites on the
//         profiler stack the same way we would call paths or baseline functions.
void Profiler::CallSiteStop(double *TotalTime, int tid, x_uint64 TraceTimeStamp)
{
  if (CallSiteFunction != NULL) {
    // Is there an important distinction between callpaths and base functions?
    if (TauEnv_get_callpath()) {
      if (AddInclCallPathFlag) {    // The first time it came on call stack
        CallSiteFunction->AddInclTime(TotalTime, tid);
      }
    } else {
      if (AddInclFlag) {
        CallSiteFunction->AddInclTime(TotalTime, tid);
      }
    }
    CallSiteFunction->AddExclTime(TotalTime, tid);
    if (TraceTimeStamp && TauEnv_get_tracing()) {
      // Tweak time stamp to preserve event order in trace
      TauTraceEvent(CallSiteFunction->GetFunctionId(), -1 /* exit */, tid, TraceTimeStamp+1, 1, TAU_TRACE_EVENT_KIND_CALLSITE);
    }
  }
  if (ParentProfiler != NULL) {
    if (ParentProfiler->CallSiteFunction != NULL) {
      ParentProfiler->CallSiteFunction->ExcludeTime(TotalTime, tid);
    }
  }

}

/*
static string getNameAndType(FunctionInfo *fi)
{
  if (strlen(fi->GetType()) > 0) {
    return string(fi->GetName() + string(" ") + fi->GetType());
  } else {
    return string(fi->GetName());
  }
}
*/

 struct CallsiteFinalThreadList : vector<bool>{
      CallsiteFinalThreadList(){
         //printf("Creating CallsiteFinalThreadList at %p\n", this);
      }
     virtual ~CallsiteFinalThreadList(){
         //printf("Destroying CallsiteFinalThreadList at %p, with size %ld\n", this, this->size());
         Tau_destructor_trigger();
     }
   };

extern "C" void finalizeCallSites_if_necessary()
{
  //static std::mutex FinCaSiVectorMutex;
  static bool callsiteFinalizationSetup = false;
  static CallsiteFinalThreadList callsiteThreadFinalized;//[TAU_MAX_THREADS];
  if (!callsiteFinalizationSetup) {
    int vecSize=RtsLayer::getTotalThreads();
    //std::lock_guard<std::mutex> guard(FinCaSiVectorMutex);
    while(callsiteThreadFinalized.size()<=vecSize){
      callsiteThreadFinalized.push_back(false);
    }

    callsiteFinalizationSetup = true;
  }
  int tid = RtsLayer::myThread();
  if (!callsiteThreadFinalized[tid]) {
    callsiteThreadFinalized[tid] = true;
  } else {
    return;
  }

  // First pass: Identify and resolve callsites into name strings.
#ifdef TAU_BFD
  RtsLayer::LockDB();
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit();
  }
  RtsLayer::UnLockDB();
#endif /* TAU_BFD */

  string delimiter = string(" --> ");
  for (unsigned int i = 0; i < getCallSiteId(tid); i++) {
    tau_cs_info_t *callsiteInfo = TheCallSiteIdVector()[i];
    if (callsiteInfo && callsiteInfo->hasName) {
      // We've already done this in the discovery phase.
      continue;
    }
    char *name;
    string *tempName = new string("");
    if (!callsiteInfo) return;

    if (callsiteInfo->resolved) {
      // resolve a single address
      unsigned long callsite = callsiteInfo->resolvedCallSite;
      name = Tau_callsite_resolveCallSite(callsite);
      *tempName = string(" [@] ") + string(name);
      callsiteInfo->resolvedName = tempName;
      free(name);
    } else {
      unsigned long *key = callsiteInfo->key;
      // One last try with the string method.
      bool success = determineCallSiteViaString(key);
      // false; // *CWL* For debugging.
      if (!success) {
        // resolve the unwound callsites as a sequence
        int keyLength = key[0];
        // Bad if not true. Also the head entry cannot be Tau_start_timer.
        if (keyLength > 0) {
          name = Tau_callsite_resolveCallSite(key[keyLength]);
          *tempName = *tempName + string(name);
          free(name);
        }
        // process until "Tau_start_timer" is encountered and stop.
        for (int j = keyLength - 1; j > 0; j--) {
          name = Tau_callsite_resolveCallSite(key[j]);
          if (strstr(name, "Tau_start_timer") == NULL) {
            *tempName = *tempName + delimiter + string(name);
            free(name);
          } else {
            free(name);
            break;
          }
        }
        *tempName = string(" [@] ") + *tempName;
        callsiteInfo->resolvedName = tempName;
        callsiteInfo->resolved = true;
      }
    }
  }

  // Do the same as EBS. Acquire candidates first. We need to create new FunctionInfo
  //   objects representing the callsites themselves.
  vector<FunctionInfo *> *candidates = new vector<FunctionInfo *>();
  // For multi-threaded applications.
  RtsLayer::LockDB();
  for (map<TAU_CALLSITE_PATH_MAP_TYPE>::iterator fI_iter = TheCallSitePathMap().begin(); fI_iter != TheCallSitePathMap().end(); fI_iter++) {
    FunctionInfo *theFunction = (*fI_iter).second;
    if (theFunction->isCallSite) {
      candidates->push_back(theFunction);
    }
  }
  RtsLayer::UnLockDB();

  vector<FunctionInfo *>::iterator cs_it;
  for (cs_it = candidates->begin(); cs_it != candidates->end(); cs_it++) {
    FunctionInfo *candidate = *cs_it;

    string *callSiteName = new string("");
    tau_cs_info_t *callsiteInfo = TheCallSiteIdVector()[candidate->callSiteKeyId];
    if (callsiteInfo->hasName) {
      *callSiteName = *callSiteName + *(callsiteInfo->resolvedName);
	}

    if (TauEnv_get_callpath()) {
      RtsLayer::LockDB();
      // Create the standalone entry for the callsite FI (no path).
      //   This is necessary only if there are callpaths involved.
      string tempName = string("[CALLSITE] ") + string(candidate->GetShortName()) + *callSiteName;
      FunctionInfo *newFunction = new FunctionInfo(tempName, "", candidate->GetProfileGroup(),
          candidate->GetAllGroups(), true);
      // CallSiteFunction data is exactly the same as the recorded data.
      newFunction->AddExclTime(candidate->GetExclTime(tid), tid);
      newFunction->AddInclTime(candidate->GetInclTime(tid), tid);
      // Has as many calls as the measured callsite.
      newFunction->SetCalls(tid, candidate->GetCalls(tid));
      newFunction->SetSubrs(tid, candidate->GetSubrs(tid));
      RtsLayer::UnLockDB();
    }

    // Now rename the candidate with the completely resolved name
    //    printf("candidate name %s\n", candidate->GetName());
    string tempName = string(candidate->GetName() + *callSiteName);
    candidate->SetName(tempName);
  }
}
// #endif /* _AIX */
#endif
