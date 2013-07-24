#ifdef __APPLE__
#define _XOPEN_SOURCE 600 /* Single UNIX Specification, Version 3 */
#endif /* __APPLE__ */

#include <stdlib.h>
#include <ctype.h>
#include <map>
#include <vector>

#include <Profile/TauSampling.h>
#include <Profile/Profiler.h>
#include <Profile/TauBfd.h>

#ifndef TAU_WINDOWS
#include <ucontext.h>

#if !defined(_AIX) && !defined(__sun) && !defined(TAU_WINDOWS)
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
    int i;

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
map<TAU_CALLSITE_KEY_ID_MAP_TYPE>& TheCallSiteKey2IdMap(void)
{
  static map<TAU_CALLSITE_KEY_ID_MAP_TYPE> callsiteKey2IdMap[TAU_MAX_THREADS];
  int tid = RtsLayer::myThread();
  return callsiteKey2IdMap[tid];
}

vector<tau_cs_info_t *>& TheCallSiteIdVector(void)
{
  static vector<tau_cs_info_t *> callsiteId2KeyVec[TAU_MAX_THREADS];
  int tid = RtsLayer::myThread();
  return callsiteId2KeyVec[tid];
}

map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE>& TheCallSiteFirstKeyMap(void)
{
  static map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE> callsiteFirstKeyMap[TAU_MAX_THREADS];
  int tid = RtsLayer::myThread();
  return callsiteFirstKeyMap[tid];
}

map<TAU_CALLSITE_PATH_MAP_TYPE>& TheCallSitePathMap(void)
{
  // to avoid initialization problems of non-local static variables
  static map<TAU_CALLSITE_PATH_MAP_TYPE> callsitePathMap[TAU_MAX_THREADS];
  int tid = RtsLayer::myThread();
  return callsitePathMap[tid];
}

static unsigned long callSiteId[TAU_MAX_THREADS];

void initializeCallSiteDiscoveryIfNecessary()
{
  static bool initialized = false;
  if (!initialized) {
    for (int i = 0; i < TAU_MAX_THREADS; i++) {
      callSiteId[i] = 0;
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
  TauBfdAddrMap const * addressMap = Tau_bfd_getAddressMap(bfdUnitHandle, addr);
  if (addressMap) {
    mapName = addressMap->name;
  }

  // Use BFD to look up the callsite info
  TauBfdInfo resolvedInfo;
  bool resolved = Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, resolvedInfo);

  // Prepare and return the callsite string
  char * resolvedBuffer = (char*)malloc(4096);
  if (resolved) {
    snprintf(resolvedBuffer, sizeof(resolvedBuffer), "[%s] [{%s} {%d}]",
        resolvedInfo.funcname, resolvedInfo.filename, resolvedInfo.lineno);
  } else {
    snprintf(resolvedBuffer, sizeof(resolvedBuffer), "[%s] UNRESOLVED ADDR", mapName);
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
  out_size = (end - str) < len - 1 ? (end - str) : len - 1;

  // Copy trimmed string and add null terminator
  memcpy(out, str, out_size);
  out[out_size] = 0;

  return out_size;
}

// *CWL* - Looking for the following pattern: "tau*/src/" where * has no "/".
//         Also look for "tau*/include/" where * has no "/".
bool nameInTau(const char *name)
{
  int offset = 0;
  int length = 0;
  // Pretty ugly hack, I foresee much trouble ahead.
  const char *strPtr = strstr(name, "tau");
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
  } else {
    return false;
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

bool nameInMPI(const char *name)
{
  int len = strlen(name);
  char *outString = (char *)malloc(sizeof(char) * (len + 1));
  trimwhitespace(outString, len, name);
  int prefixLen = 6;
  char* mpiCheckBuffer = (char*)malloc((prefixLen + 1) * sizeof(char));
  for (int i = 0; i < prefixLen; i++) {
    mpiCheckBuffer[i] = (char)tolower((int)outString[i]);
  }
  mpiCheckBuffer[prefixLen] = '\0';

  char *strPtr = NULL;
  strPtr = strstr((char *)mpiCheckBuffer, "mpi_");

  free(mpiCheckBuffer);
  free(outString);

  if (strPtr != NULL) {
    return true;
  }
  return false;
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
  char *strPtr = NULL;
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

    for (int i = 0; i < length; i++) {
      name = Tau_callsite_resolveCallSite(addresses[i + 1]);
      if (nameInTau(name)) {
        hasMPI = hasMPI | nameInMPI(name);
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
          free(name);
          if (i + 2 < length) {
            callsite = addresses[i + 2];
            name = Tau_callsite_resolveCallSite(addresses[i + 2]);
            registerNewCallsiteInfo(name, callsite, id);
            free(name);
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
void Profiler::CallSiteStart(int tid)
{

  // We do not record callsites with the top level timer.
  if (ParentProfiler == NULL) {
    CallSiteFunction = NULL;
    return;
  }

  //Initialization
  CallSiteFunction = NULL;

  // *CWL* Stub for a test for whether we wish to acquire callsites for this function.
  if (1) {
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
    void *array[TAU_SAMP_NUM_ADDRESSES];
    size_t size;
    // get void*'s for all entries on the stack
    size = backtrace(array, TAU_SAMP_NUM_ADDRESSES);
    // *CWL* NOTE: backtrace_symbols() will work for __APPLE__. Since addr2line fails
    //       there, backup information using the "-->" format could be employed for
    //       Mac OS X instead of "unresolved".
    if ((array != NULL) && (size > 0)) {
      // construct the callsite structure from the buffer.
      callsites[0] = (unsigned long)size;
      for (int i = 0; i < size; i++) {
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
    if (retVal) {
      map<TAU_CALLSITE_KEY_ID_MAP_TYPE>::iterator itCs = TheCallSiteKey2IdMap().find(callsites);

      if (itCs == TheCallSiteKey2IdMap().end()) {
        unsigned long *callsiteKey = NULL;

        //	printf("New CallSite Key %d\n", callSiteId[tid]);
        // *CWL* - It is important to make a copy of the callsiteKey for registration.
        callsiteKey = (unsigned long *)malloc(sizeof(unsigned long) * (TAU_SAMP_NUM_ADDRESSES + 1));
        for (int i = 0; i < TAU_SAMP_NUM_ADDRESSES + 1; i++) {
          //	  printf("%p ", callsites[i]);
          callsiteKey[i] = callsites[i];
        }
        //	printf("\n");
        callsiteKeyId = callSiteId[tid];
        TheCallSiteKey2IdMap().insert(map<TAU_CALLSITE_KEY_ID_MAP_TYPE>::value_type(callsiteKey, callsiteKeyId));
        tau_cs_info_t *callSiteInfo = (tau_cs_info_t *)malloc(sizeof(tau_cs_info_t));
        callSiteInfo->key = callsiteKey;
        callSiteInfo->resolved = false;
        callSiteInfo->resolvedCallSite = 0;
        callSiteInfo->hasName = false;
        callSiteInfo->resolvedName = NULL;
        TheCallSiteIdVector().push_back(callSiteInfo);
        callSiteId[tid]++;
      } else {
        // We've seen this callsite key before.
        callsiteKeyId = (*itCs).second;
        //	printf("Recalled CallSite Key %d\n", callsiteKeyId);
      }
    } else {
      // Unwind failed. Issue warning if necessary. No Callsite information.
      Tau_callsite_issueFailureNotice_ifNecessary();
      CallSiteFunction = NULL;
      return;
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

    // Has the callsite key for the base function been seen before?
    map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE>::iterator itKey = TheCallSiteFirstKeyMap().find(ThisFunction);
    if (itKey == TheCallSiteFirstKeyMap().end()) {
      // BASE Function not previously encountered. The callsite is necessarily unique.
      //   So, no callsite resolution is required.
      ThisFunction->firstSpecializedFunction = CallSiteFunction;
      TheCallSiteFirstKeyMap().insert(map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE>::value_type(ThisFunction, CallSiteFunction));
    } else {
      FunctionInfo *firstCallSiteFunction = (*itKey).second;
      if (CallSiteFunction->callSiteKeyId != firstCallSiteFunction->callSiteKeyId) {
        // Different callsite. Try to resolve it if it has not already been resolved.
        //   If it has already been resolved, the first FI must also necessarily
        //   be resolved.
        if (!CallSiteFunction->callSiteResolved) {
          // resolve the local callsite first.
          unsigned long resolvedCallSite = 0;
          resolvedCallSite = determineCallSiteViaId(CallSiteFunction->callSiteKeyId,
              firstCallSiteFunction->callSiteKeyId);
          TAU_VERBOSE("%d Got the final callsite %p\n", CallSiteFunction->callSiteKeyId, resolvedCallSite);
          // Register the resolution of this callsite key
          CallSiteFunction->callSiteResolved = true;
          TheCallSiteIdVector()[CallSiteFunction->callSiteKeyId]->resolved = true;
          TheCallSiteIdVector()[CallSiteFunction->callSiteKeyId]->resolvedCallSite = resolvedCallSite;

          if (!firstCallSiteFunction->callSiteResolved) {
            resolvedCallSite = determineCallSiteViaId(firstCallSiteFunction->callSiteKeyId,
                CallSiteFunction->callSiteKeyId);
            TAU_VERBOSE("%d Got the final master callsite %p\n", firstCallSiteFunction->callSiteKeyId,
                resolvedCallSite);
            firstCallSiteFunction->callSiteResolved = true;
            TheCallSiteIdVector()[firstCallSiteFunction->callSiteKeyId]->resolved = true;
            TheCallSiteIdVector()[firstCallSiteFunction->callSiteKeyId]->resolvedCallSite = resolvedCallSite;
          }
        }
      }
    }
    // Set up metrics. Increment number of calls and subrs
    CallSiteFunction->IncrNumCalls(tid);
  } else {    // Stub for the desire of callsites.
    // We're not interested in this function's callsite.
    CallSiteFunction = NULL;
  }
}

// *CWL* - Perform the necessary time accounting for CallSites. Note that CallSites
//         are essentially specialized mirrors of their CallPath or Base counterparts.
//         This means that we need to make adjustments to any active callsites on the
//         profiler stack the same way we would call paths or baseline functions.
void Profiler::CallSiteStop(double *TotalTime, int tid)
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
  }
  if (ParentProfiler != NULL) {
    if (ParentProfiler->CallSiteFunction != NULL) {
      ParentProfiler->CallSiteFunction->ExcludeTime(TotalTime, tid);
    }
  }
}

static string getNameAndType(FunctionInfo *fi)
{
  if (strlen(fi->GetType()) > 0) {
    return string(fi->GetName() + string(" ") + fi->GetType());
  } else {
    return string(fi->GetName());
  }
}

extern "C" void finalizeCallSites_if_necessary()
{
  static bool callsiteFinalizationSetup = false;
  static bool callsiteThreadFinalized[TAU_MAX_THREADS];
  if (!callsiteFinalizationSetup) {
    for (int i = 0; i < TAU_MAX_THREADS; i++) {
      callsiteThreadFinalized[i] = false;
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
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit();
  }
#endif /* TAU_BFD */

  //  printf("Callsites finalizing\n");
  string delimiter = string(" --> ");
  for (int i = 0; i < callSiteId[tid]; i++) {
    tau_cs_info_t *callsiteInfo = TheCallSiteIdVector()[i];
    if (callsiteInfo->hasName) {
      // We've already done this in the discovery phase.
      continue;
    }
    char *name;
    string *tempName = new string("");
    if (callsiteInfo->resolved) {
      //      printf("ID %d resolved\n", i);
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
        //      printf("ID %d not resolved\n", i);
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
  for (vector<FunctionInfo *>::iterator fI_iter = TheFunctionDB().begin(); fI_iter != TheFunctionDB().end();
      fI_iter++) {
    FunctionInfo *theFunction = *fI_iter;
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
    *callSiteName = *callSiteName + *(callsiteInfo->resolvedName);

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
#endif
