#include <Profile/TauSampling.h>
#include <Profile/Profiler.h>
#include <Profile/TauBfd.h>
#include <ucontext.h>

#include <stdlib.h>
#include <map>
#include <vector>
using namespace std;

// For BFD-based name resolution
static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;

static inline unsigned long get_pc(void *p) {
  struct ucontext *uc = (struct ucontext *)p;
  unsigned long pc;

#ifdef sun
  issueUnavailableWarningIfNecessary("Warning, TAU Sampling does not work on solaris\n");
  return 0;
#elif __APPLE__
  issueUnavailableWarningIfNecessary("Warning, TAU Sampling does not work on apple\n");
  return 0;
#elif _AIX
  issueUnavailableWarningIfNecessary("Warning, TAU Sampling does not work on AIX\n");
  return 0;
#else
  struct sigcontext *sc;
  sc = (struct sigcontext *)&uc->uc_mcontext;
#ifdef TAU_BGP
  //  pc = (unsigned long)sc->uc_regs->gregs[PPC_REG_PC];
  pc = (unsigned long)UCONTEXT_REG(uc, PPC_REG_PC);
# elif __x86_64__
  pc = (unsigned long)sc->rip;
# elif i386
  pc = (unsigned long)sc->eip;
# elif __ia64__
  pc = (unsigned long)sc->sc_ip;
# elif __powerpc64__
  // it could possibly be "link" - but that is supposed to be the return address.
  pc = (unsigned long)sc->regs->nip;
# elif __powerpc__
  // it could possibly be "link" - but that is supposed to be the return address.
  pc = (unsigned long)sc->regs->nip;
# else
#  error "profile handler not defined for this architecture"
# endif /* TAU_BGP */
  return pc;
#endif /* sun */
}

/* Only needs to be defined in FunctionInfo.h
typedef struct CombinedKey {
  bool isCallSite;
  unsigned long keyValue;
} tau_cs_path_element_t;
*/

typedef struct TauCallSiteDB {
  bool resolved;
  unsigned long resolvedCallSite;
  string *resolvedName;
  unsigned long *key;
} tau_cs_info_t;

struct TauCsPath {
  // Is v1 "less than" v2?
  bool operator() (const vector<tau_cs_path_element_t *> *v1,
		   const vector<tau_cs_path_element_t *> *v2) const {
    int i;
    int l1, l2;
    l1 = v1->size();
    l2 = v2->size();
    if (l1 != l2) {
      return (l1 < l2);
    }
    
    for (i=0; i<l1; i++) {
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

struct TauCsULong {
  bool operator() (const unsigned long *l1, const unsigned long *l2) const {
   int i;

   /* first check 0th index (size) */
   if (l1[0] != l2[0]) {
     //     printf("length differs %d %d\n", l1[0], l2[0]);
     return (l1[0] < l2[0]);
   }

   /* they're equal, see the size and iterate */
   for (i = 0; i < l1[0] ; i++) {
     //     printf("[%p] [%p] | ", l1[i+1], l2[i+1]);
     if (l1[i+1] != l2[i+1]) {
       //       printf("Element %d different [%p] [%p]\n", i, l1[i+1], l2[i+1]);
       //       printf("\nNo Match!\n");
       return (l1[i+1] < l2[i+1]);
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
map<TAU_CALLSITE_KEY_ID_MAP_TYPE >& TheCallSiteKey2IdMap(void) {
  static map<TAU_CALLSITE_KEY_ID_MAP_TYPE > callsiteKey2IdMap;
  return callsiteKey2IdMap;
}

vector<tau_cs_info_t * >& TheCallSiteIdVector(void) {
  static vector<tau_cs_info_t *> callsiteId2KeyVec;
  return callsiteId2KeyVec;
}

map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE >& TheCallSiteFirstKeyMap(void) {
  static map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE > callsiteFirstKeyMap;
  return callsiteFirstKeyMap;
}

map<TAU_CALLSITE_PATH_MAP_TYPE >& TheCallSitePathMap(void) {
  // to avoid initialization problems of non-local static variables
  static map<TAU_CALLSITE_PATH_MAP_TYPE > callsitePathMap;
  return callsitePathMap;
}

static long callsiteId = 0;

void printCallSites(unsigned long *addresses) {
  unsigned long length = addresses[0];
  printf("[length = %d] ", length);
  for (int i=0; i<length; i++) {
    printf("%p ", addresses[i+1]);
  }
  printf("\n");
}

// *CWL* - This is really a character index search on a string with unsigned long 
//         alphabets. The goal is to find the first different character.
//         This returns the callsite with respect to a1.
unsigned long determineCallSite(unsigned long *a1, unsigned long *a2) {
  // a1 and a2 will not always have the same length
  int minLength = 0;
  if (a1[0] < a2[0]) {
    minLength = a1[0];
  } else {
    minLength = a2[0];
  }

  // The TAU-specific run time prefix from the same event should always
  //   line-up for a straightforward comparison from the base address.
  for (int i=0; i<minLength; i++) {
    if (a1[i+1] != a2[i+1]) {
      return a1[i+1];
    }
  }
  return 0;
}

unsigned long determineCallSiteViaId(unsigned long a1, unsigned long a2) {
  unsigned long *key1 = NULL;
  unsigned long *key2 = NULL;

  key1 = TheCallSiteIdVector()[a1]->key;
  key2 = TheCallSiteIdVector()[a2]->key;

  return determineCallSite(key1, key2);
}

int &TauGetCallPathDepth(void);
// Make combined callsite and callpath key. We consider the callsite limit
//   only in the context of actual callpaths. In the absence of callpaths,
//   we investigate the callsites for the function itself.
vector<tau_cs_path_element_t *> *Tau_callsite_MakePathKey(Profiler *p) {
  // Default, we care only for the top event in the stack.
  int callpath_depth = 1;
  int callsite_limit = TauEnv_get_callsite_limit();
  if (TauEnv_get_callpath() == 1) {
    callpath_depth = TauGetCallPathDepth();
  }
  vector<tau_cs_path_element_t *> *key = 
    new vector<tau_cs_path_element_t *>();
  Profiler *current = p; /* argument */

  // The top-level event is special. We have not decided to specialize
  //   it yet. This is the function that is meant to do this!
  printf("FUNC: %s\n", current->ThisFunction->GetName());

  tau_cs_path_element_t *newKeyElement = NULL;

  // Process the top-level item first and a possible callsite later
  newKeyElement = new tau_cs_path_element_t;
  newKeyElement->isCallSite = false;
  newKeyElement->keyValue = (unsigned long)current->ThisFunction;
  key->push_back(newKeyElement);
  callpath_depth--;

  if (callsite_limit != 0) {
    // Do we have callsites with the CallPath Function?
    if (current->hasCallSite) {
      newKeyElement = new tau_cs_path_element_t;
      newKeyElement->isCallSite = true;
      newKeyElement->keyValue = current->callsiteKeyId;
      key->push_back(newKeyElement);
      callsite_limit--;
    }
  }
  current = current->ParentProfiler;
  
  while (current != NULL && callpath_depth != 0) {
    // No callpath functions mean no callpaths nor callsites associated
    //   with this profiler stack entry. Just go on to the next entry
    printf("FUNC: %s\n", current->ThisFunction->GetName());
    FunctionInfo *candidate = NULL;
    if (TauEnv_get_callpath() == 1) {
      if (current->CallPathFunction == NULL) {
	candidate = current->ThisFunction;
      } else {
	candidate = current->CallPathFunction;
      }
    } else {
      candidate = current->ThisFunction;
    }

    // Process callpath first and possible callsites later
    tau_cs_path_element_t *newKeyElement = NULL;
    newKeyElement = new tau_cs_path_element_t;
    newKeyElement->isCallSite = false;
    // Callpaths always draw values from ThisFunction
    newKeyElement->keyValue = (unsigned long)current->ThisFunction;
    key->push_back(newKeyElement);
    callpath_depth--;
    //   If no callsites are desired or have been computed,
    //   the value is 0.
    if (callsite_limit != 0) {
      // Do we have callsites with the CallPath Function?
      if (candidate->hasCallSite) {
	newKeyElement = new tau_cs_path_element_t;
	newKeyElement->isCallSite = true;
	newKeyElement->keyValue = candidate->callSiteKeyId;
	key->push_back(newKeyElement);
	callsite_limit--;
      }
    }
    current = current->ParentProfiler;
  }
  return key;
}

// *CWL* - This is a mirror of CallPathStart in that all FunctionInfo objects
//         on the Profiler stack. It is to be used when no Callpaths are
//         required.
bool Tau_unwind_unwindTauContext(int tid, unsigned long *addresses);
void Profiler::CallSiteStart(int tid) {

  hasCallSite = false;

  // Ensure that the base function has no defined path nor callsite.
  ThisFunction->combinedPath = NULL; 
  ThisFunction->hasCallSite = false;
  ThisFunction->callSiteResolved = false; 

  // TAU's top level event is special.
  if (ParentProfiler == NULL) {
    CallPathFunction = NULL;
    return;
  }

  // *CWL* Stub for a test for whether we wish to acquire callsites for this function.
  // *CWL* - TODO. Determine if we even want to unwind for this function in the
  //    first place.
  if (1) {
    if (Tau_unwind_unwindTauContext(tid, callsites)) {
      map<TAU_CALLSITE_KEY_ID_MAP_TYPE >::iterator itCs = TheCallSiteKey2IdMap().find(callsites);
      
      if (itCs == TheCallSiteKey2IdMap().end()) {
	unsigned long *callsiteKey = NULL;

	printf("New CallSite Key %d\n", callsiteId);
	// *CWL* - It is important to make a copy of the callsiteKey for registration.
	callsiteKey = (unsigned long *)malloc(sizeof(unsigned long)*(TAU_SAMP_NUM_ADDRESSES+1));
	for (int i=0; i<TAU_SAMP_NUM_ADDRESSES+1; i++) {
	  callsiteKey[i] = callsites[i];
	}
	callsiteKeyId = callsiteId;
	TheCallSiteKey2IdMap().insert(map<TAU_CALLSITE_KEY_ID_MAP_TYPE >::value_type(callsiteKey, 
										     callsiteKeyId));
	tau_cs_info_t *callSite = (tau_cs_info_t *)malloc(sizeof(tau_cs_info_t));
	callSite->key = callsiteKey;
	callSite->resolved = false;
	callSite->resolvedCallSite = 0;
	callSite->resolvedName = NULL;
	TheCallSiteIdVector().push_back(callSite);
	callsiteId++;
      } else {
	// We've seen this callsite key before.
	callsiteKeyId = (*itCs).second;
	printf("Recalled CallSite Key %d\n", callsiteKeyId);
      }
      // We have now determined some callsite key and have registered it.
      hasCallSite = true;
    }
  }

  // Now construct the path's key to determine if we need new measurement instances of
  //   the event.
  vector<tau_cs_path_element_t *> *comparison = Tau_callsite_MakePathKey(this);
  map<TAU_CALLSITE_PATH_MAP_TYPE >::iterator itPath = TheCallSitePathMap().find(comparison);
  if (itPath == TheCallSitePathMap().end()) {
    printf("New Path\n");
    // This is a new branch, create a new FI object for it.
    string grname;
    if (TauEnv_get_callpath()) {
      grname = string("TAU_CALLPATH | ") + 
	RtsLayer::PrimaryGroup(ThisFunction->GetAllGroups());
    } else {
      grname = RtsLayer::PrimaryGroup(ThisFunction->GetAllGroups());
    }
    CallPathFunction = new FunctionInfo(ThisFunction->GetName(), "", 
					ThisFunction->GetProfileGroup(), 
					(const char*) grname.c_str(), true);

    // Book-keeping to determine the name later
    CallPathFunction->combinedPath = comparison;
    CallPathFunction->hasCallSite = hasCallSite;
    if (hasCallSite) {
      CallPathFunction->callSiteKeyId = callsiteKeyId; 
    }
    CallPathFunction->callSiteResolved = false;
    CallPathFunction->firstSpecializedFunction = NULL; // non-base functions are always NULL
    TheCallSitePathMap().insert(map<TAU_CALLSITE_PATH_MAP_TYPE>::value_type(comparison, 
									    CallPathFunction));
  } else {
    CallPathFunction = (*itPath).second;
    // sanity check
    if (hasCallSite) {
      if (CallPathFunction->callSiteKeyId != callsiteKeyId) {
	printf("Something is wrong. FI has Id %d from Unwind %d\n", 
	       CallPathFunction->callSiteKeyId, callsiteKeyId);
      }
    }
  }

  // Identify if the callsite key associated with the base function has been repeated
  //   for this particular specialization.
  map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE >::iterator itKey = 
    TheCallSiteFirstKeyMap().find(ThisFunction);
  if (itKey == TheCallSiteFirstKeyMap().end()) {
    // BASE Function not previously encountered. The callsite is necessarily unique.
    //   So, no callsite resolution is required.
    ThisFunction->firstSpecializedFunction = CallPathFunction;
    CallPathFunction->callSiteKeyId = callsiteKeyId;
    TheCallSiteFirstKeyMap().insert(map<TAU_CALLSITE_FIRSTKEY_MAP_TYPE >::
				    value_type(ThisFunction, CallPathFunction)); 
  } else {
    FunctionInfo *firstCallSiteFunction = (*itKey).second;
    if (CallPathFunction->callSiteKeyId != firstCallSiteFunction->callSiteKeyId) {
      // Different callsite. Try to resolve it if it has not already been resolved.
      //   If it has already been resolved, the first FI must also necessarily
      //   be resolved.
      if (!CallPathFunction->callSiteResolved) {
	// resolve the local callsite first.
	unsigned long resolvedCallSite = 0;
	resolvedCallSite = 
	  determineCallSiteViaId(CallPathFunction->callSiteKeyId,
				 firstCallSiteFunction->callSiteKeyId);
	printf("%d Got the final callsite %p\n", CallPathFunction->callSiteKeyId,
	       resolvedCallSite);
	// Register the resolution of this callsite key
	CallPathFunction->callSiteResolved = true;
	TheCallSiteIdVector()[CallPathFunction->callSiteKeyId]->resolved = true;
	TheCallSiteIdVector()[CallPathFunction->callSiteKeyId]->resolvedCallSite =
	  resolvedCallSite;

	if (!firstCallSiteFunction->callSiteResolved) {
	  resolvedCallSite =
	    determineCallSiteViaId(firstCallSiteFunction->callSiteKeyId,
				   CallPathFunction->callSiteKeyId);
	  printf("%d Got the final master callsite %p\n", firstCallSiteFunction->callSiteKeyId,
		 resolvedCallSite);
	  firstCallSiteFunction->callSiteResolved = true;
	  TheCallSiteIdVector()[firstCallSiteFunction->callSiteKeyId]->resolved = true;
	  TheCallSiteIdVector()[firstCallSiteFunction->callSiteKeyId]->resolvedCallSite =
	    resolvedCallSite;
	}
      }
    }
  }
  
  // Set up metrics. Increment number of calls and subrs
  CallPathFunction->IncrNumCalls(tid);
  
  // Next, if this function is not already on the call stack, put it
  if (CallPathFunction->GetAlreadyOnStack(tid) == false) {
    AddInclCallPathFlag = true;
    // We need to add Inclusive time when it gets over as
    // it is not already on callstack.
    
    CallPathFunction->SetAlreadyOnStack(true, tid); // it is on callstack now
  } else { 
    // the function is already on callstack, no need to add inclusive time
    AddInclCallPathFlag = false;
  }
}

void Profiler::CallSiteStop(double *TotalTime, int tid) {
  if (ParentProfiler != NULL) {
    if (AddInclCallPathFlag == true) { // The first time it came on call stack
      CallPathFunction->SetAlreadyOnStack(false, tid); // while exiting
      // And its ok to add both excl and incl times
      CallPathFunction->AddInclTime(TotalTime, tid);
    }
    
    CallPathFunction->AddExclTime(TotalTime, tid);  
    if (ParentProfiler->CallPathFunction != 0) {
      /* Increment the parent's NumSubrs and decrease its exclude time */
      ParentProfiler->CallPathFunction->ExcludeTime(TotalTime, tid);
    }
  }
}

static string getNameAndType(FunctionInfo *fi) {
  if (strlen(fi->GetType()) > 0) {
    return string(fi->GetName() + string (" ") + fi->GetType());
  } else {
    return string(fi->GetName());
  }
}

char *resolveTauCallSite(unsigned long addr) {
  int bfdRet; // used only for an old interface

  char *callsiteName;
  char resolvedBuffer[4096];
  // stub. BFD is needed for final solution.
  // resolved = Tau_sampling_resolveName(addr, &name, &resolvedModuleIdx);
  TauBfdInfo *resolvedInfo = NULL;
  // backup information in case we fail to resolve the address to specific
  //   line numbers.
  TauBfdAddrMap addressMap;
  sprintf(addressMap.name, "%s", "UNKNOWN");

#ifdef TAU_BFD
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_KEEP_GLOBALS);
  }

  // Attempt to use BFD to resolve names
  resolvedInfo = 
    Tau_bfd_resolveBfdInfo(bfdUnitHandle, (unsigned long)addr);
  // backup info
  bfdRet = Tau_bfd_getAddressMap(bfdUnitHandle, (unsigned long)addr,
				 &addressMap);
  if (resolvedInfo == NULL) {
      resolvedInfo = 
	  Tau_bfd_resolveBfdExecInfo(bfdUnitHandle, (unsigned long)addr);
      sprintf(addressMap.name, "%s", "EXEC");
  }
#endif /* TAU_BFD */
  if (resolvedInfo != NULL) {
    sprintf(resolvedBuffer, "[%s] [{%s} {%d,%d}-{%d,%d}]",
	    resolvedInfo->funcname,
	    resolvedInfo->filename,
	    resolvedInfo->lineno, 0,
	    resolvedInfo->lineno, 0);
  } else {
    sprintf(resolvedBuffer, "[%s] UNRESOLVED ADDR %p", 
	    addressMap.name, (void *)addr);
  }
  //  sprintf(resolvedBuffer, "<%p>", address);

  callsiteName = strdup((char *)resolvedBuffer);

  printf("[%p] resolves to %s\n", addr, resolvedBuffer);
  return callsiteName;
}

void finalizeCallSites(int tid) {
  // First pass: Identify and resolve callsites into name strings.

  printf("finalizing\n");

  for (int i=0; i<callsiteId; i++) {
    tau_cs_info_t *callsiteInfo = TheCallSiteIdVector()[i];
    string *tempName = new string("");
    if (callsiteInfo->resolved) {
      printf("ID %d resolved\n", i);
      // resolve a single address
      unsigned long callsite = callsiteInfo->resolvedCallSite;
      *tempName = string("[TAU_CALLSITE] ") + string(resolveTauCallSite(callsite));
      callsiteInfo->resolvedName = tempName;
    } else {
      printf("ID %d not resolved\n", i);
      // resolve the unwound callsites as a sequence
      unsigned long *key = callsiteInfo->key;
      int keyLength = key[0];
      // Bad if not true
      if (keyLength > 0) {
	*tempName = *tempName + string(resolveTauCallSite(key[1]));
      }
      for (int j=1; j<keyLength; j++) {
	*tempName = string(resolveTauCallSite(key[j+1])) + string(" + ") + *tempName;
      }
      *tempName = string("[TAU_CALLSITE] ") + *tempName;
      callsiteInfo->resolvedName = tempName;
      callsiteInfo->resolved = true;
    }
  }
  
  // Do the same as EBS. Acquire candidates first. We need to create new FunctionInfo
  //   objects representing the callsites themselves.
  vector<FunctionInfo *> *candidates =
    new vector<FunctionInfo *>();
  RtsLayer::LockDB();
  for (vector<FunctionInfo *>::iterator fI_iter = TheFunctionDB().begin();
       fI_iter != TheFunctionDB().end(); fI_iter++) {
    FunctionInfo *theFunction = *fI_iter;
    if (theFunction->combinedPath != NULL) {
      candidates->push_back(theFunction);
    }
  }
  RtsLayer::UnLockDB();

  vector<FunctionInfo *>::iterator cs_it;
  for (cs_it = candidates->begin(); cs_it != candidates->end(); cs_it++) {
    FunctionInfo *candidate = *cs_it;

    // Go through the path and resolve names. Each callpath entry points to 
    //   a function info object where we can get the name through 
    //   ThisFunction->GetName(). The path is stored in reversed order.
    string *callSiteName = new string("");
    string *prefixPathName = new string("");
    string *fullPathName = new string("");
    vector<tau_cs_path_element_t *> *comparison = candidate->combinedPath;

    string delimiter(" => ");
    bool hasPrefix = false;
    int compLength = comparison->size();
    // very bad if the following is not true.
    if (compLength > 0) {
      // The last entry *must* be the current function.
      int startIndex = 0;
      if (!(*comparison)[startIndex]->isCallSite) {
	*fullPathName = *fullPathName + candidate->GetName();
	startIndex++;
      } else {
	printf("The last entry is a callsite, this cannot happen!\n");
      }
      if (startIndex < compLength) {
	// Now check if the 2nd last entry is a callsite.
	*callSiteName = *callSiteName +
	  *(TheCallSiteIdVector()[(*comparison)[startIndex]->keyValue]->resolvedName);
	*fullPathName = *callSiteName + delimiter + *fullPathName;
	startIndex++;
      }
      // Everything else is a prefix path. Get the first prefix item.
      if (startIndex < compLength) {
	unsigned long value = (*comparison)[startIndex]->keyValue;
	if ((*comparison)[startIndex]->isCallSite) {
	  // a Callsite object
	  *prefixPathName = *prefixPathName +
	    *(TheCallSiteIdVector()[value]->resolvedName);
	} else {
	  // a FunctionInfo object
	  FunctionInfo *fi = (FunctionInfo *)value;
	  *prefixPathName = *prefixPathName + getNameAndType(fi); 
	}
	hasPrefix = true;
	startIndex++;
      }
      for (int i=startIndex; i<compLength; i++) {
	unsigned long value = (*comparison)[i]->keyValue;
	if ((*comparison)[i]->isCallSite) {
	  // a Callsite object
	  *prefixPathName =
	    *(TheCallSiteIdVector()[value]->resolvedName) + delimiter + *prefixPathName;
	} else {
	  // a FunctionInfo object
	  FunctionInfo *fi = (FunctionInfo *)value;
	  *prefixPathName = getNameAndType(fi) + delimiter + *prefixPathName; 
	}
      }
    }
    if (hasPrefix) {
      *fullPathName = *prefixPathName + delimiter + *fullPathName;
    }
    candidate->SetName(*fullPathName);

    string grname = string("TAU_CALLSITE | ") + 
      RtsLayer::PrimaryGroup(candidate->GetAllGroups());
    FunctionInfo *newFunction = 
      new FunctionInfo(*callSiteName, "",
		       candidate->GetProfileGroup(),
		       (const char*) grname.c_str(), true);
    // CallSiteFunction data is exactly the same as the recorded data. So, just add
    //   inclusive time.
    newFunction->AddInclTime(candidate->GetInclTime(tid), tid);
    // Has as many calls as the measured callsite.
    newFunction->SetCalls(tid, candidate->GetCalls(tid));
    newFunction->SetSubrs(tid, candidate->GetCalls(tid));

    // Now construct the path leading to the callsite. It has exactly the same
    //   properties as the base callsite FI.
    if (hasPrefix) {
      newFunction = new FunctionInfo(*prefixPathName + delimiter + *callSiteName, "",
				     candidate->GetProfileGroup(),
				     candidate->GetAllGroups(), true);
      newFunction->AddInclTime(candidate->GetInclTime(tid), tid);
      newFunction->SetCalls(tid, candidate->GetCalls(tid));
      newFunction->SetSubrs(tid, candidate->GetCalls(tid));
    } 
  }
}
