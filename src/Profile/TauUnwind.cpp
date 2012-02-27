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

struct TauCsLong {
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

typedef struct TauFuncCsKey {
  unsigned long *masterKey;
  bool isUnique;
} tau_cs_key_t;

//////////////////////////////////////////////////////////////////////
// Global variables (wrapped in routines for static initialization)
/////////////////////////////////////////////////////////////////////////
#define TAU_CALLSITE_MAP_TYPE unsigned long *, FunctionInfo *, TauCsLong
#define TAU_CALLSITE_KEY_MAP_TYPE FunctionInfo *, tau_cs_key_t *
#define TAU_CALLSITE_KEY_ID_MAP_TYPE unsigned long *, long, TauCsLong

/////////////////////////////////////////////////////////////////////////
// We use one global map to store the callpath information
/////////////////////////////////////////////////////////////////////////
map<TAU_CALLSITE_MAP_TYPE >& TheCallSiteMap(void) {
  // to avoid initialization problems of non-local static variables
  static map<TAU_CALLSITE_MAP_TYPE > callsitemap;
  
  return callsitemap;
}

map<TAU_CALLSITE_KEY_MAP_TYPE >& TheCallSiteKeyMap(void) {
  static map<TAU_CALLSITE_KEY_MAP_TYPE > callsiteKeyMap;

  return callsiteKeyMap;
}

map<TAU_CALLSITE_KEY_ID_MAP_TYPE >& TheCallSiteKeyIdMap(void) {
  static map<TAU_CALLSITE_KEY_ID_MAP_TYPE > callsiteKeyIdMap;
  return callsiteKeyIdMap;
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
  return callsiteName;
}

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

// *CWL* - This is a mirror of CallPathStart in that all FunctionInfo objects
//         on the Profiler stack. It is to be used when no Callpaths are
//         required.
void Tau_unwind_unwindTauContext(int tid, unsigned long *addresses);
void Profiler::FindCallSite(int tid) {
  unsigned long *key = NULL;

  // Capture and record the callsite key. This is represented by a sequence of 
  //    TAU_SAMP_NUM_ADDRESSES
  //    callsite addresses starting from some location within TAU. Cloistered within
  //    these addresses potentially lie the application's callsite into a TAU event.

  // *CWL* - TODO. Determine if we even want to unwind for this function in the
  //    first place.
  Tau_unwind_unwindTauContext(tid, callsiteKey);
  printf("Base Function is %s\n", ThisFunction->GetName());
  printCallSites(callsiteKey);

  // Does my parent have a callsite? If so, I need to explicitly increment 
  //   its subroutine count. If not, the regular book-keeping routines will
  //   take care of it.
  if (ParentProfiler != NULL) {
    map<TAU_CALLSITE_KEY_MAP_TYPE >::iterator itKey = 
      TheCallSiteKeyMap().find(ParentProfiler->ThisFunction);
    if (itKey == TheCallSiteKeyMap().end()) {
      // Base Function not registered. It does not have a callsite.
    } else {
      tau_cs_key_t *masterKey = (*itKey).second;
      if (!masterKey->isUnique) {
	// It is a differentiated function. Explicitly increment the subroutine count
	
      }
    }
  }

  // Is this a new callsite key?
  map<TAU_CALLSITE_MAP_TYPE >::iterator it = TheCallSiteMap().find(callsiteKey);
  if (it == TheCallSiteMap().end()) {
    // *CWL* - It is important to make a copy of the callsiteKey for insertion into the map.
    key = (unsigned long *)malloc(sizeof(unsigned long)*(TAU_SAMP_NUM_ADDRESSES+1));
    // copy length element
    key[0] = callsiteKey[0];
    for (int i=0; i<callsiteKey[0]; i++) {
      key[i+1] = callsiteKey[i+1];
    }
    printf("New Callsite Key\n");

    // Has this Base Function encountered this particular key before?
    map<TAU_CALLSITE_KEY_MAP_TYPE >::iterator itKey = TheCallSiteKeyMap().find(ThisFunction);
    tau_cs_key_t *masterKey = NULL;
    if (itKey == TheCallSiteKeyMap().end()) {
      // BASE Function not previously encountered. Insert the master key for this Function.
      //   Do nothing else, this function does not yet have a non-unique callsite key and
      //   does not need to be distinguished any further.
      masterKey = (tau_cs_key_t *)malloc(sizeof(tau_cs_key_t));
      masterKey->masterKey = key;
      masterKey->isUnique = true;
      TheCallSiteKeyMap().insert(map<TAU_CALLSITE_KEY_MAP_TYPE >::value_type(ThisFunction, 
									     masterKey)); 
      TheCallSiteMap().insert(map<TAU_CALLSITE_MAP_TYPE >::value_type(key, ThisFunction));
    } else {
      // This function has a master key registered. This cannot be the same as the
      //   currently encountered NEW callsite key (the reason we're in this conditional). 
      // Proceed to: 
      //   1. Extract the exact callsite based on the difference between the master key
      //   and the current key.
      //   2. Create a new FunctionInfo object to represent this new measurement.
      //   3. If the master key was previously unique, then we should also compute the
      //   callsite with respect to the master key's Function and store it. We will also
      //   want to change its name. 
      masterKey = (*itKey).second;

      // Create a new FunctionInfo object specific to the newly discovered callsite. This is
      //   a path if we care about non-local callsites. Otherwise, just do it for this
      //   FunctionInfo object.
      char thisName[1024];
      sprintf(thisName, "=> %s", ThisFunction->GetName());
      FunctionInfo *CallSiteFunction = new FunctionInfo(thisName, "",
							ThisFunction->GetProfileGroup(),
							ThisFunction->GetAllGroups(), true);
      TheCallSiteMap().insert(map<TAU_CALLSITE_MAP_TYPE >::value_type(key, CallSiteFunction));
      CallSiteFunction->eventCallSite = determineCallSite(key, masterKey->masterKey);
      CallSiteFunction->IncrNumCalls(tid);
      printf("My exact callsite is [%p].\n", CallSiteFunction->eventCallSite);

      // Create a new FunctionInfo object specific to the original and replace it in the
      //   map.
      if (masterKey->isUnique) {
	char replacementName[1024];
	FunctionInfo *masterFunction = (*itKey).first;
	FunctionInfo *replacementFunction; 

	// This is still a stub name that needs to be resolved.
	sprintf(replacementName, "=> %s", masterFunction->GetName());
	replacementFunction = new FunctionInfo(replacementName, "",
					       masterFunction->GetProfileGroup(),
					       masterFunction->GetAllGroups(), true);
	replacementFunction->eventCallSite = determineCallSite(masterKey->masterKey, key);
	// replacementFunction data is set to exactly the values as the recorded data of
	//   the function being replaced. From now on, however, they will be independently
	//   measured.
	replacementFunction->SetInclTime(tid, masterFunction->GetInclTime(tid));
	replacementFunction->SetExclTime(tid, masterFunction->GetExclTime(tid));
	replacementFunction->SetCalls(tid, masterFunction->GetCalls(tid));
	replacementFunction->SetSubrs(tid, masterFunction->GetSubrs(tid));

	printf("My master callsite is [%p].\n", replacementFunction->eventCallSite);

	// Need to undo the setting of the key-to-function.
	(*it).second = replacementFunction;
    	masterKey->isUnique = false;
      }

    }
  } else {
    // This key has been found before. All subsequent access will use the key to
    //   locate the FunctionInfo object (which could be modified when it is
    //   determined to have non-unique callsites).
    printf("Callsite already exists\n");
  }
}

void Profiler::StopCallSite(double *totalTime, int tid) {
  FunctionInfo *theFunction;
  map<TAU_CALLSITE_MAP_TYPE >::iterator it = TheCallSiteMap().find(callsiteKey);
  if (it == TheCallSiteMap().end()) {
    // This is an error. Something has gone terribly wrong with the Profiler stack.
    // *CWL* TODO - handle it. For now, we ignore it.
    printf("Error!!! Key for this Profiler object cannot possibly have gone missing!\n");
    exit(-1);
  } else {
    theFunction = (*it).second;
    theFunction->AddInclTime(totalTime, tid);
    theFunction->AddExclTime(totalTime, tid);
    printf("Stopping function %s\n", theFunction->GetName());
  }
}

// 0 represents a non callsite.
// The negation of the id key is used. 
static long callsiteId = 1;

// This is replicated from TauCallPath.
int& Tau_unwind_GetCallPathDepth(void) {
  static int value = 0;

  if (value == 0) {
    value = TauEnv_get_callpath_depth();
    if (value <= 1) {
      /* minimum of 2 */
      value = 2;
    }
  }
  return value;
}

// Make combined callsite and callpath key
long *Tau_unwind_MakePathKey(Profiler *p) {
  int depth = Tau_unwind_GetCallPathDepth();

  long *key = new long [depth*2 +1];
  Profiler *current = p; /* argument */

  int index = 0;
  while (current != NULL && depth != 0) {
    // There are two parts to a key - the callsite and the callpath.
    // For now, we use the simpler, but less efficient method for
    //   including callsites. If no callsites are desired/computed,
    //   the value is 0.
    key[index*2+1] = -(current->callsiteKeyId);
    //    key[index+1] = Tau_convert_ptr_to_long(current->ThisFunction); 
    key[index*2+2] = (long)current->ThisFunction; 
    index++;
    depth--;
    current = current->ParentProfiler;
  }
  key[0] = index*2;

  return key;
}

void Profiler::CallSitePathStart(int tid) {
  // Capture and record the callsite key. This is represented by a sequence of 
  //    TAU_SAMP_NUM_ADDRESSES
  //    callsite addresses starting from some location within TAU. Cloistered within
  //    these addresses potentially lie the application's callsite into a TAU event.

  // *CWL* - TODO. Determine if we even want to unwind for this function in the
  //    first place.
  callsiteKeyId = 0; // default - we don't care about this callsite.
  Tau_unwind_unwindTauContext(tid, callsiteKey);
  map<TAU_CALLSITE_KEY_ID_MAP_TYPE >::iterator it = TheCallSiteKeyIdMap().find(callsiteKey);
  if (it == TheCallSiteKeyIdMap().end()) {
    callsiteKeyId = callsiteId;
    unsigned long *key;
    // *CWL* - It is important to make a copy of the callsiteKey for insertion into the map.
    key = (unsigned long *)malloc(sizeof(unsigned long)*(TAU_SAMP_NUM_ADDRESSES+1));
    // copy length element
    key[0] = callsiteKey[0];
    for (int i=0; i<callsiteKey[0]; i++) {
      key[i+1] = callsiteKey[i+1];
    }
    TheCallSiteKeyIdMap().insert(map<TAU_CALLSITE_KEY_ID_MAP_TYPE >::value_type(key, 
										callsiteId++));
  } else {
    // We've seen this callsite key before.
    callsiteKeyId = (*it).second;
  }

  long *comparison = 0;
  // Construct the combined callsite + callpath key
  comparison = Tau_unwind_MakePathKey(this);
  /*
  printf("key is length %d\n", comparison[0]);
  for (int i=0; i<comparison[0]; i++) {
    printf("%ld ",comparison[i+1]);
  }
  printf("\n");
  */
  // Have we seen this key before?
  
}

void Profiler::CallSitePathStop(double *totalTime, int tid) {

}

void finalizeCallSites(int tid) {
  // Do the same as EBS. Acquire candidates first. We need to create new FunctionInfo
  //   objects representing the callsites themselves.
  vector<FunctionInfo *> *candidates =
    new vector<FunctionInfo *>();
  RtsLayer::LockDB();
  for (vector<FunctionInfo *>::iterator fI_iter = TheFunctionDB().begin();
       fI_iter != TheFunctionDB().end(); fI_iter++) {
    FunctionInfo *theFunction = *fI_iter;
    if (theFunction->eventCallSite != 0) {
      candidates->push_back(theFunction);
    }
  }
  RtsLayer::UnLockDB();

#ifdef TAU_BFD
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_KEEP_GLOBALS);
  }
#endif /* TAU_BFD */

  vector<FunctionInfo *>::iterator cs_it;
  for (cs_it = candidates->begin(); cs_it != candidates->end(); cs_it++) {
    FunctionInfo *candidate = *cs_it;
    // Create TAU_CALLSITE FunctionInfo objects.
    char temp[1024];
    sprintf(temp, "[TAU_CALLSITE] %s", resolveTauCallSite(candidate->eventCallSite));
    string grname = string("TAU_CALLSITE | ") + 
      RtsLayer::PrimaryGroup(candidate->GetAllGroups());
    FunctionInfo *CallSiteFunction = new FunctionInfo(strdup(temp), "",
						      candidate->GetProfileGroup(),
						      (const char*) grname.c_str(), true);
    // CallSiteFunction data is exactly the same as the recorded data. So, just add
    //   inclusive time.
    CallSiteFunction->AddInclTime(candidate->GetInclTime(tid), tid);
    // Has as many calls as the measured callsite.
    CallSiteFunction->SetCalls(tid, candidate->GetCalls(tid));
    // Has exactly one subroutine, the measured callsite.
    CallSiteFunction->IncrNumSubrs(tid);

    // The => was added previously as a stub for error checking purposes.
    sprintf(temp, "%s %s", CallSiteFunction->GetName(), candidate->GetName());
    string nameString = string(temp);
    candidate->SetName(nameString);
  }
}
