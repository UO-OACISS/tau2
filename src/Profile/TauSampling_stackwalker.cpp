#ifdef TAU_USE_STACKWALKER

#include "Profile/TauSampling_unwind.h"
#include <TAU.h>
#include <ucontext.h>

#include <walker.h>
#include <frame.h>
#include <steppergroup.h>
using namespace Dyninst;
using namespace Stackwalker;
#include <iostream>
#include <set>
using namespace std;

extern "C" void *dlmalloc(size_t size);

extern "C" void *dlcalloc(size_t nmemb, size_t size);

extern "C" void dlfree(void *ptr);

extern "C" void *dlrealloc(void *ptr, size_t size);

extern "C" void *__libc_malloc(size_t size);

extern "C" void *__libc_calloc(size_t nmemb, size_t size);

extern "C" void __libc_free(void *ptr);

extern "C" void *__libc_realloc(void *ptr, size_t size);

/* *CWL* Seems like massive hackery going on here. Gonna disable until I can figure if this
   is some solution to the malloc-re-entrancy issues I've faced.

void *malloc(size_t size) {
  int tid = RtsLayer::myThread();
  // return __libc_malloc(size);
  if (insideSignalHandler[tid]) {
    return dlmalloc(size);
  } else {
    return __libc_malloc(size);
  }
}

void *calloc(size_t nmemb, size_t size) {
  int tid = RtsLayer::myThread();
//   printf ("Our calloc called!\n");
// return __libc_malloc(size);
  if (insideSignalHandler[tid]) {
    return dlcalloc(nmemb, size);
  } else {
    return __libc_calloc(nmemb, size);
  }
}

void free(void *ptr) {
  int tid = RtsLayer::myThread();
  // return __libc_malloc(size);
  if (insideSignalHandler[tid]) {
    dlfree(ptr);
  } else {
    __libc_free(ptr);
  }
}

void *realloc(void *ptr, size_t size) {
  int tid = RtsLayer::myThread();
  // return __libc_malloc(size);
  if (insideSignalHandler[tid]) {
    return dlrealloc(ptr, size);
  } else {
    return __libc_realloc(ptr, size);
  }
}
*/


Walker *walker = Walker::newWalker();
// Frame crapFrame(walker);
// std::vector<Frame> stackwalk(2000, crapFrame);

void show_backtrace_stackwalker(void *pc) {
  std::vector<Frame> stackwalk;

  RtsLayer::LockDB();
  printf("====\n");
  string s;
  walker->walkStack(stackwalk);

  for (unsigned i = 0; i < stackwalk.size(); i++) {
    stackwalk[i].getName(s);
    cout << "Found function " << s << endl;
  }
  RtsLayer::UnLockDB();
  exit(0);
}

void Tau_sampling_outputTraceCallstack(int tid, void *pc,
				       void *context) {
  int found = 0;
  std::vector<Frame> stackwalk;
  string s;

  // StackWalkerAPI is not thread-safe
  RtsLayer::LockDB();

  walker->walkStack(stackwalk);

  fprintf(ebsTrace[tid], " |");

  for (unsigned i = 0; i < stackwalk.size(); i++) {
    void *ip = (void *)stackwalk[i].getRA();

    if (found) {
      fprintf(ebsTrace[tid], " %p", ip);
    }
    if (ip == pc) {
      found = 1;
    }
  }

  // StackWalkerAPI is not thread-safe
  RtsLayer::UnLockDB();
}

void Tau_sampling_unwindTauContext(int tid, void **addresses) {
  
}

#endif /* TAU_USE_STACKWALKER */
