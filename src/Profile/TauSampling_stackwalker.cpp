#ifdef TAU_USE_STACKWALKER

#include "Profile/TauSampling_unwind.h"
#include <ucontext.h>

#include <walker.h>
#include <frame.h>
#include <framestepper.h>
#include <symlookup.h>
#include <steppergroup.h>
using namespace Dyninst::Stackwalker;
// using namespace Stackwalker;

#include <iostream>
#include <set>
using namespace std;

/*
extern "C" void *dlmalloc(size_t size);

extern "C" void *dlcalloc(size_t nmemb, size_t size);

extern "C" void dlfree(void *ptr);

extern "C" void *dlrealloc(void *ptr, size_t size);

extern "C" void *__libc_malloc(size_t size);

extern "C" void *__libc_calloc(size_t nmemb, size_t size);

extern "C" void __libc_free(void *ptr);

extern "C" void *__libc_realloc(void *ptr, size_t size);
*/

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
  ucontext_t context;
  int ret = getcontext(&context);

  if (ret != 0) {
    fprintf(stderr, "TAU: Error getting context\n");
    return;
  }

  std::vector<Frame> stackwalk;
  string s;
  Dyninst::MachRegisterVal compare_ip;

  vector<unsigned long> *pcStack = new vector<unsigned long>();
  int unwindDepth = 0;
  int depthCutoff = TauEnv_get_ebs_unwind_depth();

  int idx = 0;
  int skip = 0;

  // StackWalkerAPI is not thread-safe
  RtsLayer::LockDB();


  // StackWalkerAPI is not thread-safe
  RtsLayer::UnLockDB();
}

/* TODO - This ought to be common code to all unwinding */
/*
bool unwind_cutoff(void **addresses, void *address) {
  bool found = false;
  for (int i=0; i<TAU_SAMP_NUM_ADDRESSES; i++) {
    if ((unsigned long)(addresses[i]) == (unsigned long)address) {
      // printf("match found %p\n", address); 
      found = true;
      break;
    }
  }
  return found;
}
*/

// Prototype support for acquiring stack pointer from the context.
//   Only support for x86_64 for now.
static inline unsigned long get_sp(void *p) {
  struct ucontext *uc = (struct ucontext *)p;
  unsigned long sp;
  
  struct sigcontext *sc;
  sc = (struct sigcontext *)&uc->uc_mcontext;
  sp = (unsigned long)sc->rsp;

  return sp;
}

// Prototype support for acquiring frame pointer from the context.
//   Only support for x86_64 for now.
static inline unsigned long get_fp(void *p) {
  struct ucontext *uc = (struct ucontext *)p;
  unsigned long fp;
  
  struct sigcontext *sc;
  sc = (struct sigcontext *)&uc->uc_mcontext;
  fp = (unsigned long)sc->rbp;

  return fp;
}

vector<unsigned long> *Tau_sampling_unwind(int tid, Profiler *profiler,
					   void *pc, void *context) {
  std::vector<Frame> stackwalk;
  string s;
  Dyninst::MachRegisterVal compare_ip;

  vector<unsigned long> *pcStack = new vector<unsigned long>();
  int unwindDepth = 0;
  int depthCutoff = TauEnv_get_ebs_unwind_depth();

  // Add the actual PC sample into the stack
  //  printf("%p ", pc);
  pcStack->push_back((unsigned long)pc);
  //printf("StackwalkerAPI Sample PC: [%p]\n", (unsigned long)pc);

  // Commence the unwind
  // StackWalkerAPI has no interface for direct use of contexts.
  unsigned long sp = get_sp(context);
  unsigned long fp = get_fp(context);
  Frame *startFrame = Frame::newFrame((Dyninst::MachRegisterVal)pc,
				      (Dyninst::MachRegisterVal)sp,
				      (Dyninst::MachRegisterVal)fp,
				      walker);

  // StackWalkerAPI is not thread-safe
  RtsLayer::LockDB();

  // Sadly, not single-stepping here. Just read the whole stack.
  bool success = false;
  //  success = walker->walkStack(stackwalk);
  success = walker->walkStackFromFrame(stackwalk, *startFrame);
  if (success) {
    for (unsigned i = 0; i < stackwalk.size(); i++) {
      // not necessary with BFD
      stackwalk[i].getName(s);
      //    addr = stackwalk[i].getRA();
      compare_ip = stackwalk[i].getRA();
      //addr = stackwalk[i].getFP();
      //          printf("StackwalkerAPI Unwind Address: [%p] Name: [%s]\n", compare_ip, s.c_str());

      if ((unwindDepth >= depthCutoff) ||
	  (unwind_cutoff(profiler->address, (void *)compare_ip))) {
	break;
      }

      pcStack->push_back((unsigned long)compare_ip);
      unwindDepth++;
    }
  }

  // StackWalkerAPI is not thread-safe
  RtsLayer::UnLockDB();
  return pcStack;
}

#endif /* TAU_USE_STACKWALKER */
