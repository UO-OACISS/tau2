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

#define TAU_SAMP_NUM_PARENTS 0

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

  fprintf(Tau_sampling_get_ebsTrace(), " |");

  for (unsigned i = 0; i < stackwalk.size(); i++) {
    void *ip = (void *)stackwalk[i].getRA();

    if (found) {
      fprintf(Tau_sampling_get_ebsTrace(), " %p", ip);
    }
    if (ip == pc) {
      found = 1;
    }
  }

  // StackWalkerAPI is not thread-safe
  RtsLayer::UnLockDB();
}


// *CWL* Partially copied from TauSampling.cpp. Clean up later.
#if __WORDSIZE == 32
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.uc_regs->gregs[reg])
#else
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.gp_regs[reg])
#endif

#define PPC_REG_PC 32

unsigned long get_pc(void *p);

// Prototype support for acquiring stack pointer from the context.
//   Only support for x86_64 for now.
static inline unsigned long get_sp(void *p) {
  unsigned long sp;
  struct ucontext *uc = (struct ucontext *)p;

#ifdef TAU_BGP
  // *CWL* returns void ** but used directly as the SP.
  sp = (unsigned long)UCONTEXT_REG(uc, PPC_REG_SP);
#elif __x86_64__
  struct sigcontext *sc;
  sc = (struct sigcontext *)&uc->uc_mcontext;
  sp = (unsigned long)sc->rsp;
#endif /* TAU_BGP */

  return sp;
}

// Prototype support for acquiring frame pointer from the context.
//   Only support for x86_64 for now.
static inline unsigned long get_fp(void *p) {
  unsigned long fp;
  struct ucontext *uc = (struct ucontext *)p;

#ifdef TAU_BGP
  // *CWL* returns void ** but used directly as the FP.
  fp = (unsigned long)UCONTEXT_REG(uc, PPC_REG_FP);
#elif __x86_64__
  struct sigcontext *sc;
  sc = (struct sigcontext *)&uc->uc_mcontext;
  fp = (unsigned long)sc->rbp;
#endif /* TAU_BGP */

  return fp;
}

bool Tau_unwind_unwindTauContext(int tid, unsigned long *addresses) {
  ucontext_t context;
  int ret = getcontext(&context);

  if (ret != 0) {
    fprintf(stderr, "TAU: Error getting context\n");
    return false;
  }

  std::vector<Frame> stackwalk;
  string s;
  Dyninst::MachRegisterVal unwind_ip;

  unsigned long pc = get_pc(&context);
  unsigned long sp = get_sp(&context);
  unsigned long fp = get_fp(&context);
  Frame *startFrame = Frame::newFrame((Dyninst::MachRegisterVal)pc,
				      (Dyninst::MachRegisterVal)sp,
				      (Dyninst::MachRegisterVal)fp,
				      walker);

  bool success = false;
  int count = 0;
  int idx = 1;
  Dyninst::MachRegisterVal last_address = 0;
  // StackWalkerAPI is not thread-safe
  RtsLayer::LockDB();
  //  success = walker->walkStack(stackwalk);
  success = walker->walkStackFromFrame(stackwalk, *startFrame);
  if (success) {
    //    printf("Stackwalk size = %d\n", stackwalk.size());
    for (unsigned i = 0; i < stackwalk.size(); i++) {
      if (idx < TAU_SAMP_NUM_ADDRESSES) {
	//	stackwalk[i].getName(s);
	//	printf("[%d] %s\n", i, s.c_str());
	unwind_ip = stackwalk[i].getRA();
	if (unwind_ip == last_address) {
	  continue;
	}
	addresses[idx++] = (unsigned long)unwind_ip;
	last_address = unwind_ip;
	count++;
	//	printf("[%d] context unwind [%p]\n", i, (void *)unwind_ip);
      } else {
	break;
      }
    }
    if (count > 0) {
      addresses[0] = count;
      return true;
    } else {
      return false;
    }
  } else {
    return false;
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
  Dyninst::MachRegisterVal unwind_ip;

  unsigned long pc = get_pc(&context);
  unsigned long sp = get_sp(&context);
  unsigned long fp = get_fp(&context);
  Frame *startFrame = Frame::newFrame((Dyninst::MachRegisterVal)pc,
				      (Dyninst::MachRegisterVal)sp,
				      (Dyninst::MachRegisterVal)fp,
				      walker);

  bool success = false;
  int idx = 0;
  // StackWalkerAPI is not thread-safe
  RtsLayer::LockDB();
  //  success = walker->walkStack(stackwalk);
  success = walker->walkStackFromFrame(stackwalk, *startFrame);
  if (success) {
    //    printf("Stackwalk size = %d\n", stackwalk.size());
    for (unsigned i = 0; i < stackwalk.size(); i++) {
      if (idx < TAU_SAMP_NUM_ADDRESSES) {
	//	stackwalk[i].getName(s);
	//	printf("[%d] %s\n", i, s.c_str());
	unwind_ip = stackwalk[i].getRA();
	addresses[idx++] = (void *)unwind_ip;
	//	printf("[%d] context unwind [%p]\n", i, (void *)unwind_ip);
      } else {
	break;
      }
    }
  } else {
    //    printf("no success\n");
  }
  // StackWalkerAPI is not thread-safe
  RtsLayer::UnLockDB();
}

void Tau_sampling_unwind(int tid, Profiler *profiler,
			 void *pc, void *context, unsigned long pcStack[]) {
  std::vector<Frame> stackwalk;
  string s;
  Dyninst::MachRegisterVal unwind_ip;

  int unwindDepth = 0;
  int depthCutoff = TauEnv_get_ebs_unwind_depth();

  // *CWL* - The big difference between stackwalkerAPI and libunwind
  //         is that the RA from the first stack frame is the actual
  //         PC itself.

  // Add the actual PC sample into the stack
  //  printf("%p ", pc);
  //  pcStack->push_back((unsigned long)pc);

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
  int index = 1;
  // Sadly, not single-stepping here. Just read the whole stack.
  bool success = false;
  //  success = walker->walkStack(stackwalk);
  success = walker->walkStackFromFrame(stackwalk, *startFrame);
  if (success) {
    // push the PC (the top of the stackwalker stack is always it)
    unwind_ip = stackwalk[0].getRA();
    pcStack[index++] = (unsigned long)unwind_ip;
    for (unsigned i = 1; i < stackwalk.size(); i++) {
      unwind_ip = stackwalk[i].getRA();
      if ((unwindDepth >= depthCutoff) ||
	  (unwind_cutoff(profiler->address, (void *)unwind_ip))) {
	pcStack[index++] = (unsigned long)unwind_ip;
	unwindDepth++;
	break; // always break when limit is hit or cutoff reached
      } // cut-off or limit check conditional
      pcStack[index++] = (unsigned long)unwind_ip;
      unwindDepth++;
    }
  }
  // works in all cases because index is initialized to 1
  pcStack[0] = index-1;

  // StackWalkerAPI is not thread-safe
  RtsLayer::UnLockDB();
}

#endif /* TAU_USE_STACKWALKER */
