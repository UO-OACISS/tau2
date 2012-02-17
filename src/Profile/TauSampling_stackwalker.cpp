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

// Copied from TauSampling.cpp for now. Find a way to share the code later.
// *CWL* - Originally from HPCToolkit for the Power architecture. Kinda bizzarre -
//         for the stack pointer and frame pointers, void ** is returned instead
//         of void *. The double pointer is used directly, however.
#if __WORDSIZE == 32
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.uc_regs->gregs[reg])
#else
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.gp_regs[reg])
#endif

#define PPC_REG_FP   PPC_REG_R1
#define PPC_REG_PC 32
#define PPC_REG_R1   1
#define PPC_REG_SP   PPC_REG_R1

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

  vector<unsigned long> *pcStack = new vector<unsigned long>();

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

extern "C" FunctionInfo *findTopContext(Profiler *currentProfiler, void *address);
vector<unsigned long> *Tau_sampling_unwind(int tid, Profiler *profiler,
					   void *pc, void *context) {
  std::vector<Frame> stackwalk;
  string s;
  Dyninst::MachRegisterVal unwind_ip;

  vector<unsigned long> *pcStack = new vector<unsigned long>();
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

  // Sadly, not single-stepping here. Just read the whole stack.
  bool success = false;
  //  success = walker->walkStack(stackwalk);
  success = walker->walkStackFromFrame(stackwalk, *startFrame);
  if (success) {
    // push the PC
    unwind_ip = stackwalk[0].getRA();
    pcStack->push_back((unsigned long)unwind_ip);
    for (unsigned i = 1; i < stackwalk.size(); i++) {
      // not necessary with BFD
      //    addr = stackwalk[i].getRA();
      unwind_ip = stackwalk[i].getRA();

      if ((unwindDepth >= depthCutoff) ||
	  (unwind_cutoff(profiler->address, (void *)unwind_ip))) {
	if (unwind_cutoff(profiler->address, (void *)unwind_ip)) {
	  FunctionInfo *topFI;
	  pcStack->push_back((unsigned long)unwind_ip);	  
	  // Now that we have a match, we can look through the Profiler Stack
	  //    to locate the top profile.
	  topFI = findTopContext(profiler->ParentProfiler, (void *)unwind_ip);
	  // No parent shares the current match. The current profiler must
	  //    be the top entry.
	  if (topFI == NULL) {
	    if (profiler->CallPathFunction == NULL) {
	      topFI = profiler->ThisFunction;
	    } else {
	      topFI = profiler->CallPathFunction;
	    }
	  }
	  unwindDepth++;  // for accounting only
	  // add 3 more unwinds (arbitrary) // not so easy here. 
	  // And probably unnecessary.
	  // Disabling for now.
	  /*
	  for (int i=0; i<TAU_SAMP_NUM_PARENTS; i++) {
	    if (unw_step(&cursor) > 0) {
	      unw_get_reg(&cursor, UNW_REG_IP, &unwind_ip);
	      if (unwind_ip != curr_ip) {
		pcStack->push_back((unsigned long)unwind_ip);
	      }
	    } else {
	      break; // no more stack                                                                 
	    }
	  }
	  */
	} else {
	  pcStack->push_back((unsigned long)unwind_ip);	  
	}
	break; // always break when limit is hit or cutoff reached
      } // cut-off or limit check conditional

      pcStack->push_back((unsigned long)unwind_ip);
      unwindDepth++;
    }
  }

  // StackWalkerAPI is not thread-safe
  RtsLayer::UnLockDB();
  return pcStack;
}

#endif /* TAU_USE_STACKWALKER */
