#ifdef TAU_USE_LIBUNWIND

#ifdef __NEC_IA64_ABI
#define __x86_64__
#endif

#include "Profile/TauSampling_unwind.h"
#include <ucontext.h>

// Moved from header file
using namespace std;


#define UNW_LOCAL_ONLY
#include <libunwind.h>

#define TAU_SAMP_NUM_PARENTS 0

void show_backtrace_unwind(void *pc) {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_word_t ip = 0;
  unw_word_t sp = 0;

  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    // unw_get_reg(&cursor, UNW_REG_SP, &sp);
	/*
    if (ip == (unw_word_t)pc) {
      printf("ip = %lx, sp = %lx\n", (long)ip, (long)sp);
    }
	*/
    printf("ip = %lx, sp = %lx\n", (long)ip, (long)sp);
  }
}

void printStack(unsigned long *pcStack) {
  if (pcStack == NULL) {
    return;
  }
  int length = pcStack[0];
  printf("PC Stack: ");
  for (int i=0; i<length; i++) {
    printf("%lx ", pcStack[i+1]);
  }
  printf("end\n");
}

// *CWL* - *TODO*. The unwind used for trace ought to be made
//         common to the unwind for profiles. The way
//         to do this is to merge into a common function that
//         simply processes a PC stack generated from an
//         unwind directly.
//
//         Something to try to work in after Supercomputing 11
void Tau_sampling_outputTraceCallstack(int tid, void *pc, 
				       void *context) {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_word_t ip; //, sp;
  int found = 0;

  fprintf(Tau_sampling_get_ebsTrace(), " |");

  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    // unw_get_reg(&cursor, UNW_REG_SP, &sp);
    if (found) {
#ifdef __APPLE__
      fprintf(Tau_sampling_get_ebsTrace(), " %llx", ip);
#else
      fprintf(Tau_sampling_get_ebsTrace(), " %lx", ip);
#endif
    }
    if (ip == (unw_word_t)pc) {
      found = 1;
    }
  }
}

bool Tau_unwind_unwindTauContext(int tid, unsigned long *addresses) {
#if (defined(__APPLE__) || defined(__arm__) || defined(__aarch64__))
  unw_context_t context;
  int ret = unw_getcontext(&context);
#else
  ucontext_t context;
  int ret = getcontext(&context);
#endif
  
  if (ret != 0) {
    fprintf(stderr, "TAU: Error getting context\n");
    return false;
  }

  unw_cursor_t cursor;
  unw_word_t ip;
  unw_init_local(&cursor, &context);

  int count = 0;
  int idx = 1;  // we want to fill the first entry with the length later.
  unw_word_t last_address = 0;
  while (unw_step(&cursor) > 0 && idx < TAU_SAMP_NUM_ADDRESSES) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    // In the case of Unwinding from TAU context, we ignore recursion as
    //   un-helpful for the purposes of determining callsite information.
    if (ip == last_address) {
      continue;
    }
    addresses[idx++] = (unsigned long)ip;
    last_address = ip;
    count++;
  }
  if (count > 0) {
    addresses[0] = count;
    return true;
  } else {
    return false;
  }
}

void Tau_sampling_unwindTauContext(int tid, void **addresses) {
#if (defined(__APPLE__) || defined(__arm__) || defined(__aarch64__))
  unw_context_t context;
  int ret = unw_getcontext(&context);
#else
  ucontext_t context;
  int ret = getcontext(&context);
#endif
  
  if (ret != 0) {
    fprintf(stderr, "TAU: Error getting context\n");
    return;
  }

  unw_cursor_t cursor;
  unw_word_t ip;
  unw_init_local(&cursor, &context);

  int idx = 0;
  while (unw_step(&cursor) > 0 && idx < TAU_SAMP_NUM_ADDRESSES) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    addresses[idx++] = (void *)ip;
  }
}

void Tau_sampling_unwind(int tid, Profiler *profiler,
			 void *pc, void *context, unsigned long pcStack[]) {
  // stack points to valid array of max length TAU_SAMP_NUM_ADDRESSES + 1.
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_word_t unwind_ip; //, sp;

  int unwindDepth = 1; // We need to include the PC in unwind depth calculations
  int depthCutoff = TauEnv_get_ebs_unwind_depth();

  int index = 1;
  pcStack[index++] = (unsigned long)pc;

  // Commence the unwind

  // We should use the original sample context rather than the current
  // TAU EBS context for the unwind.
  //  unw_getcontext(&uc);
  uc = *(unw_context_t *)context;
  unw_init_local(&cursor, &uc);
  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &unwind_ip);
    if ((depthCutoff > 0 && unwindDepth >= depthCutoff) ||
	(unwind_cutoff(profiler->address, (void *)unwind_ip))) {
      pcStack[index++] = (unsigned long)unwind_ip;
      unwindDepth++;  // for accounting only
      break; // always break when limit or cutoff is reached.
    } // Cut-off or limit check conditional
    pcStack[index++] = (unsigned long)unwind_ip;
    unwindDepth++;
  }
  pcStack[0] = index-1;
}

#endif /* TAU_USE_LIBUNWIND */

