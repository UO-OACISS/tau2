#ifdef TAU_USE_LIBUNWIND

#include "Profile/TauSampling_unwind.h"
#include <ucontext.h>

#define UNW_LOCAL_ONLY
#include <libunwind.h>

#define TAU_SAMP_NUM_PARENTS 0

void show_backtrace_unwind(void *pc) {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_word_t ip, sp;
  int found = 0;

  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    // unw_get_reg(&cursor, UNW_REG_SP, &sp);
    if (ip == (unw_word_t)pc) {
      found = 1;
    }
    //    if (found) {
    printf("ip = %lx, sp = %lx\n", (long)ip, (long)sp);
    //    }
  }
}

void printStack(vector<unsigned long> *pcStack) {
  printf("PC Stack: ");
  vector<unsigned long>::iterator it;
  for (it = pcStack->begin(); it != pcStack->end(); it++) {
    printf("%p ", (void *)(*it));
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
  unw_word_t ip, sp;
  int found = 0;

  fprintf(ebsTrace[tid], " |");

  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    // unw_get_reg(&cursor, UNW_REG_SP, &sp);
    if (found) {
      fprintf(ebsTrace[tid], " %p", ip);
    }
    if (ip == (unw_word_t)pc) {
      found = 1;
    }
  }
}

void Tau_sampling_unwindTauContext(int tid, void **addresses) {
  ucontext_t context;
  int ret = getcontext(&context);
  
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

extern "C" FunctionInfo *findTopContext(Profiler *currentProfiler, void *address);
vector<unsigned long> *Tau_sampling_unwind(int tid, Profiler *profiler,
					   void *pc, void *context) {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_word_t unwind_ip, sp;
  unw_word_t curr_ip;

  vector<unsigned long> *pcStack = new vector<unsigned long>();
  int unwindDepth = 0;
  int depthCutoff = TauEnv_get_ebs_unwind_depth();

  pcStack->push_back((unsigned long)pc);

  // Commence the unwind

  // We should use the original sample context rather than the current
  // TAU EBS context for the unwind.
  //  unw_getcontext(&uc);
  uc = *(unw_context_t *)context;
  unw_init_local(&cursor, &uc);
  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &unwind_ip);
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
	// add 3 more unwinds (arbitrary)
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
      } else {
	pcStack->push_back((unsigned long)unwind_ip);
      } // cut-off check
      break; // always break when limit is hit or cutoff reached.
    } // Cut-off or limit check conditional
    pcStack->push_back((unsigned long)unwind_ip);
    unwindDepth++;
  }
  return pcStack;
}

#endif /* TAU_USE_LIBUNWIND */

