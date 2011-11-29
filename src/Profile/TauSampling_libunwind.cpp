#ifdef TAU_USE_LIBUNWIND

#include "Profile/TauSampling_unwind.h"
#include <ucontext.h>

#define UNW_LOCAL_ONLY
#include <libunwind.h>

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
  unw_word_t ip, sp;
  unw_word_t top_ip;
  unw_proc_info_t pip;
  unw_init_local(&cursor, &context);

  int idx = 0;
  int skip = 0; // skip the current context itself

  //  printf("Context: ");
  while (unw_step(&cursor) > 0 && idx < TAU_SAMP_NUM_ADDRESSES) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    if (skip > 0) {
      // fprintf (stderr,"skipping address %p\n", ip);
      skip--;
    } else {
      // always store the top ip (unless it is 0).
      unw_get_proc_info(&cursor, &pip);
      top_ip = pip.start_ip;
      //      printf("%p|%p ", ip, top_ip);
      if (top_ip == 0) {
	addresses[idx++] = (void *)ip;
      } else {
	addresses[idx++] = (void *)top_ip;
      }
      // fprintf (stderr,"assigning address %p to index %d\n", ip, idx-1);
    }
  }
  //  printf("\n");
}

bool unwind_cutoff(void **addresses, void *address) {
  bool found = false;
  for (int i=0; i<TAU_SAMP_NUM_ADDRESSES; i++) {
    if ((unsigned long)(addresses[i]) == (unsigned long)address) {
      //      printf("match found %p\n", address);
      found = true;
      break;
    }
  }
  return found;
}

vector<unsigned long> *Tau_sampling_unwind(int tid, Profiler *profiler,
					   void *pc, void *context) {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_word_t ip, sp;
  unw_word_t top_ip;
  unw_proc_info_t pip;

  vector<unsigned long> *pcStack = new vector<unsigned long>();
  int unwindDepth = 0;
  int depthCutoff = TauEnv_get_ebs_unwind_depth();

  // printf("cutoff depth = %d\n", depthCutoff);

  // Add the actual PC sample into the stack
  //  printf("%p ", pc);
  pcStack->push_back((unsigned long)pc);

  // Commence the unwind

  // We should use the original sample context rather than the current
  // TAU EBS context for the unwind.
  //  unw_getcontext(&uc);
  uc = *(unw_context_t *)context;
  unw_init_local(&cursor, &uc);
  // Is my sample in the immediate context?
  unw_get_proc_info(&cursor, &pip);
  top_ip = pip.start_ip;
  if (unwind_cutoff(profiler->address, (void *)top_ip)) {
    //    printf("[dropped %p|%p]", pc, top_ip);
    // Do nothing, there is no unwinding since the PC occurs
    //   in the context of the TAU context itself.
  } else {
    while (unw_step(&cursor) > 0) {
      unw_get_reg(&cursor, UNW_REG_IP, &ip);
      unw_get_proc_info(&cursor, &pip);
      top_ip = pip.start_ip;
      // unless it is 0, always compare against the top_ip 
      unw_word_t compare_ip;
      if (top_ip == 0) {
	compare_ip = ip;
      } else {
	compare_ip = top_ip;
      }
      if ((unwindDepth >= depthCutoff) ||
	  (unwind_cutoff(profiler->address, (void *)compare_ip))) {
	if (ip != top_ip) {
	  // We want to preserve the final callsite before a 
	  //   match with the top of the Tau context address.
	  pcStack->push_back((unsigned long)ip);
	  unwindDepth++;  // for accounting only
	}
	//	printf("[dropped %p|%p]", ip, top_ip);
	break;
      }
      //      printf("%p|%p ", ip, top_ip);
      pcStack->push_back((unsigned long)ip);
      unwindDepth++;
    }
  }
  //  printf("\n");

  //  printf("Unwound %d times\n", unwindDepth);
  //  printStack(pcStack);
  return pcStack;
}

#endif /* TAU_USE_LIBUNWIND */

