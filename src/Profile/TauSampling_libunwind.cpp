#ifdef TAU_USE_LIBUNWIND

#include "Profile/TauSampling.h"
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

void Tau_sampling_outputTraceCallstack(int tid, void *pc, 
				       ucontext_t *context) {
  /* context is not used in libunwind */
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

#endif /* TAU_USE_LIBUNWIND */

