#ifdef TAU_USE_HPCTOOLKIT

#include <ucontext.h>

#include <TAU.h>
#include "Profile/TauSampling.h"

extern "C" {
  #include <unwind.h>
}
#include <setjmp.h>

extern "C" sigjmp_buf *hpctoolkit_get_thread_jb();

void show_backtrace_unwind(void *pc) {
  ucontext_t *context = (ucontext_t *)pc;
  unw_cursor_t cursor;
  unw_word_t ip, sp;
  int found = 0;

  unw_init_cursor(&cursor, context);

  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    fprintf(stderr, "ip = %p ", ip);
  }
  fprintf(stderr, "\n");
}

void debug_this_try(int tid, void *in_context) {
  ucontext_t *context = (ucontext_t *)in_context;
  unw_cursor_t cursor;
  unw_word_t ip, sp;
  int found = 1;

  fprintf(stderr, "++++++++tid = %d+++++++++++\n", tid);
  Profiler *profiler = TauInternal_CurrentProfiler(tid);
  fprintf(stderr, "Function name is: %s\n", profiler->ThisFunction->GetName());

  for (int i = 0; i < TAU_SAMP_NUM_ADDRESSES; i++) {
    fprintf(stderr, "address[%d] = %p\n", i, profiler->address[i]);
  }

  // fprintf(stderr,"==========\n");
  unw_init_cursor(&cursor, context);
  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    fprintf(stderr, "step %p\n", ip);
  }
  fprintf(stderr, "+++++++++++++++++++\n");
}

void Tau_sampling_outputTraceCallstack(int tid, void *pc, 
				       void *context_in) {
  unw_cursor_t cursor;
  unw_word_t ip, sp;
  ucontext_t *context = (ucontext_t *)context_in;
  int found = 1;

  Profiler *profiler = TauInternal_CurrentProfiler(tid);

  sigjmp_buf *jmpbuf = hpctoolkit_get_thread_jb();

  int ljmp = sigsetjmp(*jmpbuf, 1);
  if (ljmp == 0) {
    // fprintf(stderr,"==========\n");
    unw_init_cursor(&cursor, context);
    while (unw_step(&cursor) > 0) {
      unw_get_reg(&cursor, UNW_REG_IP, &ip);

      for (int i = 0; i < TAU_SAMP_NUM_ADDRESSES; i++) {
        if (ip == (unw_word_t)profiler->address[i]) {
          return;
        }
      }
      // fprintf(stderr,"step %p\n", ip);

      fprintf(ebsTrace[tid], " %p", ip);
    }
  } else {
    fprintf(stderr, "*** unhandled sample:\n");
    return;
  }

  fprintf(stderr, "*** very strange, didn't find profiler\n");

  debug_this_try(tid, context);

// , profiler's address was %p\n",
//         profiler->address);
}

/*********************************************************************
 * Handler for event entry (start)
 ********************************************************************/
void Tau_sampling_event_start(int tid, void **addresses) {
  // fprintf (stderr, "[%d] SAMP: event start: ", tid);

  ucontext_t context;
  int ret = getcontext(&context);

  if (ret != 0) {
    fprintf(stderr, "TAU: Error getting context\n");
    return;
  }

  if (hpctoolkit_process_started == 0) {
    // fprintf(stderr, "nope, quitting\n");
    return;
  }

  unw_cursor_t cursor;
  unw_word_t ip, sp;
  // fprintf (stderr,"$$$$$$$$$start$$$$$$$$$\n");
  unw_init_cursor(&cursor, &context);
  int idx = 0;

  int skip = 1;
  while (unw_step(&cursor) > 0 && idx < TAU_SAMP_NUM_ADDRESSES) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);

    if (skip > 0) {
      // fprintf (stderr,"skipping address %p\n", ip);
      skip--;
    } else {
      addresses[idx++] = ip;
      // fprintf (stderr,"assigning address %p to index %d\n", ip, idx-1);
    }
  }

  // fprintf (stderr, "\n");
  // fprintf (stderr,"$$$$$$$$$$$$$$$$$$\n");
}

#endif /* TAU_USE_HPCTOOLKIT */
