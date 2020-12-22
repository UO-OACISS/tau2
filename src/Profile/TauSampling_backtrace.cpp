#ifdef TAU_USE_BACKTRACE

#include "Profile/TauSampling_unwind.h"
#include <unwind.h>

// Moved from header file
using namespace std;

/* Passed to backtrace_simple callback function.  */
/* Data passed through _Unwind_Backtrace.  */

typedef struct sdata
{
  uintptr_t *addrs;
  uintptr_t pc;
  void** profiler;
  int index;
  int max;
  bool foundpc;
  int skip;
  int ret;
} sdata_t;

/* Unwind library callback routine.  This is passed to
   _Unwind_Backtrace.  */

static _Unwind_Reason_Code
default_unwind (struct _Unwind_Context *context, void *vdata)
{
    sdata_t *data = (sdata_t*) vdata;
    uintptr_t pc;
    int ip_before_insn = 0;

#ifdef _Unwind_GetIPInfo
    pc = _Unwind_GetIPInfo (context, &ip_before_insn);
    if (!ip_before_insn) --pc;
#else
    pc = _Unwind_GetIP (context);
#endif

    if (data->skip > 0) {
        --data->skip;
        return _URC_NO_REASON;
    }

    if (data->index >= data->max) {
        data->ret = 1;
        return _URC_END_OF_STACK;
    }

    data->addrs[data->index] = pc;
    ++data->index;

    return _URC_NO_REASON;
}

static _Unwind_Reason_Code
sample_unwind (struct _Unwind_Context *context, void *vdata)
{
    sdata_t *data = (sdata_t*) vdata;
    uintptr_t pc;
    int ip_before_insn = 0;

#ifdef _Unwind_GetIPInfo
    pc = _Unwind_GetIPInfo (context, &ip_before_insn);
    if (!ip_before_insn) --pc;
#else
    pc = _Unwind_GetIP (context);
#endif

    if (data->skip > 0) {
        --data->skip;
        return _URC_NO_REASON;
    }

    if (data->index >= data->max) {
        data->ret = 1;
        return _URC_END_OF_STACK;
    }

    if (pc == data->pc) {
        data->foundpc = true;
    }

    if (unwind_cutoff(data->profiler, (void *)pc)) {
        data->ret = 1;
        return _URC_END_OF_STACK;
    }

    if (data->foundpc) {
        data->addrs[data->index] = pc;
        ++data->index;
    }

    return _URC_NO_REASON;
}

/* Used when generating a sample trace.  Not really supported any more. */
void Tau_sampling_outputTraceCallstack(int tid, void *pc, void *context) {
  uintptr_t addresses[100] = {0};
  sdata_t data;
  data.addrs = &addresses[0];
  data.index = 0;
  data.max = 100;
  data.skip = 0;

  _Unwind_Backtrace (default_unwind, &data);
  if (data.ret != 0) return;

  fprintf(Tau_sampling_get_ebsTrace(), " |");

  bool found = false;
  for (size_t index = 0 ; index < data.index ; index++) {
    if (found) {
#ifdef __APPLE__
      fprintf(Tau_sampling_get_ebsTrace(), " %llx", addresses[index]);
#else
      fprintf(Tau_sampling_get_ebsTrace(), " %lx", addresses[index]);
#endif
    }
    if (addresses[index] == (uintptr_t)pc) {
      found = true;
    }
  }
}

bool Tau_unwind_unwindTauContext(int tid, unsigned long *addresses) {
  sdata_t data;
  data.addrs = &addresses[1];
  data.index = 0;
  data.max = TAU_SAMP_NUM_ADDRESSES;
  data.skip = 2;

  _Unwind_Backtrace (default_unwind, &data);
  if (data.index > 0) {
    addresses[0] = data.index;
    return true;
  } else {
    return false;
  }
}

void Tau_sampling_unwindTauContext(int tid, void **addresses) {
  sdata_t data;
  data.addrs = (uintptr_t*)addresses;
  data.index = 0;
  data.max = TAU_SAMP_NUM_ADDRESSES;
  data.skip = 2;

  _Unwind_Backtrace (default_unwind, &data);
}

void Tau_sampling_unwind(int tid, Profiler *profiler,
			 void *pc, void *context, unsigned long pcStack[]) {
  // stack points to valid array of max length TAU_SAMP_NUM_ADDRESSES + 1.
  sdata_t data;
  data.addrs = &pcStack[1];
  data.index = 0;
  data.max = TauEnv_get_ebs_unwind_depth();
  if (data.max == 0) data.max = INT_MAX; //allow for "unlimited"
  data.pc = (uintptr_t)pc;
  data.profiler = profiler->address;
  data.foundpc = false;
  data.skip = 1;
  data.ret = 0;

  // Commence the unwind
  _Unwind_Backtrace (sample_unwind, &data);
  if (data.ret != 0) {
    pcStack[0] = 1;
  } else {
    pcStack[0] = data.index;
  }
}

#endif /* TAU_USE_LIBUNWIND */

