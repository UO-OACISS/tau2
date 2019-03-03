/*
 * Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * */


#include <iostream>
#include <hip/hip_runtime.h>
#include <roctracer.h>
using namespace std;
//#include <roctracer_hcc.h>


//#include <inc/roctracer_hcc.h>


// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                                                       \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      std::cerr << roctracer_error_string() << std::endl << std::flush;                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

// Runtime API callback function
void api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  fprintf(stdout, "<%s id(%u)\tcorrelation_id(%lu) %s> ",
    roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0),
    cid,
    data->correlation_id,
    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        fprintf(stdout, "dst(%p) src(%p) size(0x%x) kind(%u)",
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        fprintf(stdout, "ptr(%p) size(0x%x)",
          data->args.hipMalloc.ptr,
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        fprintf(stdout, "ptr(%p)",
          data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        fprintf(stdout, "kernel(\"%s\") stream(%p)",
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        break;
    }
  } else {
    switch (cid) {
      case HIP_API_ID_hipMalloc:
        fprintf(stdout, "*ptr(0x%p)",
          *(data->args.hipMalloc.ptr));
        break;
      default:
        break;
    }
  }
  fprintf(stdout, "\n"); fflush(stdout);
}

// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
  const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);
  fprintf(stdout, "\tActivity records:\n"); fflush(stdout);
  while (record < end_record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    fprintf(stdout, "\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)",
      name,
      record->correlation_id,
      record->begin_ns,
      record->end_ns
    );
    if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
      fprintf(stdout, " process_id(%u) thread_id(%u)",
        record->process_id,
        record->thread_id
      );
    } else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
      fprintf(stdout, " device_id(%d) queue_id(%lu)",
        record->device_id,
        record->queue_id
      );
    } else {
      fprintf(stderr, "Bad domain %d\n", record->domain);
      abort();
    }
    if (record->op == hc::HSA_OP_ID_COPY) fprintf(stdout, " bytes(0x%zx)", record->bytes);
    fprintf(stdout, "\n");
    fflush(stdout);
    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
}

// Init tracing routine
int TauRocTracer_init_tracing() {
  std::cout << "# START #############################" << std::endl << std::flush;
  // Allocating tracing pool
  roctracer_properties_t properties{};
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activity_callback;
  ROCTRACER_CALL(roctracer_open_pool(&properties));
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Start tracing routine
extern "C" void TauRocTracer_start_tracing() {
  static int flag = TauRocTracer_init_tracing(); 
  std::cout << "# START #############################" << std::endl << std::flush;
  // Enable HIP API callbacks
  ROCTRACER_CALL(roctracer_enable_callback(api_callback, NULL));
  // Enable HIP activity tracing
  ROCTRACER_CALL(roctracer_enable_activity());
  // Enable HIP API callbacks
}

// Stop tracing routine
extern "C" void TauRocTracer_stop_tracing() {
  ROCTRACER_CALL(roctracer_disable_callback());
  ROCTRACER_CALL(roctracer_disable_activity());
  ROCTRACER_CALL(roctracer_flush_activity());
  std::cout << "# STOP  #############################" << std::endl << std::flush;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
