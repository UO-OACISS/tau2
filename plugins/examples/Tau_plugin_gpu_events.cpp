/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>

#define GPU_PLUGIN_VERSION 0.1

typedef struct Tau_autoperf_gpu_metric_data {
  unsigned long int total_kernel_exec_time;
  unsigned long int total_bytes_transferred_HtoD;
  unsigned long int total_bytes_transferred_DtoH;
  unsigned long int total_memcpy_time;
} Tau_autoperf_gpu_metric_data;

unsigned long int total_kernel_exec_time = 0;
unsigned long int total_bytes_transferred_HtoD = 0;
unsigned long int total_bytes_transferred_DtoH = 0;
unsigned long int total_memcpy_time = 0;

int Tau_plugin_gpu_event_init(Tau_plugin_event_gpu_init_data_t* data) {
    fprintf(stderr, "TAU Plugin Event: GPU Profiling Initialized.\n");
    return 0;
}

extern "C" void Tau_darshan_export_plugin(Tau_autoperf_gpu_metric_data **data, double ver) {
    if(data == NULL) { fprintf(stderr, "TAU Plugin: Please allocate space for gpu metric data!\n"); return; }

    if(ver != GPU_PLUGIN_VERSION) { fprintf(stderr, "TAU Plugin: Version mismatch between AutoPerf and TAU Plugin. Exiting.\n"); return; }

    (*data)->total_kernel_exec_time = total_kernel_exec_time;
    (*data)->total_bytes_transferred_HtoD = total_bytes_transferred_HtoD;
    (*data)->total_bytes_transferred_DtoH = total_bytes_transferred_DtoH;
    (*data)->total_memcpy_time = total_memcpy_time;

}

int Tau_plugin_gpu_event_finalize(Tau_plugin_event_gpu_finalize_data_t* data) {
    fprintf(stderr, "TAU Plugin Event: GPU Profiling Finalized.\n");
    double total_kernel_exec_time_s = total_kernel_exec_time/1000000000.0;
    double total_memcpy_time_s = total_memcpy_time/1000000000.0;
    double total_bytes_transferred_HtoD_GB = total_bytes_transferred_HtoD/1000000000.0;
    double total_bytes_transferred_DtoH_GB = total_bytes_transferred_DtoH/1000000000.0;

    fprintf(stderr, "**********************************************\n");
    fprintf(stderr, "************ TAU AutoPerf Plugin Results      \n");
    fprintf(stderr, "**********************************************\n");
    fprintf(stderr, "Total kernel execution time (s): %lf\n", total_kernel_exec_time_s);
    fprintf(stderr, "Total CPU-GPU memory transfer time (s): %lf\n", total_memcpy_time_s);
    fprintf(stderr, "Total CPU->GPU data size (bytes): %lu\n", total_bytes_transferred_HtoD);
    fprintf(stderr, "Total GPU->CPU data size (bytes): %lu\n", total_bytes_transferred_DtoH);
    fprintf(stderr, "**********************************************\n");
    
    return 0;
}

int Tau_plugin_gpu_event_kernel_exec(Tau_plugin_event_gpu_kernel_exec_data_t* data) {
    //fprintf(stderr, "TAU Plugin Event: GPU Kernel time: %lu\n", data->time);
    RtsLayer::LockDB();
    total_kernel_exec_time += data->time;
    RtsLayer::UnLockDB();
    
    return 0;
}

int Tau_plugin_gpu_event_memcpy(Tau_plugin_event_gpu_memcpy_data_t* data) {
    //fprintf(stderr, "TAU Plugin Event: GPU Memcpy time, data, kind: %lu, %lu, %d\n", data->time, data->size, data->kind);
    RtsLayer::LockDB();
    if(data->kind == 1) {
	    total_bytes_transferred_HtoD += data->size;
    } else if(data->kind == 2) {
	    total_bytes_transferred_DtoH += data->size;
    }
    total_memcpy_time += data->time;
    RtsLayer::UnLockDB();
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks_t cb;
    TAU_VERBOSE("TAU PLUGIN GPU Init\n"); fflush(stdout);
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(&cb);
    /* Required event support */
    cb.GpuInit = Tau_plugin_gpu_event_init;
    cb.GpuFinalize = Tau_plugin_gpu_event_finalize;
    cb.GpuKernelExec = Tau_plugin_gpu_event_kernel_exec;
    cb.GpuMemcpy = Tau_plugin_gpu_event_memcpy;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(&cb, id);
    return 0;
}

