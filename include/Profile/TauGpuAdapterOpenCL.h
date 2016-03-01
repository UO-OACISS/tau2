#include "TauGpu.h"
#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <string.h>

#define TAU_MAX_FUNCTIONNAME 200

// Putting "using namespace" statements in header files can create ambiguity
// between user-defined symbols and std symbols, creating unparsable code
// or even changing the behavior of user codes.  This is also widely considered
// to be bad practice.  Here's a code PDT can't parse because of this line:
//   EX: #include <complex>
//   EX: typedef double real;
//
//using namespace std;


//The time in microseconds that the GPU is ahead of the CPU clock.
double sync_offset = 0;

class OpenCLGpuEvent : public GpuEvent {

  public:
    x_uint64 commandId;
    cl_device_id id;
    cl_event event;
    const char *name;
    FunctionInfo *callingSite;
    GpuEventAttributes *gpu_event_attr;
    int number_of_gpu_events;
    int memcpy_type;

    //double sync_offset;

    ~OpenCLGpuEvent()
    {
      free(gpu_event_attr);
    }

    bool isMemcpy()
    {
      return memcpy_type != -1;
    }

    OpenCLGpuEvent(cl_device_id i, x_uint64 cId, double sync)
    {
      id = i;
      commandId = cId;
      sync_offset = sync;
    }
    OpenCLGpuEvent *getCopy() const { 
      OpenCLGpuEvent *c = new OpenCLGpuEvent(*this);
      return c;
    }

    bool less_than(const GpuEvent *other) const
    {
      if (this->id_p1() == other->id_p1())
      {
        return this->id_p2() < other->id_p2();
      }
      else
      {
        return this->id_p1() < other->id_p1();
      }
      //return strcmp(printId(), ((OpenCLGpuEvent *)o)->printId()) < 0;
    }

    const char *getName() const { return name; }

    FunctionInfo *getCallingSite() const { 

      if (callingSite != NULL)
      {
        callingSite->SetPrimaryGroupName("TAU_REMOTE");
      }

      return callingSite; 
    }

    double syncOffset() const
    {
      return sync_offset;
    }
    // GPU attributes not implemented for OpenCL.
    void getAttributes(GpuEventAttributes *&gA, int &num) const
    {
      num = number_of_gpu_events;
      gA = gpu_event_attr;
    }

    void recordMetadata(int i) const {}

    /* CUDA Event are uniquely identified as the pair of two other ids:
     * context and call (API).
     */
    const char* gpuIdentifier() const 
    {	
      char r[40];
      sprintf(r, "%d:%lld", id, commandId);
      return strdup(r);
    }

    x_uint64 id_p1() const { return (x_uint64) id; }
    x_uint64 id_p2() const { return (x_uint64) commandId; }
};

int memcpy_data_size = sizeof(OpenCLGpuEvent);

int kernel_data_size = sizeof(OpenCLGpuEvent);

void Tau_opencl_init();

void Tau_opencl_exit();

void Tau_opencl_enter_memcpy_event(const char *name, OpenCLGpuEvent *id, int size, int MemcpyType);

void Tau_opencl_exit_memcpy_event(const char *name, OpenCLGpuEvent *id, int MemcpyType);

void Tau_opencl_register_gpu_event(OpenCLGpuEvent *id, double start, double stop);

void Tau_opencl_register_memcpy_event(OpenCLGpuEvent *id, double start, double stop, int transferSize, int MemcpyType);

OpenCLGpuEvent *Tau_opencl_enqueue_event(const char* name, cl_command_queue q, cl_event * e, FunctionInfo *fi, int memcpy_type);

void Tau_opencl_register_sync_event();

void Tau_opencl_flush();

OpenCLGpuEvent *Tau_opencl_retrive_gpu(cl_command_queue cq);

//Memcpy event callback

void CL_CALLBACK Tau_opencl_memcpy_callback(cl_event event, cl_int command_stat, void
    *memcpy_type);

void CL_CALLBACK Tau_opencl_kernel_callback(cl_event event, cl_int command_stat, void
    *kernel_type);

double Tau_opencl_sync_clocks(cl_command_queue commandQueue, cl_context context);

bool Tau_opencl_is_callbacks_supported();

//Enqueue events

enum EnqueueEvents { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER };


