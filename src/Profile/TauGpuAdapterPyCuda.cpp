#include <boost/python.hpp>
#include <Profile/TauGpuAdapterCUDA.h>

#define GET_GPU_EVENT() \
    int device; \
    cudaGetDevice(&device); \
    cudaStream_t stream = 0; \
    const char * n = strdup(name.c_str()); \
    CudaRuntimeGpuEvent * gEvt = new CudaRuntimeGpuEvent(n, device, stream);

CudaRuntimeGpuEvent * cur_kernel = NULL;

void Tau_pycuda_init()
{
	Tau_cuda_init();
}

void Tau_pycuda_exit()
{
	Tau_cuda_exit();
}

void Tau_pycuda_register_sync_event()
{
	Tau_cuda_register_sync_event();
}

void Tau_pycuda_register_memcpy_event(std::string name,
double start, double stop, int size, int type)
{
    GET_GPU_EVENT();
    Tau_cuda_register_memcpy_event(n, gEvt, start, stop, size, type);
}

void Tau_pycuda_enter_memcpy_event(std::string name, int id, int size, int MemcpyType)
{
    Tau_cuda_enter_memcpy_event(name.c_str(), id, size, MemcpyType);
}


void Tau_pycuda_exit_memcpy_event(std::string name, int id, int MemcpyType)
{
	Tau_cuda_exit_memcpy_event(name.c_str(), id, MemcpyType);
}

void Tau_pycuda_enqueue_kernel_enter_event(std::string name)
{
    GET_GPU_EVENT();
    if(cur_kernel != NULL) {
        fprintf(stderr, "Nested Tau_pycuda_enqueue_kernel_enter_event.\n");
    }
    cur_kernel = gEvt;
	Tau_cuda_enqueue_kernel_enter_event(gEvt);
}

void Tau_pycuda_enqueue_kernel_exit_event()
{
    if(cur_kernel == NULL) {
        fprintf(stderr, "Tau_pycuda_enqueue_kernel_exit_event without matching Tau_pycuda_enqueue_kernel_enter_event\n");
        return;
    }
	Tau_cuda_enqueue_kernel_exit_event(cur_kernel);
    cur_kernel = NULL;
}

BOOST_PYTHON_MODULE(pytau_cuda)
{
	using namespace boost::python;

	enum_<Memcpy>("Memcpy")
		.value("HtoD", MemcpyHtoD)
		.value("DtoH", MemcpyDtoH)
		.value("DtoD", MemcpyDtoD)
		.value("Unknown", MemcpyUnknown)
	;

	//class_<cudaGpuId, boost::noncopyable>("cudaGpuId", no_init);
	//class_<cudaRuntimeGpuId>("cudaRuntimeGpuId");

	def("Tau_pycuda_init", Tau_cuda_init);
	def("Tau_pycuda_exit", Tau_cuda_exit);
	def("Tau_pycuda_register_sync_event", Tau_cuda_register_sync_event);
	//void (*Tau_pycuda_register_memcpy_event)(std::string);
	def("Tau_pycuda_enter_memcpy_event", Tau_pycuda_enter_memcpy_event);
	def("Tau_pycuda_exit_memcpy_event", Tau_pycuda_exit_memcpy_event);
	def("Tau_pycuda_register_memcpy_event", Tau_pycuda_register_memcpy_event);
	def("Tau_pycuda_enqueue_kernel_enter_event", Tau_pycuda_enqueue_kernel_enter_event);
	def("Tau_pycuda_enqueue_kernel_exit_event", Tau_pycuda_enqueue_kernel_exit_event);
	def("Tau_pycuda_register_sync_event", Tau_pycuda_register_sync_event);

}
