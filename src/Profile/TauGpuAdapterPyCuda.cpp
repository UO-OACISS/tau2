#include <boost/python.hpp>
#include <Profile/TauGpuAdapterCUDA.h>

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

void Tau_pycuda_register_memcpy_event(std::string name, cudaRuntimeGpuId& gId,
double start, double stop, int size, int type)
{
		printf("PyCuda checkout, name: %s.\n", name.c_str());
		Tau_cuda_register_memcpy_event(name.c_str(), &gId, start, stop, size, type);
}

void Tau_pycuda_enter_memcpy_event(std::string name, int id, int size, int MemcpyType)
{
  Tau_cuda_enter_memcpy_event(name.c_str(), id, size, MemcpyType);
}


void Tau_pycuda_exit_memcpy_event(std::string name, int id, int MemcpyType)
{
	Tau_cuda_exit_memcpy_event(name.c_str(), id, MemcpyType);
}

void Tau_pycuda_enqueue_kernel_enter_event(std::string name, cudaRuntimeGpuId& gId)
{
	Tau_cuda_enqueue_kernel_enter_event(name.c_str(), &gId);
}

void Tau_pycuda_enqueue_kernel_exit_event()
{
	Tau_cuda_enqueue_kernel_exit_event();
}

BOOST_PYTHON_MODULE(libtau_pycuda)
{
	using namespace boost::python;

	enum_<Memcpy>("Memcpy")
		.value("HtoD", MemcpyHtoD)
		.value("DtoH", MemcpyDtoH)
		.value("DtoD", MemcpyDtoD)
		.value("Unknown", MemcpyUnknown)
	;

	class_<cudaGpuId, boost::noncopyable>("cudaGpuId", no_init);
	class_<cudaRuntimeGpuId>("cudaRuntimeGpuId");

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
