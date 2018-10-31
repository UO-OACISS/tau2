all: multiGpuThread

multiGpuThread: multiGpuThread.cu
	#nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_35,code=sm_35 -o multiGpuThread multiGpuThread.cu -lpthread 
	# For Volta:
	nvcc -gencode arch=compute_70,code=sm_70 -o multiGpuThread multiGpuThread.cu -lpthread 

run:
	tau_exec -T cupti,serial,pthread -cupti ./multiGpuThread

clean:
	rm -rf multiGpuThread

