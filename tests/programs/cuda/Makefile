all: multiGpuThread

multiGpuThread: multiGpuThread.cu
	nvcc -g -O2 -o multiGpuThread multiGpuThread.cu -lpthread --default-stream per-thread
	nvcc -g -O2 -o multiGpuThread_legacy multiGpuThread.cu -lpthread

test:
	tau_exec -T cupti,serial,pthread -cupti ./multiGpuThread
	test -f profile.0.0.0 || exit 1

clean:
	rm -rf multiGpuThread multiGpuThread_legacy profile.* MULTI__* traces*

