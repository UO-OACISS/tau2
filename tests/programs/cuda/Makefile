TAU_MAKEFILE ?=${tauroot}/include/Makefile
include $(TAU_MAKEFILE)

all: multiGpuThread test

multiGpuThread: multiGpuThread.cu
	nvcc -ccbin $(CONFIG_CXX) -g -O2 -o multiGpuThread multiGpuThread.cu -lpthread --default-stream per-thread
	nvcc -ccbin $(CONFIG_CXX) -g -O2 -o multiGpuThread_legacy multiGpuThread.cu -lpthread

test:
	tau_exec -T cupti,serial,pthread -cupti ./multiGpuThread
	test -f profile.0.0.0 || exit 1
	$(PPROF_CMD)

clean:
	rm -rf multiGpuThread multiGpuThread_legacy profile.* MULTI__* traces*

