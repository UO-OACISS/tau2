TAU_MAKEFILE ?=${tauroot}/include/Makefile
include $(TAU_MAKEFILE)
CCBIN=-ccbin $(CONFIG_CXX)
# Intel 20 doesn't work with Cuda 10
ifeq ("$(CONFIG_CXX)","icpc")
    CCBIN=
endif
# Clang 9 doesn't work with Cuda 10
ifeq ("$(CONFIG_CXX)","clang++")
    CCBIN=
endif

all: multiGpuThread test

multiGpuThread: multiGpuThread.cu
	nvcc $(CCBIN) -g -O2 -o multiGpuThread multiGpuThread.cu -lpthread --default-stream per-thread
	nvcc $(CCBIN) -g -O2 -o multiGpuThread_legacy multiGpuThread.cu -lpthread

test:
	tau_exec -T cupti,serial,pthread -cupti ./multiGpuThread
	test -f profile.0.0.0 || exit 1
	$(PPROF_CMD)

clean:
	rm -rf multiGpuThread multiGpuThread_legacy profile.* MULTI__* traces*

