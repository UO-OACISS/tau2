CXXFLAGS += -std=c++11 -O3
ROCM_PATH ?= /opt/rocm

#USE ROCPROF_SDK for newer ROCTx API and rocprofiler-sdk,
# use ROCPROF_LEGACY for older API and roctracer
PROF = ROCPROF_SDK
#PROF = ROCPROF_LEGACY


CXXFLAGS += -g -I$(ROCM_PATH)/include/rocprofiler-sdk-roctx/ 
LDFLAGS = -L$(ROCM_PATH)/lib -lrocprofiler-sdk-roctx


ifeq ($(PROF),ROCPROF_SDK)
    CXXFLAGS += -g -I$(ROCM_PATH)/include/rocprofiler-sdk-roctx/ -DROCPROF_SDK
    LDFLAGS = -L$(ROCM_PATH)/lib -lrocprofiler-sdk-roctx

else ifeq ($(PROF),ROCPROF_LEGACY)
    CXXFLAGS += -g -I$(ROCM_PATH)/roctracer/include -DROCPROF_LEGACY
    LDFLAGS = -L$(ROCM_PATH)/roctracer/lib/ -lroctx64 -lroctracer64
endif

all:  MT

CXX=hipcc

MT: MT.o
	$(CXX) $(LDFLAGS) $^ -o $@

MT.o : MatrixTranspose.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f   MT.* profile.*

