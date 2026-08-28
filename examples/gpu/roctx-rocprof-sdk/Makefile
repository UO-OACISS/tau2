CXXFLAGS += -std=c++11 -O3
ROCM_PATH ?= /opt/rocm
CXXFLAGS += -g -I$(ROCM_PATH)/include/rocprofiler-sdk-roctx/ 
LDFLAGS = -L$(ROCM_PATH)/lib -lrocprofiler-sdk-roctx


all:  MT

CXX=hipcc

MT: MT.o
	$(CXX) $(LDFLAGS) $^ -o $@

MT.o : MatrixTranspose.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f   MT.* profile.*

