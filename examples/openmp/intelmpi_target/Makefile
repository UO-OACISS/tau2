MPI_OPTS=$(shell ./mpiopts.sh)
# Put your own mpi -I -L -l options here. 

HIPCC=hipcc 
ICPX=icpx

SOURCES = matmult.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./matmult

.PHONY: test


all: $(EXECUTABLE) 

AMD_CXXFLAGS =  -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -DTAU_MPI -DTAU_OPENMP 
NVIDIA_CXXFLAGS=
INTEL_CXXFLAGS=-qopenmp -fopenmp-targets=spir64

CXXFLAGS=${NVIDIA_CXXFLAGS}
CXXFLAGS=${AMD_CXXFLAGS}
CXXFLAGS=${INTEL_CXXFLAGS}

CXX=$(ICPX)


$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $(OBJECTS) -o $@ ${CXXFLAGS} ${MPI_OPTS}

%.o: %.cpp
	$(CXX) ${CXXFLAGS} ${MPI_OPTS} -c $< 


run: $(EXECUTABLE)
	OMP_TARGET_OFFLOAD=MANDATORY mpirun -np 3  $(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o profile*


