MPI_OPTS=$(shell ./mpiopts.sh)
# Put your own mpi -I -L -l options here. 

HIPCC=hipcc 
ICPX=icpx
NVC=nvc++

SOURCES = matmult.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./matmult

.PHONY: test


all: $(EXECUTABLE) 

AMD_CXXFLAGS =  -g -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -DTAU_MPI -DTAU_OPENMP 
#NVIDIA_CXXFLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda,x86_64
NVIDIA_CXXFLAGS=-fopenmp  -mp=gpu -Minfo=mp
INTEL_CXXFLAGS=-qnextgen -fiopenmp -fopenmp-targets=spir64

CXXFLAGS=${NVIDIA_CXXFLAGS}
CXXFLAGS=${INTEL_CXXFLAGS}
CXXFLAGS=${AMD_CXXFLAGS}

# Pick the CXXFLAGS and the compiler: Intel, AMD, or NVIDIA
CXXFLAGS=${NVIDIA_CXXFLAGS}

CXX=${NVC}


$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $(OBJECTS) -o $@ ${CXXFLAGS} ${MPI_OPTS}

%.o: %.cpp
	$(CXX) ${CXXFLAGS} ${MPI_OPTS} -c $< 


run: $(EXECUTABLE)
	mpirun -np 3  $(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o profile*


