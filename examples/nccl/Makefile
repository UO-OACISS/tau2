# Compiler
CXX := g++

# Target
TARGET := nccl_bcast

# Source
SRC := nccl_bcast.cpp

# Paths (from your command)
NCCL_INCLUDE := /home/users/jalcaraz/nccl/nccl/build/include
NCCL_LIB     := /home/users/jalcaraz/nccl/nccl/build/lib
CUDA_LIB     := /packages/cuda/12.8.1/lib64/

# Flags
CXXFLAGS := -O2 -std=c++11 -I$(NCCL_INCLUDE)
LDFLAGS  := -L$(NCCL_LIB) -L$(CUDA_LIB)
LDLIBS   := -lnccl -lcudart -lpthread

# Optional MPI support
ifdef USE_MPI
    CXXFLAGS += -DUSE_MPI
    LDLIBS   += -lmpi
    CXX = mpicc
endif

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Clean
clean:
	rm -f $(TARGET)

.PHONY: all clean
