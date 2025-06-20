# Makefile for MPI + Level Zero Demo
#
# Assumes the Intel oneAPI environment has been sourced, e.g.,
# source /opt/intel/oneapi/setvars.sh

# --- User Configuration ---
# This Makefile is flexible and can be controlled by an environment variable.
# To build, set the GPU_ARCH variable. If not set, it defaults to 'pvc'.
#
# Example for Arc A770:   GPU_ARCH=dg2_g10 make
# Example for DG1:        GPU_ARCH=dg1 make
# Example for PVC (Max):  GPU_ARCH=pvc make
#
GPU_ARCH ?= pvc

# --- Compiler and Tool Definitions ---
# Use the oneAPI DPC++/C++ MPI compiler wrapper
MPICXX = mpiicpx
# Use the Intel OpenCL Offline Compiler to generate SPIR-V
OCLOC = ocloc

# --- Compiler and Linker Flags ---
CXXFLAGS = -std=c++17 -I/usr/include/level_zero
#CXXFLAGS = -std=c++17 -I$(ONEAPI_ROOT)/include

#LDFLAGS = -lze_loader
LDFLAGS = -L$(ONEAPI_ROOT)/lib -lze_loader

# --- Targets ---
TARGET = mpi_ze_vecadd

# Name of the kernel source and its compiled SPIR-V target
KERNEL_SRC = vector_add.cl
KERNEL_SPV = vector_add.spv
# ocloc creates a mangled name, so we define it to rename it later
OCLOC_OUTPUT_BASE = $(basename $(KERNEL_SRC))

.PHONY: all clean

# Default target
all: $(TARGET)

# Rule to build the SPIR-V kernel from the OpenCL C source
$(KERNEL_SPV): $(KERNEL_SRC)
	@echo "Compiling OpenCL kernel to SPIR-V for architecture: $(GPU_ARCH)..."
	# 1. Let ocloc create its default file (e.g., vector_add_pvc.spv)
	$(OCLOC) -file $(KERNEL_SRC) -device $(GPU_ARCH) -spv_only
	# 2. Rename the known quirky output to the simple name our program expects.
	@echo "Renaming ocloc output to $(KERNEL_SPV)..."
	mv $(OCLOC_OUTPUT_BASE)*.spv $(KERNEL_SPV)

# Rule to build the main application
# It depends on the C++ source and the compiled SPIR-V kernel
$(TARGET): mpi_ze_vecadd.cpp $(KERNEL_SPV)
	@echo "Compiling and linking MPI+L0 application..."
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) mpi_ze_vecadd.cpp $(LDFLAGS)

run-uninst: mpi_ze_vecadd
	mpirun -np 4 ./mpi_ze_vecadd

run: mpi_ze_vecadd
	mpirun -np 4 tau_exec -T level_zero -l0 ./mpi_ze_vecadd

# Rule to clean up build artifacts
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET) $(KERNEL_SPV) *.log profile.* *.trc *.edf
