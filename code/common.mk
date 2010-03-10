# This file set the paths to external packages need to build TAU Cuda

# Location of the cuda libraries and headers (default /opt/cuda).
CUDA_BASE=/opt/cuda

# TAU makefile for building the wrapper library (I think we need a TAU
# confguration build with pthread support).
TAU_MAKEFILE=/home/users/scottb/tau2/x86_64/lib/Makefile.tau-mpi

# Location of the CUDA sdk directory (we need the libcutil library).
CUDA_SDK=/home/users/scottb/NVIDIA_GPU_Computing_SDK/C

