########################################################
#                                                      #
# TAUCUDA Build Makefile                               #
########################################################

include ../common.mk
include ${TAU_MAKEFILE} 

TARGET_LIB_OLD := libtaucuda-old.so
TARGET_LIB := libtaucuda.so
TAUCUDA_SRC := taucuda.cpp 
TAUCUDA_OBJ := taucuda.o
NVIDIA_SRC := nVidia_exp_adapter.cpp
NVIDIA_OBJ := nVidia_exp_adapter.o

LIBDIR=lib

INCLUDES = -I. -I$(CUDA_BASE)/include -I$(CUDA_SDK)/common/inc -I$(TAUROOT)/include 

CPP=g++ -g
CC=gcc -g
WRAPPER_CC=$(CC) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS1) $(TAU_INCLUDE)
TAU_CXX=$(TAUROOT)/x86_64/bin/tau_cxx.sh -tau_makefile=${TAU_MAKEFILE}
LINK=$(TAU_CXX) -tau_options='-optShared -optVerbose'
CFLAGS := -fPIC -g
DEFS := -DPTHREADS -DDEBUG_PROF -DTAU_CUDA
CUDA_LIBS  := -L$(CUDA_BASE)/lib -L$(CUDA_BASE)/lib64 -L$(CUDA_SDK)/lib -L$(CUDA_SDK)/common/lib/linux -lcudart -lcutil

all: $(TARGET_LIB) libcuda_runtime_api_wrap.so libso_wrapper.so

libso_wrapper.so: so_wrapper.o 
	$(CC) -shared -o ../$(LIBDIR)/$@ $< $(TAU_SHLIBS) -ldl

so_wrapper.o: so_wrapper.c
	$(CC) -I$(CUDA_BASE)/include $(CFLAGS) -c $< -o $@

libcuda_runtime_api_wrap.so: cuda_runtime_api_wrap.o 
	$(LINK) -shared -o ../$(LIBDIR)/$@ $< $(TAU_SHLIBS) -ldl

cuda_runtime_api_wrap.o: cudart_wrapper.c
	$(WRAPPER_CC) -I$(CUDA_BASE)/include $(CFLAGS) -c $< -o $@

$(TARGET_LIB): $(NVIDIA_OBJ) $(TAUCUDA_OBJ)
	$(LINK) -shared -o ../$(LIBDIR)/$(TARGET_LIB) $(NVIDIA_OBJ) $(TAUCUDA_OBJ) $(CUDA_LIBS)

$(TAUCUDA_OBJ): $(TAUCUDA_SRC)
	$(CPP) $(CFLAGS) $(INCLUDES) $(DEFS) $(TAU_DEFS) -c $<

$(NVIDIA_OBJ): $(NVIDIA_SRC)
	$(CPP) $(CFLAGS) $(INCLUDES) $(DEFS) $(TAU_DEFS) -c $<

clean: 
	rm -f $(NVIDIA_OBJ) $(TAUCUDA_OBJ)  ../$(LIBDIR)/$(TARGET_LIB) cuda_runtime_api_wrap.o ../$(LIBDIR)/libcuda_runtime_api_wrap.so 
