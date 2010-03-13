########################################################
#                                                      #
# TAUCUDA Build Makefile                               #
########################################################

include ../common.mk
include ${TAU_MAKEFILE} 


TARGET_LIB := libtaunexus.so
CCFILES := taucuda.cc 
OBJ_FILES := taucuda.o
INCLUDES = -I. -I$(CUDA_BASE)/include -I$(CUDA_SDK)/common/inc -I$(TAUROOT)/include 

CC := g++ -g
LINK := $(TAUROOT)/x86_64/bin/tau_cxx.sh -tau_makefile=${TAU_MAKEFILE} \
-tau_options='-optShared -optVerbose'
CFLAGS := -fPIC -g
DEFS := -DPTHREADS
CUDA_LIBS  := -L$(CUDA_BASE)/lib -L$(CUDA_BASE)/lib64 -L$(CUDA_SDK)/lib -L$(CUDA_SDK)/common/lib/linux -lcudart -lcutil

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJ_FILES)
	$(LINK) -shared -o ../release/$(TARGET_LIB) $(OBJ_FILES) $(CUDA_LIBS)

$(OBJ_FILES): $(CCFILES)
	$(CC) $(CFLAGS) $(INCLUDES) $(DEFS) $(TAU_DEFS) -c $(CCFILES)



clean: 
	rm -f $(OBJ_FILES)  ../release/$(TARGET_LIB)



