########################################################
#                                                      #
# TAUCUDA Build Makefile                               #
########################################################

include ../common.mk
include ${TAU_MAKEFILE} 


TARGET_LIB := libtaunexus.so
CCFILES := taucuda.cc gpuevents.cc 
OBJ_FILES := taucuda.o gpuevents.o
INCLUDES = -I. -I$(CUDA_BASE)/include -I$(CUDA_SDK)/common/inc -I$(TAUROOT)/include 

CC := g++
LINK := g++
CFLAGS := -fPIC -g
LDFLAGS := -shared -fPIC -g
LIBS    := -L$(CUDA_BASE)/lib -L$(CUDA_SDK)/lib -L$(CUDA_SDK)/common/lib/linux -L$(TAUROOT)/x86_64/lib  -lcudart -lcutil -lTAU_tf 
TAU_FLAGS :=  -DUNIX  -DPROFILING_ON -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -DTAU_LARGEFILE -D_LARGEFILE64_SOURCE -O3

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJ_FILES)
	$(LINK) $(TAU_MPI_INC) $(TAU_MPI_NOWRAP_LIB) $(LDFLAGS) -o ../release/$(TARGET_LIB) $(OBJ_FILES) $(LIBS) 

$(OBJ_FILES): $(CUFILES)
	$(CC) $(CFLAGS) $(INCLUDES) $(TAU_MPI_INC) $(TAU_MPI_NOWRAP_LIB) $(TAU_FLAGS) -c $(CCFILES)



clean: 
	rm -f $(OBJ_FILES)  ../release/$(TARGET_LIB)



