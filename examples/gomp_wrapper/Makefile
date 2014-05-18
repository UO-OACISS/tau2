TAU_MAKEFILE ?=../../include/Makefile

include $(TAU_MAKEFILE)

#CXX=opencc -fopenmp
CXX=TAU_MAKEFILE=$(TAU_MAKEFILE) tau_cc.sh
LD=$(CXX)
AR=ar
ARFLAGS=rcv
LDFLAGS         = -g $(TAU_OPENMP_OPTION) -O0
CFLAGS          = -g $(TAU_OPENMP_OPTION) -O0
TARGET=matmult

OBJS=$(TARGET).o 
$(TARGET): $(OBJS) $(TAU_LIB_DIR)/libTAU.so
	$(LD) -g $(TARGET).o -o $@ $(LDFLAGS) 

.c.o:
	$(CXX) $(CFLAGS) -c $< -I$(HDFDIR)/include

test: $(TARGET)
	./matmult

clean:
	/bin/rm -rf $(OBJS) $(TARGET) profile.* MULT* core.*
