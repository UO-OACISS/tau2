TAU_MAKEFILE ?=../../include/Makefile

include $(TAU_MAKEFILE)

#CXX=gcc
CXX=TAU_MAKEFILE=$(TAU_MAKEFILE) tau_cc.sh
LD=$(CXX)
AR=ar
ARFLAGS=rcv
LDFLAGS         = -g -fopenmp -O0
CFLAGS          = -g -fopenmp -O0
TARGET=matmult

OBJS=$(TARGET).o 
$(TARGET): $(OBJS) 
	$(LD) -g $(TARGET).o -o $@ $(LDFLAGS) 

.c.o:
	$(CXX) $(CFLAGS) -c $< -I$(HDFDIR)/include

test: $(TARGET)
	./matmult

clean:
	/bin/rm -rf $(OBJS) $(TARGET) profile.* MULT* core.*
