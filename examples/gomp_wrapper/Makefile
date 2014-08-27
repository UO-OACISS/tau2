TAU_MAKEFILE ?=../../include/Makefile

include $(TAU_MAKEFILE)

#CXX=opencc -fopenmp
CC=TAU_MAKEFILE=$(TAU_MAKEFILE) tau_cc.sh
LD=$(CC)
AR=ar
ARFLAGS=rcv
LDFLAGS         = -g $(TAU_OPENMP_OPTION)
CFLAGS          = -g $(TAU_OPENMP_OPTION)
TARGET=matmult

OBJS=$(TARGET).o 
$(TARGET): $(OBJS) $(TAU_LIB_DIR)/libTAU.so
	$(LD) -g $(TARGET).o -o $@ $(LDFLAGS) 

.c.o:
	$(CC) $(CFLAGS) -c $< -I$(HDFDIR)/include

test: $(TARGET)
	./matmult

clean:
	/bin/rm -rf $(OBJS) $(TARGET) profile.* MULT* core.*
