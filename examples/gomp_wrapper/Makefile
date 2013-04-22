TAU_MAKEFILE ?=../../include/Makefile
include $(TAU_MAKEFILE)

#CXX=gcc
CXX=TAU_MAKEFILE=$(TAU_MAKEFILE) tau_cc.sh
LD=$(CXX)
AR=ar
ARFLAGS=rcv
LDFLAGS=-fopenmp -g 
CFLAGS=-fopenmp -g 
TARGET=matmult
WRAPPER_PATH=$(shell pwd)/libgomp_g_wrapper
WRAPPER=$(WRAPPER_PATH)/libgomp_g_wrap$(TAU_SHLIBX)

OBJS=$(TARGET).o 
$(TARGET): $(OBJS) 
	# to link the wrapper in directly, do this (CXX should be tau_cxx.sh):
	#$(LD) $(TARGET).o $(WRAPPER) -o $@ $(LDFLAGS) 
	# otherwise, just link normally
	$(LD) -g $(TARGET).o -o $@ $(LDFLAGS) 

.c.o:
	# to make a high overhead multiply call, do this:
	#$(CXX) $(CFLAGS) -c $< -I$(HDFDIR)/include -DAPP_USE_INLINE_MULTIPLY
	# otherwise, compile normally.
	$(CXX) $(CFLAGS) -c $< -I$(HDFDIR)/include

test: $(TARGET)
	./matmult

$(WRAPPER):
	cd libgomp_g_wrapper ; $(MAKE)

testlib: $(TARGET) $(WRAPPER)
	#LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(WRAPPER_PATH) tau_exec -v -T pdt,serial,openmp -loadlib=$(WRAPPER) ./matmult
	tau_exec -v -T pdt,serial,openmp -loadlib=$(WRAPPER) ./matmult

clean:
	/bin/rm -rf $(OBJS) $(TARGET) profile.* MULT* core.*
