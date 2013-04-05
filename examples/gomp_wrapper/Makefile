TAU_MAKEFILE ?=../../include/Makefile
include $(TAU_MAKEFILE)

#CXX=gcc
CXX=TAU_MAKEFILE=$(TAU_MAKEFILE) tau_cxx.sh
LD=$(CXX)
AR=ar
ARFLAGS=rcv
LDFLAGS=-fopenmp -g 
CFLAGS=-fopenmp -g 
TARGET=matmult
WRAPPER=libgomp_g_wrapper/libgomp_g_wrap.so

OBJS=$(TARGET).o 
$(TARGET): $(OBJS) 
	# to link the wrapper in directly, do this (CXX should be tau_cxx.sh):
	$(LD) $(TARGET).o $(WRAPPER) -o $@ $(LDFLAGS) 
	# otherwise, just link normally
	# $(LD) -g $(TARGET).o -o $@ $(LDFLAGS) 

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
	tau_exec -v -T pdt,mpi,papi,openmp -loadlib=$(WRAPPER) ./matmult

clean:
	/bin/rm -rf $(OBJS) $(TARGET) profile.* MULT* core.*
