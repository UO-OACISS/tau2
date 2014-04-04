# set this to your TBB library location
TBB_HOME=/usr/local/packages/intel/14.0/tbb
INTEL_ARCH=intel64
TBBFLAGS=-I$(TBB_HOME)/include -L$(TBB_HOME)/lib/$(INTEL_ARCH)/gcc4.4 -Wl,-rpath,$(TBB_HOME)/lib/$(INTEL_ARCH)/gcc4.4 -g

CXX=icpc $(TBBFLAGS)
#BITS=-m64
INCLUDES=-I.
CXXFLAGS=-O2 $(INCLUDES) $(BITS)

# If you are using a version of binutils prior to 2.17, 
# uncomment following line to work around a known issue. 
# please reference the Release Notes for more information. 
#LIBS=-lcilkrts

PROGS=matrix-multiply

all:: $(PROGS) test

matrix-multiply: matrix-multiply.o
	$(CXX) $(BITS) -o $@ matrix-multiply.o $(LIBS)

test:: matrix-multiply
	tau_exec -v -T tbb,serial,icpc -ebs ./matrix-multiply
clean::
	-rm -f $(PROGS) *.o profile.* MULTI__*
