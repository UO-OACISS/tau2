include ../../../include/Makefile
include ../Makefile


CXX=$(TAU_CXX) 
CXXFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -fPIC 
EXTRA_FLAGS=

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: $(TAU_ZEINIT_SHARED)
	$(TAU_SHARED_INSTALL_CMD)

$(TAU_ZEINIT_SHARED): zeinit_wrapper.o
	$(CXX) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

zeinit_wrapper.o: zeinit_wrapper.cpp
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	/bin/rm -f zeinit_wrapper.o libTAU-zeinit-wrapper.so