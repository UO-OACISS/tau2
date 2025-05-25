include ../../../include/Makefile
include ../Makefile


CXX=$(TAU_CXX) 
CXXFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -fPIC 
EXTRA_FLAGS=

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTAU-ittnotify-collector.so
	mkdir -p $(INSTALLDEST)/wrappers/ittnotify_wrapper
	$(TAU_SHARED_INSTALL_CMD)

libTAU-ittnotify-collector.so: ittnotify_tau_collector.o
	$(CXX) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

ittnotify_tau_collector.o: ittnotify_tau_collector.cpp
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	/bin/rm -f ittnotify_tau_collector.o libTAU-ittnotify-collector.so

