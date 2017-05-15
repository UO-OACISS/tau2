###############################################################################
## Top level Makefile for Plugins #
###############################################################################

include ../../include/Makefile
include ../../utils/include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/plugins

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

CFLAGS = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -fPIC

LDFLAGS = $(TAU_MPI_LIB)

OBJS = libtau_plugin_example_disable_instrumentation_runtime.so tau_selective.o tau_instrument.o tau_instrumentor.o tau_plugin_example_disable_instrumentation_runtime.o

SHAREDLIBS = libtau_plugin_example_disable_instrumentation_runtime.so

all: install
	@echo

install: $(SHAREDLIBS)
	@mkdir -p $(PLUGININSTALLDIR)
	$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

libtau_plugin_example_disable_instrumentation_runtime.so: tau_plugin_example_disable_instrumentation_runtime.o tau_selective.o tau_instrument.o tau_instrumentor.o
	$(CXX) $(LDFLAGS) -shared -o libtau_plugin_example_disable_instrumentation_runtime.so tau_plugin_example_disable_instrumentation_runtime.o tau_selective.o tau_instrument.o tau_instrumentor.o $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a

tau_selective.o: $(TAUROOT)/utils/tau_selective.cpp
	 $(PDT_CXX_COMP) $(PDT_CFLAGS) $(CFLAGS) -I$(PDTDIR)/include -c $(TAUROOT)/utils/tau_selective.cpp -o $(shell pwd)/tau_selective.o

tau_instrument.o: $(TAUROOT)/utils/tau_instrument.cpp
	$(PDT_CXX_COMP) $(PDT_CFLAGS) $(CFLAGS) -I$(PDTDIR)/include -c $(TAUROOT)/utils/tau_instrument.cpp -o $(shell pwd)/tau_instrument.o

tau_instrumentor.o: $(TAUROOT)/utils/tau_instrumentor.cpp $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a
	$(PDT_CXX_COMP) $(PDT_CFLAGS) $(CFLAGS) -I$(PDTDIR)/include -c $(TAUROOT)/utils/tau_instrumentor.cpp -o $(shell pwd)/tau_instrumentor.o $(PDTOPTS) -lm $(LEXTRAUTIL)

tau_plugin_example_disable_instrumentation_runtime.o: Tau_plugin_example_disable_instrumentation_runtime.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_disable_instrumentation_runtime.cpp -o tau_plugin_example_disable_instrumentation_runtime.o

clean:
	/bin/rm -rf *.o $(SHAREDLIBS)
