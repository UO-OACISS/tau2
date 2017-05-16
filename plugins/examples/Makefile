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

OBJS = libtau_plugin_example_disable_instrumentation_runtime.so libtau_plugin_example_mpit_recommend_sharp_usage.so tau_plugin_example_disable_instrumentation_runtime.o tau_plugin_example_mpit_recommend_sharp_usage.o

SHAREDLIBS = libtau_plugin_example_disable_instrumentation_runtime.so libtau_plugin_example_mpit_recommend_sharp_usage.so

all: install
	@echo

install: $(SHAREDLIBS)
	@mkdir -p $(PLUGININSTALLDIR)
	$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

libtau_plugin_example_disable_instrumentation_runtime.so: tau_plugin_example_disable_instrumentation_runtime.o 
	$(CXX) $(LDFLAGS) -shared -o libtau_plugin_example_disable_instrumentation_runtime.so tau_plugin_example_disable_instrumentation_runtime.o 

tau_plugin_example_disable_instrumentation_runtime.o: Tau_plugin_example_disable_instrumentation_runtime.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_disable_instrumentation_runtime.cpp -o tau_plugin_example_disable_instrumentation_runtime.o

libtau_plugin_example_mpit_recommend_sharp_usage.so: tau_plugin_example_mpit_recommend_sharp_usage.o
	 $(CXX) $(LDFLAGS) -shared -o libtau_plugin_example_mpit_recommend_sharp_usage.so tau_plugin_example_mpit_recommend_sharp_usage.o

tau_plugin_example_mpit_recommend_sharp_usage.o: Tau_plugin_example_mpit_recommend_sharp_usage.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_mpit_recommend_sharp_usage.cpp -o tau_plugin_example_mpit_recommend_sharp_usage.o
clean:
	/bin/rm -rf *.o $(SHAREDLIBS)
