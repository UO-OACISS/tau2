###############################################################################
## Top level Makefile for Plugins #
###############################################################################

include ../../include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/plugins

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

CFLAGS = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -fPIC

LDFLAGS = $(TAU_MPI_LIB)

OBJS = libtau_plugin_function_registration_complete.so libtau_plugin_atomic_event_trigger.so tau_plugin_function_registration_complete.o tau_plugin_atomic_event_trigger.o

SHAREDLIBS = libtau_plugin_function_registration_complete.so libtau_plugin_atomic_event_trigger.so

all: install
	@echo

install: $(SHAREDLIBS)
	@mkdir -p $(PLUGININSTALLDIR)
	$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

libtau_plugin_function_registration_complete.so: tau_plugin_function_registration_complete.o
	$(CXX) $(LDFLAGS) -shared -o libtau_plugin_function_registration_complete.so tau_plugin_function_registration_complete.o

tau_plugin_function_registration_complete.o:      Tau_plugin_function_registration_complete.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_function_registration_complete.cpp -o tau_plugin_function_registration_complete.o

libtau_plugin_atomic_event_trigger.so: tau_plugin_atomic_event_trigger.o
	$(CXX) $(LDFLAGS) -shared -o libtau_plugin_atomic_event_trigger.so tau_plugin_atomic_event_trigger.o

tau_plugin_atomic_event_trigger.o:      Tau_plugin_atomic_event_trigger.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_atomic_event_trigger.cpp -o tau_plugin_atomic_event_trigger.o

clean:
	/bin/rm -rf *.o $(SHAREDLIBS)
