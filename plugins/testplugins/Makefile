###############################################################################
## Top level Makefile for Plugins #
###############################################################################

include ../../include/Makefile
include ../../utils/include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/shared$(TAU_CONFIG)/test_plugins

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

CFLAGS = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) $(FPIC) $(TAU_STDCXX11_FLAG)

LDFLAGS = $(TAU_MPI_LIB)

OBJS = libtau_plugin_phases.so libtau_plugin_function_registration_complete.so libtau_plugin_atomic_event_trigger.so libtau_plugin_atomic_event_registration_complete.so libtau_plugin_end_of_execution.so libtau_plugin_interrupt_trigger.so tau_plugin_function_registration_complete.o tau_plugin_phases.o tau_plugin_atomic_event_trigger.o tau_plugin_atomic_event_registration_complete.o tau_plugin_end_of_execution.o tau_plugin_interrupt_trigger.o

SHAREDLIBS = libtau_plugin_phases.so libtau_plugin_function_registration_complete.so libtau_plugin_atomic_event_trigger.so libtau_plugin_atomic_event_registration_complete.so libtau_plugin_end_of_execution.so libtau_plugin_interrupt_trigger.so

all: install
	@echo

install: $(SHAREDLIBS)
	@mkdir -p $(PLUGININSTALLDIR)
	$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

libtau_plugin_function_registration_complete.so: tau_plugin_function_registration_complete.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libtau_plugin_function_registration_complete.so tau_plugin_function_registration_complete.o

libtau_plugin_phases.so: tau_plugin_phases.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libtau_plugin_phases.so tau_plugin_phases.o

tau_plugin_phases.o: Tau_plugin_phases.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_phases.cpp -o tau_plugin_phases.o

tau_plugin_function_registration_complete.o: Tau_plugin_function_registration_complete.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_function_registration_complete.cpp -o tau_plugin_function_registration_complete.o

libtau_plugin_atomic_event_trigger.so: tau_plugin_atomic_event_trigger.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libtau_plugin_atomic_event_trigger.so tau_plugin_atomic_event_trigger.o

tau_plugin_atomic_event_trigger.o: Tau_plugin_atomic_event_trigger.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_atomic_event_trigger.cpp -o tau_plugin_atomic_event_trigger.o

libtau_plugin_atomic_event_registration_complete.so: tau_plugin_atomic_event_registration_complete.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libtau_plugin_atomic_event_registration_complete.so tau_plugin_atomic_event_registration_complete.o

tau_plugin_atomic_event_registration_complete.o: Tau_plugin_atomic_event_registration_complete.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_atomic_event_registration_complete.cpp -o tau_plugin_atomic_event_registration_complete.o

libtau_plugin_end_of_execution.so: tau_plugin_end_of_execution.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libtau_plugin_end_of_execution.so tau_plugin_end_of_execution.o

tau_plugin_end_of_execution.o: Tau_plugin_end_of_execution.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_end_of_execution.cpp -o tau_plugin_end_of_execution.o

libtau_plugin_interrupt_trigger.so: tau_plugin_interrupt_trigger.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libtau_plugin_interrupt_trigger.so tau_plugin_interrupt_trigger.o

tau_plugin_interrupt_trigger.o: Tau_plugin_interrupt_trigger.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_interrupt_trigger.cpp -o tau_plugin_interrupt_trigger.o

clean:
	/bin/rm -rf *.o $(SHAREDLIBS)
