###############################################################################
## Top level Makefile for Plugins #
###############################################################################

include ../../include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/plugins

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

TAUINC = -I$(TAUROOT)/include -I$(TAUROOT)/include/Profile -I$(TAUROOT)/plugins/json-c/inst/include/json-c

CFLAGS = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAUINC) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -fPIC -Wall -Werror

#LDFLAGS =  -L/home/users/aurelem/tau/tau2_beacon/plugins/json-c/inst/lib -ljson-c -L/home/users/aurelem/tau/tau2_beacon/x86_64/lib -ltau-mvapich2-beacon-mpi-pdt-mpit -L/home/users/aurelem/tau/tau2_beacon/x86_64/lib -lTauMpi-mvapich2-beacon-mpi-pdt-mpit


OBJS = tuning_policies.o tuning_policies_json.o

SHAREDLIBS = tuning_policies.so

all: install
	@echo

install: $(SHAREDLIBS)
	#@mkdir -p $(PLUGININSTALLDIR)
	#$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

tuning_policies.so: 
#tuning_policies.so: tuning_policies.o
#	$(CC) -shared -o tuning_policies.so tuning_policies.o

tuning_policies.o: 
#tuning_policies.o:      tuning_policies.c
#	$(CC) $(CFLAGS)  -c tuning_policies.c 

clean:
	/bin/rm -rf *.o $(SHAREDLIBS)

