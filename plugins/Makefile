###############################################################################
## Makefile for just the basic utility tools #
###############################################################################

include ../include/Makefile
#include include/Makefile

HDRS	      =

MAKEFILE      = Makefile

TAUINC        = -I/home/users/aurelem/tau/tau2_beacon/include -I/home/users/aurelem/tau/tau2_beacon/include/Profile -I/home/users/aurelem/tau/tau2_beacon/plugins/json-c/inst/include/json-c

CC            = $(TAU_CC)

#CFLAGS        = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAUINC) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -fPIC
CFLAGS        = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAUINC) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -fPIC

LDFLAGS       = -ljson-c -L/home/users/aurelem/tau/tau2_beacon/plugins/json-c/inst/lib 

OBJS	      = tuning_policies.o

PRINT	      = pr

SHAREDLIBS    = tuning_policies.so

all:		$(SHAREDLIBS)

printvar:	
		@echo "CFLAGS "$(CFLAGS)" LEXTRAUTIL "$(LEXTRAUTIL)""

tuning_policies.so:	tuning_policies.o
			$(CC) -shared -fPIC -o tuning_policies.so tuning_policies.o

tuning_policies.o:      tuning_policies.c
			$(CC) $(CFLAGS)  -c tuning_policies.c 

#tau2vtf: 	tau2vtf.o
#		$(CXX) $(CFLAGS) tau2vtf.o -o tau2vtf -lm $(LEXTRAUTIL) -L$(VTFDIR)/$(TAU_ARCH)/lib -lvtf3 $(TAU_TRACE_INPUT_LIB)

#tau2vtf.o:	tau2vtf.cpp
#		$(CXX) $(CFLAGS)  tau2vtf.cpp -c -I$(VTFDIR)/include 

clean:
		/bin/rm -rf *.o $(SHAREDLIBS) 

install:	$(INSTALLDEST)/tuning_policies.so

program:        $(PROGRAM)
