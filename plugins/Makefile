###############################################################################
## Makefile for just the basic utility tools #
###############################################################################

TAUROOT       = /home/users/aurelem/tau/tau2_beacon 
include ../include/Makefile
#include include/Makefile

HDRS	      =

MAKEFILE      = Makefile

TAUINC        = -I$(TAUROOT)/include -I$(TAUROOT)/include/Profile -I$(TAUROOT)/plugins/json-c/inst/include/json-c

CC            = $(TAU_CC)

#CFLAGS        = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAUINC) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -fPIC
CFLAGS        = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAUINC) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -g 

#LDFLAGS       =  -L/home/users/aurelem/tau/tau2_beacon/x86_64/lib -ltau-mvapich2-beacon-mpi-pdt-mpit -L/home/users/aurelem/tau/tau2_beacon/x86_64/lib -lTauMpi-mvapich2-beacon-mpi-pdt-mpit
LDFLAGS       =  -L/home/users/aurelem/tau/tau2_beacon/plugins/json-c/inst/lib -ljson-c -L/home/users/aurelem/tau/tau2_beacon/x86_64/lib -ltau-mvapich2-beacon-mpi-pdt-mpit -L/home/users/aurelem/tau/tau2_beacon/x86_64/lib -lTauMpi-mvapich2-beacon-mpi-pdt-mpit

OBJS	      = tuning_policies.o tuning_policies_json.o alpha.o beta.o delta.o gamma.o

PRINT	      = pr

SHAREDLIBS    = tuning_policies.so alpha.so beta.so delta.so gamma.so
EXE           = tuning_policies_json

all:		$(SHAREDLIBS) 

printvar:	
		@echo "CFLAGS "$(CFLAGS)" LEXTRAUTIL "$(LEXTRAUTIL)""

tuning_policies.so:	tuning_policies.o
			$(CC) -shared -fPIC -o tuning_policies.so tuning_policies.o

tuning_policies.o:      tuning_policies.c
			$(CC) $(CFLAGS)  -c tuning_policies.c 

alpha.so:		alpha.o
			$(CC) -shared -fPIC -o alpha.so alpha.o

alpha.o:	 	alpha.c
			$(CC) $(CFLAGS)  -c alpha.c 

beta.so:		beta.o
			$(CC) -shared -fPIC -o beta.so beta.o

beta.o:		 	beta.c
			$(CC) $(CFLAGS)  -c beta.c 

delta.so:		delta.o
			$(CC) -shared -fPIC -o beta.so beta.o

delta.o:		delta.c
			$(CC) $(CFLAGS)  -c delta.c 

gamma.so:		gamma.o
			$(CC) -shared -fPIC -o gamma.so gamma.o

gamma.o:		gamma.c
			$(CC) $(CFLAGS)  -c gamma.c 



#tuning_policies_json:	tuning_policies_json_test.o
#			$(CC) $(LDFLAGS) -o tuning_policies_json tuning_policies_json_test.o

tuning_policies_json_test.o: tuning_policies_json_test.c
			$(CC) $(CFLAGS) -c tuning_policies_json_test.c 

#tau2vtf: 	tau2vtf.o
#		$(CXX) $(CFLAGS) tau2vtf.o -o tau2vtf -lm $(LEXTRAUTIL) -L$(VTFDIR)/$(TAU_ARCH)/lib -lvtf3 $(TAU_TRACE_INPUT_LIB)

#tau2vtf.o:	tau2vtf.cpp
#		$(CXX) $(CFLAGS)  tau2vtf.cpp -c -I$(VTFDIR)/include 

clean:
		/bin/rm -rf *.o $(SHAREDLIBS) 

install:	$(INSTALLDEST)/tuning_policies.so $(INSTALLDEST)/alpha.so $(INSTALLDEST)/beta.so $(INSTALLDEST)/delta.so $(INSTALLDEST)/gamma.so 

program:        $(PROGRAM)
