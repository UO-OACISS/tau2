###############################################################################
## Makefile for just the basic utility tools #
###############################################################################

include ../include/Makefile
include include/Makefile

HDRS	      =

MAKEFILE      = Makefile

OBJS	      = control_policies.o

PRINT	      = pr

SHAREDLIBS    = control_policies

all:		$(SHAREDLIBS)

printvar:	
		@echo "CFLAGS "$(CFLAGS)" LEXTRAUTIL "$(LEXTRAUTIL)""

control_policies.so:	control_policies.o
			$(CC) -shared -fPIC -o control_policies.so control_policies.o

control_policies.o:     control_policies.c
			$(CC) $(CFLAGS) -fPIC control_policies.c 

#tau2vtf: 	tau2vtf.o
#		$(CXX) $(CFLAGS) tau2vtf.o -o tau2vtf -lm $(LEXTRAUTIL) -L$(VTFDIR)/$(TAU_ARCH)/lib -lvtf3 $(TAU_TRACE_INPUT_LIB)

#tau2vtf.o:	tau2vtf.cpp
#		$(CXX) $(CFLAGS)  tau2vtf.cpp -c -I$(VTFDIR)/include 

clean:
		/bin/rm -rf *.o $(SHAREDLIBS) 

install:	$(INSTALLDEST)/control_policies

program:        $(PROGRAM)
