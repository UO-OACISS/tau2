#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 2001  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
 
TAUROOTDIR	= ../..

include $(TAUROOTDIR)/include/Makefile

USE_TAU = 1
# Comment above line to disable TAU 


CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

CPARSE          = $(PDTDIR)/$(PDTARCHDIR)/bin/cparse

F90PARSE        = $(PDTDIR)/$(PDTARCHDIR)/bin/f90parse

TAUINSTR        = $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_instrumentor

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)

#FFLAGS          = -W132

LIBS            = $(TAU_LIBS) $(TAU_FORTRANLIBS) 

LDFLAGS	      	= $(USER_OPT)  $(TAU_LDFLAGS)

TARGET	      	= app

EXTRAOBJS     	= 

RM		= /bin/rm -f

##############################################
ifdef USE_TAU
COMPC_RULE = $(CPARSE) $< ; \
	$(TAUINSTR) $*.pdb $< -o $*.inst.c ; \
	$(CC) $(CFLAGS) -c $*.inst.c -o $@ ; \
	rm -f $*.pdb ;      
COMPF_RULE1 = $(F90PARSE) $< ; \
	$(TAUINSTR) $*.pdb $< -o $*.inst.f90 -g Fort1; \
	$(TAU_F90) $(FFLAGS) -c $*.inst.f90 -o $@ ; \
	rm -f $*.pdb ;      
COMPF_RULE2 = $(F90PARSE) $< ; \
	$(TAUINSTR) $*.pdb $< -o $*.inst.f90 -g Fort2; \
	$(TAU_F90) $(FFLAGS) -c $*.inst.f90 -o $@ ; \
	rm -f $*.pdb ;      
else
# DISABLE TAU INSTRUMENTATION
TAU_DEFS = 
COMPC_RULE = $(CC) $(CFLAGS) -c $< -o $@ ;
COMPF_RULE1 = $(TAU_F90) $(FFLAGS) -c $< -o $@ ;
COMPF_RULE2 = $(TAU_F90) $(FFLAGS) -c $< -o $@ ;
LIBS = $(TAU_FORTRANLIBS) 
endif

##############################################

all:		$(TARGET)	

install: 	$(TARGET)


OBJS	= main.o f1.o f2.o

$(TARGET):	$(OBJS) 
	$(TAU_LINKER) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

# Compilation rule
main.o : main.c
	$(COMPC_RULE)
f1.o : f1.f90
	$(COMPF_RULE1)
f2.o : f2.f90
	$(COMPF_RULE2)

clean: 	
	$(RM) $(OBJS) $(TARGET) f1.inst.f90 f2.inst.f90  main.inst.c core\
	profile.* tautrace.* events.* *~
##############################################
