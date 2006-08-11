#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 2001  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
 
TAUROOTDIR	= ../../..

.SUFFIXES : .f90 

include $(TAU_TEST_MAKEFILE)

USE_TAU = 1
# Comment above line to disable TAU 


CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

F90		= $(TAU_F90)

PDTF90PARSE        = $(PDTDIR)/$(PDTARCHDIR)/bin/f95parse

TAUINSTR        = $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_instrumentor

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)

FFLAGS          = $(TAU_INCLUDE) $(TAU_MPI_INCLUDE) $(TAU_F90_SUFFIX)

LIBS            = $(TAU_MPI_FLIBS) $(TAU_LIBS) $(TAU_CXXLIBS) 

LDFLAGS	      	= $(USER_OPT)  $(TAU_LDFLAGS)

LINKER 		= $(TAU_LINKER)
LINKER 		= $(TAU_F90)

TARGET	      	= simple

EXTRAOBJS     	= 

RM		= /bin/rm -f


AR = ar
ARFLAGS = rcv


##############################################

all:		$(TARGET)	

install: 	$(TARGET)


OBJS	= simple.o

$(TARGET):	$(OBJS)
	$(LINKER) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

# Compilation rule
.f90.o:
	$(F90) $(FFLAGS) -c $(TARGET).f90


clean: 	
	$(RM) $(OBJS) $(TARGET) $(OBJS:.o=.inst.f90) \
	profile.* tautrace.* events.* *.elg
##############################################
