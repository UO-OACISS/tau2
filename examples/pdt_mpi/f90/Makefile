#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 2001  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
 
TAUROOTDIR	= ../../..

.SUFFIXES : .f90 

include $(TAUROOTDIR)/include/Makefile

USE_TAU = 1
# Comment above line to disable TAU 


CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

F90		= $(TAU_F90)

PDTF90PARSE        = $(PDTDIR)/$(PDTARCHDIR)/bin/f90parse

TAUINSTR        = $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_instrumentor

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)

FFLAGS          = $(TAU_INCLUDE) $(TAU_MPI_INCLUDE) $(TAU_F90_SUFFIX)

LIBS            = $(TAU_MPI_FLIBS) $(TAU_LIBS) $(TAU_CXXLIBS) 

LDFLAGS	      	= $(USER_OPT)  $(TAU_LDFLAGS)

LINKER 		= $(TAU_LINKER)
LINKER 		= $(TAU_F90)

TARGET	      	= ring

EXTRAOBJS     	= 

RM		= /bin/rm -f

##############################################
ifdef USE_TAU
COMP_RULE = $(PDTF90PARSE) $< $(FFLAGS); \
	$(TAUINSTR) $*.pdb $< -o $*.inst.f90 ; \
	$(F90) $(FFLAGS) -c $*.inst.f90 -o $@ ; \
	rm -f $*.pdb ;      
else
# DISABLE TAU INSTRUMENTATION
TAU_DEFS = 
# Don't use TAU MPI wrapper library
TAU_MPI_LIBS = -L/usr/local/lib -lmpich -lgm
TAU_LIBS = 
TAU_WRAPPER_LIB = 
TAU_INCLUDE = 
COMP_RULE = $(F90) $(FFLAGS) -c $< -o $@ ;
LIBS = $(TAU_MPI_LIBS)
LINKER = $(F90)
endif

AR = ar
ARFLAGS = rcv


##############################################

all:		$(TARGET)	

install: 	$(TARGET)


OBJS	= ring.o

$(TARGET):	$(OBJS)
	$(LINKER) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

# Compilation rule
.f90.o:
	$(COMP_RULE)


#
#$(TARGET).o : $(TARGET).cpp
#	$(CXX) $(CFLAGS) -c $(TARGET).c

clean: 	
	$(RM) $(OBJS) $(TARGET) $(OBJS:.o=.inst.f90) \
	profile.* tautrace.* events.* *.elg
##############################################
