#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 2001  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
 
TAUROOTDIR	= ../../..

include $(TAUROOTDIR)/include/Makefile

USE_TAU = 1
# Comment above line to disable TAU 


CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

PDTCPARSE        = $(PDTDIR)/$(CONFIG_ARCH)/bin/cparse

TAUINSTR        = $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_instrumentor

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)

LIBS            = $(TAU_MPI_LIBS) $(TAU_LIBS) $(LEXTRA1) $(EXTRALIBS) -lm -lgm

LDFLAGS	      	= $(USER_OPT)  $(TAU_LDFLAGS)

TARGET	      	= ring

EXTRAOBJS     	= 

RM		= /bin/rm -f

##############################################
ifdef USE_TAU
COMP_RULE = $(PDTCPARSE) $< $(CFLAGS); \
	$(TAUINSTR) $*.pdb $< -o $*.inst.c -g "COMPUTATION" ; \
	$(CC) $(CFLAGS) -c $*.inst.c -o $@ ; \
	rm -f $*.pdb ;      
else
# DISABLE TAU INSTRUMENTATION
TAU_DEFS = 
# Don't use TAU MPI wrapper library
TAU_MPI_LIBS = -L/usr/local/lib -lmpich
TAU_LIBS = 
TAU_WRAPPER_LIB = 
TAU_INCLUDE = 
COMP_RULE = $(CC) $(CFLAGS) -c $< -o $@ ;
endif

AR = ar
ARFLAGS = rcv


##############################################

all:		$(TARGET)	

install: 	$(TARGET)


OBJS	= ring.o

$(TARGET):	$(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

# Compilation rule
.c.o:
	$(COMP_RULE)


#
#$(TARGET).o : $(TARGET).cpp
#	$(CXX) $(CFLAGS) -c $(TARGET).c

clean: 	
	$(RM) $(OBJS) $(TARGET) $(OBJS:.o=.inst.c) \
	profile.* tautrace.* events.* *.elg
##############################################
