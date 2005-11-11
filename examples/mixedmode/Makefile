#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
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

PDTPARSE        = $(PDTDIR)/$(PDTARCHDIR)/bin/cxxparse

TAUINSTR        = $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_instrumentor

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)

LIBS            = $(TAU_MPI_LIBS) $(TAU_LIBS) -lm

LDFLAGS	      	= $(USER_OPT)  $(TAU_LDFLAGS)

TARGET	      	= app

EXTRAOBJS     	= 

TAU_WRAPPER_LIB = libTauWrapper.a 

RM		= /bin/rm -f

##############################################
ifdef USE_TAU
COMP_RULE = $(PDTPARSE) $< $(CFLAGS); \
	$(TAUINSTR) $*.pdb $< -o $*.inst.cpp ; \
	$(CXX) $(CFLAGS) -c $*.inst.cpp -o $@ ; \
	rm -f $*.pdb ;      
else
# DISABLE TAU INSTRUMENTATION
TAU_DEFS = 
# Don't use TAU MPI wrapper library
TAU_MPI_LIBS = -L/usr/local/packages/mpich/lib -lmpich
TAU_LIBS = 
TAU_WRAPPER_LIB = 
COMP_RULE = $(CXX) $(CFLAGS) -c $< -o $@ ;
endif

AR = ar
ARFLAGS = rcv


############# For KAI KCC compiler, replace AR accordingly #######
#KAI#AR                 = $(CXX)                           #ENDIF#
#KAI#ARFLAGS                    = --one_per -o                     #ENDIF#
##################################################################

############# For SGI CC compiler, replace AR accordingly #######
#SGICC#AR                       = $(CXX) -ar               #ENDIF#
#SGICC#ARFLAGS                  = -o                       #ENDIF#
##################################################################

############# For FUJITSU FCC compiler, replace AR accordingly #######
#FUJITSU#AR                     = $(CXX) --ar              #ENDIF#
#FUJITSU#ARFLAGS                = -o                       #ENDIF#
##################################################################

############# For SUN CC compiler, replace AR accordingly #######
#SOL2CC#AR                      = $(CXX) -xar              #ENDIF#
#SOL2CC#ARFLAGS                 = -o                       #ENDIF#
##################################################################


##############################################

all:		$(TARGET)	

install: 	$(TARGET)


OBJS	= app.o  thr.o processing.o 

$(TARGET):	$(OBJS) $(TAU_WRAPPER_LIB)
	$(CXX) $(LDFLAGS) $(OBJS) $(TAU_WRAPPER_LIB) -o $@ $(LIBS)

# Compilation rule
.cpp.o:
	$(COMP_RULE)

# Exception to the above rule:
# Thread file cannot use PDT (static analysis limits: TAU_REGISTER_THREAD )
thr.o: thr.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

# Make the library 
libTauWrapper.a: TauWrapper.o
	$(AR) $(ARFLAGS) libTauWrapper.a TauWrapper.o

TauWrapper.o: TauWrapper.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

#
#$(TARGET).o : $(TARGET).cpp
#	$(CXX) $(CFLAGS) -c $(TARGET).cpp

clean: 	
	$(RM) $(OBJS) $(TARGET) $(OBJS:.o=.inst.cpp) \
	TauWrapper.o libTauWrapper.a  profile.* tautrace.* events.*
##############################################
