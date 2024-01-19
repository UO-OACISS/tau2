#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 2024  						   	   **
#*    University of Oregon                                                 **
#****************************************************************************
 
include ../../include/Makefile 
include ../Makefile



CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

HDF5_ROOT	= /storage/users/jalcaraz/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/hdf5-1.14.3-ct7dv7muxqd6beytmokixsfiblbc3qjq

CFLAGS		= -I$(HDF5_ROOT)/include
#CFLAGS        	= $(TAU_INCLUDE) $(TAU_DEFS)
#CFLAGS        	= 

LIBS		= -L$(HDF5_ROOT)/lib -lhdf5 -ldl -lm
#LIBS 	      	= $(TAU_LIBS) 
#LIBS 	      	= $(TAUROOTDIR)/src/Profile/TauHooks.o $(TAU_LIBS) -ldl

LDFLAGS	      	= $(USER_OPT)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= parallel2darray

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o
	$(CC) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).c
	$(CC) $(CFLAGS) -c $(TARGET).c

clean: 	
	$(RM) $(TARGET).o $(TARGET) profile.* 
##############################################
