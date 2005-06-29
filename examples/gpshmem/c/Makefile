#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################
 
TAUROOTDIR	= ../../..

include $(TAUROOTDIR)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

CFLAGS	= $(TAU_SHMEM_INCLUDE)  $(TAU_DEFS) $(TAU_INCLUDE)

# Pass these options using -shmemlibrary='-lgpshmem -L... -larmci ...' to 
# TAU's configure script.
#LIBS = $(TAU_SHMEM_LIBS)  -L/usr/local/packages/armci-1.1/lib/LINUX -L/usr/lib/gcc-lib/i386-pc-linux/3.2.3/ -lstdc++ -lgcc_s -larmci -L/opt/mpich-mpd/gnu/lib -lmpich $(TAU_LIBS) $(TAU_CXXLIBS) 
LIBS = $(TAU_SHMEM_LIBS) $(TAU_LIBS) $(TAU_CXXLIBS) 

MAKEFILE      	= Makefile

RM 	      	= /bin/rm -f

TARGET	      	= fox

##############################################
# Original Rules
##############################################
all:		$(TARGET)	

$(TARGET):	$(TARGET).o
	$(CXX) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).c
	$(CC) $(CFLAGS) -c $(TARGET).c

clean:
	$(RM) $(TARGET) $(TARGET).o $(TARGET).pdb *.vpt *.gz *.slog2 *.trc *.edf

##############################################
