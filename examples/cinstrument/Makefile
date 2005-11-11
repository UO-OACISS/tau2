#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################
 
TAUROOTDIR	= ../..

include $(TAUROOTDIR)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

PDTPARSE 	= $(PDTDIR)/$(PDTARCHDIR)/bin/cparse

TAUINSTR 	= $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_instrumentor

CFLAGS        	= $(TAU_INCLUDE) $(TAU_DEFS) -w -UTAU_MPI

LIBS 	      	= $(TAU_LIBS) 

LDFLAGS	      	= $(USER_OPT)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= klargest

EXTRAOBJS     	= 

##############################################
# Original Rules
##############################################
#all:		$(TARGET)	
#
#install: 	$(TARGET)
#
#$(TARGET):	$(TARGET).o
#	$(CXX) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)
#
#$(TARGET).o : $(TARGET).c
#	$(CXX) $(CFLAGS) -c $(TARGET).c
#
#clean: 	
#	$(RM) $(TARGET).o $(TARGET)
##############################################



##############################################
# Modified Rules
##############################################

all: 	$(TARGET) $(PDTPARSE) $(TAUINSTR)

$(TARGET): $(TARGET).o 
	$(CXX) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

# Use the instrumented source code to generate the object code
$(TARGET).o : $(TARGET).inst.c 
	$(CC) -c $(CFLAGS) $(TARGET).inst.c  -o $(TARGET).o

# Generate the instrumented source from the original source and the pdb file
$(TARGET).inst.c : $(TARGET).pdb $(TARGET).c $(TAUINSTR) 
	$(TAUINSTR) $(TARGET).pdb $(TARGET).c -o $(TARGET).inst.c

# Parse the source file to generate the pdb file
$(TARGET).pdb : $(PDTPARSE) $(TARGET).c 
	$(PDTPARSE) $(TARGET).c $(CFLAGS) 

clean: 
	$(RM) $(TARGET).o $(TARGET).inst.c $(TARGET) $(TARGET).pdb
##############################################

$(PDTPARSE):
	@echo "*********************************************************"
	@echo "Download and Install Program Database Toolkit "
	@echo "ERROR: Cannot find $(PDTPARSE)"
	@echo "*********************************************************"

$(TAUINSTR):
	@echo "*********************************************************"
	@echo "Configure TAU with -pdt=<dir> configuration option to use" 
	@echo "C++ instrumentation with PDT"
	@echo "ERROR: Cannot find $(TAUINSTR)"
	@echo "*********************************************************"

