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
 
TAUROOTDIR	= ../..

include $(TAUROOTDIR)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

PDTPARSE 	= $(PDTDIR)/$(PDTARCHDIR)/bin/cxxparse

TAUINSTR 	= $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_instrumentor

TAUREDUCE	= $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_reduce

TAUPPROF	= $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/pprof

CFLAGS        	= $(TAU_INCLUDE) $(TAU_DEFS)

LIBS 	      	= $(TAU_LIBS) $(LEXTRA1)

LDFLAGS	      	= $(USER_OPT)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= klargest

ARGS		= 1000000 2324

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
#$(TARGET).o : $(TARGET).cpp
#	$(CXX) $(CFLAGS) -c $(TARGET).cpp
#
#clean: 	

##############################################



##############################################
# Modified Rules
##############################################

all: 	showdata $(TARGET) $(PDTPARSE) $(TAUINSTR)

showdata: run2
	$(TAUPPROF)

run2: $(TARGET).refined
	./$(TARGET).refined $(ARGS)

$(TARGET).refined: $(TARGET).refined.o 
	$(CXX) $(LDFLAGS) $< -o $@ $(LIBS)
	
$(TARGET).refined.o: $(TARGET).inst.refined.cpp 
	$(CXX) -c $(CFLAGS) $<  -o $@

$(TARGET).inst.refined.cpp: select.refined $(TARGET).pdb
	$(TAUINSTR) $(TARGET).pdb $(TARGET).cpp -o $(TARGET).inst.refined.cpp -f select.refined

select.refined: pprof.dat
	$(TAUREDUCE) -f pprof.dat -r rules.prop -o select.refined

pprof.dat: run1
	$(TAUPPROF) -d > pprof.dat

run1:	$(TARGET)
	./$(TARGET) $(ARGS)
	

$(TARGET): $(TARGET).o 
	$(CXX) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

# Use the instrumented source code to generate the object code
$(TARGET).o : $(TARGET).inst.cpp 
	$(CXX) -c $(CFLAGS) $(TARGET).inst.cpp  -o $(TARGET).o

# Generate the instrumented source from the original source and the pdb file
$(TARGET).inst.cpp : $(TARGET).pdb $(TARGET).cpp $(TAUINSTR) 
	$(TAUINSTR) $(TARGET).pdb $(TARGET).cpp -o $(TARGET).inst.cpp 

# Parse the source file to generate the pdb file
$(TARGET).pdb : $(PDTPARSE) $(TARGET).cpp 
	$(PDTPARSE) $(TARGET).cpp $(CFLAGS) 

clean: 
	$(RM) $(TARGET).o $(TARGET).inst.cpp $(TARGET) $(TARGET).pdb *.refined* profile* pprof.dat
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

