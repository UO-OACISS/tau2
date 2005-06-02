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

F90		= $(TAU_COMPILER) $(TAU_F90) 

CXX		= $(TAU_COMPILER) $(TAU_CXX) 

CC		= $(TAU_COMPILER) $(TAU_CC) 

OPARI_TOOL	= $(TAU_OPARI_TOOL)

CFLAGS 		= 

FFLAGS        	= $(TAU_F90_SUFFIX) 

LIBS            =  
#LIBS            = $(TAU_DISABLE)  

LDFLAGS	      	= $(USER_OPT) 

MAKEFILE      	= Makefile

RM 	      	= /bin/rm -f

TARGET	      	= mandel

##############################################

all:		$(TARGET) $(OPARI_TOOL)

install: 	$(TARGET)

$(TARGET):	ppm.o $(TARGET).o mytimer.o opari.tab.o
	@echo "*********************************************************"
	@echo "LINKING: "
	$(F90) $(LDFLAGS) $(TARGET).o ppm.o mytimer.o opari.tab.o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).f90 ppm.o 
	@echo "*********************************************************"
	@echo "Creating $(TARGET).o:"
	$(RM) opari.rc
	$(OPARI_TOOL) -nosrc -table opari.tab.c $*.f90 $*.pomp.f90
	$(F90) $(FFLAGS) -c $*.pomp.f90 -o $@

ppm.o : ppm.f90
	@echo "*********************************************************"
	@echo "Creating ppm.o: "
	$(OPARI_TOOL) -nosrc -table opari.tab.c $*.f90 $*.pomp.f90
	$(F90) $(FFLAGS) -c $*.pomp.f90 -o $@

opari.tab.o: $(TARGET).o
	@echo "*********************************************************"
	@echo "Creating opari.tab.o:"
	$(CC) $(CFLAGS) -c opari.tab.c

mytimer.o : mytimer.c
	@echo "*********************************************************"
	@echo "Creating mytimer.o:"
	$(OPARI_TOOL) -nosrc -table opari.tab.c $*.c $*.pomp.c
	$(CC) $(CFLAGS) -c $*.pomp.c -o $@

clean: 	
	@echo "*********************************************************"
	@echo "Cleaning... :"
	$(RM) -r $(TARGET).o $(TARGET) mytimer.o ppm.o *.mod* *.pdb *.tab.* *.pomp.f90 *.inst.* opari.rc *.inc *.pomp.c
##############################################

$(OPARITOOL):
	@echo "*********************************************************"
	@echo "Configure TAU with -opari=<dir> configuration option to use"
	@echo "Opari OpenMP directive rewriting tool"
	@echo "ERROR: Cannot find $(OPARITOOL)"
	@echo "*********************************************************"
