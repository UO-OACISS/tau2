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

PDTCPARSE       = $(PDTDIR)/$(PDTARCHDIR)/bin/cparse

PDTF90PARSE     = $(PDTDIR)/$(PDTARCHDIR)/bin/f95parse

TAUINSTR        = $(TAUROOTDIR)/$(CONFIG_ARCH)/bin/tau_instrumentor

OPARI_TOOL	= $(TAU_OPARI_TOOL)

CFLAGS 		= $(TAU_INCLUDE) $(TAU_DEFS)  

FFLAGS        	= $(TAU_F90_SUFFIX) -g

LIBS            =  $(TAU_LIBS)  $(TAU_FORTRANLIBS)  $(LEXTRA1)
#LIBS            = $(TAU_DISABLE)  $(TAU_FORTRANLIBS)

LDFLAGS	      	= $(USER_OPT) 

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

F90		= $(TAU_F90) 

TARGET	      	= mandel

EXTRAOBJS     	= 

##############################################

all:		$(TARGET) $(PDTF90PARSE) $(TAUINSTR) $(OPARI_TOOL)

install: 	$(TARGET)

$(TARGET):	ppm.o $(TARGET).o mytimer.o opari.tab.o
	@echo "*********************************************************"
	@echo "LINKING: "
	$(TAU_LINKER) $(LDFLAGS) $(TARGET).o ppm.o mytimer.o opari.tab.o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).f90 ppm.o 
	@echo "*********************************************************"
	@echo "Creating $(TARGET).o:"
	$(RM) opari.rc
	$(OPARI_TOOL) -nosrc -table opari.tab.c $*.f90 $*.pomp.f90
	$(PDTF90PARSE) $*.pomp.f90 -MPDT_MODULES
	$(TAUINSTR) $*.pomp.pdb $*.pomp.f90 -o $*.inst.f90
	$(F90) $(FFLAGS) -c $*.inst.f90 -o $@

ppm.o : ppm.f90
	@echo "*********************************************************"
	@echo "Creating ppm.o: "
	$(PDTF90PARSE) $<
	if [ -d PDT_MODULES ] ; then true; \
	  else mkdir PDT_MODULES ; fi
	if [ ! -f PPM.mod ] ; then true ; \
	  else mv PPM.mod PDT_MODULES ; fi 
	   
	$(TAUINSTR) $*.pdb $< -o $*.inst.f90
	$(F90) $(FFLAGS) -c $*.inst.f90 -o $@

opari.tab.o: $(TARGET).o
	@echo "*********************************************************"
	@echo "Creating opari.tab.o:"
	$(CC) $(CFLAGS) -c opari.tab.c

mytimer.o : mytimer.c
	@echo "*********************************************************"
	@echo "Creating mytimer.o:"
	$(PDTCPARSE) $<
	$(TAUINSTR) $*.pdb $< -o $*.inst.c
	$(CC) $(CFLAGS) -c $*.inst.c -o $@

clean: 	
	@echo "*********************************************************"
	@echo "Cleaning... :"
	$(RM) -r $(TARGET).o $(TARGET) mytimer.o ppm.o *.mod* *.pdb *.tab.* *.pomp.f90 *.inst.* opari.rc *.inc PDT_MODULES
##############################################

$(PDTF90PARSE):
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

$(OPARITOOL):
	@echo "*********************************************************"
	@echo "Configure TAU with -opari=<dir> configuration option to use"
	@echo "Opari OpenMP directive rewriting tool"
	@echo "ERROR: Cannot find $(OPARITOOL)"
	@echo "*********************************************************"
