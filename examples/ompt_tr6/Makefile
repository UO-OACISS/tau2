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
 

TAU_MAKEFILE ?=../../include/Makefile

include $(TAU_MAKEFILE)

#CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh -optKeepFiles -optNoCompInst -optLinkOnly
CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh -optCompInst
#CC		= opencc -fopenmp
#CC		= icc -pthread -DPTHREADS
#CC      = $(TAU_CC) -fopenmp

PRINT	      	= pr

RM 	      	= /bin/rm -rf

TARGET	      	= ompt_tr6_example

OBJS     	= ompt_tr6_example.o
CFLAGS=-g -O0 $(TAU_OPENMP_OPTION) 
LDFLAGS=-g $(TAU_OPENMP_OPTION)

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS) 
	$(CC) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

ompt_tr6_example.o: ompt_tr6_example.c
	$(CC) $(CFLAGS) -c $< -o $@
run: 
	tau_exec -T serial,ompt,tr6 -ompt ./$(TARGET)

clean: 	
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog
##############################################
