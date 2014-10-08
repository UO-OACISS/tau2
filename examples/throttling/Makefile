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
CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh
#CC		= opencc -fopenmp
#CC		= icc -pthread -DPTHREADS
#CC      = $(TAU_CC) -fopenmp

PRINT	      	= pr

RM 	      	= /bin/rm -rf

TARGET	      	= throttled

OBJS     	= throttled.o
CFLAGS=-g $(TAU_OPENMP_OPTION)
LDFLAGS=-g $(TAU_OPENMP_OPTION)

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS) $(TAU_LIB_DIR)/libTAU$(TAU_SHLIBX)
	$(CC) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

throttled.o: throttled.c
	$(CC) $(CFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog
##############################################
