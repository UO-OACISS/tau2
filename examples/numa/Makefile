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

CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh

PRINT	      	= pr

RM 	      	= /bin/rm -rf

TARGET	      	= numa

OBJS     	= numa.o 
LIBS		= -lnuma

CFLAGS=-g -O0 $(TAU_OPENMP_OPTION)

LDFLAGS=-g $(TAU_OPENMP_OPTION)

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS) 
	$(CC) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

numa.o: numa.c
	$(CC) $(CFLAGS) -c $< -o $@

run: 
	mpirun -np 4 tau_exec -T papi -numa ./$(TARGET)

clean: 	
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog
##############################################
