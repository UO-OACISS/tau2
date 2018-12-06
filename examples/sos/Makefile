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

TARGET	      	= matmult

MPI_OPTS = $(shell mpicc -show | awk '{$$1 = ""; print $$0}')
OBJS     	= matmult.o matmult_initialize.o
CFLAGS=-g -O0 $(TAU_OPENMP_OPTION) -DSOS_MPI $(MPI_OPTS) -I../../include
LDFLAGS=-dynamic -g $(TAU_OPENMP_OPTION)

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS) $(TAU_LIB_DIR)/libTAU.so Makefile
	$(CC) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

matmult.o: matmult.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

matmult_initialize.o: matmult_initialize.c matmult_initialize.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog dump.* sosd.* core.*
##############################################
