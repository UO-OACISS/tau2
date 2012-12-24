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

CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh -optKeepFiles -optTauSelectFile="./select.tau" -optShared
CFLAGS = -g -O0 -fopenmp -fPIC
LDFLAGS = -g -O0 -fopenmp -rdynamic

PRINT	      	= pr

RM 	      	= /bin/rm -rf

TARGET	      	= matmult

OBJS     	= matmult.o matmult_initialize.o

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAU.so
	$(CC) $(LDFLAGS) $(OBJS) -o $@ $(LIBS) -Wl,-rpath=/home3/khuck/src/tau2/x86_64/lib/shared$(TAU_CONFIG)

matmult.o: matmult.c
	$(CC) $(CFLAGS) -c $< -o $@

matmult_initialize.o: matmult_initialize.c matmult_initialize.h
	$(CC) $(CFLAGS) -c $< -o $@

notau:
	gcc $(CFLAGS) *.c -o $(TARGET)

clean: 	
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o

test: $(TARGET)
	OMP_NUM_THREADS=2 ./$(TARGET) 
##############################################
