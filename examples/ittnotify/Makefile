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

CC      = $(TAU_CC)

PRINT	      	= pr

RM 	      	= /bin/rm -rf

TARGET	      	= matmult

OBJS     	= matmult.o matmult_initialize.o 
LIBS        = libittnotify.a
CFLAGS      = -g -O0 -no-pie $(TAU_OPENMP_OPTION)
LDFLAGS     = -g -no-pie $(TAU_OPENMP_OPTION)

##############################################

all:		$(TARGET)

install: 	$(TARGET)

$(TARGET):	$(OBJS) $(LIBS)
	$(CC) $(LDFLAGS) $(OBJS) $(LIBS) -o $@ $(LIBS)

matmult.o: matmult.c
	$(CC) $(CFLAGS) -c $< -o $@

matmult_initialize.o: matmult_initialize.c matmult_initialize.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) $(TARGET) core core.* profile.* scorep-* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog
##############################################
