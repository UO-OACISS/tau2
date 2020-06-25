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

TAU_MAKEFILE ?=${tauroot}/include/Makefile

include $(TAU_MAKEFILE)

CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh
RM 	    = /bin/rm -rf
TARGET	= matmult
OBJS    = matmult.o matmult_initialize.o
CFLAGS  =-g $(TAU_OPENMP_OPTION)
LDFLAGS =-g $(TAU_OPENMP_OPTION)

##############################################

all: clean test

test:
	tau_python -T serial -tau-python-interpreter=python3 ./firstprime.py
	test -f profile.0.0.0 || exit 1
	$(PPROF_CMD)

#$(shell if [ ! -f profile.0.0.0 ] then ; exit 1 )

clean:
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog

##############################################
