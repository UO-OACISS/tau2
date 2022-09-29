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


TAU_MAKEFILE ?=../../../include/Makefile

include $(TAU_MAKEFILE)

#CXX		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cxx.sh
#CXX             = tau_cxx.sh
CXX		= $(TAU_CXX)
PRINT	      	= pr
CFLAGS = -g $(TAU_OPENMP_OPTION)
LDFLAGS = -g $(TAU_OPENMP_OPTION)
RM 	      	= /bin/rm -rf

TARGET	      	= multitask_openmp

OBJS     	= multitask_openmp.o

##############################################

all:		$(TARGET)

install: 	$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

multitask_openmp.o: multitask_openmp.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

#matmult_initialize.o: matmult_initialize.c matmult_initialize.h
#	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o
##############################################
