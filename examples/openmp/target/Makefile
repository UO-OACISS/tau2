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

CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh
CC      = $(TAU_CC)
PRINT	      	= pr
RM 	      	= /bin/rm -rf
TARGET	      	= matmult
OBJS     	= matmult.o
#TARGET_FLAGS =-fopenmp -Xopenmp-target -march=sm_50 -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA}
#TARGET_FLAGS =-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA}
#TARGET_FLAGS =-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda,x86_64
TARGET_FLAGS =-qnextgen -fiopenmp -fopenmp-targets=spir64 
TARGET_FLAGS =-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
#TARGET_FLAGS =-fopenmp  -mp=gpu,ompt -Minfo=mp 
#TARGET_FLAGS =-fopenmp -lomptarget
CFLAGS=-g -O2 $(TARGET_FLAGS)
LDFLAGS=-g $(TARGET_FLAGS)


##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS) 
	$(CC) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

matmult.o: matmult.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog
##############################################
