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
CFLAGS = -ggdb3 -O0 -fopenmp -fPIC
LDFLAGS = -ggdb3 -O0 -fopenmp -rdynamic

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

testinst: $(TARGET)
	TAU_THROTTLE=1 TAU_SAMPLING=0 TAU_VERBOSE=1 OMP_NUM_THREADS=4 ./$(TARGET) 

test: $(TARGET)
	TAU_METRICS=TIME TAU_SAMPLING=1 TAU_VERBOSE=0 OMP_NUM_THREADS=4 TAU_EBS_SOURCE=TIME TAU_EBS_PERIOD=12345 TAU_EBS_KEEP_UNRESOLVED_ADDR=1 ./$(TARGET) 

testpapi: $(TARGET)
	TAU_METRICS=PAPI_TOT_CYC TAU_SAMPLING=1 TAU_VERBOSE=0 OMP_NUM_THREADS=4 TAU_EBS_SOURCE=PAPI_TOT_CYC TAU_EBS_PERIOD=12345678 TAU_EBS_KEEP_UNRESOLVED_ADDR=1 ./$(TARGET) 

testcallpath: $(TARGET)
	TAU_METRICS=PAPI_TOT_CYC TAU_VERBOSE=0 TAU_CALLPATH=1 TAU_CALLPATH_DEPTH=15 OMP_NUM_THREADS=4 ./$(TARGET) 
##############################################
