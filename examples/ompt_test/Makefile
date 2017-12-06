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

CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh -optLinkOnly
CFLAGS=-g -O2 $(TAU_OPENMP_OPTION) 
LDFLAGS=-g -O2 $(TAU_OPENMP_OPTION)
SRC=$(wildcard *.c)
TESTS=$(patsubst %.c, %_test, $(SRC))

##############################################

all: $(TESTS)

$(TESTS): %_test : %.o
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJ): %.o : %.c
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f *.o profile.* *_test

.DEFAULT: all

##############################################
