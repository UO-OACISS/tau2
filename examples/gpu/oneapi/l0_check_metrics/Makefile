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
 

CC      = icpx
PRINT	      	= pr
RM 	      	= /bin/rm -rf
TARGET	      	= metric_avail
OBJS     	= metric_avail.o

CFLAGS=-I./utils -std=c++11
LDFLAGS=-lze_loader


##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS) 
	$(CC) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

metric_avail.o: metric_avail.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJS) $(TARGET) 
##############################################
