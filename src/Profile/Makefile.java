#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################

TAUROOTDIR	= ../..
JDKROOT		= /research/paraducks/apps/jdk1.2.2

include $(TAUROOTDIR)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

CFLAGS          = $(TAU_INCLUDE) -I$(JDKROOT)/include -I$(JDKROOT)/include/solaris $(TAU_DEFS) 

RM		= /bin/rm -f

AR		= $(TAU_CXX)

ARFLAGS		= -G -o

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

TARGET		= libTAU.so
##############################################

all : 		$(TARGET)

install:	$(INSTALLDEST)/$(TARGET) 

$(TARGET) : 	TauJava.o 
	$(AR) $(ARFLAGS) $(TARGET) TauJava.o $(TAU_LIBS)

TauJava.o :	TauJava.cpp
	$(CXX) $(CFLAGS) -c TauJava.cpp

$(INSTALLDEST)/$(TARGET): $(TARGET)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)
clean: 	
	$(RM) core TauJava.o $(TARGET)
##############################################
