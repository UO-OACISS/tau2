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
JDKDIR		= /usr/local/packages/jdk1.2

include $(TAUROOTDIR)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

CEXTRA		= 

CFLAGS          = $(TAU_INCLUDE) -I$(JDKDIR)/include -I$(JDKDIR)/include/linux $(TAU_DEFS) $(CEXTRA)

RM		= /bin/rm -f

AR		= $(TAU_CXX)

#ARFLAGS		= -shared -o 
# On Solaris, use -G, on linux, use -shared
ARFLAGS		= -o 
# KCC doesn't need any special -G or -shared flag to build a .so

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

TARGET		= libTAU.so
##############################################

all : 		$(TARGET)

install:	$(INSTALLDEST)/$(TARGET) 

$(TARGET) : 	TauJava.o 
	$(PRELINK_PHASE)
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
