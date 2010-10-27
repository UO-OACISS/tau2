#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 2001  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
 
TAUROOTDIR	= ../..

CXX		= tau_cxx.sh

RM		= /bin/rm -f

TARGET	= app

##############################################
OBJS	= app.o  thr.o processing.o 

all:		$(TARGET)	


$(TARGET):	$(OBJS) 
	$(CXX) $(LDFLAGS) $(OBJS) -o $@

# Compilation rule
.cpp.o:
	tau_cxx.sh -c $< -DTAU_PTHREAD_BARRIER_AVAILABLE

clean: 	
	$(RM) $(OBJS) $(TARGET) $(OBJS:.o=.inst.cpp) \
	profile.* tautrace.* events.*
##############################################
