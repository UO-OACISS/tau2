#****************************************************************************
#*                      TAU Portable Profiling Package                     **
#*                      http://www.cs.uoregon.edu/research/paracomp/tau    **
#****************************************************************************

OTFDIR=
CONFIG_ARCH=default
CXX=g++
CC=gcc
LIBS=
INCLUDES= -I.
OTF_SOURCE=$(OTFDIR)/include
OTF_UTIL_INCLUDE=$(OTFDIR)/include
RM=/bin/rm -f

include ../../include/Makefile

#The following are variables for all the files and their associated variations.
HANDLERS=handlers.o handlers.*.~*
READTRACE=readTrace.o readTrace.*.~*
MAKEFILE=Makefile.~*

CXXFLAGS = $(USER_OPT)
OTF2FLAGS = -DNOCROSS_BUILD -D__STDC_FORMAT_MACROS -DFRONTEND_BUILD  -fpermissive

OTFLIBS=-L$(OTFDIR)/lib -lotf2 -lrt

all:    otf2profile

install:       $(TAU_BIN_DIR)/otf2profile

$(TAU_BIN_DIR)/otf2profile: otf2profile
	/bin/cp $< $(TAU_BIN_DIR)

otf2profile:     handlers.o otf2_hash_table.o otf2_vector.o readOTF2.o trace2profile.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ handlers.o otf2_hash_table.o otf2_vector.o readOTF2.o trace2profile.o $(LIBS) $(OTFLIBS)    

handlers.o: handlers.h trace2profile.h handlers.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c handlers.cpp

trace2profile.o: trace2profile.h trace2profile.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c trace2profile.cpp
	
readOTF2.o: handlers.h trace2profile.h readOTF2.cpp
	$(CXX) $(CXXFLAGS) $(OTF2FLAGS) $(INCLUDES) -I$(OTFDIR)/include -I$(OTF_SOURCE) -c readOTF2.cpp
	
otf2_hash_table.o: otf2_hash_table.h
	$(CC) $(CXXFLAGS) $(OTF2FLAGS) $(INCLUDES) -I$(OTFDIR)/include -I$(OTF_SOURCE) -I$(OTF_UTIL_INCLUDE) -c otf2_hash_table.c
	
otf2_vector.o: otf2_vector.h
	$(CC) $(CXXFLAGS) $(OTF2FLAGS) $(INCLUDES) -I$(OTFDIR)/include -I$(OTF_SOURCE) -I$(OTF_UTIL_INCLUDE) -c otf2_vector.c

clean:
	$(RM) $(TRACE2PROFILE) $(HANDLERS) $(READTRACE) $(MAKEFILE) *.o otf2profile

