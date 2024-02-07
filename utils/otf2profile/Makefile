#****************************************************************************
#*                      TAU Portable Profiling Package                     **
#*                      http://www.cs.uoregon.edu/research/paracomp/tau    **
#****************************************************************************

include ../include/Makefile
include ../../include/Makefile

INCLUDES=-I.
OTF_SOURCE=$(OTFDIR)/include
OTF_UTIL_INCLUDE=$(OTFDIR)/include

#The following are variables for all the files and their associated variations.
HANDLERS=handlers.o handlers.*.~*
READTRACE=readTrace.o readTrace.*.~*
MAKEFILE=Makefile.~*

CXXFLAGS = $(USER_OPT) 
OTF2FLAGS = -DNOCROSS_BUILD -D__STDC_FORMAT_MACROS -DFRONTEND_BUILD 

OTFLIBS=-L$(OTFLIB) -lotf2 $(TAU_LINKER_OPT16)  #Only include -lrt (opt16) when it is found by TAU

all:    otf2profile

install:       $(TAU_BIN_DIR)/otf2profile

$(TAU_BIN_DIR)/otf2profile: otf2profile
	/bin/cp $< $(TAU_BIN_DIR)

otf2profile:    handlers.o otf2_hash_table.o otf2_vector.o UTILS_Error.o UTILS_IO_Tools.o  otf2profile.o trace2profile.o handlers.o
		$(TAU_CXX_FE) $(CXXFLAGS) -o $@  otf2_hash_table.o otf2profile.o trace2profile.o handlers.o otf2_vector.o UTILS_Error.o UTILS_IO_Tools.o $(LIBS) $(OTFLIBS) 
	
otf2profile.o: otf2profile.cpp 
	$(TAU_CXX_FE) $(CXXFLAGS) $(OTF2FLAGS) $(INCLUDES) -I$(OTFDIR)/include -I$(OTF_SOURCE) -I$(OTF_UTIL_INCLUDE) -c otf2profile.cpp
		
handlers.o: handlers.h trace2profile.h handlers.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c handlers.cpp

trace2profile.o: trace2profile.h trace2profile.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c trace2profile.cpp
	
otf2_hash_table.o: otf2_hash_table.h
	$(CC) $(TAU_CC_SPECIAL_OPTS) $(CXXFLAGS) $(OTF2FLAGS) $(INCLUDES) -I$(OTFDIR)/include -I$(OTF_SOURCE) -I$(OTF_UTIL_INCLUDE) -c otf2_hash_table.c
	
otf2_vector.o: otf2_vector.h
	$(CC) $(TAU_CC_SPECIAL_OPTS) $(CXXFLAGS) $(OTF2FLAGS) $(INCLUDES) -I$(OTFDIR)/include -I$(OTF_SOURCE) -I$(OTF_UTIL_INCLUDE) -c otf2_vector.c
	
UTILS_Error.o: UTILS_Error.h
	$(CC) $(TAU_CC_SPECIAL_OPTS) $(CXXFLAGS) $(OTF2FLAGS) $(INCLUDES) -I$(OTFDIR)/include -I$(OTF_SOURCE) -I$(OTF_UTIL_INCLUDE) -c UTILS_Error.c
	
UTILS_IO_Tools.o: UTILS_IO.h
	$(CC) $(TAU_CC_SPECIAL_OPTS) $(CXXFLAGS) $(OTF2FLAGS) $(INCLUDES) -I$(OTFDIR)/include -I$(OTF_SOURCE) -I$(OTF_UTIL_INCLUDE) -c UTILS_IO_Tools.c

clean:
	$(RM) $(TRACE2PROFILE) $(HANDLERS) $(READTRACE) $(MAKEFILE) *.o otf2profile

