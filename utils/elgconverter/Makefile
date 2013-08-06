include ../../include/Makefile
include ../include/Makefile

CC = $(TAU_CC_FE) $(ABI) $(ISA)
CXX = $(TAU_CXX_FE) $(ABI) $(ISA)

RM               = /bin/rm -f
LIBS		= $(TAU_TRACE_INPUT_LIB) -L$(ELG_LIB_DIR)  -lelg.base $(EPILOGEXTRALINKCMD)
CFLAGS        	= $(TAU_INCLUDE) -I$(ELG_INC_DIR) 


all:	tau2elg
install:	tau2elg

tau2elg: tau2elg.o
	$(CXX) $(CFLAGS) -o $@ tau2elg.o $(LIBS)

tau2elg.o: tau2elg.cpp 
	$(CXX) $(CFLAGS) -c tau2elg.cpp

clean:
	$(RM) tau2elg.o tau2elg

