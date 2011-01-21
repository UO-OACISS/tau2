include ../../include/Makefile

CC               = $(TAU_CC)
CXX		 = $(TAU_CXX)

#CRAYCNL#CC = $(TAU_CC_FE) $(ABI) $(ISA) #ENDIF#
#CRAYCNL#CXX = $(TAU_CXX_FE) $(ABI) $(ISA) #ENDIF#

ELG_INC_DIR=$(EPILOGINCDIR)
#CRAYCNL#ELG_INC_DIR=$(EPILOGINCDIR)/../fe #ENDIF#

ELG_LIB_DIR=$(EPILOGLIBDIR) 
#CRAYCNL#ELG_LIB_DIR=$(EPILOGLIBDIR)/../fe #ENDIF#

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

