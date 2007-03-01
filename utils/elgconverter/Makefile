include ../../include/Makefile

CC               = $(TAU_CC)
CXX		 = $(TAU_CXX)
RM               = /bin/rm -f
LIBS		= $(TAU_TRACE_INPUT_LIB) -L$(EPILOGLIBDIR)  -lelg.base $(EPILOGEXTRALINKCMD)
CFLAGS        	= $(TAU_INCLUDE) -I$(EPILOGINCDIR) 


all:	tau2elg
install:	tau2elg

tau2elg: tau2elg.o
	$(CXX) $(CFLAGS) -o $@ tau2elg.o $(LIBS)

tau2elg.o: tau2elg.cpp 
	$(CXX) $(CFLAGS) -c tau2elg.cpp

clean:
	$(RM) tau2elg.o tau2elg

