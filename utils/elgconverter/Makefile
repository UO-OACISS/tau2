include ../../include/Makefile

CC               = $(CONFIG_CC)
CXX		 = $(CONFIG_CXX)
RM               = /bin/rm -f
LIBS		= $(TAU_TRACE_INPUT_LIB) -L$(EPILOGDIR)/lib -lelg
CFLAGS        	= $(TAU_INCLUDE)  -I$(EPILOGDIR)/include -I$(EPILOGDIR)/src 


all:	tau2elg
install:	tau2elg

tau2elg: tau2elg.o
	$(CXX) $(CFLAGS) -o $@ tau2elg.o $(LIBS)

tau2elg.o: tau2elg.cpp 
	$(CXX) $(CFLAGS) -c tau2elg.cpp

clean:
	$(RM) tau2elg.o tau2elg

