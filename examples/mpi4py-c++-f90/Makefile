TAU = 0

ifeq ($(TAU),0)
CXX           = mpicxx
CC            = mpicc
F90           = ifort
MPIF90        = mpif90
else
CXX           = tau_cxx.sh
CC            = tau_cc.sh
F90           = tau_f90.sh
MPIF90        = tau_f90.sh
endif
SWIG          = swig
CP            = cp -f

#### Defns ##########
CPPFLAGS      = 
CXXFLAGS      = -fPIC -fpermissive -g
CFLAGS        = -fPIC -g
FFLAGS        = -fPIC -g 
LDFLAGS       = -lifcore


PYTHON_INC    = $(shell python-config --includes)
NUMPY_INC     = -I$(shell python-config --prefix)/lib/python2.7/site-packages/numpy/core/include

#### Codes ##########
CXX_OBJS  = SAMINT.o

F90_OBJS  = fortmthds.o

PY_OBJS   = pycintfc.o samint_wrap.o

#### Rules ##########
all:	$(CXX_OBJS) $(F90_OBJS) $(PY_OBJS)
	$(CXX) -fPIC -shared $(PY_OBJS) $(CXX_OBJS) $(F90_OBJS) -o _samint.so $(LDFLAGS)

pycintfc.o: pycintfc.C
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c pycintfc.C $(PYTHON_INC)

samint_wrap.o: samint_wrap.c
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c samint_wrap.c $(PYTHON_INC) $(NUMPY_INC)

samint_wrap.c: samint.i
	$(SWIG) -python -I$(SWIG_LIB) -I$(SWIG_LIB)/python samint.i

fortmthds.o: fortmthds.f
	$(F90) $(FFLAGS) -c fortmthds.f

clean:
	$(RM) samint.py samint_wrap.c _samint.so
	$(RM) *.o *.pyc *~
	$(RM) samint_wrap.c samint.py

