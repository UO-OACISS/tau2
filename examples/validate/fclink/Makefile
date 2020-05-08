include $(TAU_TEST_MAKEFILE)

CXX		= $(TAU_CXX)
CC		= $(TAU_CC)
FFLAGS        	= $(TAU_INCLUDE) $(TAU_F90_FIXED) $(TAU_SHMEM_INC)
LIBS            = $(shell tau_cc.sh -tau:showsharedlibs) $(TAU_FORTRANLIBS)
#$(TAU_MPI_FLIBS) $(TAU_LIBS) $(TAU_FORTRANLIBS) $(TAU_CXXLIBS) $(LEXTRA1) $(EXTRALIBS) $(TAU_SHMEM_LIB)

LDFLAGS	      	= $(USER_OPT) $(TAU_F90_OPT)
RM 	      	= /bin/rm -f
F90		= $(TAU_F90)  
TARGET	      	= simple
EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o  
	$(CC) $(LDFLAGS) $(TARGET).o  -o $@ $(LIBS)

$(TARGET).o : $(TARGET).f
	$(F90) $(FFLAGS) -c $(TARGET).f

clean: 	
	$(RM) $(TARGET).o $(TARGET)
##############################################
