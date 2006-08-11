include $(TAU_TEST_MAKEFILE)

CXX		= $(TAU_CXX)
CC		= $(TAU_CC)
FFLAGS        	= $(TAU_INCLUDE)  $(TAU_F90_FIXED)
LIBS            = $(TAU_LIBS)  $(TAU_CXXLIBS) $(LEXTRA1)
LDFLAGS	      	= $(USER_OPT) 
RM 	      	= /bin/rm -f
F90		= $(TAU_F90) 
TARGET	      	= simple
EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o 
	$(F90) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).f
	$(F90) $(FFLAGS) -c $(TARGET).f

clean: 	
	$(RM) $(TARGET).o $(TARGET)
##############################################
