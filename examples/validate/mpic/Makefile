
include $(TAU_TEST_MAKEFILE)


CXX		= $(TAU_CXX)
CC		= $(TAU_CC)

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)
LIBS            = $(TAU_MPI_LIBS) $(TAU_LIBS) $(LEXTRA1) $(EXTRALIBS) -lm
LDFLAGS	      	= $(USER_OPT)  $(TAU_LDFLAGS)

TARGET	      	= simple

EXTRAOBJS     	= 

RM		= /bin/rm -f


##############################################

all:		$(TARGET)	

install: 	$(TARGET)


OBJS	= simple.o

$(TARGET):	$(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

# Compilation rule
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

#
#$(TARGET).o : $(TARGET).cpp
#	$(CXX) $(CFLAGS) -c $(TARGET).c

clean: 	
	$(RM) $(OBJS) $(TARGET) $(OBJS:.o=.inst.c) \
	profile.* tautrace.* events.* *.elg
##############################################
