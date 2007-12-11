
#SELFISHLIBS	= -L /home/anataraj/APPS/Selfish/BGW2006-prep-02-cmpd/ -ldetour-regular

TAUROOTDIR	= ../..

include $(TAUROOTDIR)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE) -DMPICH_IGNORE_CXX_SEEK

LIBS            = $(TAU_MPI_LIBS) $(TAU_LIBS) $(SELFISHLIBS) -lm 

LDFLAGS	      	= $(USER_OPT) $(TAU_LDFLAGS)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= bsp

EXTRAOBJS     	= workperiod.o

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(EXTRAOBJS) $(TARGET).o
	$(CXX) $(LDFLAGS) $(TARGET).o $(EXTRAOBJS) -o $@ $(LIBS)

%.o:%.cpp
	$(CXX) $(CFLAGS) -c $<

%.o:%.c
	$(CC) $(CFLAGS) -c $<

clean: 	
	$(RM) $(TARGET).o $(EXTRAOBJS) $(TARGET) *.trc *.edf
##############################################
