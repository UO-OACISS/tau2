include $(TAU_TEST_MAKEFILE)

# I'm using -optLinkReset because I don't want to link in MPI, otherwise I may not
# be able to run the resulting executable
CC		= $(TAU_COMPILER) $(TAU_CC) -optCompInst -optNoRevert -optVerbose 
# -optLinkReset="$(TAU_LIBS) $(TAU_CXXLIBS)"
RM 	      	= /bin/rm -f
TARGET	      	= simple

##############################################

all:		$(TARGET)	

$(TARGET):	$(TARGET).o
	$(CC) $(TARGET).o -o $@

$(TARGET).o : $(TARGET).c
	$(CC) -c $(TARGET).c

clean:
	$(RM) $(TARGET) $(TARGET).o $(TARGET).pdb $(TARGET).inst.c


##############################################
