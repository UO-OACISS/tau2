include $(TAU_TEST_MAKEFILE)

# I'm using -optLinkReset because I don't want to link in MPI, otherwise I may not
# be able to run the resuling executable
CC		= $(TAU_COMPILER) $(TAU_CC) -optKeepFiles -optNoRevert -optVerbose -optTauSelectFile=select.tau
# -optLinkReset="$(TAU_LIBS) $(TAU_CXXLIBS)" -optTauSelectFile=select.tau
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
