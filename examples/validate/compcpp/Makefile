include $(TAU_TEST_MAKEFILE)

# I'm using -optLinkReset because I don't want to link in MPI, otherwise I may not
# be able to run the resuling executable
CXX		= $(TAU_COMPILER) $(TAU_CXX) -optCompInst -optNoRevert -optVerbose  -optLinkReset="$(shell tau_cc.sh -tau:showsharedlibs)"
RM 	      	= /bin/rm -f
TARGET	      	= simple

##############################################

all:		$(TARGET)	

$(TARGET):	$(TARGET).o
	$(CXX) $(TARGET).o -o $@

$(TARGET).o : $(TARGET).cc
	$(CXX) -c $(TARGET).cc

clean:
	$(RM) $(TARGET) $(TARGET).o $(TARGET).pdb $(TARGET).inst.cc


##############################################
