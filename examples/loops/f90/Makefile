TAUROOTDIR      = ../../..

.SUFFIXES : .f90

include $(TAUROOTDIR)/include/Makefile

# We use TAU_COMPILER (defined in the above stub Makefile) for instrumenting
# the source code. We can pass it optional arguments. See tau_compiler.sh -help

OPTS = -optVerbose  -optTauSelectFile=select.tau -optCompile=$(TAU_F90_SUFFIX) -optKeepFiles
F90=$(TAU_COMPILER) $(OPTS) $(TAU_F90)

APP=loop_test
$(APP): $(APP).o
	@$(F90) $(APP).o -o $@ 
$(APP).o: $(APP).f90
	@$(F90) -c $(APP).f90 
clean:
	/bin/rm -f *.o *.pdb *.inst.* $(APP) profile.*
