include ../../../include/Makefile
TAUOPTS = -optVerbose -optKeepFiles -optDetectMemoryLeaks
CC= $(TAU_COMPILER) $(TAUOPTS) $(TAU_CC)

simple: simple.o
	@$(CC) $< -o $@
simple.o: simple.c
	@$(CC) -c $< 
clean:
	/bin/rm -rf simple simple.o *.inst.* *.pdb  profile.* *.trc MULTI_* *.edf
	
	

