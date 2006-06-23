include ../../include/Makefile
TAUOPTS = -optVerbose -optKeepFiles -optDetectMemoryLeaks
CXX= $(TAU_COMPILER) $(TAUOPTS) $(TAU_CXX)

simple: simple.o
	@$(CXX) $< -o $@
simple.o: simple.cpp
	@$(CXX) -c $< 
clean:
	/bin/rm -rf simple simple.o *.inst.* *.pdb  profile.* *.trc MULTI_* *.edf
	
	

