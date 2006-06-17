include ../../include/Makefile
TAUOPTS = -optVerbose -optKeepFiles -optCompile=$(TAU_INCLUDE_MEMORY) 
CXX= $(TAU_COMPILER) $(TAUOPTS) $(TAU_CXX)

simple: simple.o
	@$(CXX) $< -o $@
simple.o: simple.cpp
	@$(CXX) -c $< 
clean:
	/bin/rm -rf simple simple.o *.inst.* *.pdb  profile.* *.trc MULTI_* *.edf
	
	

