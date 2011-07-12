include ../../../include/Makefile
CXX=$(TAU_COMPILER) $(TAU_CXX)

INCLUDE=$(TAU_SHMEM_INCLUDE) 
LIB=$(TAU_SHMEM_LIBS)
# IMPORTANT NOTE: The <taudir>/<arch>/bin directory should be in your path.
# Please set TAU_MAKEFILE environment variable to point to your stub Makefile
# that is typically found in <taudir>/<arch>/lib/Makefile.tau-<options> 
# You may also pass parameters to the above script by setting the environment
# variable TAU_OPTIONS. Please see tau_compiler.sh -help for a full listing
# and refer to the README file in this directory for more information. 

simple: simple.cpp
	$(CXX) $(INCLUDE)  simple.cpp -o simple $(LIB)

clean:
	/bin/rm -rf simple.o simple *.trc *.edf profile* MULT*
