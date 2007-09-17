include /home/sameer/tau2/include/Makefile
OPT=-optVerbose -optTauSelectFile=select.tau -optKeepFiles
CC=$(TAU_COMPILER) $(OPT) $(TAU_CC)
CXX=$(TAU_COMPILER) $(OPT) $(TAU_CXX)
F90=$(TAU_COMPILER) $(OPT) $(TAU_F90)

all: foo 

foo: foo.o 
	$(CXX) foo.o -o foo
foo.o: foo.cpp
	$(CXX) -c foo.cpp
clean:
	/bin/rm -f foo *.o *.inst.* *.pdb profile.*
run: foo
	./foo
	pprof 
	cat select.tau
