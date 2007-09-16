include ../../../../../include/Makefile
OPT=-optVerbose -optTauSelectFile=select.tau -optKeepFiles
CC=$(TAU_COMPILER) $(OPT) icc
CXX=$(TAU_COMPILER) $(OPT) icpc
F90=$(TAU_COMPILER) $(OPT) ifort

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
