F90=tau_f90.sh -tau_makefile=../../../include/Makefile

simple: simple.f90
	$(F90) $(INCLUDE)  simple.f90 -o simple $(LIB)

clean:
	/bin/rm -rf simple.o simple *.trc *.edf profile* MULT*
