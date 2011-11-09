F90=tau_f90.sh -tau_makefile=../../../include/Makefile

rotput: rotput.f90
	$(F90) $(INCLUDE)  rotput.f90 -o rotput $(LIB)

clean:
	/bin/rm -rf rotput.o rotput *.trc *.edf profile* MULT*
