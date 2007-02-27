F90= tau_f90.sh -tau_makefile=../../../include/Makefile -tau_options='-optVerbose -optKeepFiles -optTauSelectFile=select.tau '
# You may also invoke it 
#F90= tau_f90.sh -tau_makefile=../../../include/Makefile -tau_options='-optVerbose -optKeepFiles -optDetectMemoryLeaks'

foo: foo.f90
	$(F90) foo.f90 -o foo
clean:
	/bin/rm -f foo.o foo *.pdb *.inst.f90 profile.*
