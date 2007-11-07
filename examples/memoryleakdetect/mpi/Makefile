F90= tau_f90.sh -tau_makefile=../../../include/Makefile -tau_options='-optVerbose -optKeepFiles -optTauSelectFile=select.tau '
# You may also invoke it 
#F90= tau_f90.sh -tau_makefile=../../../include/Makefile -tau_options='-optVerbose -optKeepFiles -optDetectMemoryLeaks'

matrix: matrix.f90
	$(F90) matrix.f90 -o matrix
clean:
	/bin/rm -f matrix.o matrix *.pdb *.inst.f90 profile.*
