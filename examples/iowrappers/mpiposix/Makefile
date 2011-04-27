# With TAU
include ../../../include/Makefile
TAUOP='-optTrackIO -optVerbose'

CC=tau_cc.sh -tau_makefile=$(TAU_MAKEFILE) -tau_options=$(TAUOP)

run: $(TAU_LIB_DIR)/libtau$(TAU_CONFIG) foo
	export TAU_TRACK_IO_PARAMS=1
	mpirun -np 2 ./foo
	pprof 

$(TAU_LIB_DIR)/libtau$(TAU_CONFIG): clean

foo: foo.c
	$(CC) foo.c -o foo
clean: 
	/bin/rm -rf foo.o foo out*.dat profile.* *.trc *.edf MULTI__*
