# With TAU
include ../../../include/Makefile
TAUMAKEFILE=../../../include/Makefile
TAUOP='-optLinking=@../../../$(TAU_ARCH)/lib/wrappers/io_wrapper/link_options.tau'

CC=tau_cc.sh -tau_makefile=$(TAUMAKEFILE) -tau_options=$(TAUOP)

foo: foo.c
	$(CC) foo.c -o foo
clean: 
	/bin/rm -rf foo.o foo out.dat profile.* *.trc *.edf MULTI__*
