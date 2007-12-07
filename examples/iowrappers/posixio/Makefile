# With TAU
include ../../../include/Makefile
TAUARCH=rs6000
TAUMAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(TAUARCH)/lib/Makefile.tau-pdt
TAUOP='-optCompile=-I$(TAU_PREFIX_INSTALL_DIR)/include/wrappers/posixio -optLinking=-lTauIOWrapper'

CC=tau_cc.sh -tau_makefile=$(TAUMAKEFILE) -tau_options=$(TAUOP)

foo: foo.c
	$(CC) foo.c -o foo
clean: 
	/bin/rm -rf foo.o foo out.dat profile.* *.trc *.edf MULTI__*
