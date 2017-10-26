# With TAU
#include ../../../../include/include/Makefile 
include ../../../include/Makefile
TAUMAKEFILE=../../../include/Makefile
TAUOP='-optLinking=@../../../$(TAU_ARCH)/lib/wrappers/io_wrapper/link_options.tau -optQuiet'

CC=tau_cc.sh -tau_makefile=$(TAUMAKEFILE) -tau_options=$(TAUOP) -g -O0
CC2=$(CONFIG_CC)

all: reader writer reader2 writer2

writer: writer.c
	$(CC) writer.c -o writer

reader: reader.c
	$(CC) reader.c -o reader

writer2: writer.c
	$(CC2) writer.c -o writer2

reader2: reader.c
	$(CC2) reader.c -o reader2

clean: 
	/bin/rm -rf writer.o writer writer2 reader.o reader reader2 *_out.dat reader*_profiles writer*_profiles linked.json preload.json
