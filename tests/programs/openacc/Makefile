include ../../../include/Makefile
CC=$(strip $(TAU_CC))
CFLAGS=-fopenacc  -g
ifeq ("$(CC)","pgcc")
	CFLAGS=-acc  -g
endif
LIBS=

all: jacobi test

jacobi: jacobi.o
	$(CC) $(CFLAGS) -o $@ $<  $(LIBS)

jacobi.o: jacobi.c
	$(CC) -c $(CFLAGS) $<

clean:
	/bin/rm -f prof* MULT* *.trc *.edf *.o jacobi

test:
	../../../$(CONFIG_ARCH)/bin/tau_exec -T serial,cupti -cupti -openacc ./jacobi
	test -f profile.0.0.0 || exit 1
	$(PPROF_CMD)

