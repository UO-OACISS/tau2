include ../../include/Makefile
CC=$(TAU_CC)
CFLAGS=-acc  -g
LIBS=

jacobi: jacobi.o
	$(CC) $(CFLAGS) -o $@ $<  $(LIBS)

jacobi.o: jacobi.c
	$(CC) -c $(CFLAGS) $< 
clean:
	/bin/rm -f prof* MULT* *.trc *.edf *.o jacobi

run:
	tau_exec -T serial,pgi,pdt -openacc ./jacobi





