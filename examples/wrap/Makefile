CC=tau_cc.sh
CC_UNINST=gcc
LDFLAGS=
OBJS=main.o foo.o wb.o
LIBS=-L. -ldgemm -Wl,--wrap,dgemm

app: $(OBJS) libdgemm.a
	$(CC) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)
libdgemm.a: dgemm.o
	ar rcv libdgemm.a $< 
dgemm.o: dgemm.c
	$(CC_UNINST) -c $<  -I. 
.c.o: 
	$(CC) -c $<
clean:
	/bin/rm -f app *.o libdgemm.a
