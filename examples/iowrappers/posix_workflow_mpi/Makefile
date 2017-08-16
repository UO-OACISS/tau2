# With TAU

CC=mpicc

all: reader writer

writer: writer.c common.h
	$(CC) writer.c -o writer -I.

reader: reader.c common.h
	$(CC) reader.c -o reader -I.

clean: 
	/bin/rm -rf writer.o writer reader.o reader *.dat reader*_profiles writer*_profiles *.json
