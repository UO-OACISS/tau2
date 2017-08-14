# With TAU

CC=mpicc

all: reader writer

writer: writer.c
	$(CC) writer.c -o writer

reader: reader.c
	$(CC) reader.c -o reader

clean: 
	/bin/rm -rf writer.o writer reader.o reader *.dat reader*_profiles writer*_profiles *.json
