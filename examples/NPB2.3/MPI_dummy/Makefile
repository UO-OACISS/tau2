# Makefile for MPI dummy library. 
# Must be edited for a specific machine. Does NOT read in 
# the make.def file of NPB 2.3

# Enable if either Cray or IBM: (no such flag for most machines: see wtime.h)
# MACHINE	=	-DCRAY
# MACHINE	=	-DIBM

libmpi.a: mpi_dummy.o mpi_dummy_c.o wtime.o
	ar r libmpi.a mpi_dummy.o mpi_dummy_c.o wtime.o

mpi_dummy.o: mpi_dummy.f mpif.h
	f77 -c mpi_dummy.f
# For a Cray C90, try:
#	cf77 -dp -c mpi_dummy.f
# For an IBM 590, try:
#	xlf -c mpi_dummy.f

mpi_dummy_c.o: mpi_dummy.c mpi.h
	cc -c ${MACHINE} -o mpi_dummy_c.o mpi_dummy.c

wtime.o: wtime.c
# For most machines or CRAY or IBM
	cc -c ${MACHINE} wtime.c
# For a precise timer on an SGI Power Challenge, try:
#	cc -o wtime.o -c wtime_sgi64.c

test: test.f
	f77 -o test -I. test.f -L. -lmpi



clean: 
	- rm -f *~ *.o
	- rm -f test libmpi.a
