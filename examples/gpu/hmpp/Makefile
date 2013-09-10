include ../../../../include/include/Makefile 
include ../../../include/Makefile

F90 = hmpp $(TAU_F90) -finstrument-functions
FLINK = hmpp tau_f90.sh

all: matmult

matmult: matmult.o
	$(FLINK) $(FLAGS) $< -o $@

matmult.o: matmult.f90
	$(F90) $(FLAGS) -c $< -o $@

clean:
	rm -f matmult matmult.o *.mod *.cu* *.so

run:
	tau_exec -T serial -cuda ./matmult
