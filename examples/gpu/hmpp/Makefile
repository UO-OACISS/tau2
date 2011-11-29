include ../../../include/Makefile

F90 = hmpp $(TAU_F90)

all: matmult

matmult: matmult.o
	$(F90) $(FLAGS) $< -o $@

matmult.o: matmult.f90
	$(F90) $(FLAGS) -c $< -o $@

clean:
	rm -f matmult matmult.o *.mod *.cu* *.so

run:
	tau_exec -T serial -cuda ./matmult
