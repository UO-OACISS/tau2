#remember to load the craype-accel-nvidia* module.

F90 = ftn
FLAGS = -rm -dynamic

all: matmult

matmult: matmult.o
	$(F90) $(FLAGS) $< -o $@

matmult.o: matmult.f90
	$(F90) $(FLAGS) -c $< -o $@

clean:
	rm -f matmult matmult.o

run:
	./matmult
