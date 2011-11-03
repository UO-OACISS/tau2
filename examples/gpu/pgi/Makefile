F90 = tau_f90.sh
FLAGS = -ta=nvidia,time -Minfo

all: matmult

matmult: matmult.o
	$(F90) $(FLAGS) $< -o $@

matmult.o: matmult.f90
	$(F90) $(FLAGS) -c $< -o $@

clean:
	rm -f matmult matmult.o

run:
	./matmult
