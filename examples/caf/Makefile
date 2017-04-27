F90=caf
FFLAGS=-g -O3

pi: pi_ca.f90
	$(F90)  $< -o $@ $(FFLAGS)
run: pi
	@echo "Running uninstrumented binary:"
	cafrun -n 4 ./pi
tau: pi
	@echo "Running uninstrumented binary with tau_exec:"
	cafrun -n 4 tau_exec -T mpi,pdt -ebs ./pi
	paraprof &
	
clean:
	/bin/rm -f profile* foo pi
