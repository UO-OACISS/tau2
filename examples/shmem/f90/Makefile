include ../../../include/Makefile
F90=$(TAU_COMPILER) $(TAU_F90)

INCLUDE=$(TAU_SHMEM_INCLUDE) 
LIB=$(TAU_SHMEM_LIBS)

simple: simple.f90
	$(F90) $(INCLUDE)  simple.f90 -o simple $(LIB)

clean:
	/bin/rm -f simple simple.o
