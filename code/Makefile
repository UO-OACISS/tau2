

all: build run


build:
	cd wrapper ; make clean ; make
	cd tau_nexus ; make clean ; make

run:
	cd release ; rm -rf profile.0.0.* gpu_* job.pbs.*
	cd release ; qsub job.pbs
