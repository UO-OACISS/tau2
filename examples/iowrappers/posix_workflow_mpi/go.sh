#!/bin/bash -x
set -e

make clean
make

export TAU_TRACK_MESSAGE=1
#export TAU_VERBOSE=1
export PROFILEDIR=reader_profiles
mpirun -np 4 tau_exec -T mpi,pdt -io ./reader &

export PROFILEDIR=writer_profiles
mpirun -np 4 tau_exec -T mpi,pdt -io ./writer

wait
tau_prof2json.py -o merged_mpi_io.json writer_profiles reader_profiles

rm -f *.dat
