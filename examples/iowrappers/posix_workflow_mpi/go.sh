#!/bin/bash -x
set -e

make

#export TAU_VERBOSE=1
export PROFILEDIR=reader_profiles
mpirun -np 1 tau_exec -T mpi,pthread -io ./reader &

export PROFILEDIR=writer_profiles
mpirun -np 1 tau_exec -T mpi,pthread -io ./writer

wait
tau_prof2json.py -o preload.json writer_profiles reader_profiles

