#!/bin/bash -e

make clean
make

set -x
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=100
export TAU_COMM_MATRIX=1
mpirun -np 4 ./matmult
pprof
