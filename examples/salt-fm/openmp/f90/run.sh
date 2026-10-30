#!/bin/bash -e

make clean
make

set -x
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=100
export OMP_NUM_THREADS=4
export TAU_OMPT_SUPPORT_LEVEL=full
./matmult
pprof
